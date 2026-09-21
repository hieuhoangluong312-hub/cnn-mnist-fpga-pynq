#!/usr/bin/env python3
"""Bit-accurate integer model of the Verilog CNN accelerator (numpy only).

Mirrors the arithmetic in the RTL so accuracy can be checked without a board or
a simulator:

  conv1 : sum(pixel_u8 * w_s8) >> 8  (12-bit wrap) + bias
  pool1 : 2x2 max, then ReLU
  conv2 : sum over 3 input channels, >> 6 (14-bit), then >> 1, 12-bit wrap, + bias
  pool2 : 2x2 max, then ReLU
  fc    : (sum(w_s8 * act) + bias) >> 7  (12-bit)
  out   : first index of the maximum (comparator.v)

Usage
-----
  # accuracy of a set of .mem files on the MNIST test set
  python tools/golden_model.py --mem-dir vivado/NhanDienChuVietTay/NhanDienChuVietTay.srcs/sources_1/imports/module \
      --mnist-dir model/data/MNIST/raw

  # additionally compare with the RTL simulation log of axis_cnn_mnist_1000_tb
  python tools/golden_model.py --mem-dir <dir> --vectors sim/testvector/input_1000.txt \
      --sim-log sim/results/simulate_1000_axis_cnn_mnist_1000_tb.log

  # accuracy of a PyTorch checkpoint quantised like the notebook (trunc(w*128))
  python tools/golden_model.py --pt model/cnn_mnist.pt --mnist-dir model/data/MNIST/raw
"""
import argparse
import pickle
import re
import struct
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv


# ----------------------------------------------------------------- weights
def _read_mem(path):
    v = np.array([int(t, 16) for t in Path(path).read_text().split()], dtype=np.int64)
    return np.where(v > 127, v - 256, v)


def load_mem_dir(d):
    d = Path(d)
    c1w = np.stack([_read_mem(d / f"conv1_weight_{i}.mem").reshape(5, 5) for i in (1, 2, 3)])[:, None]
    c2w = np.stack([np.stack([_read_mem(d / f"conv2_weight_{o}{i}.mem").reshape(5, 5) for i in (1, 2, 3)])
                    for o in (1, 2, 3)])
    return dict(c1w=c1w, c1b=_read_mem(d / "conv1_bias.mem"),
                c2w=c2w, c2b=_read_mem(d / "conv2_bias.mem"),
                fw=_read_mem(d / "fc_weight.mem").reshape(10, 48), fb=_read_mem(d / "fc_bias.mem"))


class _Stub:  # placeholder for torch classes while unpickling
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


class _Unpickler(pickle.Unpickler):
    _DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64}

    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix = zf, prefix

    def find_class(self, mod, name):
        if name == "_rebuild_tensor_v2":
            def rebuild(storage, offset, size, stride, *_):
                s = [x * storage.itemsize for x in stride]
                return np.lib.stride_tricks.as_strided(storage[offset:], shape=size, strides=s).copy()
            return rebuild
        if name == "_rebuild_parameter":
            return lambda data, *_: data
        if name in self._DT:
            return name
        if mod.startswith("torch") or mod in ("__main__", "cnn_mnist"):
            return type(name, (_Stub,), {})
        return super().find_class(mod, name)

    def persistent_load(self, pid):
        _, storage_type, key, _, _ = pid
        name = storage_type if isinstance(storage_type, str) else storage_type.__name__
        return np.frombuffer(self.zf.read(f"{self.prefix}/data/{key}"), dtype=self._DT[name])


def load_pt_quantised(path):
    """Read a torch.save(model) checkpoint without torch and quantise as the notebook does."""
    zf = zipfile.ZipFile(path)
    pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
    model = _Unpickler(zf.open(pkl), zf, pkl.rsplit("/data.pkl", 1)[0]).load()
    mods = model.__dict__["_modules"]

    def q(a):
        return np.trunc(np.asarray(a) * 128).astype(np.int64)

    p = {k: mods[k].__dict__["_parameters"] for k in ("conv1", "conv2", "fc_1")}
    return dict(c1w=q(p["conv1"]["weight"]), c1b=q(p["conv1"]["bias"]),
                c2w=q(p["conv2"]["weight"]), c2b=q(p["conv2"]["bias"]),
                fw=q(p["fc_1"]["weight"]), fb=q(p["fc_1"]["bias"]))


# ------------------------------------------------------------------- model
def _wrap(x, bits):
    m = 1 << bits
    x = np.asarray(x, dtype=np.int64) & (m - 1)
    return np.where(x >= (m >> 1), x - m, x)


def _pool_relu(a):
    n, c, h, w = a.shape
    return np.maximum(a.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5)), 0)


def _conv(x, w):
    return np.einsum("ncijkl,ockl->noij", swv(x, (5, 5), axis=(2, 3)), w)


def predict(ws, images_u8):
    """images_u8: (N, 28, 28) integers 0..255 -> predicted digits (N,)."""
    x = np.asarray(images_u8, dtype=np.int64)[:, None]
    c1 = _wrap(_wrap(_conv(x, ws["c1w"]) >> 8, 12) + ws["c1b"][None, :, None, None], 12)
    a1 = _pool_relu(c1)
    c2 = _wrap((_wrap(_conv(a1, ws["c2w"]) >> 6, 14) >> 1) + ws["c2b"][None, :, None, None], 12)
    a2 = _pool_relu(c2).reshape(len(x), -1)
    fc = _wrap((a2 @ ws["fw"].T + ws["fb"]) >> 7, 12)
    return fc.argmax(1)


# -------------------------------------------------------------------- data
def load_idx(path):
    b = Path(path).read_bytes()
    magic, n = struct.unpack(">II", b[:8])
    if magic == 2051:
        r, c = struct.unpack(">II", b[8:16])
        return np.frombuffer(b, np.uint8, offset=16).reshape(n, r, c)
    return np.frombuffer(b, np.uint8, offset=8)


def load_vectors(path):
    return np.array([int(t, 16) for t in Path(path).read_text().split()], dtype=np.int64).reshape(-1, 28, 28)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mem-dir", help="directory with the RTL .mem files")
    src.add_argument("--pt", help="torch.save(model) checkpoint, quantised as trunc(w*128)")
    ap.add_argument("--mnist-dir", help="directory with the raw MNIST idx files (t10k-*-ubyte)")
    ap.add_argument("--vectors", help="input_1000.txt used by axis_cnn_mnist_1000_tb")
    ap.add_argument("--sim-log", help="simulate.log of axis_cnn_mnist_1000_tb, to compare decision by decision")
    a = ap.parse_args()

    ws = load_mem_dir(a.mem_dir) if a.mem_dir else load_pt_quantised(a.pt)

    if a.mnist_dir:
        d = Path(a.mnist_dir)
        pred = predict(ws, load_idx(d / "t10k-images-idx3-ubyte"))
        y = load_idx(d / "t10k-labels-idx1-ubyte")
        print(f"MNIST test (10000 images): accuracy = {np.mean(pred == y) * 100:.2f}%")

    if a.vectors:
        pred = predict(ws, load_vectors(a.vectors))
        print(f"input_1000.txt: accuracy vs label (j%10) = {np.mean(pred == np.arange(len(pred)) % 10) * 100:.1f}%")
        if a.sim_log:
            dec = {int(m[1]): int(m[2]) for m in
                   (re.match(r"Input image (\d+): original value = \d+, decision = (\d+)", l)
                    for l in Path(a.sim_log).read_text().splitlines()) if m}
            sim = np.array([dec[i] for i in range(len(pred))])
            print(f"RTL simulation log: {int((pred == sim).sum())}/{len(pred)} decisions identical to this model")


if __name__ == "__main__":
    main()
