# Handwritten Digit Recognition with a CNN on FPGA (PYNQ-Z2)

A small CNN (796 parameters) is trained with PyTorch on MNIST, quantized to int8, implemented as an RTL accelerator
(Verilog) with an AXI-Stream / AXI DMA interface, and deployed on the **PYNQ-Z2** board (Zynq-7000 `xc7z020clg400-1`).

```
28x28 image (uint8) -> conv1 5x5x3 -> maxpool 2x2 + ReLU -> conv2 5x5 (3->3) -> maxpool 2x2 + ReLU -> FC 48->10 -> argmax
                          24x24x3          12x12x3               8x8x3              4x4x3            10
```

## Results

| Model | Accuracy (10,000 MNIST test images) |
|---|---|
| PyTorch float (`model/cnn_mnist.pt`) | 96.35 % |
| **Hardware RTL (int8, `.mem` set in the Vivado project)** | **86.50 %** |

- The 1000 images used by the testbench (`sim/testvector/input_1000.txt`) are taken from the MNIST **training** set; the RTL
  scores 90.0 % on them (`sim/results/simulate_1000_axis_cnn_mnist_1000_tb.log`). This is not a general figure; use the
  86.5 % above.
- The gap to the float model comes from post-training quantization (`trunc(w*128)` int8 weights, no QAT).

Resources and timing (PYNQ-Z2, Vivado 2025.1, PL clock **50 MHz**, `impl_1`):
LUT 21,297 (40.0 %), FF 14,278 (13.4 %), BRAM 2.5 (1.8 %), **DSP 205/220 (93.2 %)**, WNS +4.27 ns, estimated power 1.58 W.
The core needs about 1281 cycles per image (~25.6 us at 50 MHz).

### Verification status

| Item | Status |
|---|---|
| RTL vs. integer model `tools/golden_model.py` | Verified: 1000/1000 decisions identical to the simulation log |
| PYNQ-Z2 bitstream uses the `.mem` set in `vivado/` | Inferred from the Vivado synthesis log; not checked directly |
| Run on a physical PYNQ-Z2 | Not done in this release (no board available); `deploy/pynq_z2/` has not been run |
| Testbenches `top_tb.v`, `top_tb_1000.v`, `axis_cnn_mnist_tb.v` (paths fixed) | Not re-run; only `axis_cnn_mnist_1000_tb.v` has a simulation log |

### Note on the weights

The training notebook (`model/cnn_mnist.ipynb`) exports a `.mem` set to `model/mem_export/` that **differs** from the `.mem`
set inside the RTL / bitstream. Loading the notebook export into the RTL would drop hardware accuracy to **70.6 %**
(measured with `golden_model.py`). The repository therefore keeps the RTL and bitstream as built (86.5 %); the `.pt` that
produced the `.mem` set in the RTL is not part of this repository. Retraining with better quantization (e.g. QAT) and
rebuilding the bitstream is a natural next step.

## Repository layout

```
model/                  PyTorch training and weight export
  cnn_mnist.ipynb         train, save .pt, export .mem (x128, two's complement, hex)
  base64_test.ipynb       inference from a base64 image (same data format the web demo sends)
  cnn_mnist.pt            float model
  mem_export/             .mem files exported by the notebook (differ from the RTL set, see above)
  bmp/                    20 sample MNIST images (train_0..19)
vivado/NhanDienChuVietTay/   Vivado 2025.1 project for PYNQ-Z2 (sources only)
  *.srcs/sources_1/imports/module   RTL + .mem (the weights used by the bitstream)
  *.srcs/sources_1/bd               block design (PS7, AXI DMA, FIFOs, axis_cnn_mnist)
  *.srcs/sim_1/imports/testbench    testbenches
sim/
  testvector/             input_1000.txt (1000 images), 3_0.txt (one image of digit 3)
  results/                RTL simulation log
deploy/pynq_z2/         cnn_mnist.bit + cnn_mnist.hwh, hw_test.ipynb, app/ (Flask demo)
tools/golden_model.py   bit-accurate integer model of the RTL (numpy only)
```

## Usage

**Check hardware accuracy without a board or simulator**
```bash
# needs the raw MNIST files: run the dataset download cell of model/cnn_mnist.ipynb (creates model/data/MNIST/raw)
python tools/golden_model.py \
    --mem-dir vivado/NhanDienChuVietTay/NhanDienChuVietTay.srcs/sources_1/imports/module \
    --mnist-dir model/data/MNIST/raw \
    --vectors sim/testvector/input_1000.txt \
    --sim-log sim/results/simulate_1000_axis_cnn_mnist_1000_tb.log
```
Expected output: `86.50%`, `90.0%`, `1000/1000`.

**Windows:** some IP files in `vivado/` have long paths (~160 characters). If `git clone` reports *Filename too long*, run
`git config --global core.longpaths true` or clone into a short directory (for example `C:\cnn`).

**Training and weight export:** `pip install -r requirements.txt`, then open `model/cnn_mnist.ipynb`.

**RTL simulation (Vivado xsim):** open `vivado/NhanDienChuVietTay/NhanDienChuVietTay.xpr` (first time: *Generate Output
Products* for the block design). The testbenches read their data files by relative name, so copy
`sim/testvector/input_1000.txt` (and `3_0.txt`) into the simulation run directory
`NhanDienChuVietTay.sim/sim_1/behav/xsim/` before running `axis_cnn_mnist_1000_tb`.

**Run on the PYNQ-Z2:** copy the repository to the board and open `deploy/pynq_z2/hw_test.ipynb` (loads the bitstream, tests
one image, compares 1000 images against the simulation, measures timing). Web demo: `python3 deploy/pynq_z2/app/app.py`,
then open `http://<board-ip>:5000` and draw a digit.

## Core interface (`axis_cnn_mnist`)

- Input: 8-bit AXI-Stream, 784 pixels (0..255) in raster order; `tlast` is not used.
- Output: one 8-bit beat (digit 0-9) with `tlast`.
- The core ignores `m_axis_tready` (no back-pressure); the block design places an `axis_data_fifo` on the output to buffer
  the result.

## Origin and project scope

- **Origin:** the RTL CNN core and the train -> export `.mem` -> simulate flow are based on the design by Bo Young Kang
  (weenslab), developed from the `boaaaang/CNN-Implementation-in-Verilog` repository (2021). The copyright headers in the
  `.v` files are kept unchanged.
- **CNN training:** used as provided (`model/cnn_mnist.ipynb`); no change to the architecture or the algorithm.
- **Project changes:** the design was moved to the **PYNQ-Z2** board (Zynq-7000, `xc7z020`); the original targeted the Kria
  KV260 (Zynq UltraScale+). Only the integration had to be redone: a block design with `processing_system7` instead of
  `zynq_ultra_ps_e`, AXI DMA, interconnect / SmartConnect, AXI-Stream FIFOs, a 50 MHz PL clock, and a Vivado 2025.1 build.
  The CNN core (`axis_cnn_mnist`) is unchanged.
- **Added for this release:** `tools/golden_model.py`, the notebook and app for the PYNQ-Z2 (`deploy/pynq_z2/`), testbench
  path fixes, and this document.
