"""Flask demo: draw a digit in the browser, classify it on the FPGA accelerator.

Run on the PYNQ-Z2 board (needs the `pynq` package):

    python3 app.py                    # http://<board-ip>:5000
    CNN_BIT=/path/to/cnn_mnist.bit python3 app.py

cnn_mnist.hwh must sit next to the .bit file (same base name).
"""
import base64
import io
import os
import re

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from pynq import Overlay, allocate

BIT_PATH = os.environ.get(
    "CNN_BIT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cnn_mnist.bit"))

app = Flask(__name__)

overlay = Overlay(BIT_PATH)
dma_send = overlay.axi_dma_0.sendchannel
dma_recv = overlay.axi_dma_0.recvchannel
input_buffer = allocate(shape=(784,), dtype=np.uint8)
output_buffer = allocate(shape=(1,), dtype=np.uint8)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predict():
    base64str = request.get_json(force=True)["base64str"]
    imgstr = re.search(r"base64,(.*)", str(base64str)).group(1)
    img = Image.open(io.BytesIO(base64.b64decode(imgstr))).convert("L").resize((28, 28))

    input_buffer[:] = np.array(img, dtype=np.uint8).reshape(784)
    # Arm the receive channel first, then stream the image in.
    dma_recv.transfer(output_buffer)
    dma_send.transfer(input_buffer)
    dma_send.wait()
    dma_recv.wait()

    return jsonify({"prediction": str(int(output_buffer[0]))})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
