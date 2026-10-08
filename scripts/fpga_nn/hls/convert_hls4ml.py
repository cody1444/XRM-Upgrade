#!/usr/bin/env python3

"""
Convert an exported network (export_network.py) to an HLS project with hls4ml.

Run in the hls4ml environment (hls4ml 1.1, Keras 2):

    conda run -n hls4ml python scripts/fpga_nn/hls/convert_hls4ml.py

Rebuilds the network in Keras 2 from the exported weights, checks it
against the original network's outputs, converts it for the ZCU216 FPGA
(Vitis backend, fully parallel: io_parallel, Latency strategy, reuse
factor 1) and compares hls4ml's bit-accurate C simulation with Keras on
real held-out bunches, for two fixed-point choices:

  wide     ap_fixed<32,12> everywhere: checks the conversion itself
  18-bit   one FPGA multiplier per weight; precision per stage chosen from
           the value ranges (see PRECISION_18)

Input: the 16 channel amplitudes divided by their sum. Output: sigma_y and
mu in um. Projects go to data/hls/<name>_<choice>/.
"""

import argparse
from pathlib import Path

import numpy as np
import hls4ml
from tensorflow import keras

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PART = "xczu49dr-ffvf1760-2-e"    # ZCU216

# Per stage: (weight, bias, result) precision, plus the input.
# Ranges on real data: input 0.005-0.16; dense0 weights up to 86 (std folded in),
# dense1/2 weights up to 1.6, dense3 weights/bias up to 34/107 (output scale
# folded in); hidden values up to 7.6; outputs 38-136 um (range kept to +-256 um).
PRECISION_18 = {
    "input": "ap_fixed<18,1>",
    "dense0": ("ap_fixed<18,8>", "ap_fixed<18,8>", "ap_fixed<18,5>"),
    "dense1": ("ap_fixed<18,2>", "ap_fixed<18,2>", "ap_fixed<18,5>"),
    "dense2": ("ap_fixed<18,2>", "ap_fixed<18,2>", "ap_fixed<18,5>"),
    "dense3": ("ap_fixed<18,8>", "ap_fixed<18,8>", "ap_fixed<18,9>"),
    "hidden": "ap_fixed<18,5>",
}
WIDE = "ap_fixed<32,12>"


def parse_args():
    parser = argparse.ArgumentParser(prog="convert_hls4ml", description="Convert an exported network with hls4ml.")
    parser.add_argument("--network", default=str(DATA_DIR / "hls" / "centre_12_27_deviations.npz"))
    return parser.parse_args()


def rebuild(d):
    """The network as a Keras 2 model, with every layer named so its precision can be set."""
    n_layers = len(d["activations"])
    inputs = keras.Input(shape=(d["w0"].shape[0],), name="amplitudes")
    h = inputs
    for i in range(n_layers):
        h = keras.layers.Dense(d[f"w{i}"].shape[1], name=f"dense{i}")(h)
        if d["activations"][i] == "relu":
            h = keras.layers.Activation("relu", name=f"relu{i}")(h)
    model = keras.Model(inputs, h)
    for i in range(n_layers):
        model.get_layer(f"dense{i}").set_weights([d[f"w{i}"], d[f"b{i}"]])
    return model


def accum(result):
    """Sums inside a layer: the result's integer bits + 6, 6 more fractional bits."""
    width, integer = (int(v) for v in result[len("ap_fixed<"):-1].split(","))
    return f"ap_fixed<{width + 12},{integer + 6}>"


def hls_config(model, choice):
    config = hls4ml.utils.config_from_keras_model(model, granularity="name", backend="Vitis")
    config["Model"]["Strategy"] = "Latency"
    config["Model"]["ReuseFactor"] = 1
    layers = config["LayerName"]
    if choice == "wide":
        config["Model"]["Precision"] = WIDE
        for name in layers:
            layers[name]["Precision"] = WIDE
        return config

    config["Model"]["Precision"] = PRECISION_18["hidden"]
    layers["amplitudes"]["Precision"] = PRECISION_18["input"]
    for name, cfg in layers.items():
        if name.endswith("_linear"):
            # hls4ml's pass-through activation after each dense layer
            cfg["Precision"] = {"result": PRECISION_18[name[:-len("_linear")]][2]}
        elif name.startswith("dense"):
            weight, bias, result = PRECISION_18[name]
            cfg["Precision"] = {"weight": weight, "bias": bias, "result": result, "accum": accum(result)}
        elif name.startswith("relu"):
            cfg["Precision"] = {"result": PRECISION_18["hidden"]}
    return config


def compare(label, pred, ref):
    err = pred - ref
    print(f"  {label:>8s}: sigma_y error mean {err[:, 0].mean():+.4f} um, rms {np.sqrt(np.mean(err[:, 0] ** 2)):.4f} um, "
          f"max {np.abs(err[:, 0]).max():.4f} um | mu rms {np.sqrt(np.mean(err[:, 1] ** 2)):.4f} um")
    return err


def main():
    args = parse_args()
    network = Path(args.network)
    d = np.load(network)
    x = np.ascontiguousarray(d["check_x"], dtype=np.float32)
    ref = d["check_y"].astype(float)
    fit_sigma = d["check_fit_sigma"]

    model = rebuild(d)
    keras2 = model.predict(x, verbose=0, batch_size=4096)
    print(f"Rebuilt network ({network.name}) vs original on {len(x)} real bunches: max difference "
          f"{np.abs(keras2 - ref).max():.2e} um")

    for choice in ("wide", "18bit"):
        output_dir = DATA_DIR / "hls" / f"{network.stem}_{choice}"
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=hls_config(model, choice), output_dir=str(output_dir),
            backend="Vitis", part=PART, io_type="io_parallel")
        hls_model.compile()
        pred = hls_model.predict(x).reshape(ref.shape)
        print(f"\n{choice}: HLS C simulation vs Keras ({output_dir})")
        compare("vs Keras", pred, ref)
        spread_keras = 1.4826 * np.median(np.abs(ref[:, 0] - fit_sigma - np.median(ref[:, 0] - fit_sigma)))
        spread_hls = 1.4826 * np.median(np.abs(pred[:, 0] - fit_sigma - np.median(pred[:, 0] - fit_sigma)))
        print(f"  network - offline fit, per-bunch spread: Keras {spread_keras:.3f} um, HLS {spread_hls:.3f} um")


if __name__ == "__main__":
    main()
