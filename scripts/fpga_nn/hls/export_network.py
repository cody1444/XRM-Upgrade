#!/usr/bin/env python3

"""
Export a trained network (train_nn.py) for hls4ml, as plain weights.

Run in the xrm-tf environment. The hls4ml environment has Keras 2, which
can't read Keras 3 .keras files, so the weights travel as an .npz and
convert_hls4ml.py rebuilds the network there.

The input standardization and output scaling around the network are folded
into the first and last layer, so the exported network takes the 16
amplitudes divided by their sum and returns sigma_y and mu in um:

    first layer:  W x' + b  with  x' = (x - mean) / std
               =  (W / std) x + (b - W mean / std)
    last layer:   (W h + b) * scale + centre
               =  (W * scale) h + (b * scale + centre)

Also saves real held-out bunches (inputs and the original network's
outputs) to check the HLS version against.
"""

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from xrm_generator import DATA_DIR, DEFAULT_CALIBRATION
from train_nn import load_real, normalize

HLS_DIR = DATA_DIR / "hls"


def parse_args():
    parser = argparse.ArgumentParser(prog="export_network", description="Export a network for hls4ml.")
    parser.add_argument("--model-dir", default=str(DATA_DIR / "nn_deviations"), help="Folder with the trained network.")
    parser.add_argument("--roster", default="centre_12_27")
    parser.add_argument("--n-check", type=int, default=20000, help="Real bunches saved for checking.")
    parser.add_argument("-o", "--output", default=str(HLS_DIR / "centre_12_27_deviations.npz"))
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    model = keras.models.load_model(model_dir / f"{args.roster}.keras")
    scaler = np.load(model_dir / f"{args.roster}_scaler.npz")
    roster = list(scaler["roster"])
    mean, std = scaler["mean"], scaler["std"]
    scale, centre = scaler["label_scale"], scaler["label_centre"]

    dense = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    weights = [layer.get_weights() for layer in dense]          # [(W (n_in, n_out), b), ...]
    activations = [layer.get_config()["activation"] for layer in dense]
    print("Layers: " + " -> ".join([str(len(roster))] + [f"{w.shape[1]} ({a})" for (w, _), a in zip(weights, activations)]))

    # Fold the standardization and the output scaling.
    w, b = weights[0]
    weights[0] = (w / std[:, None], b - (mean / std) @ w)
    w, b = weights[-1]
    weights[-1] = (w * scale[None, :], b * scale + centre)

    # Real held-out bunches, as in train_nn.py, and what the original network says.
    held_out = list(np.load(DEFAULT_CALIBRATION)["held_out_sweeps"])[::2]
    amps, fit_sigma, _, _, _ = load_real(held_out)
    pick = np.random.default_rng(0).choice(len(amps), size=min(args.n_check, len(amps)), replace=False)
    x = normalize(amps[pick], roster)
    original = model.predict((x - mean) / std, verbose=0, batch_size=4096) * scale + centre

    # Check the folded weights give the same result in plain numpy.
    h = x
    for (w, b), a in zip(weights, activations):
        h = h @ w + b
        if a == "relu":
            h = np.maximum(h, 0)
    diff = np.abs(h - original).max(axis=0)
    print(f"Folded vs original network on {len(x)} real bunches: max difference "
          f"sigma_y {diff[0]:.2e} um, mu {diff[1]:.2e} um")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"w{i}": w for i, (w, _) in enumerate(weights)} | {f"b{i}": b for i, (_, b) in enumerate(weights)}
    np.savez(out, **arrays, activations=np.array(activations), roster=roster,
             check_x=x.astype(np.float32), check_y=original.astype(np.float32), check_fit_sigma=fit_sigma[pick],
             source=str(model_dir / f"{args.roster}.keras"))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
