#!/usr/bin/env python3

"""
True vs predicted sigma_y, current generator vs the archived step-3 generator.

Top row:    centre network in data/nn/, current generator (the benchmark)
Bottom row: centre network in data/nn_step3/, step-3 generator
            (made by archive/train_nn_step3.py)
Panels as in plot_true_vs_predicted.py; each network's generated test
bunches come from its own generator.
"""

import sys
from pathlib import Path

FPGA_NN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FPGA_NN_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from xrm_generator import XrmGenerator, DEFAULT_CALIBRATION  # noqa: E402
from plot_nn_results import load_model, INK  # noqa: E402
from plot_true_vs_predicted import ROSTER, draw_row, load_real_test  # noqa: E402
from train_nn_step3 import CALIBRATION as STEP3_CALIBRATION, OUTPUT_DIR as STEP3_DIR, load_step3  # noqa: E402

OUTPUT = FPGA_NN_DIR / "data" / "nn_step3" / "true_vs_predicted_compare.png"
N_GENERATED, SEED = 30000, 99


def main():
    real = load_real_test()
    rows = [
        ("Current generator", load_model(ROSTER), XrmGenerator(DEFAULT_CALIBRATION)),
        ("Step-3 generator", load_model(ROSTER, STEP3_DIR), load_step3().XrmGenerator(STEP3_CALIBRATION)),
    ]

    fig, ax = plt.subplots(2, 3, figsize=(19, 13))
    for row, (label, predict, gen) in zip(ax, rows):
        m = draw_row(fig, row, predict, gen, real, N_GENERATED, SEED, label)
        print(f"{label}: generated error spread {m['generated error spread']:.2f} um; real: network - fit "
              f"{m['network - fit median']:+.2f} um median, {m['network - fit spread']:.2f} um spread, "
              f"within-sweep {m['within-sweep scatter']:.2f} um, vs CMOS {m['vs CMOS'][0]:+.2f} +- {m['vs CMOS'][1]:.2f} um")

    fig.suptitle("Centre 16-channel network (12-27): current generator (top) vs step-3 generator (bottom), seed 0",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=110, facecolor="white")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
