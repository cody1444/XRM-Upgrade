#!/usr/bin/env python3

"""
True vs predicted sigma_y for the centre 16-channel network (train_nn.py).

A  generated bunches (fresh draws, not used in training): true sigma_y is known
B  real held-out bunches: no true value, so the offline template fit is on x
C  real held-out sweeps: CMOS sigma_y on x, sweep medians of network and fit on y
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from xrm_generator import XrmGenerator, DEFAULT_CALIBRATION
from train_nn import OUTPUT_DIR, SIGMA_RANGE_UM, load_real, robust_spread
from plot_nn_results import load_model, style, INK, MUTED, GRID

ROSTER = "centre_12_27"
NETWORK, FIT = "#2a78d6", "#eb6834"


def parse_args():
    parser = argparse.ArgumentParser(prog="plot_true_vs_predicted", description="True vs predicted sigma_y.")
    parser.add_argument("-o", "--output", default=str(OUTPUT_DIR / "true_vs_predicted.png"))
    parser.add_argument("--n-generated", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=99, help="Seed for the generated test bunches.")
    return parser.parse_args()


def diagonal(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls="--", label="predicted = true")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")


def density(ax, x, y, lim):
    """2D histogram of bunches, one hue light->dark, log count scale."""
    *_, image = ax.hist2d(x, y, bins=100, range=[lim, lim], cmap="Blues", norm=LogNorm(), cmin=1)
    return image


def main():
    args = parse_args()
    predict = load_model(ROSTER)

    # A: generated
    gen = XrmGenerator(DEFAULT_CALIBRATION, np.random.default_rng(args.seed))
    rng = np.random.default_rng(args.seed + 1)
    x_gen, _, sig_gen, _ = gen.sample(rng.uniform(*SIGMA_RANGE_UM, args.n_generated))
    pred_gen = predict(x_gen)[:, 0]
    real_lo, real_hi = gen.donor_params[:, 1].min(), gen.donor_params[:, 1].max()

    # B, C: real held-out sweeps (same half as train_nn.py)
    held_out = list(np.load(DEFAULT_CALIBRATION)["held_out_sweeps"])[::2]
    amps, fit_sigma, _, sweep_idx, cmos = load_real(held_out)
    pred_real = predict(amps)[:, 0]
    sweeps = np.unique(sweep_idx)
    nn_sweep = np.array([np.median(pred_real[sweep_idx == s]) for s in sweeps])
    fit_sweep = np.array([np.median(fit_sigma[sweep_idx == s]) for s in sweeps])
    cmos_sweep = cmos[sweeps]
    ok = np.isfinite(cmos_sweep)

    fig, ax = plt.subplots(1, 3, figsize=(19, 6.6))

    a = ax[0]
    a.axvspan(real_lo, real_hi, color=GRID, alpha=0.5, lw=0, label=f"range of real beam sizes ({real_lo:.0f}-{real_hi:.0f} um)")
    image = density(a, sig_gen, pred_gen, SIGMA_RANGE_UM)
    diagonal(a, *SIGMA_RANGE_UM)
    err = pred_gen - sig_gen
    style(a, f"A  Generated bunches: true value known\n"
             f"   error {np.mean(err):+.2f} um mean, {np.std(err):.2f} um spread ({len(err)} bunches)",
          "true sigma_y (um)", "network sigma_y (um)")
    a.legend(frameon=False, fontsize=9, loc="upper left")
    fig.colorbar(image, ax=a, shrink=0.75, label="bunches per bin")

    a = ax[1]
    lim = (30, 110)
    image = density(a, fit_sigma, pred_real, lim)
    diagonal(a, *lim)
    diff = pred_real - fit_sigma
    style(a, f"B  Real bunches: no true value, offline fit instead\n"
             f"   network - fit {np.median(diff):+.2f} um median, {robust_spread(diff):.2f} um spread "
             f"({len(diff)} bunches)",
          "offline fit sigma_y (um)", "network sigma_y (um)")
    a.legend(frameon=False, fontsize=9, loc="upper left")
    fig.colorbar(image, ax=a, shrink=0.75, label="bunches per bin")

    a = ax[2]
    lim = (40, 80)
    d_nn, d_fit = nn_sweep[ok] - cmos_sweep[ok], fit_sweep[ok] - cmos_sweep[ok]
    a.plot(cmos_sweep[ok], fit_sweep[ok], "o", ms=8, color=FIT, mec="white", mew=1.5,
           label=f"offline fit: {np.median(d_fit):+.2f} +- {np.std(d_fit):.2f} um")
    a.plot(cmos_sweep[ok], nn_sweep[ok], "o", ms=8, color=NETWORK, mec="white", mew=1.5,
           label=f"network: {np.median(d_nn):+.2f} +- {np.std(d_nn):.2f} um")
    diagonal(a, *lim)
    style(a, f"C  Real sweeps: CMOS camera on x\n   median per sweep ({ok.sum()} sweeps)",
          "CMOS sigma_y (um)", "sweep median sigma_y (um)")
    a.legend(frameon=False, fontsize=9, loc="upper left", title="vs CMOS:", title_fontsize=9)

    fig.suptitle("Centre 16-channel network (12-27), trained on the current generator",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(args.output, dpi=110, facecolor="white")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
