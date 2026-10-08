#!/usr/bin/env python3

"""
True vs predicted sigma_y for the centre 16-channel network (train_nn.py).

A  generated bunches (fresh draws, not used in training): true sigma_y is known
B  real held-out bunches: no true value, so the offline template fit is on x
C  real held-out sweeps: CMOS sigma_y on x, sweep medians of network and fit on y

draw_row() draws these three panels for any network and generator; it is
also used by archive/plot_compare_step3.py.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from xrm_generator import XrmGenerator, DEFAULT_CALIBRATION
from train_nn import OUTPUT_DIR, SIGMA_RANGE_UM, load_real, robust_spread
from plot_nn_results import load_model, style, INK, GRID

ROSTER = "centre_12_27"
NETWORK, FIT = "#2a78d6", "#eb6834"


def parse_args():
    parser = argparse.ArgumentParser(prog="plot_true_vs_predicted", description="True vs predicted sigma_y.")
    parser.add_argument("-o", "--output", default=str(OUTPUT_DIR / "true_vs_predicted.png"))
    parser.add_argument("--n-generated", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=99, help="Seed for the generated test bunches.")
    parser.add_argument("--model-dir", default=str(OUTPUT_DIR), help="Folder with the trained network.")
    return parser.parse_args()


def load_real_test():
    """Real held-out bunches (same half as train_nn.py)."""
    held_out = list(np.load(DEFAULT_CALIBRATION)["held_out_sweeps"])[::2]
    return load_real(held_out)


def diagonal(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls="--", label="predicted = true")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")


def density(ax, x, y, lim, vmax):
    """2D histogram of bunches, one hue light->dark, log count scale."""
    *_, image = ax.hist2d(x, y, bins=100, range=[lim, lim], cmap="Blues", norm=LogNorm(vmin=1, vmax=vmax), cmin=1)
    return image


def draw_row(fig, ax, predict, gen, real, n_generated, seed, label=""):
    """Panels A-C for one network; gen makes its generated test bunches."""
    amps, fit_sigma, _, sweep_idx, cmos = real
    prefix = f"{label}: " if label else ""

    # A: generated
    rng = np.random.default_rng(seed + 1)
    gen.rng = np.random.default_rng(seed)
    x_gen, _, sig_gen, _ = gen.sample(rng.uniform(*SIGMA_RANGE_UM, n_generated))
    pred_gen = predict(x_gen)[:, 0]
    real_lo, real_hi = gen.donor_params[:, 1].min(), gen.donor_params[:, 1].max()

    a = ax[0]
    a.axvspan(real_lo, real_hi, color=GRID, alpha=0.5, lw=0, label=f"range of real beam sizes ({real_lo:.0f}-{real_hi:.0f} um)")
    image = density(a, sig_gen, pred_gen, SIGMA_RANGE_UM, vmax=300)
    diagonal(a, *SIGMA_RANGE_UM)
    err = pred_gen - sig_gen
    style(a, f"A  {prefix}generated bunches, true value known\n"
             f"   error {np.mean(err):+.2f} um mean, {np.std(err):.2f} um spread ({len(err)} bunches)",
          "true sigma_y (um)", "network sigma_y (um)")
    a.legend(frameon=False, fontsize=9, loc="upper left")
    fig.colorbar(image, ax=a, shrink=0.75, label="bunches per bin")

    # B: real bunches vs offline fit
    pred_real = predict(amps)[:, 0]
    a = ax[1]
    lim = (30, 110)
    image = density(a, fit_sigma, pred_real, lim, vmax=1500)
    diagonal(a, *lim)
    diff = pred_real - fit_sigma
    style(a, f"B  {prefix}real bunches, offline fit instead of truth\n"
             f"   network - fit {np.median(diff):+.2f} um median, {robust_spread(diff):.2f} um spread "
             f"({len(diff)} bunches)",
          "offline fit sigma_y (um)", "network sigma_y (um)")
    a.legend(frameon=False, fontsize=9, loc="upper left")
    fig.colorbar(image, ax=a, shrink=0.75, label="bunches per bin")

    # C: real sweeps vs CMOS
    sweeps = np.unique(sweep_idx)
    nn_sweep = np.array([np.median(pred_real[sweep_idx == s]) for s in sweeps])
    fit_sweep = np.array([np.median(fit_sigma[sweep_idx == s]) for s in sweeps])
    cmos_sweep = cmos[sweeps]
    ok = np.isfinite(cmos_sweep)
    d_nn, d_fit = nn_sweep[ok] - cmos_sweep[ok], fit_sweep[ok] - cmos_sweep[ok]

    a = ax[2]
    lim = (40, 80)
    a.plot(cmos_sweep[ok], fit_sweep[ok], "o", ms=8, color=FIT, mec="white", mew=1.5,
           label=f"offline fit: {np.median(d_fit):+.2f} +- {np.std(d_fit):.2f} um")
    a.plot(cmos_sweep[ok], nn_sweep[ok], "o", ms=8, color=NETWORK, mec="white", mew=1.5,
           label=f"network: {np.median(d_nn):+.2f} +- {np.std(d_nn):.2f} um")
    diagonal(a, *lim)
    style(a, f"C  {prefix}real sweeps, CMOS camera on x\n   median per sweep ({ok.sum()} sweeps)",
          "CMOS sigma_y (um)", "sweep median sigma_y (um)")
    a.legend(frameon=False, fontsize=9, loc="upper left", title="vs CMOS:", title_fontsize=9)

    within = np.median([robust_spread(pred_real[sweep_idx == s]) for s in sweeps])
    return {"generated error spread": np.std(err), "network - fit median": np.median(diff),
            "network - fit spread": robust_spread(diff), "within-sweep scatter": within,
            "vs CMOS": (np.median(d_nn), np.std(d_nn))}


def generator_for(model_dir):
    """The generator, with the extras switched on as they were when the network was trained."""
    scaler = np.load(model_dir / f"{ROSTER}_scaler.npz")
    factors = bool(scaler["factors"]) if "factors" in scaler.files else False
    deviations = bool(scaler["deviations"]) if "deviations" in scaler.files else False
    return XrmGenerator(DEFAULT_CALIBRATION, factors=factors, deviations=deviations), factors, deviations


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    gen, factors, deviations = generator_for(model_dir)
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.6))
    draw_row(fig, ax, load_model(ROSTER, model_dir), gen, load_real_test(), args.n_generated, args.seed)
    extras = [name for name, on in (("channel factors", factors), ("deviations", deviations)) if on]
    fig.suptitle(f"Centre 16-channel network (12-27), trained on the generator "
                 f"({' + '.join(extras) if extras else 'no extras'})",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(args.output, dpi=110, facecolor="white")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
