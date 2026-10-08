#!/usr/bin/env python3

"""
Summary plots for the networks trained by train_nn.py.

A  real held-out bunches: network sigma_y vs offline fit, per bunch
B  network - fit per bunch, for each channel set
C  one real sweep: sigma_y along the turn, fit and network
D  per sweep: median sigma_y vs CMOS, network and fit
E  generated data: predicted vs true sigma_y
F  generated data: error spread vs true sigma_y, for each channel set
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

import xrm_raw
from xrm_generator import XrmGenerator, AMPLITUDE_DIR, DEFAULT_CALIBRATION
from train_nn import OUTPUT_DIR, ROSTERS, SIGMA_RANGE_UM, load_real, normalize

MAIN = "centre_12_27"
LABELS = {"centre_12_27": "16 ch, centre 12-27",
          "every_other_6_36": "16 ch, every other 6-36",
          "all_measured_4_39": "36 ch, 4-39 (reference only)"}
COLORS = {"centre_12_27": "#2a78d6", "every_other_6_36": "#eb6834", "all_measured_4_39": "#1baf7a"}
FIT_COLOR = "#52514e"   # neutral: the comparison, not one of the networks
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def parse_args():
    parser = argparse.ArgumentParser(prog="plot_nn_results", description="Summary plots for the trained networks.")
    parser.add_argument("-o", "--output", default=str(OUTPUT_DIR / "nn_results.png"))
    parser.add_argument("--n-generated", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=99)
    return parser.parse_args()


def load_model(name, model_dir=OUTPUT_DIR):
    scaler = np.load(model_dir / f"{name}_scaler.npz")
    model = keras.models.load_model(model_dir / f"{name}.keras")

    def predict(amplitudes):
        x = (normalize(amplitudes, list(scaler["roster"])) - scaler["mean"]) / scaler["std"]
        return model.predict(x, verbose=0, batch_size=4096) * scaler["label_scale"] + scaler["label_centre"]

    return predict


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)


def main():
    args = parse_args()
    models = {name: load_model(name) for name in ROSTERS}

    held_out = list(np.load(DEFAULT_CALIBRATION)["held_out_sweeps"])[::2]
    amps, fit_sigma, _, sweep_idx, cmos = load_real(held_out)
    pred_real = {name: m(amps)[:, 0] for name, m in models.items()}

    gen = XrmGenerator(DEFAULT_CALIBRATION, np.random.default_rng(args.seed))
    real_lo, real_hi = gen.donor_params[:, 1].min(), gen.donor_params[:, 1].max()
    rng = np.random.default_rng(args.seed)
    x_gen, _, sig_gen, _ = gen.sample(rng.uniform(*SIGMA_RANGE_UM, args.n_generated))
    pred_gen = {name: m(x_gen)[:, 0] for name, m in models.items()}

    fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
    fig.patch.set_facecolor("#fcfcfb")
    plt.rcParams["font.size"] = 10

    # A: network vs fit per bunch
    a = ax[0, 0]
    rng_a = (40, 90)
    a.hist2d(fit_sigma, pred_real[MAIN], bins=120, range=[rng_a, rng_a], cmap="Blues", cmin=1)
    a.plot(rng_a, rng_a, color=INK, lw=1, ls="--")
    diff = pred_real[MAIN] - fit_sigma
    a.text(0.03, 0.95, f"{len(diff):,} bunches, {len(held_out)} sweeps\n"
                       f"difference: median {np.median(diff):+.1f} µm, "
                       f"spread {1.4826 * np.median(np.abs(diff - np.median(diff))):.1f} µm",
           transform=a.transAxes, va="top", color=INK)
    style(a, "A. Real held-out bunches: network vs offline fit", "offline fit σ_y [µm]", "network σ_y [µm]")

    # B: difference per channel set
    a = ax[0, 1]
    bins = np.linspace(-20, 20, 161)
    for name in ROSTERS:
        d = pred_real[name] - fit_sigma
        spread = 1.4826 * np.median(np.abs(d - np.median(d)))
        a.hist(d, bins=bins, histtype="step", lw=2, color=COLORS[name], density=True,
               label=f"{LABELS[name]}  (spread {spread:.1f} µm)")
    a.axvline(0, color=MUTED, lw=1)
    a.legend(loc="upper left", fontsize=9, frameon=False)
    style(a, "B. Network minus fit, per real bunch", "network − fit σ_y [µm]", "fraction of bunches (density)")

    # C: one sweep along the turn
    a = ax[0, 2]
    k = len(held_out) // 2
    d = np.load(AMPLITUDE_DIR / f"{held_out[k]}.npz")
    ok = np.all(np.isfinite(d["firmware"]), axis=1) & np.all(np.isfinite(d["fit_params"]), axis=1)
    t = d["bunch_times_ns"][ok]
    a.scatter(t, d["fit_params"][ok, 1], s=3, color=FIT_COLOR, alpha=0.6, label="offline fit (42 ch, careful processing)")
    a.scatter(t, models[MAIN](d["firmware"][ok])[:, 0], s=3, color=COLORS[MAIN], alpha=0.6,
              label="network (16 ch, firmware amplitudes)")
    c = float(np.load(xrm_raw.REAL_DIR / "runs" / f"{held_out[k]}.npz")["cmos_sigma_median"])
    if np.isfinite(c):
        a.axhline(c, color=INK, lw=1.2, ls="--", label=f"CMOS ({c:.1f} µm)")
    a.set_ylim(20, 110)
    a.legend(loc="upper right", fontsize=9, markerscale=4, frameon=False)
    style(a, f"C. One real sweep, every bunch ({held_out[k]})", "time in the turn [ns]", "σ_y [µm]")

    # D: per sweep vs CMOS
    a = ax[1, 0]
    sweeps = np.unique(sweep_idx)
    nn_med = np.array([np.median(pred_real[MAIN][sweep_idx == s]) for s in sweeps])
    fit_med = np.array([np.median(fit_sigma[sweep_idx == s]) for s in sweeps])
    ok = np.isfinite(cmos[sweeps])
    rng_d = (45, 75)
    a.plot(rng_d, rng_d, color=INK, lw=1, ls="--")
    a.scatter(cmos[sweeps][ok], fit_med[ok], s=22, color=FIT_COLOR, edgecolor="#fcfcfb", lw=1,
              label=f"offline fit  ({np.median(fit_med[ok] - cmos[sweeps][ok]):+.1f} ± {np.std(fit_med[ok] - cmos[sweeps][ok]):.1f} µm)")
    a.scatter(cmos[sweeps][ok], nn_med[ok], s=22, color=COLORS[MAIN], edgecolor="#fcfcfb", lw=1,
              label=f"network  ({np.median(nn_med[ok] - cmos[sweeps][ok]):+.1f} ± {np.std(nn_med[ok] - cmos[sweeps][ok]):.1f} µm)")
    a.set_xlim(rng_d)
    a.set_ylim(rng_d)
    a.legend(loc="upper left", fontsize=9, frameon=False)
    style(a, "D. Per sweep: median σ_y vs CMOS (one point per sweep)", "CMOS σ_y [µm]", "median σ_y in the sweep [µm]")

    # E: generated predicted vs true
    a = ax[1, 1]
    a.axvspan(real_lo, real_hi, color="#f0efec", zorder=0)
    a.hist2d(sig_gen, pred_gen[MAIN], bins=120, range=[SIGMA_RANGE_UM, SIGMA_RANGE_UM], cmap="Blues", cmin=1)
    a.plot(SIGMA_RANGE_UM, SIGMA_RANGE_UM, color=INK, lw=1, ls="--")
    a.text(real_lo + 2, 185, "real data\n47–83 µm", color=MUTED, fontsize=9)
    style(a, "E. Generated bunches: predicted vs true σ_y", "true σ_y [µm]", "network σ_y [µm]")

    # F: error spread vs true sigma
    a = ax[1, 2]
    a.axvspan(real_lo, real_hi, color="#f0efec", zorder=0)
    edges = np.arange(10, 201, 10)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for name in ROSTERS:
        err = pred_gen[name] - sig_gen
        spread = [np.std(err[(sig_gen >= lo) & (sig_gen < hi)]) for lo, hi in zip(edges[:-1], edges[1:])]
        a.plot(centres, spread, "o-", lw=2, ms=5, color=COLORS[name], label=LABELS[name])
    a.set_ylim(0, None)
    a.legend(loc="upper left", fontsize=9, frameon=False)
    style(a, "F. Generated bunches: error spread vs beam size", "true σ_y [µm]", "spread of network − true [µm]")

    fig.tight_layout()
    fig.savefig(args.output, dpi=110, facecolor=fig.get_facecolor())
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
