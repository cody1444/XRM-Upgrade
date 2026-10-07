#!/usr/bin/env python3

"""
How beam size, intensity and beam position relate to each other in the
stable-period data (Dec 4-15), before studying how template deviations
depend on them.

Per sweep (medians) and per bunch:
  sigma_y   fitted vertical beam size
  intensity fitted brightness (norm); checked against the logged
            current per bunch (DCCT current / number of bunches)
  mu        fitted vertical beam position
"""

import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xrm_raw
from xrm_generator import AMPLITUDE_DIR, STABLE_FROM

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
VARS = {"sigma": ("σ_y [µm]", 1), "norm": ("intensity (fitted norm) [×10⁵]", 4), "mu": ("μ [µm]", 0)}


def parse_args():
    parser = argparse.ArgumentParser(prog="plot_condition_correlations",
                                     description="Relations between beam size, intensity and position.")
    parser.add_argument("-o", "--output", default=str(xrm_raw.REAL_DIR.parent / "condition_correlations.png"))
    return parser.parse_args()


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def main():
    args = parse_args()

    sweeps, med, bunch, current, n_bunch, when = [], [], [], [], [], []
    for path in sorted(AMPLITUDE_DIR.glob("*.npz")):
        if path.stem < STABLE_FROM:
            continue
        d = np.load(path)
        if str(d["category"]) != "good":
            continue
        p = d["fit_params"]
        p = p[np.all(np.isfinite(p), axis=1)]
        sweeps.append(path.stem)
        med.append(np.median(p, axis=0))
        bunch.append(p[::10])
        current.append(float(d["current_ma"]))
        n_bunch.append(len(d["bunch_times_ns"]))
        when.append(datetime.strptime(path.stem, "%Y-%m-%d.%H%M%S"))

    med, bunch = np.array(med), np.concatenate(bunch)
    current, n_bunch = np.array(current), np.array(n_bunch)
    days = np.array([(w - when[0]).total_seconds() / 86400 for w in when])
    per_bunch_ma = current / n_bunch

    def col(arr, name):
        v = arr[:, VARS[name][1]]
        return v / 1e5 if name == "norm" else v

    pairs = [("sigma", "norm"), ("sigma", "mu"), ("norm", "mu")]

    fig, ax = plt.subplots(2, 4, figsize=(20, 9.5))
    fig.patch.set_facecolor(SURFACE)
    plt.rcParams["font.size"] = 10

    print(f"{len(sweeps)} stable-period good sweeps, {len(bunch)} bunches (every 10th)")
    print("\nCorrelation (Pearson r):          per sweep   per bunch   within sweeps")
    for k, (a, b) in enumerate(pairs):
        xa, ya = col(med, a), col(med, b)
        r_sweep = np.corrcoef(xa, ya)[0, 1]
        xb, yb = col(bunch, a), col(bunch, b)
        r_bunch = np.corrcoef(xb, yb)[0, 1]

        # within-sweep: remove each sweep's median first
        sweep_of_bunch = np.repeat(np.arange(len(sweeps)),
                                   [len(np.load(AMPLITUDE_DIR / f"{s}.npz")["fit_params"][::10]
                                        [np.all(np.isfinite(np.load(AMPLITUDE_DIR / f"{s}.npz")["fit_params"][::10]), axis=1)])
                                    for s in sweeps])
        r_within = np.corrcoef(xb - xa[sweep_of_bunch], yb - ya[sweep_of_bunch])[0, 1]
        print(f"  {a:>6s} vs {b:<6s}               {r_sweep:+6.2f}      {r_bunch:+6.2f}      {r_within:+6.2f}")

        sc = ax[0, k].scatter(xa, ya, c=days, cmap="Blues", vmin=-3, vmax=days.max(), s=22,
                              edgecolor="#52514e", lw=0.3)
        style(ax[0, k], f"{chr(65 + k)}. Per sweep: {a} vs {b}   (r = {r_sweep:+.2f})", VARS[a][0], VARS[b][0])

        ax[1, k].hexbin(xb, yb, gridsize=60, cmap="Blues", mincnt=1, bins="log")
        style(ax[1, k], f"{chr(69 + k)}. Per bunch: {a} vs {b}   (r = {r_bunch:+.2f}, within sweeps {r_within:+.2f})",
              VARS[a][0], VARS[b][0])

    cb = fig.colorbar(sc, ax=ax[0, 2], fraction=0.05, pad=0.02)
    cb.set_label("days since Dec 4", color=MUTED)

    a = ax[0, 3]
    r_int = np.corrcoef(per_bunch_ma, col(med, "norm"))[0, 1]
    a.scatter(per_bunch_ma, col(med, "norm"), c=days, cmap="Blues", vmin=-3, vmax=days.max(), s=22,
              edgecolor="#52514e", lw=0.3)
    style(a, f"D. Fitted intensity vs logged current per bunch (r = {r_int:+.2f})",
          "DCCT current / number of bunches [mA]", VARS["norm"][0])
    print(f"\n  fitted intensity vs logged current per bunch (per sweep): r = {r_int:+.2f}")

    a = ax[1, 3]
    for name, colour in (("sigma", "#2a78d6"), ("norm", "#eb6834"), ("mu", "#1baf7a")):
        v = col(med, name)
        z = (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))))
        a.plot(days, z, "o", ms=3, color=colour, label=VARS[name][0].split(" [")[0])
    a.legend(loc="upper left", fontsize=9, frameon=False)
    style(a, "H. Per sweep over time (each in units of its own spread)", "days since Dec 4", "deviation from median [spreads]")

    fig.tight_layout()
    fig.savefig(args.output, dpi=110, facecolor=fig.get_facecolor())

    print("\nRanges per sweep (5th / 50th / 95th percentile):")
    for name in VARS:
        v = col(med, name)
        print(f"  {name:>6s}: {np.percentile(v, 5):7.2f} {np.median(v):7.2f} {np.percentile(v, 95):7.2f}")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
