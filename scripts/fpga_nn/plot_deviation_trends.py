#!/usr/bin/env python3

"""
Does the deviation between real profiles and templates change with beam
size or beam position?

One data point per sweep (stable period, excluded sweeps removed):
  consistent deviation[ch] = median over bunches of (reference - v_offset) /
                             (template - v_offset) - 1, in %
  random part[ch]          = bunch-to-bunch robust spread of that ratio, in %
  sigma_y, mu              = the sweep's median fitted values
Sweeps, not bunches, are used so that bunch-level fit errors cannot create
trends. Uncertainties come from resampling whole days.

Figure 1: deviation vs sigma_y and vs mu for a few channels, with a fitted
          line extended beyond the data and its uncertainty band
Figure 2: every channel's slope vs sigma_y and vs mu, from a joint fit
"""

import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xrm_raw
from xrm_generator import AMPLITUDE_DIR, STABLE_FROM, EXCLUDED_SWEEPS

MIN_SIGNAL_ADC = 100.0

CHANNELS = [10, 15, 18, 20, 24, 34]
MIN_BUNCHES = 200
N_BOOT = 300
N_DISPLAY_BINS = 6
EXTRAP = {"sigma": (10, 200), "mu": (40, 140)}
LABEL = {"sigma": "sweep median σ_y [µm]", "mu": "sweep median μ [µm]"}
INK, MUTED, GRID, SURFACE, BLUE, ORANGE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#2a78d6", "#eb6834"


def robust_std(values):
    """Standard deviation estimated from the median absolute deviation (ignores outliers)."""
    values = values[np.isfinite(values)]
    if len(values) < 30:
        return np.nan
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def parse_args():
    parser = argparse.ArgumentParser(prog="plot_deviation_trends",
                                     description="Template deviation vs beam size and position.")
    parser.add_argument("-o", "--output-prefix", default=str(xrm_raw.REAL_DIR.parent / "deviation_trends"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def per_sweep_deviations():
    rows = defaultdict(list)
    for path in sorted(AMPLITUDE_DIR.glob("*.npz")):
        if path.stem < STABLE_FROM or path.stem in EXCLUDED_SWEEPS:
            continue
        d = np.load(path)
        if str(d["category"]) != "good":
            continue

        params = d["fit_params"]
        v_offset = params[:, 2:3]
        signal = d["model"] - v_offset
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (d["reference"] - v_offset) / signal
        ratio[~(signal > MIN_SIGNAL_ADC) | ~np.isfinite(ratio)] = np.nan

        enough = np.sum(np.isfinite(ratio), axis=0) >= MIN_BUNCHES
        consistent = np.where(enough, 100 * (np.nanmedian(ratio, axis=0) - 1), np.nan)
        random = np.array([100 * robust_std(ratio[:, ch]) if enough[ch] else np.nan
                           for ch in range(ratio.shape[1])])

        ok = np.all(np.isfinite(params), axis=1)
        rows["sweep"].append(path.stem)
        rows["day"].append(path.stem[:10])
        rows["sigma"].append(np.median(params[ok, 1]))
        rows["mu"].append(np.median(params[ok, 0]))
        rows["consistent"].append(consistent)
        rows["random"].append(random)

    return {k: np.array(v) for k, v in rows.items()}


def day_bootstrap(days, rng, n=N_BOOT):
    """Indices of resampled data sets, drawing whole days with replacement."""
    unique = np.unique(days)
    members = {u: np.nonzero(days == u)[0] for u in unique}
    for _ in range(n):
        yield np.concatenate([members[u] for u in rng.choice(unique, len(unique))])


def line_band(x, y, days, grid, rng):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, days = x[ok], y[ok], days[ok]
    best = np.polyfit(x, y, 1)
    curves, slopes = [], []
    for idx in day_bootstrap(days, rng):
        c = np.polyfit(x[idx], y[idx], 1)
        curves.append(np.polyval(c, grid))
        slopes.append(c[0])
    lo, hi = np.percentile(curves, [16, 84], axis=0)
    # slope and its error per 10 um
    return np.polyval(best, grid), lo, hi, 10 * best[0], 10 * np.std(slopes)


def joint_slopes(data, rng):
    """deviation = a + b*sigma + c*mu per channel; slopes per 10 um with day-bootstrap errors."""
    n_ch = data["consistent"].shape[1]
    out = np.full((n_ch, 2, 2), np.nan)      # channel, (sigma, mu), (value, error)
    for ch in range(n_ch):
        y = data["consistent"][:, ch]
        ok = np.isfinite(y)
        if ok.sum() < 50:
            continue
        X = np.column_stack([np.ones(ok.sum()), data["sigma"][ok], data["mu"][ok]])
        best = np.linalg.lstsq(X, y[ok], rcond=None)[0]
        boots = []
        days = data["day"][ok]
        for idx in day_bootstrap(days, rng):
            boots.append(np.linalg.lstsq(X[idx], y[ok][idx], rcond=None)[0])
        boots = np.array(boots)
        out[ch, :, 0] = 10 * best[1:]
        out[ch, :, 1] = 10 * boots[:, 1:].std(axis=0)
    return out


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    data = per_sweep_deviations()
    print(f"{len(data['sweep'])} sweeps from {len(np.unique(data['day']))} days")

    # Figure 1
    fig, ax = plt.subplots(2, len(CHANNELS), figsize=(3.3 * len(CHANNELS), 7.6), sharey="col")
    fig.patch.set_facecolor(SURFACE)
    for row, var in enumerate(("sigma", "mu")):
        x = data[var]
        lo_data, hi_data = np.nanpercentile(x, [1, 99])
        grid = np.linspace(*EXTRAP[var], 200)
        edges = np.nanpercentile(x, np.linspace(0, 100, N_DISPLAY_BINS + 1))
        for col, ch in enumerate(CHANNELS):
            a = ax[row, col]
            y = data["consistent"][:, ch]
            rnd = data["random"][:, ch]

            centres, means, errs, bands = [], [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (x >= lo) & (x <= hi) & np.isfinite(y)
                if m.sum() >= 5:
                    centres.append(np.median(x[m]))
                    means.append(np.mean(y[m]))
                    errs.append(np.std(y[m]) / np.sqrt(m.sum()))
                    bands.append(np.nanmedian(rnd[m]))
            centres, means, bands = map(np.array, (centres, means, bands))

            fit, band_lo, band_hi, slope, slope_err = line_band(x, y, data["day"], grid, rng)
            a.fill_between(centres, means - bands, means + bands, color="#e1e0d9", label="random part (±1 spread)")
            a.fill_between(grid, band_lo, band_hi, color=BLUE, alpha=0.15, lw=0, label="line uncertainty (day resampling)")
            inside = (grid >= lo_data) & (grid <= hi_data)
            a.plot(grid[inside], fit[inside], color=BLUE, lw=2)
            a.plot(grid, fit, color=BLUE, lw=1.2, ls="--", label="fitted line (dashed: extrapolated)")
            a.errorbar(centres, means, yerr=errs, fmt="o", color=INK, ms=5, capsize=2, label="consistent deviation (sweeps binned)")
            a.axvspan(lo_data, hi_data, color="#f0efec", zorder=0)
            a.axhline(0, color=MUTED, lw=0.8)
            a.set_xlim(*EXTRAP[var])
            style(a, f"ch {ch}: {slope:+.1f} ± {slope_err:.1f} % per 10 µm", LABEL[var], "real ÷ template − 1 [%]")
    ax[0, 0].legend(loc="lower left", fontsize=7, frameon=False)
    fig.suptitle("Template deviation vs beam size (top) and beam position (bottom); shaded column = range covered by real sweeps",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(args.output_prefix + "_channels.png", dpi=110, facecolor=fig.get_facecolor())

    # Figure 2
    slopes = joint_slopes(data, rng)
    measured = np.isfinite(slopes[:, 0, 0])
    chs = np.nonzero(measured)[0]
    fig, ax = plt.subplots(2, 1, figsize=(15, 7.5), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for row, (var, colour) in enumerate((("sigma", BLUE), ("mu", ORANGE))):
        a = ax[row]
        a.bar(chs, slopes[chs, row, 0], yerr=slopes[chs, row, 1], color=colour, width=0.7,
              error_kw={"ecolor": INK, "capsize": 2, "lw": 1})
        a.axhline(0, color=INK, lw=0.8)
        name = "beam size σ_y" if var == "sigma" else "beam position μ"
        style(a, f"{chr(65 + row)}. Change in deviation per 10 µm of {name} (other variable held fixed)",
              "channel" if row == 1 else "", "% per 10 µm")
    fig.tight_layout()
    fig.savefig(args.output_prefix + "_slopes.png", dpi=110, facecolor=fig.get_facecolor())

    print("\nJoint-fit slopes (% per 10 um), median |slope| over channels and how many exceed 2 errors:")
    for k, var in enumerate(("sigma", "mu")):
        s, e = slopes[chs, k, 0], slopes[chs, k, 1]
        print(f"  {var:>5s}: median |slope| {np.median(np.abs(s)):.2f}, "
              f"{np.sum(np.abs(s) > 2 * e)} of {len(chs)} channels beyond 2 errors, "
              f"largest {s[np.argmax(np.abs(s))]:+.2f} (ch {chs[np.argmax(np.abs(s))]})")
    print(f"\nData range (1-99% of sweeps): sigma {np.nanpercentile(data['sigma'], 1):.1f}-{np.nanpercentile(data['sigma'], 99):.1f} um, "
          f"mu {np.nanpercentile(data['mu'], 1):.1f}-{np.nanpercentile(data['mu'], 99):.1f} um")
    print(f"Saved {args.output_prefix}_channels.png and {args.output_prefix}_slopes.png")


if __name__ == "__main__":
    main()
