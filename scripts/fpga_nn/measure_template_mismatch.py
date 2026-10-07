#!/usr/bin/env python3

"""
Is the reference-vs-template mismatch a gain error or a shape error?

Mismatch ratio per bunch and channel, with the fit's constant offset removed:
    R = (reference - v_offset) / (template - v_offset)
A wrong gain constant multiplies a channel's signal, so its R is the same
whatever the beam does, and it stays with the STRIP when the image moves.
A template shape error depends on beam size and position, and moves with
the IMAGE when the beam moves.

Checks:
  1. per channel, how much the median R changes with sigma_y, mu, day, current
  2. split bunches by image position; compare mismatch patterns aligned by
     strip vs shifted along with the image
"""

import argparse

import numpy as np

import xrm_raw

AMPLITUDE_DIR = xrm_raw.REAL_DIR.parent / "amplitudes"
MIN_SIGNAL_ADC = 100.0
N_QUANTILE_BINS = 5
MIN_PER_BIN = 200


def parse_args():
    parser = argparse.ArgumentParser(
            prog="measure_template_mismatch",
            description="Test whether the template mismatch is a gain or a shape error.",
            )

    parser.add_argument(
            "--every",
            type=int,
            default=5,
            help="Keep every n-th bunch of each sweep.",
            )

    parser.add_argument(
            "-o",
            "--output",
            default=str(xrm_raw.REAL_DIR.parent / "template_mismatch.npz"),
            help="Output .npz file.",
            )

    return parser.parse_args()


def load(every):
    parts = {k: [] for k in ("reference", "model", "fit_params", "current_ma", "day")}

    for path in sorted(AMPLITUDE_DIR.glob("*.npz")):
        d = np.load(path)
        if str(d["category"]) != "good":
            continue
        n = len(d["reference"][::every])
        parts["reference"].append(d["reference"][::every])
        parts["model"].append(d["model"][::every])
        parts["fit_params"].append(d["fit_params"][::every])
        parts["current_ma"].append(np.full(n, float(d["current_ma"])))
        parts["day"].append(np.full(n, str(d["run"])[:10]))

    return {k: np.concatenate(v) for k, v in parts.items()}


def binned_medians(values, factor, bins):
    """Median of values (n, n_ch) in each factor bin -> (n_bins, n_ch), NaN if too few."""
    out = np.full((len(bins), values.shape[1]), np.nan)

    for b, mask in enumerate(bins):
        sub = values[mask]
        enough = np.sum(np.isfinite(sub), axis=0) >= MIN_PER_BIN
        med = np.nanmedian(sub, axis=0)
        out[b, enough] = med[enough]

    return out


def quantile_bins(factor, n):
    edges = np.nanpercentile(factor, np.linspace(0, 100, n + 1))
    return [(factor >= lo) & (factor <= hi) for lo, hi in zip(edges[:-1], edges[1:])], edges


def main():
    args = parse_args()

    d = load(args.every)
    v_offset = d["fit_params"][:, 2:3]
    signal = d["model"] - v_offset
    log_ratio = np.log((d["reference"] - v_offset) / signal)
    log_ratio[~(signal > MIN_SIGNAL_ADC)] = np.nan
    log_ratio[~np.isfinite(log_ratio)] = np.nan

    n_ch = log_ratio.shape[1]
    channels = np.arange(n_ch)
    usable = np.sum(np.isfinite(log_ratio), axis=0) >= 5000
    print(f"{len(log_ratio)} bunches; {usable.sum()} channels with enough signal (template > {MIN_SIGNAL_ADC:.0f} ADC)")

    overall = np.nanmedian(log_ratio, axis=0)
    print("\nOverall mismatch ratio per channel (reference / template):")
    print("  " + ", ".join(f"ch{c}: {np.exp(overall[c]):.2f}" for c in channels[usable]))

    # 1. Does each channel's ratio change with the beam or over time?
    centroid = np.nansum(np.clip(signal, 0, None) * channels, axis=1) / np.nansum(np.clip(signal, 0, None), axis=1)
    factors = {
        "beam size sigma_y": quantile_bins(d["fit_params"][:, 1], N_QUANTILE_BINS)[0],
        "image position": quantile_bins(centroid, N_QUANTILE_BINS)[0],
        "day": [d["day"] == day for day in np.unique(d["day"])],
        "beam current": [(d["current_ma"] >= lo) & (d["current_ma"] < hi)
                         for lo, hi in [(0, 600), (600, 900), (900, 2000)]],
    }

    print("\n1. How much each channel's ratio changes across bins of each factor")
    print("   (max/min of the binned median ratio; 1.00 = no change), median and worst over channels:")
    results = {}
    for name, bins in factors.items():
        med = binned_medians(log_ratio, None, bins)[:, usable]
        swing = np.exp(np.nanmax(med, axis=0) - np.nanmin(med, axis=0))
        results[name] = med
        print(f"   {name:>18s}: median {np.nanmedian(swing):.2f}, worst {np.nanmax(swing):.2f}")

    # 2. Does the pattern follow the strip or the image?
    pos_bins, pos_edges = quantile_bins(centroid, 3)
    patterns = binned_medians(log_ratio, None, pos_bins)
    centres = [np.nanmedian(centroid[m]) for m in pos_bins]
    print(f"\n2. Image position groups (centroid, in channels): "
          + ", ".join(f"{c:.1f}" for c in centres))

    def corr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() >= 8 else np.nan

    def shifted(pattern, shift):
        """Pattern moved by `shift` channels (linear interpolation)."""
        ok = np.isfinite(pattern)
        return np.interp(channels - shift, channels[ok], pattern[ok], left=np.nan, right=np.nan)

    for i, j in [(0, 1), (1, 2), (0, 2)]:
        shift = centres[j] - centres[i]
        by_strip = corr(patterns[i], patterns[j])
        by_image = corr(shifted(patterns[i], shift), patterns[j])
        print(f"   groups {i}->{j} (image moved {shift:+.1f} channels): "
              f"pattern agreement aligned by strip {by_strip:.2f}, shifted with the image {by_image:.2f}")

    np.savez(
        args.output,
        overall_log_ratio=overall,
        usable=usable,
        position_patterns=patterns,
        position_centres=np.array(centres),
        **{f"binned_{k.replace(' ', '_')}": v for k, v in results.items()},
    )
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
