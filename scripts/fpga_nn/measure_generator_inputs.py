#!/usr/bin/env python3

"""
Measure the generator's inputs from data/amplitudes.

Layer 1, real fluctuation:  residual = reference - template model
  per channel, spread vs model amplitude, fitted as
      spread^2 = a^2 + k * A + (f * A)^2
  (a: constant, k: grows like sqrt(A) as photon counting would, f: proportional)
  plus how strongly residuals of nearby channels move together.

Layer 2, firmware response:  firmware vs reference
  per channel, firmware = g * reference + o, and the spread around that line
  vs reference amplitude, fitted the same way, split by beam current.

Beam parameters: fitted (mu, sigma, v_offset, x0, norm, scale) and the
DCCT current, kept as samples for the generator to draw from.
"""

import argparse
from pathlib import Path

import numpy as np

import xrm_raw

AMPLITUDE_DIR = xrm_raw.REAL_DIR.parent / "amplitudes"
AMPLITUDE_BINS = np.array([-50, 0, 25, 50, 100, 200, 400, 800, 1600, 3200])
CURRENT_CLASSES_MA = [(0, 600), (600, 900), (900, 2000)]
FIT_PARAM_NAMES = ["mu", "sigma", "v_offset", "x0", "norm", "scale"]


def parse_args():
    parser = argparse.ArgumentParser(
            prog="measure_generator_inputs",
            description="Measure template residuals and firmware response for the generator.",
            )

    parser.add_argument(
            "--every",
            type=int,
            default=10,
            help="Keep every n-th bunch of each sweep (memory).",
            )

    parser.add_argument(
            "--category",
            default="good",
            choices=["good", "low_fill"],
            help="Which sweeps to use.",
            )

    parser.add_argument(
            "-o",
            "--output",
            default=str(xrm_raw.REAL_DIR.parent / "generator_inputs.npz"),
            help="Output .npz file.",
            )

    return parser.parse_args()


def load(category, every):
    keys = ["reference", "firmware", "model", "fit_params"]
    parts = {k: [] for k in keys + ["current_ma"]}

    for path in sorted(AMPLITUDE_DIR.glob("*.npz")):
        d = np.load(path)
        if str(d["category"]) != category:
            continue
        for k in keys:
            parts[k].append(d[k][::every])
        parts["current_ma"].append(np.full(len(d["reference"][::every]), float(d["current_ma"])))

    return {k: np.concatenate(v) for k, v in parts.items()}


def robust_std(values):
    values = values[np.isfinite(values)]
    if len(values) < 30:
        return np.nan
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def binned_spread(amplitude, residual):
    """Robust spread of residual in amplitude bins: (bin centres, spread, counts)."""
    centres, spreads, counts = [], [], []

    for lo, hi in zip(AMPLITUDE_BINS[:-1], AMPLITUDE_BINS[1:]):
        mask = (amplitude >= lo) & (amplitude < hi) & np.isfinite(residual)
        if mask.sum() >= 30:
            centres.append(np.median(amplitude[mask]))
            spreads.append(robust_std(residual[mask]))
            counts.append(mask.sum())

    return np.array(centres), np.array(spreads), np.array(counts)


def fit_noise_model(centres, spreads):
    """Least-squares fit of spread^2 = a^2 + k*A + (f*A)^2 with non-negative terms."""
    # Weighted by 1/spread^2 so small-amplitude bins count as much as large ones.
    a_pos = np.clip(centres, 0, None)
    weight = 1.0 / spreads ** 2
    design = np.column_stack([np.ones_like(a_pos), a_pos, a_pos ** 2]) * weight[:, None]
    coeffs, *_ = np.linalg.lstsq(design, spreads ** 2 * weight, rcond=None)
    coeffs = np.clip(coeffs, 0, None)

    return np.sqrt(coeffs[0]), coeffs[1], np.sqrt(coeffs[2])


def describe_noise(label, a, k, f):
    def at(amp):
        return np.sqrt(a ** 2 + k * amp + (f * amp) ** 2)

    return (f"{label}: constant {a:5.1f} ADC, sqrt-term k {k:6.2f}, proportional {f * 100:4.1f}% "
            f"-> spread at 50/200/800 ADC: {at(50):5.1f} / {at(200):5.1f} / {at(800):5.1f}")


def neighbour_correlation(residual):
    """Mean correlation of residuals between channels 1, 2, 3 and >5 apart."""
    ok = np.all(np.isfinite(residual), axis=1)
    z = residual[ok]
    z = (z - np.median(z, axis=0)) / (np.std(z, axis=0) + 1e-9)
    z = np.clip(z, -5, 5)
    corr = np.corrcoef(z.T)

    n = corr.shape[0]
    i, j = np.triu_indices(n, 1)
    distance = j - i

    return {d: np.nanmean(corr[i[distance == d], j[distance == d]]) for d in (1, 2, 3)} | \
           {">5": np.nanmean(corr[i[distance > 5], j[distance > 5]])}


def main():
    args = parse_args()

    d = load(args.category, args.every)
    n_bunches, n_channels = d["reference"].shape
    print(f"{args.category} sweeps: {n_bunches} bunches (every {args.every}th), {n_channels} channels")

    out = {"amplitude_bins": AMPLITUDE_BINS}

    # Layer 1: real profile vs template.
    residual1 = d["reference"] - d["model"]
    layer1 = np.full((n_channels, 3), np.nan)
    for ch in range(n_channels):
        c, s, _ = binned_spread(d["model"][:, ch], residual1[:, ch])
        if len(c) >= 3:
            layer1[ch] = fit_noise_model(c, s)

    c, s, _ = binned_spread(d["model"].ravel(), residual1.ravel())
    pooled1 = fit_noise_model(c, s)
    out["layer1_per_channel"] = layer1
    out["layer1_pooled"] = np.array(pooled1)
    out["layer1_bias_per_channel"] = np.nanmedian(residual1, axis=0)

    print("\nLayer 1: reference - template (real fluctuation + template mismatch)")
    print("  " + describe_noise("all channels", *pooled1))
    print("  binned spread (ADC):", ", ".join(f"{ci:.0f}: {si:.1f}" for ci, si in zip(c, s)))
    corr1 = neighbour_correlation(residual1)
    out["layer1_neighbour_corr"] = np.array([corr1[1], corr1[2], corr1[3], corr1[">5"]])
    print("  residual correlation, channels 1/2/3/>5 apart: "
          + " / ".join(f"{v:.2f}" for v in corr1.values()))

    # Layer 2: firmware vs reference.
    gain2 = np.full(n_channels, np.nan)
    offset2 = np.full(n_channels, np.nan)
    for ch in range(n_channels):
        ok = np.isfinite(d["reference"][:, ch]) & np.isfinite(d["firmware"][:, ch])
        gain2[ch], offset2[ch] = np.polyfit(d["reference"][ok, ch], d["firmware"][ok, ch], 1)

    residual2 = d["firmware"] - (gain2 * d["reference"] + offset2)
    c, s, _ = binned_spread(d["reference"].ravel(), residual2.ravel())
    pooled2 = fit_noise_model(c, s)
    out["layer2_gain"] = gain2
    out["layer2_offset"] = offset2
    out["layer2_pooled"] = np.array(pooled2)

    signal = np.nanmedian(d["reference"], axis=0) > 100
    print("\nLayer 2: firmware = g * reference + o, then the spread around it")
    print(f"  g on channels with signal: {gain2[signal].min():.3f} .. {gain2[signal].max():.3f}, "
          f"median {np.median(gain2[signal]):.3f}")
    print(f"  o: {np.nanmin(offset2):+.1f} .. {np.nanmax(offset2):+.1f} ADC, median {np.nanmedian(offset2):+.1f}")
    print("  " + describe_noise("all channels", *pooled2))
    print("  binned spread (ADC):", ", ".join(f"{ci:.0f}: {si:.1f}" for ci, si in zip(c, s)))
    corr2 = neighbour_correlation(residual2)
    out["layer2_neighbour_corr"] = np.array([corr2[1], corr2[2], corr2[3], corr2[">5"]])
    print("  residual correlation, channels 1/2/3/>5 apart: "
          + " / ".join(f"{v:.2f}" for v in corr2.values()))

    print("  by beam current:")
    for lo, hi in CURRENT_CLASSES_MA:
        mask = (d["current_ma"] >= lo) & (d["current_ma"] < hi)
        if mask.sum() < 1000:
            continue
        g_cls = []
        for ch in np.nonzero(signal)[0]:
            ok = mask & np.isfinite(d["reference"][:, ch]) & np.isfinite(d["firmware"][:, ch])
            g_cls.append(np.polyfit(d["reference"][ok, ch], d["firmware"][ok, ch], 1)[0])
        c_cls, s_cls, _ = binned_spread(d["reference"][mask].ravel(), residual2[mask].ravel())
        a, k, f = fit_noise_model(c_cls, s_cls)
        print(f"    {lo}-{hi} mA: median g {np.median(g_cls):.3f}; " + describe_noise("spread", a, k, f))

    # Beam parameters to sample from.
    params = d["fit_params"]
    ok = np.all(np.isfinite(params), axis=1)
    out["fit_params"] = params[ok]
    out["fit_param_names"] = np.array(FIT_PARAM_NAMES)
    out["current_ma"] = d["current_ma"][ok]

    print("\nBeam parameters (fits), 5th / 50th / 95th percentile:")
    for i, name in enumerate(FIT_PARAM_NAMES):
        p5, p50, p95 = np.percentile(params[ok, i], [5, 50, 95])
        print(f"  {name:>8s}: {p5:9.2f} {p50:9.2f} {p95:9.2f}")

    np.savez(args.output, **out)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
