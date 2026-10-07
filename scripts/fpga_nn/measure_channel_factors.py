#!/usr/bin/env python3

"""
Per-channel correction factors between real profiles and the templates,
measured on the stable setup period (Dec 4-15).

factor[ch] = median over bunches of (reference - v_offset) / (template - v_offset)

Factors are measured on every other sweep and tested on the rest:
  - mismatch before and after correction (average, sweep-to-sweep, bunch-to-bunch)
  - refit of test sweeps with reference / factor: remaining mismatch,
    change of the factors on a second pass, change of sigma_y, and
    agreement with CMOS before and after
"""

import argparse
from multiprocessing import Pool

import numpy as np

import xrm_raw
from xrm_raw import ffr

AMPLITUDE_DIR = xrm_raw.REAL_DIR.parent / "amplitudes"
DEFAULT_GRID = xrm_raw.REPO_DIR / "beam_profiles" / "smeared_grid_CMOS_upto200um.npz"
STABLE_FROM = "2025-12-04"
MIN_SIGNAL_ADC = 100.0
MIN_ENTRIES = 2000

_grid = None
_factors = None


def parse_args():
    parser = argparse.ArgumentParser(
            prog="measure_channel_factors",
            description="Measure per-channel template correction factors on the stable period.",
            )

    parser.add_argument(
            "--n-refit",
            type=int,
            default=30,
            help="Number of test sweeps to refit with corrected amplitudes.",
            )

    parser.add_argument(
            "-o",
            "--output",
            default=str(xrm_raw.REAL_DIR.parent / "channel_factors.npz"),
            help="Output .npz file.",
            )

    parser.add_argument(
            "-j",
            "--jobs",
            type=int,
            default=8,
            help="Number of sweeps refitted in parallel.",
            )

    return parser.parse_args()


def stable_sweeps():
    paths = []
    for path in sorted(AMPLITUDE_DIR.glob("*.npz")):
        if path.stem >= STABLE_FROM:
            d = np.load(path)
            if str(d["category"]) == "good":
                paths.append(path)
    return paths


def log_ratio(reference, model, fit_params):
    """log((reference - v_offset) / (template - v_offset)), NaN where the template is small."""
    v_offset = fit_params[:, 2:3]
    signal = model - v_offset
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log((reference - v_offset) / signal)
    out[~(signal > MIN_SIGNAL_ADC) | ~np.isfinite(out)] = np.nan
    return out


def load(paths, every=1):
    rows = {"log_ratio": [], "sweep": []}
    for i, path in enumerate(paths):
        d = np.load(path)
        lr = log_ratio(d["reference"][::every], d["model"][::every], d["fit_params"][::every])
        rows["log_ratio"].append(lr)
        rows["sweep"].append(np.full(len(lr), i))
    return {k: np.concatenate(v) for k, v in rows.items()}


def mismatch_summary(lr, sweep, channels):
    """Average, sweep-to-sweep and bunch-to-bunch mismatch, in %, median over channels."""
    lr = lr[:, channels]
    average = np.abs(np.nanmedian(lr, axis=0))

    sweeps = np.unique(sweep)
    per_sweep = np.array([np.nanmedian(lr[sweep == s], axis=0) for s in sweeps])
    within = np.array([1.4826 * np.nanmedian(np.abs(lr[sweep == s] - per_sweep[k]), axis=0)
                       for k, s in enumerate(sweeps)])

    pct = lambda x: 100 * (np.exp(np.nanmedian(x)) - 1)
    return pct(average), pct(np.nanstd(per_sweep, axis=0)), pct(np.nanmedian(within, axis=0)), \
        100 * (np.exp(np.nanmax(average)) - 1)


def init_worker(grid_path, factors):
    global _grid, _factors
    _grid = ffr.SmearedGridModel.from_npz(grid_path)
    _factors = factors


def refit_sweep(path):
    d = np.load(path)
    corrected = d["reference"] / _factors
    channels = np.arange(corrected.shape[1], dtype=float)

    params = np.full((len(corrected), 6), np.nan)
    model = np.full_like(corrected, np.nan)
    for i, heights in enumerate(corrected):
        if np.all(np.isfinite(heights)):
            params[i], _ = ffr.fit_one_bunch_index(heights, _grid)
            model[i] = _grid.model_profile_for_channels_index(channels, *params[i])

    run = str(d["run"])
    cmos = float(np.load(xrm_raw.REAL_DIR / "runs" / f"{run}.npz")["cmos_sigma_median"])

    return {
        "run": run,
        "cmos": cmos,
        "sigma_before": np.nanmedian(d["fit_params"][:, 1]),
        "sigma_after": np.nanmedian(params[:, 1]),
        "sigma_shift": np.nanmedian(params[:, 1] - d["fit_params"][:, 1]),
        "log_ratio_after": log_ratio(corrected, model, params),
    }


def main():
    args = parse_args()

    paths = stable_sweeps()
    train, test = paths[0::2], paths[1::2]
    print(f"Stable period from {STABLE_FROM}: {len(paths)} good sweeps "
          f"({len(train)} to measure factors, {len(test)} to test)")

    tr = load(train, every=2)
    n_ch = tr["log_ratio"].shape[1]
    counts = np.sum(np.isfinite(tr["log_ratio"]), axis=0)
    measured = counts >= MIN_ENTRIES
    factors = np.ones(n_ch)
    factors[measured] = np.exp(np.nanmedian(tr["log_ratio"][:, measured], axis=0))

    print(f"\nFactors (reference / template), {measured.sum()} channels measured; "
          f"others set to 1.00 (too little signal):")
    for start in range(0, n_ch, 14):
        print("  " + "  ".join(f"ch{c:<2d} {factors[c]:.2f}{'' if measured[c] else '*'}"
                               for c in range(start, min(start + 14, n_ch))))

    te = load(test, every=2)
    channels = np.nonzero(measured)[0]
    before = mismatch_summary(te["log_ratio"], te["sweep"], channels)
    after = mismatch_summary(te["log_ratio"] - np.log(factors), te["sweep"], channels)

    print("\nTest sweeps, mismatch as % of signal (median over channels):")
    print(f"  {'':>28s} {'average':>8s} {'sweep-to-sweep':>15s} {'bunch-to-bunch':>15s} {'worst average':>14s}")
    print(f"  {'before correction':>28s} {before[0]:7.1f}% {before[1]:14.1f}% {before[2]:14.1f}% {before[3]:13.1f}%")
    print(f"  {'after correction (no refit)':>28s} {after[0]:7.1f}% {after[1]:14.1f}% {after[2]:14.1f}% {after[3]:13.1f}%")

    refit_paths = [test[i] for i in np.linspace(0, len(test) - 1, args.n_refit).astype(int)]
    print(f"\nRefitting {len(refit_paths)} test sweeps with corrected amplitudes...")
    with Pool(args.jobs, initializer=init_worker, initargs=(str(DEFAULT_GRID), factors)) as pool:
        refits = pool.map(refit_sweep, refit_paths)

    lr_after = np.concatenate([r["log_ratio_after"] for r in refits])
    sweep_after = np.concatenate([np.full(len(r["log_ratio_after"]), k) for k, r in enumerate(refits)])
    refit = mismatch_summary(lr_after, sweep_after, channels)
    print(f"  {'after correction + refit':>28s} {refit[0]:7.1f}% {refit[1]:14.1f}% {refit[2]:14.1f}% {refit[3]:13.1f}%")

    second_pass = np.exp(np.nanmedian(lr_after[:, channels], axis=0))
    print(f"  second-pass factor change: median {100 * np.median(np.abs(second_pass - 1)):.1f}%, "
          f"worst {100 * np.max(np.abs(second_pass - 1)):.1f}%")

    shift = np.array([r["sigma_shift"] for r in refits])
    cmos = np.array([r["cmos"] for r in refits])
    s_before = np.array([r["sigma_before"] for r in refits])
    s_after = np.array([r["sigma_after"] for r in refits])
    ok = np.isfinite(cmos)
    print(f"\nsigma_y change from the correction (median per sweep): {np.median(shift):+.2f} um "
          f"(range {shift.min():+.2f} .. {shift.max():+.2f})")
    print(f"Fit - CMOS over {ok.sum()} sweeps: before {np.median(s_before[ok] - cmos[ok]):+.2f} um "
          f"(spread {np.std(s_before[ok] - cmos[ok]):.2f}), "
          f"after {np.median(s_after[ok] - cmos[ok]):+.2f} um (spread {np.std(s_after[ok] - cmos[ok]):.2f})")

    np.savez(
        args.output,
        factors=factors,
        measured=measured,
        stable_from=STABLE_FROM,
        train_sweeps=np.array([p.stem for p in train]),
        second_pass=second_pass,
    )
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
