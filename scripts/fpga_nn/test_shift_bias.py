#!/usr/bin/env python3

"""
Test whether the signal-dependent baseline shift biases the fitted sigma_y.

The same bunches are fitted with the same template fit, using three sets of
per-channel amplitudes:
  reference  : full processing (cubic, alignment, local baseline, gain);
               stored in data/real/runs as fit_sigma
  firmware   : (beam-off pedestal - raw valley sample) * gain
  fw_local   : (local baseline - raw valley sample) * gain

firmware - fw_local isolates the baseline handling (same raw samples);
fw_local - reference isolates interpolation and alignment.
"""

import argparse
from multiprocessing import Pool

import numpy as np

import xrm_raw
from xrm_raw import ffr
from measure_baseline_shift import train_start_times

DEFAULT_GRID = xrm_raw.REPO_DIR / "beam_profiles" / "smeared_grid_CMOS_upto200um.npz"

CURRENT_CLASSES_MA = [(0, 600), (600, 900), (900, 2000)]
TIME_BIN_EDGES_NS = np.array([0, 20, 60, 100, 200, 500, 5000])

_grid = None
_pedestal = None


def parse_args():
    parser = argparse.ArgumentParser(
            prog="test_shift_bias",
            description="Test whether the baseline shift biases the fitted sigma_y.",
            )

    parser.add_argument(
            "--stride",
            type=int,
            default=4,
            help="Use every n-th good run.",
            )

    parser.add_argument(
            "-o",
            "--output",
            default=str(xrm_raw.REAL_DIR.parent / "shift_bias.npz"),
            help="Output .npz file.",
            )

    parser.add_argument(
            "-j",
            "--jobs",
            type=int,
            default=8,
            help="Number of runs processed in parallel.",
            )

    return parser.parse_args()


def init_worker(grid_path, pedestal):
    global _grid, _pedestal
    _grid = ffr.SmearedGridModel.from_npz(grid_path)
    _pedestal = pedestal


def fit_sigma(heights):
    sigma = np.full(heights.shape[0], np.nan)

    for i in range(heights.shape[0]):
        params, _ = ffr.fit_one_bunch_index(heights[i], _grid)
        sigma[i] = params[1]

    return sigma


def process_run(run):
    data = np.load(xrm_raw.REAL_DIR / "runs" / f"{run}.npz")

    order = np.argsort(data["bunch_times_ns"])
    times = data["bunch_times_ns"][order]
    valley = data["valley_raw"][order]
    baseline = data["baseline"][order]
    gain = data["gain"]

    _, _, own_start = train_start_times(times)

    return {
        "run": run,
        "t_since_start": times - own_start,
        "reference": data["fit_sigma"][order],
        "firmware": fit_sigma((_pedestal - valley) * gain),
        "fw_local": fit_sigma((baseline - valley) * gain),
    }


def robust_summary(diff):
    diff = diff[np.isfinite(diff)]

    if len(diff) == 0:
        return "      -"

    median = np.median(diff)
    spread = 1.4826 * np.median(np.abs(diff - median))

    return f"{median:+6.2f} ({spread:4.1f})"


def print_table(title, groups):
    print(f"\n{title}")
    print(f"{'':>22s} {'n bunches':>10s} | {'firmware - ref':>15s} | {'fw_local - ref':>15s} | {'firmware - fw_local':>19s}")

    for label, mask, d in groups:
        print(f"{label:>22s} {mask.sum():>10d} | "
              f"{robust_summary(d['firmware'][mask] - d['reference'][mask]):>15s} | "
              f"{robust_summary(d['fw_local'][mask] - d['reference'][mask]):>15s} | "
              f"{robust_summary(d['firmware'][mask] - d['fw_local'][mask]):>19s}")


def main():
    args = parse_args()

    print("Computing beam-off pedestals...")
    pedestal = xrm_raw.beam_off_pedestals()

    runs = xrm_raw.good_runs()[::args.stride]
    dcct = xrm_raw.DcctCurrent()

    print(f"Fitting {len(runs)} runs three ways...")
    with Pool(args.jobs, initializer=init_worker, initargs=(str(DEFAULT_GRID), pedestal)) as pool:
        results = pool.map(process_run, runs)

    current_per_run = {r["run"]: dcct(r["run"]) for r in results}

    d = {key: np.concatenate([r[key] for r in results])
         for key in ("t_since_start", "reference", "firmware", "fw_local")}
    d["current"] = np.concatenate([np.full(len(r["reference"]), current_per_run[r["run"]]) for r in results])
    d["run"] = np.concatenate([np.full(len(r["reference"]), r["run"]) for r in results])

    np.savez(args.output, **d)

    print("Differences in fitted sigma_y, in um: median (robust spread)")

    everything = np.isfinite(d["current"])
    print_table("All bunches", [("all", everything, d)])

    print_table("By beam current", [
        (f"{lo}-{hi} mA", (d["current"] >= lo) & (d["current"] < hi), d)
        for lo, hi in CURRENT_CLASSES_MA
    ])

    high = d["current"] >= 900
    print_table("By time since train start, > 900 mA", [
        (f"{lo}-{hi} ns", high & (d["t_since_start"] >= lo) & (d["t_since_start"] < hi), d)
        for lo, hi in zip(TIME_BIN_EDGES_NS[:-1], TIME_BIN_EDGES_NS[1:])
    ])

    ref_median = np.nanmedian(d["reference"])
    print(f"\nFor scale: median reference sigma_y = {ref_median:.1f} um")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
