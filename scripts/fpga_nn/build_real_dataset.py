#!/usr/bin/env python3

"""
Build a per-bunch real-data evaluation set from cleaned XRM ROOT files.

For every cleaned run this stores, per bunch:
  - firmware-like inputs: raw valley sample (no interpolation, no alignment,
    no gain) and the moving-average baseline at that bunch, for all 42 channels
  - reference labels: the full-processing template fit
    (cubic + alignment + baseline + gain), same as fit_results/
  - the CMOS sigma_y averaged over a window around the run timestamp

After all runs are processed, each run is put in a category
(good / low_fill / corrupted / failed) and per-channel pedestals are
estimated from the baselines of the good runs.

Note: "baseline" is the fit's baseline (overshoot maxima around each pulse),
so in channels with signal it grows with pulse size. It is not a pure
electronics pedestal.
"""

import argparse
import contextlib
import csv
import io
import re
import sys
import traceback
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import fit_from_rootfile as ffr  # noqa: E402

REPO_DIR = SCRIPTS_DIR.parent
DEFAULT_DATA_DIR = REPO_DIR / "scripts_dev" / "data" / "HER"
DEFAULT_GRID = REPO_DIR / "beam_profiles" / "smeared_grid_CMOS_upto200um.npz"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "real"

# Bunches from the fit path and the firmware-like path are matched by time.
MATCH_TOLERANCE_NS = 0.5

# Runs with fewer bunches are kept but separated from the main sample.
LOW_FILL_MAX_BUNCHES = 1000

# Runs whose median baseline sits this low are saturated or garbage.
CORRUPTED_BASELINE_ADC = -500.0

# CMOS readings that look like limits or stuck values rather than measurements.
CMOS_SUSPECT_VALUES = (10.0, 30.0, 200.0)
CMOS_SUSPECT_TOL = 0.01

RUN_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.(\d{6})")

# Full runs only. Variant files such as "<run>.28_cleaned.root" or
# "<run>.pilot.1_cleaned.root" are small, likely cancelled acquisitions.
FULL_RUN_FILE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\.\d{6}_cleaned\.root")

_grid = None


def parse_args():
    parser = argparse.ArgumentParser(
            prog="build_real_dataset",
            description="Build a per-bunch real-data evaluation set from cleaned XRM ROOT files.",
            )

    parser.add_argument(
            "--data-dir",
            default=str(DEFAULT_DATA_DIR),
            help="HER data directory containing <date>/cleaned/*_cleaned.root and CMOS files.",
            )

    parser.add_argument(
            "--grid",
            default=str(DEFAULT_GRID),
            help="Smeared template grid used by the fit.",
            )

    parser.add_argument(
            "-o",
            "--output-dir",
            default=str(DEFAULT_OUTPUT_DIR),
            help="Folder to store per-run .npz files and the summary.",
            )

    parser.add_argument(
            "-j",
            "--jobs",
            type=int,
            default=6,
            help="Number of runs processed in parallel.",
            )

    parser.add_argument(
            "--cmos-window-s",
            type=float,
            default=10.0,
            help="Half-width of the CMOS averaging window around the run timestamp, in seconds.",
            )

    parser.add_argument(
            "--dates",
            nargs="*",
            default=None,
            help="Only process these date folders (e.g. 2025-12-01).",
            )

    parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Process at most this many runs (for testing).",
            )

    parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Reprocess runs that already have an output file.",
            )

    return parser.parse_args()


def run_name_from_path(root_path):
    # Keep any suffix (e.g. ".28", ".pilot.1") so variant files of the same
    # timestamp get their own output.
    if RUN_NAME_RE.match(root_path.name) is None:
        raise ValueError(f"Cannot parse run name from {root_path.name}")

    return root_path.name.removesuffix(".root").removesuffix("_cleaned")


def run_timestamp(run_name):
    match = RUN_NAME_RE.match(run_name)

    return datetime.strptime(f"{match.group(1)}.{match.group(2)}", "%Y-%m-%d.%H%M%S")


def find_cleaned_runs(data_dir, dates=None):
    runs = []

    for date_dir in sorted(Path(data_dir).iterdir()):
        if not date_dir.is_dir():
            continue

        if dates is not None and date_dir.name not in dates:
            continue

        runs.extend(sorted(
            path for path in (date_dir / "cleaned").glob("*_cleaned.root")
            if FULL_RUN_FILE_RE.fullmatch(path.name)
        ))

    return runs


def load_cmos_file(path):
    """Return (timestamps, values) from a .cmos_sigmay.HER file, skipping status lines."""
    times = []
    values = []

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")

            if len(parts) != 3:
                continue

            try:
                value = float(parts[2])
            except ValueError:
                continue

            times.append(datetime.fromisoformat(parts[0]))
            values.append(value)

    return np.array(times), np.array(values)


def cmos_is_suspect(values):
    values = np.asarray(values, dtype=float)
    suspect = np.zeros(values.shape, dtype=bool)

    for v in CMOS_SUSPECT_VALUES:
        suspect |= np.abs(values - v) < CMOS_SUSPECT_TOL

    return suspect


def cmos_for_run(root_path, run_name, window_s):
    date_dir = root_path.parent.parent
    cmos_files = sorted(date_dir.glob("*.cmos_sigmay.HER"))

    result = {
        "cmos_sigma_median": np.nan,
        "cmos_n": 0,
        "cmos_n_suspect": 0,
        "cmos_values": np.array([]),
        "cmos_dt_s": np.array([]),
    }

    if len(cmos_files) == 0:
        return result

    times, values = load_cmos_file(cmos_files[0])
    t_run = run_timestamp(run_name)
    window = timedelta(seconds=window_s)

    in_window = (times >= t_run - window) & (times <= t_run + window)
    values = values[in_window]
    dt_s = np.array([(t - t_run).total_seconds() for t in times[in_window]])

    suspect = cmos_is_suspect(values)
    good = values[~suspect]

    result["cmos_n"] = len(values)
    result["cmos_n_suspect"] = int(suspect.sum())
    result["cmos_values"] = values
    result["cmos_dt_s"] = dt_s

    if len(good) > 0:
        result["cmos_sigma_median"] = float(np.median(good))

    return result


def match_bunches(times_a, times_b, tolerance_ns):
    """Return index pairs (ia, ib) of bunches whose times agree within tolerance."""
    idx = np.searchsorted(times_b, times_a)
    idx = np.clip(idx, 1, len(times_b) - 1)

    left = times_b[idx - 1]
    right = times_b[idx]
    nearest = np.where(np.abs(times_a - left) <= np.abs(times_a - right), idx - 1, idx)

    close = np.abs(times_a - times_b[nearest]) <= tolerance_ns
    ia = np.nonzero(close)[0]
    ib = nearest[close]

    # Drop duplicate matches to the same bunch, keeping the first.
    ib, first = np.unique(ib, return_index=True)
    ia = ia[first]

    return ia, ib


def fit_bunches(heights, grid):
    n_events = heights.shape[0]

    fit_params = np.full((n_events, 6), np.nan)
    fit_cost = np.full(n_events, np.nan)

    for i in range(n_events):
        fit_params[i], fit_cost[i] = ffr.fit_one_bunch_index(heights[i], grid)

    return fit_params, fit_cost


def init_worker(grid_path):
    global _grid
    _grid = ffr.SmearedGridModel.from_npz(grid_path)


def process_run(task):
    root_path, output_path, cmos_window_s = task
    run_name = run_name_from_path(root_path)

    try:
        # Silence the [INFO] prints from fit_from_rootfile.
        with contextlib.redirect_stdout(io.StringIO()):
            # Reference: full processing, identical to fit_results/.
            fit_heights, fit_times = ffr.extract_peak_heights_from_root(
                root_path,
                use_cubic=True,
                use_alignment=True,
                use_baseline=True,
                use_gain=True,
            )

            # Firmware-like: raw samples, no interpolation or alignment.
            neg_valley, fw_times = ffr.extract_peak_heights_from_root(
                root_path,
                use_cubic=False,
                use_alignment=False,
                use_baseline=False,
                use_gain=False,
            )
            baseline_minus_valley, _ = ffr.extract_peak_heights_from_root(
                root_path,
                use_cubic=False,
                use_alignment=False,
                use_baseline=True,
                use_gain=False,
            )

        valley_raw = -neg_valley
        baseline = baseline_minus_valley - neg_valley

        i_fw, i_fit = match_bunches(fw_times, fit_times, MATCH_TOLERANCE_NS)

        fit_params, fit_cost = fit_bunches(fit_heights[i_fit], _grid)

        cmos = cmos_for_run(root_path, run_name, cmos_window_s)

        np.savez(
            output_path,
            run=run_name,
            root_path=str(root_path),
            channels=np.array(ffr.USING_CHANNEL),
            gain=ffr.GAIN_CALIB_FACTOR,
            bunch_times_ns=fw_times[i_fw],
            valley_raw=valley_raw[i_fw],
            baseline=baseline[i_fw],
            fit_params=fit_params,
            fit_mu=fit_params[:, 0],
            fit_sigma=fit_params[:, 1],
            fit_cost=fit_cost,
            n_bunches_fit_path=len(fit_times),
            n_bunches_fw_path=len(fw_times),
            **cmos,
        )

        return {
            "run": run_name,
            "status": "ok",
            "n_matched": len(i_fw),
            "n_fit_path": len(fit_times),
            "n_fw_path": len(fw_times),
            "fit_sigma_median": float(np.nanmedian(fit_params[:, 1])),
            "baseline_median": float(np.nanmedian(baseline[i_fw])),
            "cmos_sigma_median": cmos["cmos_sigma_median"],
            "cmos_n": cmos["cmos_n"],
            "cmos_n_suspect": cmos["cmos_n_suspect"],
            "error": "",
        }

    except Exception as exc:
        return {
            "run": run_name,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def classify_run(result):
    if result["status"] != "ok":
        return "failed"

    if result["baseline_median"] < CORRUPTED_BASELINE_ADC:
        return "corrupted"

    if result["n_matched"] < LOW_FILL_MAX_BUNCHES:
        return "low_fill"

    return "good"


def summarize_existing(output_path):
    data = np.load(output_path)

    return {
        "run": str(data["run"]),
        "status": "ok",
        "n_matched": len(data["bunch_times_ns"]),
        "n_fit_path": int(data["n_bunches_fit_path"]),
        "n_fw_path": int(data["n_bunches_fw_path"]),
        "fit_sigma_median": float(np.nanmedian(data["fit_sigma"])),
        "baseline_median": float(np.nanmedian(data["baseline"])),
        "cmos_sigma_median": float(data["cmos_sigma_median"]),
        "cmos_n": int(data["cmos_n"]),
        "cmos_n_suspect": int(data["cmos_n_suspect"]),
        "error": "",
    }


def write_summary(results, path):
    fields = [
        "run",
        "status",
        "category",
        "n_matched",
        "n_fit_path",
        "n_fw_path",
        "fit_sigma_median",
        "baseline_median",
        "cmos_sigma_median",
        "cmos_n",
        "cmos_n_suspect",
        "error",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()

        for result in sorted(results, key=lambda r: r["run"]):
            writer.writerow(result)


def estimate_pedestals(output_paths, path):
    """Fixed per-channel pedestal: median baseline over every bunch of every run."""
    run_medians = []
    runs = []

    for output_path in output_paths:
        data = np.load(output_path)
        run_medians.append(np.nanmedian(data["baseline"], axis=0))
        runs.append(str(data["run"]))

    run_medians = np.array(run_medians)
    pedestal = np.nanmedian(run_medians, axis=0)

    np.savez(
        path,
        pedestal=pedestal,
        run_medians=run_medians,
        runs=np.array(runs),
        channels=np.array(ffr.USING_CHANNEL),
    )

    return pedestal, run_medians


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    root_paths = find_cleaned_runs(args.data_dir, args.dates)

    if args.limit is not None:
        root_paths = root_paths[:args.limit]

    tasks = []
    results = []

    for root_path in root_paths:
        output_path = runs_dir / f"{run_name_from_path(root_path)}.npz"

        if output_path.exists() and not args.overwrite:
            results.append(summarize_existing(output_path))
        else:
            tasks.append((root_path, output_path, args.cmos_window_s))

    print(f"Found {len(root_paths)} cleaned runs, {len(tasks)} to process, "
          f"{len(results)} already done.")

    with Pool(args.jobs, initializer=init_worker, initargs=(args.grid,)) as pool:
        for i, result in enumerate(pool.imap_unordered(process_run, tasks), start=1):
            results.append(result)

            if result["status"] == "ok":
                print(f"[{i}/{len(tasks)}] {result['run']}: "
                      f"{result['n_matched']} bunches, "
                      f"fit sigma {result['fit_sigma_median']:.1f}, "
                      f"CMOS {result['cmos_sigma_median']:.1f}", flush=True)
            else:
                print(f"[{i}/{len(tasks)}] {result['run']}: FAILED {result['error']}",
                      flush=True)

    for result in results:
        result["category"] = classify_run(result)

    write_summary(results, output_dir / "runs_summary.csv")

    failed = [r for r in results if r["status"] != "ok"]

    if failed:
        with open(output_dir / "failed_runs.log", "w") as f:
            for result in failed:
                f.write(f"== {result['run']}\n{result.get('traceback', result['error'])}\n")

    ok_paths = sorted(runs_dir / f"{r['run']}.npz" for r in results if r["category"] == "good")

    if ok_paths:
        pedestal, run_medians = estimate_pedestals(ok_paths, output_dir / "pedestals.npz")
        drift = np.nanmax(run_medians, axis=0) - np.nanmin(run_medians, axis=0)

        print(f"\nPedestals (ADC): {np.round(pedestal, 1)}")
        print(f"Pedestal drift across runs, max - min of run medians (ADC): {np.round(drift, 1)}")

    counts = {c: sum(r["category"] == c for r in results)
              for c in ("good", "low_fill", "corrupted", "failed")}

    print(f"\nRun categories: {counts}. Output in {output_dir}")


if __name__ == "__main__":
    main()
