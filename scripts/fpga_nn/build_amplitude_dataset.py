#!/usr/bin/env python3

"""
Per-bunch, per-channel amplitudes for the generator study.

For every good and low-fill sweep, on the bunches in data/real/runs:
  reference : full-processing heights (cubic, alignment, baseline, gain);
              what the template fit in fit_results was made on
  firmware  : E50 firmware-model amplitudes from the raw waveform
  model     : template prediction from the bunch's reference fit parameters

reference - model  = how real bunches deviate from a template
firmware - reference = what the detector readout + firmware does to them
"""

import argparse
import contextlib
import io
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import firmware_model
import xrm_raw
from xrm_raw import ffr
from build_real_dataset import match_bunches, MATCH_TOLERANCE_NS

DEFAULT_GRID = xrm_raw.REPO_DIR / "beam_profiles" / "smeared_grid_CMOS_upto200um.npz"
DEFAULT_OUTPUT_DIR = xrm_raw.REAL_DIR.parent / "amplitudes"

_grid = None
_pedestal = None


def parse_args():
    parser = argparse.ArgumentParser(
            prog="build_amplitude_dataset",
            description="Per-bunch reference, firmware and template amplitudes for the generator study.",
            )

    parser.add_argument(
            "-o",
            "--output-dir",
            default=str(DEFAULT_OUTPUT_DIR),
            help="Folder for one .npz per sweep.",
            )

    parser.add_argument(
            "-j",
            "--jobs",
            type=int,
            default=8,
            help="Number of sweeps processed in parallel.",
            )

    parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Reprocess sweeps that already have an output file.",
            )

    return parser.parse_args()


def init_worker(grid_path, pedestal):
    global _grid, _pedestal
    _grid = ffr.SmearedGridModel.from_npz(grid_path)
    _pedestal = pedestal


def template_prediction(fit_params):
    channels = np.arange(len(ffr.USING_CHANNEL), dtype=float)
    model = np.full((fit_params.shape[0], len(channels)), np.nan)

    for i, params in enumerate(fit_params):
        if np.all(np.isfinite(params)):
            model[i] = _grid.model_profile_for_channels_index(channels, *params)

    return model


def process_sweep(task):
    run, category, current, output_path = task
    root_path = xrm_raw.cleaned_root_path(run)
    data = np.load(xrm_raw.REAL_DIR / "runs" / f"{run}.npz")

    order = np.argsort(data["bunch_times_ns"])
    times = data["bunch_times_ns"][order]
    fit_params = data["fit_params"][order]

    x = xrm_raw.read_raw_waveforms(root_path)
    peaks = np.rint(times * ffr.ADC_GSPS).astype(int)
    keep = firmware_model.valid_peaks(peaks, x.shape[1])
    times, peaks, fit_params = times[keep], peaks[keep], fit_params[keep]

    firmware = firmware_model.firmware_amplitudes(x, peaks, _pedestal, data["gain"])

    with contextlib.redirect_stdout(io.StringIO()):
        ref_heights, ref_times = ffr.extract_peak_heights_from_root(
            root_path, use_cubic=True, use_alignment=True, use_baseline=True, use_gain=True,
        )

    reference = np.full_like(firmware, np.nan)
    i_ours, i_ref = match_bunches(times, ref_times, MATCH_TOLERANCE_NS)
    reference[i_ours] = ref_heights[i_ref]

    np.savez(
        output_path,
        run=run,
        category=category,
        current_ma=current,
        bunch_times_ns=times,
        fit_params=fit_params,
        reference=reference,
        firmware=firmware,
        model=template_prediction(fit_params),
    )

    return run, len(times), len(i_ours)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Computing beam-off pedestals...")
    pedestal = xrm_raw.beam_off_pedestals()
    dcct = xrm_raw.DcctCurrent()

    tasks = []
    for category in ("good", "low_fill"):
        for run in xrm_raw.good_runs(category):
            output_path = output_dir / f"{run}.npz"
            if args.overwrite or not output_path.exists():
                tasks.append((run, category, dcct(run), output_path))

    print(f"{len(tasks)} sweeps to process")

    with Pool(args.jobs, initializer=init_worker, initargs=(str(DEFAULT_GRID), pedestal)) as pool:
        for i, (run, n, n_matched) in enumerate(pool.imap_unordered(process_sweep, tasks), start=1):
            if i % 50 == 0 or n_matched < 0.95 * n:
                print(f"[{i}/{len(tasks)}] {run}: {n} bunches, {n_matched} matched to reference", flush=True)

    np.savez(output_dir.parent / "amplitudes_pedestal.npz", pedestal=pedestal)
    print(f"Done. Output in {output_dir}")


if __name__ == "__main__":
    main()
