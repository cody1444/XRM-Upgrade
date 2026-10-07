#!/usr/bin/env python3

"""
Measure how much pulse depth is lost by taking the minimum raw sample
(firmware-style, 2.7 GSPS) instead of the minimum of a cubic-spline
interpolation (x10), for the same bunches.

Both depths are measured from the beam-off pedestal, with no alignment,
baseline subtraction or gain, so the only difference is interpolation:
  loss = 1 - depth_raw / depth_interp

The loss is split into
  - a fixed per-channel part (pulse shape / timing of that channel),
  - a per-bunch part common to all channels (bunch arrival phase),
  - the remaining channel-and-bunch-specific scatter.
Only the parts that differ between channels change the normalized shape.
"""

import argparse
import contextlib
import io
from multiprocessing import Pool

import numpy as np

import xrm_raw
from xrm_raw import ffr
from build_real_dataset import match_bunches, MATCH_TOLERANCE_NS

MIN_DEPTH_ADC = 100.0
DEPTH_BINS_ADC = [100, 200, 400, 800, 1600, 5000]

_pedestal = None


def parse_args():
    parser = argparse.ArgumentParser(
            prog="measure_sampling_loss",
            description="Measure pulse depth lost by not interpolating between samples.",
            )

    parser.add_argument(
            "-n",
            "--n-runs",
            type=int,
            default=60,
            help="Number of good runs to use, spread evenly over the run list.",
            )

    parser.add_argument(
            "-o",
            "--output",
            default=str(xrm_raw.REAL_DIR.parent / "sampling_loss.npz"),
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


def init_worker(pedestal):
    global _pedestal
    _pedestal = pedestal


def process_run(run):
    root_path = xrm_raw.cleaned_root_path(run)

    with contextlib.redirect_stdout(io.StringIO()):
        neg_valley_interp, times_interp = ffr.extract_peak_heights_from_root(
            root_path, use_cubic=True, use_alignment=False, use_baseline=False, use_gain=False,
        )
        neg_valley_raw, times_raw = ffr.extract_peak_heights_from_root(
            root_path, use_cubic=False, use_alignment=False, use_baseline=False, use_gain=False,
        )

    i_raw, i_interp = match_bunches(times_raw, times_interp, MATCH_TOLERANCE_NS)

    depth_raw = _pedestal + neg_valley_raw[i_raw]
    depth_interp = _pedestal + neg_valley_interp[i_interp]

    return run, depth_raw, depth_interp


def main():
    args = parse_args()

    print("Computing beam-off pedestals...")
    pedestal = xrm_raw.beam_off_pedestals()

    runs = xrm_raw.good_runs()
    runs = [runs[i] for i in np.linspace(0, len(runs) - 1, args.n_runs).astype(int)]

    with Pool(args.jobs, initializer=init_worker, initargs=(pedestal,)) as pool:
        results = pool.map(process_run, runs)

    depth_raw = np.concatenate([r[1] for r in results])
    depth_interp = np.concatenate([r[2] for r in results])

    np.savez(args.output, runs=np.array(runs), depth_raw=depth_raw, depth_interp=depth_interp)

    loss = 1.0 - depth_raw / depth_interp
    loss[depth_interp < MIN_DEPTH_ADC] = np.nan

    print(f"{len(runs)} runs, {len(depth_raw)} bunches, "
          f"{np.isfinite(loss).sum()} channel-bunch entries with interpolated depth > {MIN_DEPTH_ADC:.0f} ADC")
    print(f"Overall loss: mean {np.nanmean(loss) * 100:.1f}%, std {np.nanstd(loss) * 100:.1f}%, "
          f"16-84%: {np.nanpercentile(loss, 16) * 100:.1f}..{np.nanpercentile(loss, 84) * 100:.1f}%")

    channel_mean = np.nanmean(loss, axis=0)
    resid = loss - channel_mean
    bunch_common = np.nanmean(resid, axis=1)
    specific = resid - bunch_common[:, None]

    print("\nDecomposition:")
    print(f"  fixed per-channel part: spread across channels (std) {np.nanstd(channel_mean) * 100:.1f}%")
    print(f"  per-bunch part common to all channels: std {np.nanstd(bunch_common) * 100:.1f}%")
    print(f"  channel-and-bunch-specific scatter: std {np.nanstd(specific) * 100:.1f}%")

    print("\nMean loss per channel (%), channels with enough signal:")
    counts = np.isfinite(loss).sum(axis=0)
    for ch in range(42):
        if counts[ch] > 1000:
            print(f"  ch {ch:2d}: {channel_mean[ch] * 100:5.1f}%  (scatter {np.nanstd(specific[:, ch]) * 100:4.1f}%, n={counts[ch]})")

    print("\nLoss vs interpolated depth:")
    for lo, hi in zip(DEPTH_BINS_ADC[:-1], DEPTH_BINS_ADC[1:]):
        mask = (depth_interp >= lo) & (depth_interp < hi)
        if mask.any():
            print(f"  {lo:5d}-{hi:<5d} ADC: mean {np.nanmean(loss[mask]) * 100:5.1f}%, "
                  f"std {np.nanstd(loss[mask]) * 100:4.1f}%  (n={mask.sum()})")

    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
