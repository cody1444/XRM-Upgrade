#!/usr/bin/env python3

"""
Measure the board-wide offset of each beam-on sweep from the edge channels,
and how it varies between sweeps.

Offset of a sweep: median level of the edge channels (almost no X-rays)
relative to the beam-off pedestals, averaged over the edge channels.
  - jitter: RMS of differences between consecutive sweeps / sqrt(2),
    split by the time between the two sweeps (slow drift makes the
    differences grow with the time gap)
  - drift: median offset per day

Also checks that the edge channels carry no signal: with no pulse, the
firmware-style amplitude (pedestal - lowest sample in the window) is set
by noise alone, so the edge amplitudes should match the beam-off value.
"""

import argparse
from datetime import datetime
from multiprocessing import Pool

import numpy as np

import xrm_raw
from xrm_raw import ffr

TIME_GAP_BINS_MIN = [(0, 2), (2, 10), (10, 60), (60, 24 * 60)]

_pedestal = None


def parse_args():
    parser = argparse.ArgumentParser(
            prog="measure_sweep_offset",
            description="Measure the board-wide offset per sweep from the edge channels.",
            )

    parser.add_argument(
            "-o",
            "--output",
            default=str(xrm_raw.REAL_DIR.parent / "sweep_offset.npz"),
            help="Output .npz file.",
            )

    parser.add_argument(
            "-j",
            "--jobs",
            type=int,
            default=8,
            help="Number of sweeps processed in parallel.",
            )

    return parser.parse_args()


def init_worker(pedestal):
    global _pedestal
    _pedestal = pedestal


def window_amplitude_no_pulse(x, n_windows=500):
    """Firmware-style amplitude of random windows: pedestal-free noise floor."""
    width = int(round(2 * ffr.WINDOW_HALF_NS * ffr.ADC_GSPS)) + 1
    starts = np.linspace(0, x.shape[1] - width - 1, n_windows).astype(int)
    level = np.median(x, axis=1, keepdims=True)

    return np.mean([level[:, 0] - x[:, s:s + width].min(axis=1) for s in starts], axis=0)


def process_sweep(run):
    edge = xrm_raw.EDGE_CHANNELS
    x = xrm_raw.read_raw_waveforms(xrm_raw.cleaned_root_path(run))[edge] - _pedestal[edge, None]

    data = np.load(xrm_raw.REAL_DIR / "runs" / f"{run}.npz")
    samples = np.sort(data["bunch_times_ns"]) * ffr.ADC_GSPS
    g = np.argmax(np.diff(samples))
    gap = x[:, int(samples[g]) + 300:int(samples[g + 1]) - 30]

    whole_level = np.median(x, axis=1)
    edge_amplitude = (_pedestal[edge] - data["valley_raw"][:, edge]).mean(axis=0)

    return {
        "run": run,
        "offset_gap": np.median(gap, axis=1),
        "offset_whole": whole_level,
        # Pulse depth above this sweep's own level, minus what noise alone gives.
        "edge_signal": edge_amplitude + whole_level - window_amplitude_no_pulse(x),
    }


def sweep_time(run):
    return datetime.strptime(run, "%Y-%m-%d.%H%M%S")


def main():
    args = parse_args()

    print("Computing beam-off pedestals...")
    pedestal = xrm_raw.beam_off_pedestals()

    runs = sorted(xrm_raw.good_runs(), key=sweep_time)

    with Pool(args.jobs, initializer=init_worker, initargs=(pedestal,)) as pool:
        results = pool.map(process_sweep, runs)

    offset_gap = np.array([r["offset_gap"] for r in results])
    offset_whole = np.array([r["offset_whole"] for r in results])
    edge_signal = np.array([r["edge_signal"] for r in results])
    times = np.array([sweep_time(r) for r in runs])

    offset = offset_whole.mean(axis=1)

    np.savez(
        args.output,
        runs=np.array(runs),
        edge_channels=np.array(xrm_raw.EDGE_CHANNELS),
        offset_gap=offset_gap,
        offset_whole=offset_whole,
        edge_signal=edge_signal,
    )

    print(f"{len(runs)} good sweeps, edge channels {xrm_raw.EDGE_CHANNELS}")

    print("\n1. Offset per sweep (mean over edge channels, ADC relative to beam-off pedestal)")
    print(f"   range {offset.min():.1f} .. {offset.max():.1f}, median {np.median(offset):.1f}, "
          f"std {offset.std():.1f}")
    print(f"   abort gap vs whole sweep: median difference "
          f"{np.median(offset_gap.mean(1) - offset):+.1f}, std {np.std(offset_gap.mean(1) - offset):.1f}")
    spread = offset_whole.std(axis=1)
    print(f"   spread across the 8 edge channels within a sweep: median {np.median(spread):.1f}, "
          f"max {spread.max():.1f}")

    print("\n2. Jitter: consecutive sweeps, RMS of differences / sqrt(2)")
    dt_min = np.array([(b - a).total_seconds() / 60 for a, b in zip(times[:-1], times[1:])])
    diff = np.diff(offset)
    same_day = np.array([a.date() == b.date() for a, b in zip(times[:-1], times[1:])])
    for lo, hi in TIME_GAP_BINS_MIN:
        mask = same_day & (dt_min >= lo) & (dt_min < hi)
        if mask.sum() >= 3:
            rms = np.sqrt(np.mean(diff[mask] ** 2))
            print(f"   {lo:>4d}-{hi:<5d} min apart: {mask.sum():4d} pairs, "
                  f"RMS diff {rms:5.1f}  ->  per-sweep {rms / np.sqrt(2):5.1f} ADC")

    print("\n3. Drift: median offset per day")
    days = np.array([t.date().isoformat() for t in times])
    for day in np.unique(days):
        mask = days == day
        print(f"   {day}: {mask.sum():4d} sweeps, median {np.median(offset[mask]):+6.1f}, "
              f"spread within the day (std) {offset[mask].std():5.1f}")

    print("\n4. Do the edge channels carry signal? Pulse depth beyond noise alone (ADC)")
    for i, ch in enumerate(xrm_raw.EDGE_CHANNELS):
        print(f"   ch {ch:2d}: median {np.median(edge_signal[:, i]):5.1f}, "
              f"95th percentile {np.percentile(edge_signal[:, i], 95):5.1f}")

    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
