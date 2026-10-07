#!/usr/bin/env python3

"""
Measure the signal-dependent baseline shift along the bunch train.

For every good run, relative to the beam-off pedestal P:
  - firmware-style amplitude of each bunch:  A = P - raw valley sample
  - waveform level L: median of each 64-sample readout window
Both are binned by time since the start of the bunch train and averaged
over runs. If the shift is real, L rises along the train in channels with
signal while A drops by the same amount, so A + L stays roughly constant.
Edge channels (almost no signal) act as the control.
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import xrm_raw
from xrm_raw import ffr

# Time since train start, ns. Fine bins early on, where the shift builds up.
TIME_BIN_EDGES_NS = np.array([0, 10, 20, 40, 60, 100, 150, 200, 300, 500, 1000, 2000, 3000, 5000])
STEADY_STATE_NS = 500.0

CURRENT_CLASSES_MA = [(0, 600), (600, 900), (900, 2000)]

_pedestal = None


def parse_args():
    parser = argparse.ArgumentParser(
            prog="measure_baseline_shift",
            description="Measure the signal-dependent baseline shift along the bunch train.",
            )

    parser.add_argument(
            "-o",
            "--output",
            default=str(xrm_raw.REAL_DIR.parent / "baseline_shift.npz"),
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


def train_start_times(bunch_times_ns):
    """Start time of the train each bunch belongs to."""
    new_train = np.concatenate(([True], np.diff(bunch_times_ns) > ffr.GAP_THRESHOLD_NS))
    train_index = np.cumsum(new_train) - 1
    starts = bunch_times_ns[new_train]
    ends = bunch_times_ns[np.concatenate((new_train[1:], [True]))]

    return starts, ends, starts[train_index]


def binned_mean(times, values, edges):
    """Mean of values (n, n_ch) in each time bin; NaN for empty bins."""
    out = np.full((len(edges) - 1, values.shape[1]), np.nan)
    which = np.digitize(times, edges) - 1

    for b in range(len(edges) - 1):
        mask = which == b
        if mask.any():
            out[b] = values[mask].mean(axis=0)

    return out


def process_run(run):
    data = np.load(xrm_raw.REAL_DIR / "runs" / f"{run}.npz")

    bunch_times = data["bunch_times_ns"]
    order = np.argsort(bunch_times)
    bunch_times = bunch_times[order]
    amplitude = _pedestal - data["valley_raw"][order]

    starts, ends, own_start = train_start_times(bunch_times)
    amp_binned = binned_mean(bunch_times - own_start, amplitude, TIME_BIN_EDGES_NS)

    # Level per 64-sample readout window, placed on the same time axis.
    x = xrm_raw.read_raw_waveforms(xrm_raw.cleaned_root_path(run)) - _pedestal[:, None]
    n_windows = x.shape[1] // ffr.NSAMPLE_PER_WINDOW
    level = np.median(
        x[:, :n_windows * ffr.NSAMPLE_PER_WINDOW].reshape(42, n_windows, ffr.NSAMPLE_PER_WINDOW),
        axis=2,
    ).T
    window_t = (np.arange(n_windows) + 0.5) * ffr.NSAMPLE_PER_WINDOW / ffr.ADC_GSPS

    t_since = np.full(n_windows, np.nan)
    for start, end in zip(starts, ends):
        inside = (window_t >= start) & (window_t <= end)
        t_since[inside] = window_t[inside] - start

    in_train = np.isfinite(t_since)
    level_binned = binned_mean(t_since[in_train], level[in_train], TIME_BIN_EDGES_NS)

    steady = (bunch_times - own_start) >= STEADY_STATE_NS

    return {
        "run": run,
        "n_bunches": len(bunch_times),
        "amp_binned": amp_binned,
        "level_binned": level_binned,
        "amp_steady": amplitude[steady].mean(axis=0) if steady.any() else np.full(42, np.nan),
        "level_steady": level[in_train & (t_since >= STEADY_STATE_NS)].mean(axis=0),
    }


def print_group_table(amp, level, label, mask):
    print(f"\n=== {label}: {mask.sum()} runs ===")
    header = f"{'t since train start':>20s}"
    for group in xrm_raw.CHANNEL_GROUPS:
        header += f" | {group + ' A':>9s} {'L':>6s} {'A+L':>6s}"
    print(header)

    amp_mean = np.nanmean(amp[mask], axis=0)
    level_mean = np.nanmean(level[mask], axis=0)

    for b in range(len(TIME_BIN_EDGES_NS) - 1):
        row = f"{TIME_BIN_EDGES_NS[b]:>8d}-{TIME_BIN_EDGES_NS[b + 1]:<5d} ns    "
        for chs in xrm_raw.CHANNEL_GROUPS.values():
            a = amp_mean[b, chs].mean()
            l = level_mean[b, chs].mean()
            row += f" | {a:9.1f} {l:6.1f} {a + l:6.1f}"
        print(row)


def main():
    args = parse_args()

    print("Computing beam-off pedestals...")
    pedestal = xrm_raw.beam_off_pedestals()

    runs = xrm_raw.good_runs()
    dcct = xrm_raw.DcctCurrent()

    with Pool(args.jobs, initializer=init_worker, initargs=(pedestal,)) as pool:
        results = pool.map(process_run, runs)

    current = np.array([dcct(r["run"]) for r in results])
    n_bunches = np.array([r["n_bunches"] for r in results])
    amp = np.array([r["amp_binned"] for r in results])
    level = np.array([r["level_binned"] for r in results])
    amp_steady = np.array([r["amp_steady"] for r in results])
    level_steady = np.array([r["level_steady"] for r in results])

    np.savez(
        args.output,
        runs=np.array(runs),
        pedestal=pedestal,
        current=current,
        n_bunches=n_bunches,
        time_bin_edges_ns=TIME_BIN_EDGES_NS,
        amp_binned=amp,
        level_binned=level,
        amp_steady=amp_steady,
        level_steady=level_steady,
    )

    print("A = firmware-style amplitude (pedestal - raw valley), L = waveform level above pedestal, ADC")

    for lo, hi in CURRENT_CLASSES_MA:
        mask = (current >= lo) & (current < hi)
        if mask.any():
            print_group_table(amp, level, f"DCCT {lo}-{hi} mA", mask)

    # Dose-response: steady-state level vs the channel's signal per turn,
    # using A + L as the shift-corrected pulse depth.
    signal_per_turn = (amp_steady + level_steady) * n_bunches[:, None]
    ok = np.isfinite(signal_per_turn) & np.isfinite(level_steady)
    slope, intercept = np.polyfit(signal_per_turn[ok], level_steady[ok], 1)
    corr = np.corrcoef(signal_per_turn[ok], level_steady[ok])[0, 1]

    print(f"\nSteady-state level vs signal per turn, all channels x runs: "
          f"L = {slope:.3e} * (A+L) * N_bunches + {intercept:.1f} ADC, corr {corr:.3f}")

    frac = level_steady / (amp_steady + level_steady)
    print("Steady-state level as a fraction of the corrected pulse depth, L / (A+L), median over runs:")
    for lo, hi in CURRENT_CLASSES_MA:
        mask = (current >= lo) & (current < hi)
        if mask.any():
            parts = [f"{g} {np.nanmedian(frac[mask][:, chs]):.2f}" for g, chs in xrm_raw.CHANNEL_GROUPS.items()]
            print(f"  DCCT {lo}-{hi} mA: " + ", ".join(parts))

    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
