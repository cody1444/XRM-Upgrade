#!/usr/bin/env python3

"""
Same question as study_baseline_donors.py, with the instantaneous baseline.

For a few calibration sweeps, recompute from the raw waveforms what
firmware_model.py uses per bunch and channel:
  line     the instantaneous baseline: the line between the highest
           sample before and after the pulse, at the pulse (raw ADC)
  running  the E50 running average of line, which the firmware subtracts
and check the result by recomputing the stored firmware amplitudes.

Then, per channel, how much of the firmware deviation
    deviation = firmware - (gain * corrected template + offset)
is explained by a straight line in
    line - pedestal       (the instantaneous baseline itself)
    running - line        (the firmware's baseline error for this bunch)
fitted on half of the sweeps and tested on the other half.

With --train, also train the centre 16-channel network with and without a
donor term  a + b * (line - pedestal)  as in study_baseline_donors.py.
"""

import argparse

import numpy as np

import firmware_model
import xrm_raw
from xrm_raw import ffr
from xrm_generator import (XrmGenerator, AMPLITUDE_DIR, DATA_DIR, DEFAULT_CALIBRATION, FIT_LOWER, FIT_UPPER,
                           corrected_template)
from train_nn import ROSTERS
from study_baseline_donors import ROSTER, robust_var, fit_lines

CACHE_DIR = DATA_DIR / "instant_baseline"


def parse_args():
    parser = argparse.ArgumentParser(prog="study_instant_baseline", description=__doc__.strip().splitlines()[0])
    parser.add_argument("--n-sweeps", type=int, default=10, help="Calibration sweeps to use.")
    parser.add_argument("--train", action="store_true", help="Also run the network comparison.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def instant_baselines(sweep, pedestal):
    """line and running baseline (n_bunches, 42) for the bunches of data/amplitudes/<sweep>.npz."""
    cache = CACHE_DIR / f"{sweep}.npz"
    if cache.exists():
        c = np.load(cache)
        return c["line"], c["running"]

    # Same bunches and peak positions as build_amplitude_dataset.py.
    run = np.load(xrm_raw.REAL_DIR / "runs" / f"{sweep}.npz")
    times = np.sort(run["bunch_times_ns"])
    x = xrm_raw.read_raw_waveforms(xrm_raw.cleaned_root_path(sweep))
    peaks = np.rint(times * ffr.ADC_GSPS).astype(int)
    keep = firmware_model.valid_peaks(peaks, x.shape[1])
    times, peaks = times[keep], peaks[keep]

    # Same steps as firmware_model.firmware_amplitudes.
    left_max, right_max, t_left, t_right = firmware_model.side_maxima(x, peaks)
    span = np.where(t_right == t_left, 1, t_right - t_left)
    line = left_max + (right_max - left_max) * (peaks[None, :] - t_left) / span
    running = firmware_model.exponential_average(line, firmware_model.BASELINE_N_BUNCHES, pedestal)

    # Check: these reproduce the stored bunches and firmware amplitudes.
    amp = np.load(AMPLITUDE_DIR / f"{sweep}.npz")
    assert np.array_equal(times, amp["bunch_times_ns"]), sweep
    firmware = firmware_model.firmware_amplitudes(x, peaks, pedestal, run["gain"])
    assert np.allclose(firmware, amp["firmware"], equal_nan=True), sweep

    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(cache, line=line.T, running=running.T)
    return line.T, running.T


def load_sweep(sweep, gen, pedestal):
    """Per good bunch: fit parameters, firmware deviation, line - pedestal, running - line."""
    line, running = instant_baselines(sweep, pedestal)
    amp = np.load(AMPLITUDE_DIR / f"{sweep}.npz")
    params, firmware = amp["fit_params"], amp["firmware"]
    ok = np.all(np.isfinite(params), axis=1) & np.all(np.isfinite(firmware), axis=1)
    ok &= ~np.any((np.abs(params - FIT_LOWER) < 1e-3) | (np.abs(params - FIT_UPPER) < 1e-3), axis=1)

    deviation = np.array([f - (gen.fw_gain * corrected_template(gen.grid, p, gen.factors, gen.channels) + gen.fw_offset)
                          for f, p in zip(firmware[ok], params[ok])])
    return params[ok], deviation, line[ok] - pedestal, running[ok] - line[ok]


def explained_table(name, dev, var, roster):
    """Fit on the first half, test on the second; print per channel and return the median explained."""
    a, b = fit_lines(dev["fit"], var["fit"])
    pred = a + b * var["test"]
    print(f"\n{name}  (test half, {len(dev['test'])} bunches)")
    print(f"{'ch':>4s} {'median':>8s} {'rms':>7s} {'dev rms':>8s} {'slope b':>8s} {'corr':>6s} {'explained':>9s}")
    explained = []
    for ch in roster:
        v, d = var["test"][:, ch], dev["test"][:, ch]
        e = 1 - robust_var(d - pred[:, ch]) / robust_var(d)
        explained.append(e)
        print(f"{ch:>4d} {np.median(v):8.1f} {np.sqrt(robust_var(v)):7.1f} {np.sqrt(robust_var(d)):8.1f} "
              f"{b[ch]:+8.3f} {np.corrcoef(v, d)[0, 1]:+6.2f} {100 * e:8.0f}%")
    print(f"Median over channels {roster[0]}-{roster[-1]}: {100 * np.median(explained):.0f}% explained")
    return np.median(explained)


def main():
    args = parse_args()
    gen = XrmGenerator(DEFAULT_CALIBRATION)
    pedestal = np.load(DATA_DIR / "amplitudes_pedestal.npz")["pedestal"]
    cal_sweeps = np.load(DEFAULT_CALIBRATION)["calibration_sweeps"]
    sweeps = cal_sweeps[np.linspace(0, len(cal_sweeps) - 1, args.n_sweeps).astype(int)]
    roster = ROSTERS[ROSTER]

    print(f"Reading raw waveforms of {len(sweeps)} calibration sweeps...", flush=True)
    data = []
    for s in sweeps:
        data.append(load_sweep(s, gen, pedestal))
        print(f"  {s}: {len(data[-1][0])} bunches (firmware amplitudes reproduced)", flush=True)

    half = {"fit": data[0::2], "test": data[1::2]}
    dev = {k: np.concatenate([d[1] for d in v]) for k, v in half.items()}
    line = {k: np.concatenate([d[2] for d in v]) for k, v in half.items()}
    error = {k: np.concatenate([d[3] for d in v]) for k, v in half.items()}

    print("\ndeviation = firmware - (gain * corrected template + offset), ADC; baselines in raw ADC")
    explained_table("Instantaneous baseline: line - pedestal", dev, line, roster)
    explained_table("Firmware baseline error: running - line", dev, error, roster)

    if args.train:
        import study_baseline_donors as sbd
        all_dev = np.concatenate([d[1] for d in data])
        all_line = np.concatenate([d[2] for d in data])
        a, b = fit_lines(all_dev, all_line)
        pool_params = np.concatenate([d[0] for d in data])
        terms = {"control": np.zeros_like(all_line), "instant": a + b * all_line}
        sbd.train_and_compare(sbd.pool_versions(gen, pool_params, terms), args.seeds, [ROSTER])


if __name__ == "__main__":
    main()
