#!/usr/bin/env python3

"""
Compare firmware-friendly baseline methods by how well the fitted sigma_y
agrees with the full-processing reference fit.

For every bunch and channel: amplitude = (baseline - raw valley sample) * gain,
then the same template fit as the reference. Methods:
  A    fixed beam-off pedestal
  B    exponential running average of the raw waveform (tau ~200 ns),
       read just before each pulse window
  C    highest sample in the 2.5 ns before each pulse, exponentially
       averaged over ~50 bunches
  D<N> fit-style baseline (line between the highest samples before and
       after the pulse), averaged over N bunches within a train
  E<N> the same line, exponentially averaged over ~N bunches (causal,
       one stored value per channel, like C)
"""

import argparse
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter

import xrm_raw
from xrm_raw import ffr

DEFAULT_GRID = xrm_raw.REPO_DIR / "beam_profiles" / "smeared_grid_CMOS_upto200um.npz"

B_TAU_NS = 200.0
C_N_BUNCHES = 50
D_N_BUNCHES = [1, 8, 50]
E_N_BUNCHES = [8, 50]
ALL_METHODS = ["A", "B", "C"] + [f"D{n}" for n in D_N_BUNCHES] + [f"E{n}" for n in E_N_BUNCHES]

CURRENT_CLASSES_MA = [(0, 600), (600, 900), (900, 2000)]

_grid = None
_pedestal = None
_methods = None


def parse_args():
    parser = argparse.ArgumentParser(
            prog="test_baseline_methods",
            description="Compare firmware-friendly baseline methods via the fitted sigma_y.",
            )

    parser.add_argument(
            "-n",
            "--n-sweeps",
            type=int,
            default=40,
            help="Number of good sweeps, spread evenly over beam current.",
            )

    parser.add_argument(
            "-o",
            "--output",
            default=str(xrm_raw.REAL_DIR.parent / "baseline_methods.npz"),
            help="Output .npz file.",
            )

    parser.add_argument(
            "--methods",
            nargs="+",
            default=ALL_METHODS,
            choices=ALL_METHODS,
            help="Baseline methods to fit.",
            )

    parser.add_argument(
            "-j",
            "--jobs",
            type=int,
            default=8,
            help="Number of sweeps processed in parallel.",
            )

    return parser.parse_args()


def init_worker(grid_path, pedestal, methods):
    global _grid, _pedestal, _methods
    _grid = ffr.SmearedGridModel.from_npz(grid_path)
    _pedestal = pedestal
    _methods = methods


def fit_sigma(heights):
    sigma = np.full(heights.shape[0], np.nan)

    for i in range(heights.shape[0]):
        params, _ = ffr.fit_one_bunch_index(heights[i], _grid)
        sigma[i] = params[1]

    return sigma


def exponential_average(values, n_eff, start):
    """Causal exponential running average along axis 1, starting from `start`."""
    alpha = 1.0 / n_eff
    zi = ((1.0 - alpha) * start)[:, None]
    out, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], values, axis=1, zi=zi)

    return out


def side_maxima(x, peaks, half_width):
    """Highest sample and its index before and after each peak (ffr's windows)."""
    left_idx = peaks[:, None] + np.arange(-half_width, 0)[None, :]
    right_idx = peaks[:, None] + np.arange(0, half_width + 1)[None, :]

    left = x[:, left_idx]
    right = x[:, right_idx]

    t_left = np.take_along_axis(np.broadcast_to(left_idx, left.shape), left.argmax(axis=2)[..., None], 2)[..., 0]
    t_right = np.take_along_axis(np.broadcast_to(right_idx, right.shape), right.argmax(axis=2)[..., None], 2)[..., 0]

    return left.max(axis=2), right.max(axis=2), t_left, t_right


def train_segments(times_ns):
    breaks = np.nonzero(np.diff(times_ns) > ffr.GAP_THRESHOLD_NS)[0] + 1
    return np.split(np.arange(len(times_ns)), breaks)


def process_sweep(run):
    data = np.load(xrm_raw.REAL_DIR / "runs" / f"{run}.npz")

    order = np.argsort(data["bunch_times_ns"])
    times = data["bunch_times_ns"][order]
    valley = data["valley_raw"][order].T          # (42, n_bunches)
    gain = data["gain"][:, None]
    peaks = np.rint(times * ffr.ADC_GSPS).astype(int)

    x = xrm_raw.read_raw_waveforms(xrm_raw.cleaned_root_path(run))
    half_width = int(ffr.WINDOW_HALF_NS * ffr.ADC_GSPS)

    keep = (peaks - half_width >= 0) & (peaks + half_width < x.shape[1])
    times, valley, peaks = times[keep], valley[:, keep], peaks[keep]

    baselines = {"A": np.broadcast_to(_pedestal[:, None], valley.shape)}

    n_tau = B_TAU_NS * ffr.ADC_GSPS
    running = exponential_average(x, n_tau, _pedestal)
    baselines["B"] = running[:, peaks - half_width - 1]

    left_max, right_max, t_left, t_right = side_maxima(x, peaks, half_width)
    baselines["C"] = exponential_average(left_max, C_N_BUNCHES, _pedestal)

    span = np.where(t_right == t_left, 1, t_right - t_left)
    line = left_max + (right_max - left_max) * (peaks[None, :] - t_left) / span
    for n in D_N_BUNCHES:
        smoothed = np.empty_like(line)
        for seg in train_segments(times):
            smoothed[:, seg] = uniform_filter1d(line[:, seg], size=n, axis=1, mode="nearest")
        baselines[f"D{n}"] = smoothed
    for n in E_N_BUNCHES:
        baselines[f"E{n}"] = exponential_average(line, n, _pedestal)

    result = {"run": run, "reference": data["fit_sigma"][order][keep]}
    for name in _methods:
        result[name] = fit_sigma(((baselines[name] - valley) * gain).T)

    return result


def robust_spread(values):
    values = values[np.isfinite(values)]
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def main():
    args = parse_args()

    print("Computing beam-off pedestals...")
    pedestal = xrm_raw.beam_off_pedestals()

    dcct = xrm_raw.DcctCurrent()
    runs = sorted(xrm_raw.good_runs(), key=dcct)
    runs = [runs[i] for i in np.linspace(0, len(runs) - 1, args.n_sweeps).astype(int)]
    current = {run: dcct(run) for run in runs}

    methods = args.methods

    print(f"Fitting {len(runs)} sweeps with baseline methods {methods}...")
    with Pool(args.jobs, initializer=init_worker, initargs=(str(DEFAULT_GRID), pedestal, methods)) as pool:
        results = pool.map(process_sweep, runs)

    np.savez(
        args.output,
        runs=np.array(runs),
        current=np.array([current[r] for r in runs]),
        **{f"{key}_{r['run']}": r[key] for r in results for key in ["reference"] + methods},
    )

    print("\nsigma_y in um. Bias = median(method - reference). "
          "Scatter = typical bunch-to-bunch spread of sigma_y within a sweep.")
    header = f"{'method':>10s} | {'scatter':>7s} | {'bias all':>8s} | {'spread vs ref':>13s}"
    for lo, hi in CURRENT_CLASSES_MA:
        header += f" | {f'bias {lo}-{hi} mA':>16s}"
    print(header)

    for name in ["reference"] + methods:
        scatter = np.median([robust_spread(r[name]) for r in results])
        diff = np.concatenate([r[name] - r["reference"] for r in results])
        row = f"{name:>10s} | {scatter:7.1f} | {np.nanmedian(diff):+8.2f} | {robust_spread(diff):13.1f}"

        for lo, hi in CURRENT_CLASSES_MA:
            sel = [r for r in results if lo <= current[r["run"]] < hi]
            if sel:
                d = np.concatenate([r[name] - r["reference"] for r in sel])
                row += f" | {np.nanmedian(d):+16.2f}"
            else:
                row += f" | {'-':>16s}"
        print(row)

    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
