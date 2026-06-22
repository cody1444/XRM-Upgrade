#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass

from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import uproot
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares


plt.rcParams["font.size"] = 14

# =========================================================
# Fixed config
# =========================================================
REF_CH = 18
WINDOW_HALF_NS = 2.5
YLIM = (0, 120)

NCHANNEL = 128
NWINDOW = 426 + 1
NSAMPLE_PER_WINDOW = 64
NSAMPLE_HALFWINDOW = NSAMPLE_PER_WINDOW // 2
TOTAL_SAMPLES = NWINDOW * NSAMPLE_PER_WINDOW

ABORT_GAP_DELAY = 11200
UP_FACTOR = 10
ADC_GSPS = 2.7

PEAK_DISTANCE_NSAMPLES = 5 * UP_FACTOR
PEAK_PROMINENCE = 100
BASELINE_MA_WINDOW = 50
GAP_THRESHOLD_NS = 100.0

USING_CHANNEL = [
    2, 1, 8, 11, 10, 17, 20, 19, 26, 25, 32, 35, 34, 41,
    44, 43, 50, 49, 56, 59, 58, 69, 68, 71, 74, 77, 84, 83,
    86, 93, 92, 95, 98, 101, 108, 107, 110, 117, 116, 119,
    122, 125,
]

GAIN_CALIB_FACTOR = np.array([
    1.06287682, 1.5290054, 1.06563394, 1.07568872, 1.94462831,
    1.07973691, 1.03603836, 2.27749265, 1.14501594, 1.78006029,
    1.16016389, 1.10321282, 2.11273431, 1.07316967, 1.08100949,
    2.19036947, 1.12708313, 1.84267252, 1.2341472, 1.22696029,
    2.8415352, 2.00185434, 1.26469336, 1.29550866, 1.16993316,
    1.08341741, 1.38272889, 1.06297357, 1.23192728, 1.60515329,
    1.0621062, 1.21726133, 1.11997346, 1.14444874, 1.03856805,
    1.09660115, 1.182108, 1.30326687, 0.9111637, 1.0,
    1.02606163, 1.64743662,
])


# =========================================================
# Grid model
# =========================================================
@dataclass
class SmearedGridModel:
    smeprf: np.ndarray
    mu_axis: np.ndarray
    sig_axis: np.ndarray
    y_positions: np.ndarray

    @classmethod
    def from_npz(cls, path: str | Path) -> "SmearedGridModel":
        d = np.load(path)
        return cls(
            smeprf=d["smeprf"],
            mu_axis=d["mu_axis"],
            sig_axis=d["sig_axis"],
            y_positions=d["y_positions"],
        )

    @staticmethod
    def _interp1d(x, xp, fp):
        return np.interp(x, xp, fp, left=fp[0], right=fp[-1])

    def _profile_mu_sigma(self, mu: float, sigma: float) -> np.ndarray:
        mu_axis = self.mu_axis
        sig_axis = self.sig_axis
        W = self.smeprf

        mu_c = np.clip(mu, mu_axis[0], mu_axis[-1])
        sg_c = np.clip(sigma, sig_axis[0], sig_axis[-1])

        i1 = np.searchsorted(mu_axis, mu_c)
        if i1 == 0:
            i0, i1 = 0, 1
        elif i1 >= len(mu_axis):
            i0, i1 = len(mu_axis) - 2, len(mu_axis) - 1
        else:
            i0 = i1 - 1

        t_mu = 0.0 if mu_axis[i1] == mu_axis[i0] else (
            mu_c - mu_axis[i0]
        ) / (mu_axis[i1] - mu_axis[i0])

        j1 = np.searchsorted(sig_axis, sg_c)
        if j1 == 0:
            j0, j1 = 0, 1
        elif j1 >= len(sig_axis):
            j0, j1 = len(sig_axis) - 2, len(sig_axis) - 1
        else:
            j0 = j1 - 1

        t_sig = 0.0 if sig_axis[j1] == sig_axis[j0] else (
            sg_c - sig_axis[j0]
        ) / (sig_axis[j1] - sig_axis[j0])

        p00 = W[i0, j0, :]
        p10 = W[i1, j0, :]
        p01 = W[i0, j1, :]
        p11 = W[i1, j1, :]

        p0 = p00 * (1.0 - t_mu) + p10 * t_mu
        p1 = p01 * (1.0 - t_mu) + p11 * t_mu

        return p0 * (1.0 - t_sig) + p1 * t_sig

    def model_profile_for_channels_index(
        self,
        ch_indices: np.ndarray,
        mu: float,
        sigma: float,
        v_offset: float,
        x0: float,
        norm: float,
        scale: float,
    ) -> np.ndarray:

        base_prof = self._profile_mu_sigma(mu, sigma)
        n_y = base_prof.size

        ch_indices = np.asarray(ch_indices, dtype=float)
        center_ch = (len(ch_indices) - 1) / 2.0

        x_ch = scale * (ch_indices - center_ch) + x0
        k_axis = np.arange(n_y, dtype=float)

        prof_at_ch = self._interp1d(x_ch, k_axis, base_prof)
        return v_offset + norm * prof_at_ch


# =========================================================
# Fit one bunch
# =========================================================
def fit_one_bunch_index(
    y_obs: np.ndarray,
    grid: SmearedGridModel,
) -> tuple[np.ndarray, float]:

    y_obs = np.asarray(y_obs, dtype=float)
    n_ch = y_obs.size

    finite_mask = np.isfinite(y_obs)
    y_fit = y_obs[finite_mask]

    if y_fit.size < 5:
        return np.full(6, np.nan), np.nan

    mu0 = 0.5 * (grid.mu_axis[0] + grid.mu_axis[-1])
    sigma0 = 0.5 * (grid.sig_axis[0] + grid.sig_axis[-1])

    ref_prof = grid._profile_mu_sigma(mu0, sigma0)
    amp_ref = ref_prof.max() - ref_prof.min()
    if amp_ref <= 0:
        amp_ref = 1e-6

    amp_data = np.nanmax(y_fit) - np.nanmin(y_fit)
    if amp_data <= 0:
        amp_data = 1.0

    norm0 = amp_data / amp_ref

    x0_init = np.array([
        mu0,
        sigma0,
        -50.0,
        505.0,
        norm0,
        14.0,
    ])

    bounds_low = np.array([
        grid.mu_axis[0],
        grid.sig_axis[0],
        -150.0,
        465.0,
        norm0 * 0.25,
        13.0,
    ])

    bounds_high = np.array([
        grid.mu_axis[-1],
        grid.sig_axis[-1],
        50.0,
        545.0,
        norm0 * 5.0,
        17.0,
    ])

    ch_all = np.arange(n_ch, dtype=float)

    def residuals(p):
        mu, sigma, v_off, x0, norm, scale = p
        model_all = grid.model_profile_for_channels_index(
            ch_indices=ch_all,
            mu=mu,
            sigma=sigma,
            v_offset=v_off,
            x0=x0,
            norm=norm,
            scale=scale,
        )
        return model_all[finite_mask] - y_obs[finite_mask]

    y_med = np.median(y_fit)
    mad = np.median(np.abs(y_fit - y_med))
    f_scale = max(5.0, 1.4826 * mad)

    res = least_squares(
        residuals,
        x0_init,
        bounds=(bounds_low, bounds_high),
        method="trf",
        loss="soft_l1",
        f_scale=f_scale,
        max_nfev=500,
        xtol=1e-6,
        ftol=1e-6,
    )

    return res.x, 2.0 * res.cost


# =========================================================
# Fit all bunches
# =========================================================
def fit_all_bunches(
    peak_height_calib: np.ndarray,
    peak_event_times_ns: np.ndarray,
    grid_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    grid_path = Path(grid_path).expanduser()

    if not grid_path.exists():
        raise FileNotFoundError(grid_path)

    print(f"[INFO] Reading GRID PATH file: {grid_path}")

    grid = SmearedGridModel.from_npz(grid_path)

    n_events, _ = peak_height_calib.shape

    fit_params = np.full((n_events, 6), np.nan)
    fit_chi2 = np.full(n_events, np.nan)

    for i in range(n_events):
        params_i, chi2_i = fit_one_bunch_index(
            peak_height_calib[i, :],
            grid,
        )
        fit_params[i, :] = params_i
        fit_chi2[i] = chi2_i

    mu_fit = fit_params[:, 0]
    sigma_fit = fit_params[:, 1]

    return peak_event_times_ns, sigma_fit, mu_fit


# =========================================================
# ROOT to peak-height array
# =========================================================
def extract_peak_heights_from_root(
    root_path: str | Path,
    use_cubic: bool = True,
    use_alignment: bool = True,
    use_baseline: bool = True,
    use_gain: bool = True,
) -> tuple[np.ndarray, np.ndarray]:

    root_path = Path(root_path).expanduser()

    if not root_path.exists():
        raise FileNotFoundError(root_path)

    print(f"[INFO] Reading ROOT file: {root_path}")

    tree = uproot.open(str(root_path))["waveforms"]
    branches = tree.arrays(library="np")

    wf = np.zeros((NCHANNEL, TOTAL_SAMPLES), dtype=np.int16)

    channel_pre = -1

    for i in range(len(branches["waveform"])):
        lin_ch = int(branches["lin_ch"][i])
        physical_window = int(branches["physical_window"][i])
        waveform = branches["waveform"][i]

        if lin_ch == channel_pre:
            for j in range(NSAMPLE_HALFWINDOW):
                sample = physical_window * NSAMPLE_PER_WINDOW + NSAMPLE_HALFWINDOW + j
                if sample < TOTAL_SAMPLES:
                    wf[lin_ch, sample] = waveform[j]
        else:
            for j in range(NSAMPLE_HALFWINDOW):
                sample = physical_window * NSAMPLE_PER_WINDOW + j
                if sample < TOTAL_SAMPLES:
                    wf[lin_ch, sample] = waveform[j]

        channel_pre = lin_ch

    plot_range = len(branches["waveform"]) // 4
    plot_range = min(plot_range, TOTAL_SAMPLES)

    wf = wf[:, :plot_range]
    wf = wf[USING_CHANNEL].astype(np.float64)

    n_ch = wf.shape[0]

    if REF_CH < 0 or REF_CH >= n_ch:
        raise ValueError(f"REF_CH must be 0..{n_ch - 1}")

    # roll
    for ch in range(n_ch):
        wf[ch] = np.roll(wf[ch], ABORT_GAP_DELAY)

    # cubic interpolation
    if use_cubic:
        UP_FACTOR_EFFECTIVE = UP_FACTOR
        x_coarse = np.arange(plot_range, dtype=float)
        x_fine = np.linspace(
            0,
            plot_range - 1,
            (plot_range - 1) * UP_FACTOR_EFFECTIVE + 1,
        )

        y_fine_all = np.empty((n_ch, len(x_fine)), dtype=float)

        for ch in range(n_ch):
            cs = CubicSpline(x_coarse, wf[ch])
            y_fine_all[ch] = cs(x_fine)
    else:
        UP_FACTOR_EFFECTIVE = 1
        x_fine = np.arange(plot_range, dtype=float)
        y_fine_all = wf.copy()

    peak_distance_nsamples = 5 * UP_FACTOR_EFFECTIVE

    # alignment to REF_CH
    if use_alignment:
        # dt_ns = (1.0 / ADC_GSPS) / UP_FACTOR
        dt_ns = (1.0 / ADC_GSPS) / UP_FACTOR_EFFECTIVE
        max_shift_samples = int(round(1.0 / dt_ns))

        ref_wave = y_fine_all[REF_CH].copy()

        for ch in range(n_ch):
            if ch == REF_CH:
                continue

            y_ch = y_fine_all[ch]
            best_shift = 0
            best_corr = -np.inf

            for shift in range(-max_shift_samples, max_shift_samples + 1):
                if shift > 0:
                    a = ref_wave[shift:]
                    b = y_ch[:-shift]
                elif shift < 0:
                    s = -shift
                    a = ref_wave[:-s]
                    b = y_ch[s:]
                else:
                    a = ref_wave
                    b = y_ch

                if a.size < 50:
                    continue

                c = np.corrcoef(a, b)[0, 1]

                if np.isfinite(c) and c > best_corr:
                    best_corr = c
                    best_shift = shift

            if best_shift > 0:
                y_shifted = np.empty_like(y_ch)
                y_shifted[best_shift:] = y_ch[:-best_shift]
                y_shifted[:best_shift] = 0.0
            elif best_shift < 0:
                s = -best_shift
                y_shifted = np.empty_like(y_ch)
                y_shifted[:-s] = y_ch[s:]
                y_shifted[-s:] = 0.0
            else:
                y_shifted = y_ch

            y_fine_all[ch] = y_shifted
    else:
        pass

    # peak search on REF_CH
    y_ref = y_fine_all[REF_CH]
    noise = np.std(wf[REF_CH])

    if noise == 0:
        raise RuntimeError("Noise is zero. Cannot search peaks.")

    ref_peaks_fine, _ = find_peaks(
        -y_ref,
        height=noise,
        distance= peak_distance_nsamples,
        prominence=PEAK_PROMINENCE,
    )

    if len(ref_peaks_fine) == 0:
        raise RuntimeError("No peaks found in REF_CH.")

    x_peaks = x_fine[ref_peaks_fine]
    peak_event_times_ns = x_peaks / ADC_GSPS

    print(f"[INFO] Number of detected bunches: {len(peak_event_times_ns)}")

    n_events = len(ref_peaks_fine)
    window_half_in_coarse = WINDOW_HALF_NS * ADC_GSPS

    # valley values
    valley_values = np.full((n_events, n_ch), np.nan)

    for e, pk_idx in enumerate(ref_peaks_fine):
        x_peak = x_fine[pk_idx]

        i_left = np.searchsorted(
            x_fine,
            x_peak - window_half_in_coarse,
            side="left",
        )
        i_right = np.searchsorted(
            x_fine,
            x_peak + window_half_in_coarse,
            side="right",
        )

        i_left = max(i_left, 0)
        i_right = min(i_right, len(x_fine))

        for ch in range(n_ch):
            seg = y_fine_all[ch, i_left:i_right]
            if seg.size > 0:
                valley_values[e, ch] = np.min(seg)

    if use_baseline:
        # baseline from left/right maxima
        base_level = np.full((n_events, n_ch), np.nan)

        for e in range(n_events):
            t_pk_coarse = x_peaks[e]
            w = window_half_in_coarse

            iL_start = np.searchsorted(x_fine, t_pk_coarse - w, side="left")
            iL_end = np.searchsorted(x_fine, t_pk_coarse, side="left")

            iR_start = np.searchsorted(x_fine, t_pk_coarse, side="left")
            iR_end = np.searchsorted(x_fine, t_pk_coarse + w, side="right")

            iL_start = max(iL_start, 0)
            iL_end = min(iL_end, len(x_fine))
            iR_start = max(iR_start, 0)
            iR_end = min(iR_end, len(x_fine))

            for ch in range(n_ch):
                y_fine = y_fine_all[ch]

                segL = y_fine[iL_start:iL_end]
                segR = y_fine[iR_start:iR_end]

                if segL.size == 0 or segR.size == 0:
                    continue

                idxL = iL_start + np.argmax(segL)
                idxR = iR_start + np.argmax(segR)

                yL = y_fine[idxL]
                yR = y_fine[idxR]
                tL = x_fine[idxL]
                tR = x_fine[idxR]

                if tR == tL:
                    y_base = 0.5 * (yL + yR)
                else:
                    y_base = yL + (yR - yL) * (t_pk_coarse - tL) / (tR - tL)

                base_level[e, ch] = y_base

        # baseline moving average within each train segment
        base_level_ma = np.empty_like(base_level)

        gaps = np.diff(peak_event_times_ns)
        split_indices = np.where(gaps > GAP_THRESHOLD_NS)[0]

        seg_starts = np.concatenate(([0], split_indices + 1))
        seg_ends = np.concatenate((split_indices + 1, [n_events]))

        for s, e in zip(seg_starts, seg_ends):
            base_level_ma[s:e, :] = uniform_filter1d(
                base_level[s:e, :],
                size=BASELINE_MA_WINDOW,
                axis=0,
                mode="nearest",
            )
        peak_height_calib = base_level_ma - valley_values
    else:
        peak_height_calib = -valley_values

    # gain calibration
    if use_gain:
        for ch in range(n_ch):
            peak_height_calib[:, ch] *= GAIN_CALIB_FACTOR[ch]

    return peak_height_calib, peak_event_times_ns


# =========================================================
# Plot
# =========================================================
def plot_sigma_vs_ns(
    times_ns: np.ndarray,
    sigma_fit: np.ndarray,
    root_path: Path,
    out_png: Path,
):
    valid = np.isfinite(times_ns) & np.isfinite(sigma_fit)
    mean_sigma_fit = np.mean(sigma_fit[valid])

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.scatter(
        times_ns[valid],
        sigma_fit[valid],
        s=8
    )

    ax.axhline(mean_sigma_fit, linestyle="--", color='r', label=r"Average $\sigma_y$")
    ax.axhline(6.117538e+01, linestyle='--', color='g', label=r"CMOS $\sigma_y$")

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel(r"Fitted $\sigma_y$ [$\mu$m]")
    ax.set_title(f"Fitted sigma vs time: {root_path.stem}")
    ax.grid(True)
    ax.set_ylim(*YLIM)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    print(f"[INFO] Saved: {out_png}")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Fit SiXRM ROOT file and output only fitted sigma vs ns plot."
    )

    parser.add_argument(
        "root_path",
        type=str,
        help="Input ROOT file path.",
    )
    
    parser.add_argument(
        "grid_path",
        type=str,
        help="Input grid file path.",
    )

    args = parser.parse_args()

    root_path = Path(args.root_path).expanduser()
    grid_path = Path(args.grid_path).expanduser()

    output_dir = Path("./png")
    output_dir.mkdir(parents=True, exist_ok=True)

    RUN_ALL_FLAG_COMBINATIONS = True

    if RUN_ALL_FLAG_COMBINATIONS:
        flag_combinations = product([False,True], repeat=4)
    else:
        flag_combinations = [
            # use_cubic, use_alignment, use_baseline, use_gain
            (True, True, True, True)
        ]

    for use_cubic, use_alignment, use_baseline, use_gain in flag_combinations:
        tag = (
            f"cubic{int(use_cubic)}_"
            f"align{int(use_alignment)}_"
            f"base{int(use_baseline)}_"
            f"gain{int(use_gain)}"
        )
        out_png = output_dir / f"{root_path.stem}_fitted_sigma_vs_ns_{tag}.png"

        peak_height_calib, peak_event_times_ns = extract_peak_heights_from_root(
            root_path=root_path,
            use_cubic=use_cubic,
            use_alignment=use_alignment,
            use_baseline=use_baseline,
            use_gain=use_gain
        )

        times_ns, sigma_fit, _ = fit_all_bunches(
            peak_height_calib=peak_height_calib,
            peak_event_times_ns=peak_event_times_ns,
            grid_path=grid_path,
        )

        plot_sigma_vs_ns(
            times_ns=times_ns,
            sigma_fit=sigma_fit,
            root_path=root_path,
            out_png=out_png,
        )


if __name__ == "__main__":
    main()
