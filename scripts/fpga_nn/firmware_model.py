"""
Python model of the firmware amplitude extraction (E50 baseline).

Per channel, for each bunch at sample index t:
  1. highest sample in the window before the pulse [t - w, t) and after it [t, t + w]
  2. baseline line between those two points, evaluated at t
  3. running baseline <- running baseline + (line - running baseline) / N
  4. amplitude = (running baseline - lowest sample in [t - w, t + w]) * gain

This is the reference the FPGA implementation should reproduce.
"""

import numpy as np
from scipy.signal import lfilter

from xrm_raw import ffr

BASELINE_N_BUNCHES = 50
HALF_WIDTH = int(ffr.WINDOW_HALF_NS * ffr.ADC_GSPS)


def exponential_average(values, n_eff, start):
    """Causal exponential running average along axis 1, starting from `start`."""
    alpha = 1.0 / n_eff
    zi = ((1.0 - alpha) * start)[:, None]
    out, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], values, axis=1, zi=zi)

    return out


def side_maxima(x, peaks, half_width=HALF_WIDTH):
    """Highest sample and its index before and after each peak, per channel."""
    left_idx = peaks[:, None] + np.arange(-half_width, 0)[None, :]
    right_idx = peaks[:, None] + np.arange(0, half_width + 1)[None, :]

    left = x[:, left_idx]
    right = x[:, right_idx]

    t_left = np.take_along_axis(np.broadcast_to(left_idx, left.shape), left.argmax(axis=2)[..., None], 2)[..., 0]
    t_right = np.take_along_axis(np.broadcast_to(right_idx, right.shape), right.argmax(axis=2)[..., None], 2)[..., 0]

    return left.max(axis=2), right.max(axis=2), t_left, t_right


def valid_peaks(peaks, n_samples, half_width=HALF_WIDTH):
    return (peaks - half_width >= 0) & (peaks + half_width < n_samples)


def firmware_amplitudes(x, peaks, pedestal, gain, n_bunches=BASELINE_N_BUNCHES, half_width=HALF_WIDTH):
    """
    x: raw waveforms (n_channels, n_samples); peaks: sorted bunch sample indices,
    all with a full window inside x. Returns amplitudes (n_bunches, n_channels).
    """
    left_max, right_max, t_left, t_right = side_maxima(x, peaks, half_width)

    span = np.where(t_right == t_left, 1, t_right - t_left)
    line = left_max + (right_max - left_max) * (peaks[None, :] - t_left) / span
    baseline = exponential_average(line, n_bunches, pedestal)

    window = peaks[:, None] + np.arange(-half_width, half_width + 1)[None, :]
    lowest = x[:, window].min(axis=2)

    return ((baseline - lowest) * gain[:, None]).T
