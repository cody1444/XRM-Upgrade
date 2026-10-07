"""
Shared helpers for studies on raw SiXRM waveforms.
"""

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import uproot

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import fit_from_rootfile as ffr  # noqa: E402

REPO_DIR = SCRIPTS_DIR.parent
DATA_DIR = REPO_DIR / "scripts_dev" / "data" / "HER"
REAL_DIR = Path(__file__).resolve().parent / "data" / "real"

# Captures taken with DCCT current ~0 mA; waveforms are pure noise.
BEAM_OFF_RUNS = [
    "2025-11-28.143757",
    "2025-11-28.143829",
    "2025-11-28.145502",
    "2025-11-28.150108",
    "2025-12-03.115629",
]

EDGE_CHANNELS = [0, 1, 2, 3, 38, 39, 40, 41]
MID_CHANNELS = [8, 9, 10, 11]
CENTRE_CHANNELS = [16, 17, 18, 19, 20]

CHANNEL_GROUPS = {
    "edge": EDGE_CHANNELS,
    "mid": MID_CHANNELS,
    "centre": CENTRE_CHANNELS,
}


def cleaned_root_path(run):
    return DATA_DIR / run[:10] / "cleaned" / f"{run}_cleaned.root"


def read_raw_waveforms(root_path):
    """
    Raw ADC waveforms for the 42 used channels, rolled the same way as
    fit_from_rootfile, so sample index = bunch_times_ns * ADC_GSPS.
    """
    branches = uproot.open(str(root_path))["waveforms"].arrays(library="np")

    wf = np.zeros((ffr.NCHANNEL, ffr.TOTAL_SAMPLES), dtype=np.int16)
    channel_pre = -1

    for i in range(len(branches["waveform"])):
        lin_ch = int(branches["lin_ch"][i])
        physical_window = int(branches["physical_window"][i])
        waveform = branches["waveform"][i]

        offset = ffr.NSAMPLE_HALFWINDOW if lin_ch == channel_pre else 0
        start = physical_window * ffr.NSAMPLE_PER_WINDOW + offset
        n = max(0, min(ffr.NSAMPLE_HALFWINDOW, ffr.TOTAL_SAMPLES - start))
        wf[lin_ch, start:start + n] = waveform[:n]

        channel_pre = lin_ch

    n_samples = min(len(branches["waveform"]) // 4, ffr.TOTAL_SAMPLES)
    wf = wf[ffr.USING_CHANNEL, :n_samples].astype(np.float64)

    return np.roll(wf, ffr.ABORT_GAP_DELAY, axis=1)


def beam_off_pedestals():
    """Per-channel pedestal: mean over beam-off captures of each channel's median."""
    medians = [
        np.median(read_raw_waveforms(cleaned_root_path(run)), axis=1)
        for run in BEAM_OFF_RUNS
    ]

    return np.mean(medians, axis=0)


def good_runs(category="good"):
    with open(REAL_DIR / "runs_summary.csv") as f:
        return [row["run"] for row in csv.DictReader(f) if row["category"] == category]


def load_her_series(path, column=2):
    """(timestamps, values) from a .HER log file, skipping non-numeric lines."""
    times = []
    values = []

    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            try:
                values.append(float(parts[column]))
                times.append(datetime.fromisoformat(parts[0]))
            except (ValueError, IndexError):
                continue

    return np.array(times), np.array(values)


class DcctCurrent:
    """Median DCCT beam current (mA) around a run timestamp."""

    def __init__(self, window_s=30.0):
        self.window = timedelta(seconds=window_s)
        self._cache = {}

    def __call__(self, run):
        day = run[:10]

        if day not in self._cache:
            files = list((DATA_DIR / day).glob("*.current.HER"))
            self._cache[day] = load_her_series(files[0]) if files else None

        if self._cache[day] is None:
            return np.nan

        times, values = self._cache[day]
        t_run = datetime.strptime(run[:17], "%Y-%m-%d.%H%M%S")
        mask = (times >= t_run - self.window) & (times <= t_run + self.window)

        return float(np.median(values[mask])) if mask.any() else np.nan
