#!/usr/bin/env python3

"""
train_nn.py, but with the archived step-3 generator (channel factors, real
donor deviations, measured scatter model) instead of the current one.

Calibrates that generator into data/generator_calibration_step3.npz (if not
there yet) and saves the networks in data/nn_step3/, so the current
calibration and the benchmark networks in data/nn/ are left alone.
All train_nn.py options work, e.g.

    python archive/train_nn_step3.py --seed 0 --rosters centre_12_27
"""

import importlib.util
import sys
from pathlib import Path

FPGA_NN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FPGA_NN_DIR))

import train_nn  # noqa: E402
from xrm_generator import DATA_DIR  # noqa: E402

CALIBRATION = DATA_DIR / "generator_calibration_step3.npz"
OUTPUT_DIR = DATA_DIR / "nn_step3"


def load_step3():
    spec = importlib.util.spec_from_file_location("xrm_generator_step3", Path(__file__).parent / "xrm_generator_step3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    step3 = load_step3()
    if not CALIBRATION.exists():
        step3.calibrate(CALIBRATION)

    # train_nn.main() looks these up when it runs, so swapping them here is enough.
    # (the archived generator has no factors/deviations switches; it always uses both)
    train_nn.XrmGenerator = lambda calibration, rng, *switches: step3.XrmGenerator(calibration, rng)
    train_nn.DEFAULT_CALIBRATION = CALIBRATION
    train_nn.OUTPUT_DIR = OUTPUT_DIR
    train_nn.main()


if __name__ == "__main__":
    main()
