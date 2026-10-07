#!/usr/bin/env python3

"""
What do the per-channel factors do for the networks?

Trains the 16-channel networks on two versions of the current generator:
  with factors   xrm_generator.py as it is
  no factors     every channel factor set to 1 (template goes straight to gain/offset + noise)
The random draws (beam sizes, donors, noise) are identical between the
two, so the factors are the only difference. Tested on the real held-out
sweeps, as in train_nn.py.
"""

import argparse

import numpy as np

from xrm_generator import XrmGenerator, DEFAULT_CALIBRATION
from study_baseline_donors import train_and_compare

ROSTERS_16 = ["centre_12_27", "every_other_6_36"]


def parse_args():
    parser = argparse.ArgumentParser(prog="study_no_factors", description=__doc__.strip().splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=200000)
    parser.add_argument("--n-val", type=int, default=40000)
    return parser.parse_args()


def recipe(factors):
    gen = XrmGenerator(DEFAULT_CALIBRATION)
    gen.factors = factors

    def make(sigma_y, rng):
        gen.rng = rng
        x, mu, sigma_y, _ = gen.sample(sigma_y)
        return x, mu, sigma_y

    return make


def main():
    args = parse_args()
    factors = np.load(DEFAULT_CALIBRATION)["factors"]
    versions = {"with factors": recipe(factors), "no factors": recipe(np.ones_like(factors))}
    train_and_compare(versions, args.seeds, ROSTERS_16, args.n_train, args.n_val)


if __name__ == "__main__":
    main()
