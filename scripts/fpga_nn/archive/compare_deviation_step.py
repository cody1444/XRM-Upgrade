#!/usr/bin/env python3

"""
Does the generator need step 3 (adding a real donor's deviation)?

Trains the same network on two versions of the generated data:
  full            the generator as it is
  no deviation    template x channel factors + firmware response only
Both use the same random draws (beam sizes, donors, firmware scatter), so
the deviation is the only difference. Each is trained with several seeds
to show the ordinary network-to-network variation.

Tests:
  - real held-out sweeps (same half as train_nn.py): network vs fit per
    bunch, scatter within a sweep, per sweep vs CMOS
  - generated validation data from both versions, by beam-size band
"""

import argparse

import numpy as np
from tensorflow import keras

from xrm_generator import XrmGenerator, DEFAULT_CALIBRATION
from train_nn import (ROSTERS, SIGMA_RANGE_UM, LABEL_CENTRE, LABEL_SCALE,
                      normalize, build_model, load_real, robust_spread)

BANDS = [(10, 30), (30, 48), (48, 83), (83, 120), (120, 200)]


def parse_args():
    parser = argparse.ArgumentParser(prog="compare_deviation_step", description=__doc__.strip().splitlines()[0])
    parser.add_argument("--roster", default="centre_12_27", choices=list(ROSTERS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=200000)
    parser.add_argument("--n-val", type=int, default=40000)
    return parser.parse_args()


def make_data(seed, n_train, n_val):
    """Training and validation data for both versions, from identical random draws."""
    data = {}
    for version in ("full", "no deviation"):
        rng = np.random.default_rng(seed)
        gen = XrmGenerator(DEFAULT_CALIBRATION, rng)
        if version == "no deviation":
            gen.donor_deviation = np.zeros_like(gen.donor_deviation)
        train = gen.sample(rng.uniform(*SIGMA_RANGE_UM, n_train))
        val = gen.sample(rng.uniform(*SIGMA_RANGE_UM, n_val))
        data[version] = (train, val)
    return data


def labels(sample):
    _, mu, sigma, _ = sample
    return (np.column_stack([sigma, mu]) - LABEL_CENTRE) / LABEL_SCALE


def real_metrics(pred, fit_sigma, sweep_idx, cmos):
    sweeps = np.unique(sweep_idx)
    diff = pred - fit_sigma
    scatter = np.median([robust_spread(pred[sweep_idx == s]) for s in sweeps])
    per_sweep = np.array([np.median(pred[sweep_idx == s]) for s in sweeps])
    ok = np.isfinite(cmos[sweeps])
    vs_cmos = per_sweep[ok] - cmos[sweeps][ok]
    return {"nn-fit median": np.median(diff), "nn-fit spread": robust_spread(diff),
            "within-sweep scatter": scatter,
            "vs CMOS median": np.median(vs_cmos), "vs CMOS spread": np.std(vs_cmos)}


def main():
    args = parse_args()
    roster = ROSTERS[args.roster]
    held_out = list(np.load(DEFAULT_CALIBRATION)["held_out_sweeps"])
    amps, fit_sigma, _, sweep_idx, cmos = load_real(held_out[::2])
    sweeps = np.unique(sweep_idx)
    fit_scatter = np.median([robust_spread(fit_sigma[sweep_idx == s]) for s in sweeps])
    fit_cmos = np.array([np.median(fit_sigma[sweep_idx == s]) for s in sweeps]) - cmos[sweeps]
    fit_cmos = fit_cmos[np.isfinite(fit_cmos)]
    print(f"Roster {args.roster}; real test: {len(sweeps)} held-out sweeps, {len(amps)} bunches")

    real = {v: [] for v in ("full", "no deviation")}
    generated = {}   # (trained on, tested on) -> list over seeds of per-band (bias, spread)

    for seed in args.seeds:
        print(f"\n--- seed {seed} ---", flush=True)
        data = make_data(seed, args.n_train, args.n_val)
        for version, (train, _) in data.items():
            keras.utils.set_random_seed(seed)
            xt = normalize(train[0], roster)
            mean, std = xt.mean(axis=0), xt.std(axis=0) + 1e-9
            own_val = data[version][1]
            model = build_model(len(roster))
            model.fit((xt - mean) / std, labels(train),
                      validation_data=((normalize(own_val[0], roster) - mean) / std, labels(own_val)),
                      epochs=100, batch_size=256, verbose=0,
                      callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                               restore_best_weights=True)])

            def predict(x):
                return (model.predict((normalize(x, roster) - mean) / std, verbose=0, batch_size=4096)
                        * LABEL_SCALE + LABEL_CENTRE)[:, 0]

            m = real_metrics(predict(amps), fit_sigma, sweep_idx, cmos)
            real[version].append(m)
            print(f"  trained on {version:>12s}: real nn-fit {m['nn-fit median']:+.2f} um, spread "
                  f"{m['nn-fit spread']:.2f}, within-sweep {m['within-sweep scatter']:.2f}, "
                  f"vs CMOS {m['vs CMOS median']:+.2f} +- {m['vs CMOS spread']:.2f}", flush=True)

            for tested_on, (_, val) in data.items():
                err = predict(val[0]) - val[2]
                generated.setdefault((version, tested_on), []).append(
                    [(np.mean(err[b]), np.std(err[b]))
                     for b in [(val[2] >= lo) & (val[2] < hi) for lo, hi in BANDS]])

    print(f"\n=== Real held-out sweeps: mean over {len(args.seeds)} seeds (min..max) ===")
    print(f"Reference fit: within-sweep scatter {fit_scatter:.2f} um, "
          f"vs CMOS {np.median(fit_cmos):+.2f} +- {np.std(fit_cmos):.2f} um")
    print(f"{'':>22s} {'full':>20s} {'no deviation':>20s}")
    for key in real["full"][0]:
        cells = []
        for version in ("full", "no deviation"):
            v = np.array([m[key] for m in real[version]])
            cells.append(f"{v.mean():+6.2f} ({v.min():+.2f}..{v.max():+.2f})")
        print(f"{key:>22s} {cells[0]:>20s} {cells[1]:>20s}")

    print(f"\n=== Generated validation data: sigma_y error, mean bias / spread (um), over seeds ===")
    print(f"{'trained on -> tested on':>30s} " + " ".join(f"{f'{lo}-{hi}':>13s}" for lo, hi in BANDS))
    for (trained, tested), runs in generated.items():
        runs = np.array(runs).mean(axis=0)
        print(f"{trained + ' -> ' + tested:>30s} " + " ".join(f"{b:+5.2f} / {s:5.2f}" for b, s in runs))


if __name__ == "__main__":
    main()
