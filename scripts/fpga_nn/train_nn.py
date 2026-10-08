#!/usr/bin/env python3

"""
Train small networks on generator data and test them on real data.

For each channel set (roster):
  inputs  = firmware amplitudes of those channels, divided by their sum
  outputs = sigma_y and mu
Tests:
  - generated validation data, inside vs outside the beam-size range
    covered by real data
  - real held-out sweeps (stable period, not used for calibration):
    per bunch vs the reference fit, per sweep vs CMOS, and the
    bunch-to-bunch scatter within sweeps
  - real good sweeps from before the stable period (different setup)
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

import xrm_raw
from xrm_generator import XrmGenerator, AMPLITUDE_DIR, DEFAULT_CALIBRATION, STABLE_FROM

OUTPUT_DIR = xrm_raw.REAL_DIR.parent / "nn"

ROSTERS = {
    "every_other_6_36": list(range(6, 37, 2)),
    "centre_12_27": list(range(12, 28)),
    "all_measured_4_39": list(range(4, 40)),
}

SIGMA_RANGE_UM = (10.0, 200.0)
LABEL_CENTRE = np.array([105.0, 90.0])     # sigma_y, mu
LABEL_SCALE = np.array([55.0, 30.0])


def parse_args():
    parser = argparse.ArgumentParser(
            prog="train_nn",
            description="Train networks on generator data and test them on real data.",
            )
    parser.add_argument("--n-train", type=int, default=200000)
    parser.add_argument("--n-val", type=int, default=40000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rosters", nargs="+", default=list(ROSTERS), choices=list(ROSTERS))
    parser.add_argument("--factors", action="store_true", help="Generator extra: channel factors.")
    parser.add_argument("--deviations", action="store_true", help="Generator extra: donors' real deviations.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Folder for the trained networks.")
    return parser.parse_args()


def normalize(amplitudes, roster):
    x = amplitudes[:, roster]
    return x / x.sum(axis=1, keepdims=True)


def build_model(n_inputs):
    model = keras.Sequential([
        keras.layers.Input(shape=(n_inputs,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(2),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def robust_spread(values):
    values = values[np.isfinite(values)]
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def load_real(sweeps):
    """Firmware amplitudes, reference-fit sigma_y and mu, sweep index and CMOS per sweep."""
    amps, sigma, mu, sweep_idx, cmos = [], [], [], [], []
    for k, sweep in enumerate(sweeps):
        d = np.load(AMPLITUDE_DIR / f"{sweep}.npz")
        ok = np.all(np.isfinite(d["firmware"]), axis=1) & np.all(np.isfinite(d["fit_params"]), axis=1)
        amps.append(d["firmware"][ok])
        sigma.append(d["fit_params"][ok, 1])
        mu.append(d["fit_params"][ok, 0])
        sweep_idx.append(np.full(ok.sum(), k))
        cmos.append(float(np.load(xrm_raw.REAL_DIR / "runs" / f"{sweep}.npz")["cmos_sigma_median"]))
    return (np.concatenate(amps), np.concatenate(sigma), np.concatenate(mu),
            np.concatenate(sweep_idx), np.array(cmos))


def evaluate_real(label, pred_sigma, fit_sigma, sweep_idx, cmos):
    diff = pred_sigma - fit_sigma
    sweeps = np.unique(sweep_idx)
    nn_scatter = np.median([robust_spread(pred_sigma[sweep_idx == s]) for s in sweeps])
    fit_scatter = np.median([robust_spread(fit_sigma[sweep_idx == s]) for s in sweeps])
    per_sweep = np.array([np.median(pred_sigma[sweep_idx == s]) for s in sweeps])
    fit_per_sweep = np.array([np.median(fit_sigma[sweep_idx == s]) for s in sweeps])
    ok = np.isfinite(cmos[sweeps])

    print(f"  {label}: {len(sweeps)} sweeps, {len(diff)} bunches")
    print(f"    network - fit, per bunch: median {np.median(diff):+.2f} um, spread {robust_spread(diff):.2f} um")
    print(f"    bunch-to-bunch scatter within a sweep: network {nn_scatter:.2f} um, fit {fit_scatter:.2f} um")
    if ok.any():
        d_nn = per_sweep[ok] - cmos[sweeps][ok]
        d_fit = fit_per_sweep[ok] - cmos[sweeps][ok]
        print(f"    per sweep vs CMOS ({ok.sum()} sweeps): network {np.median(d_nn):+.2f} um (spread {np.std(d_nn):.2f}), "
              f"fit {np.median(d_fit):+.2f} um (spread {np.std(d_fit):.2f})")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(args.seed)
    print("TensorFlow", tf.__version__, "GPUs:", tf.config.list_physical_devices("GPU"))

    rng = np.random.default_rng(args.seed)
    gen = XrmGenerator(DEFAULT_CALIBRATION, rng, args.factors, args.deviations)
    print(f"Generator: factors {'on' if args.factors else 'off'}, deviations {'on' if args.deviations else 'off'}")
    real_lo, real_hi = gen.donor_params[:, 1].min(), gen.donor_params[:, 1].max()

    print(f"Generating {args.n_train} training and {args.n_val} validation bunches...", flush=True)
    x_train, mu_train, sig_train, _ = gen.sample(rng.uniform(*SIGMA_RANGE_UM, args.n_train))
    x_val, mu_val, sig_val, _ = gen.sample(rng.uniform(*SIGMA_RANGE_UM, args.n_val))
    in_range_val = (sig_val >= real_lo) & (sig_val <= real_hi)

    y_train = (np.column_stack([sig_train, mu_train]) - LABEL_CENTRE) / LABEL_SCALE
    y_val = (np.column_stack([sig_val, mu_val]) - LABEL_CENTRE) / LABEL_SCALE

    held_out = list(np.load(DEFAULT_CALIBRATION)["held_out_sweeps"])
    early = sorted(p.stem for p in AMPLITUDE_DIR.glob("*.npz")
                   if p.stem < STABLE_FROM and str(np.load(p)["category"]) == "good")
    real_sets = {"real held-out sweeps (Dec 4-15)": load_real(held_out[::2]),
                 f"real early sweeps (before {STABLE_FROM}, other setup)": load_real(early)}

    for name in args.rosters:
        roster = ROSTERS[name]
        print(f"\n=== {name}: {len(roster)} channels {roster[0]}..{roster[-1]} ===", flush=True)

        xt, xv = normalize(x_train, roster), normalize(x_val, roster)
        mean, std = xt.mean(axis=0), xt.std(axis=0) + 1e-9

        model = build_model(len(roster))
        model.fit((xt - mean) / std, y_train, validation_data=((xv - mean) / std, y_val),
                  epochs=args.epochs, batch_size=args.batch_size, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                           restore_best_weights=True)])

        pred = model.predict((xv - mean) / std, verbose=0) * LABEL_SCALE + LABEL_CENTRE
        err = pred[:, 0] - sig_val
        print(f"  generated validation, sigma_y error (mean / spread):")
        print(f"    inside {real_lo:.0f}-{real_hi:.0f} um: {np.mean(err[in_range_val]):+.2f} / {np.std(err[in_range_val]):.2f} um")
        print(f"    outside:            {np.mean(err[~in_range_val]):+.2f} / {np.std(err[~in_range_val]):.2f} um")
        for lo, hi in [(10, 30), (30, 47), (47, 83), (83, 120), (120, 200)]:
            m = (sig_val >= lo) & (sig_val < hi)
            print(f"      {lo:3d}-{hi:<3d} um: spread {np.std(err[m]):5.2f} um")

        for label, (amps, fit_sigma, _, sweep_idx, cmos) in real_sets.items():
            x_real = (normalize(amps, roster) - mean) / std
            pred_real = model.predict(x_real, verbose=0, batch_size=4096) * LABEL_SCALE + LABEL_CENTRE
            evaluate_real(label, pred_real[:, 0], fit_sigma, sweep_idx, cmos)

        model.save(output_dir / f"{name}.keras")
        np.savez(output_dir / f"{name}_scaler.npz", roster=np.array(roster), mean=mean, std=std,
                 factors=args.factors, deviations=args.deviations,
                 label_centre=LABEL_CENTRE, label_scale=LABEL_SCALE)

    print(f"\nModels saved in {output_dir}")


if __name__ == "__main__":
    main()
