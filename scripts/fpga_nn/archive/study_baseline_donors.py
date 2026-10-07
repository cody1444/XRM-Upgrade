#!/usr/bin/env python3

"""
Can a bunch's measured baselines stand in for part of its deviation from the template?

1. For a limited set of calibration sweeps, take each bunch's per-channel
   baseline (offline-style moving average, raw ADC, stored by
   build_real_dataset.py in data/real/runs) and its firmware deviation:
       deviation = firmware - (gain * corrected template + offset)
2. Per channel, fit  deviation = a + b * (baseline - pedestal)  on half of
   these sweeps, and report how much of the deviation it explains on the
   other half.
3. Donor pool = the bunches of these sweeps. Train the centre 16-channel
   network on two toy datasets made from identical random draws:
       control    the current generator (with this donor pool)
       baseline   + a + b * (donor's baseline - pedestal), per channel
   and test both on the real held-out sweeps, as in train_nn.py.
"""

import argparse

import numpy as np
from tensorflow import keras

import xrm_raw
from xrm_generator import (XrmGenerator, AMPLITUDE_DIR, DATA_DIR, DEFAULT_CALIBRATION, FIT_LOWER, FIT_UPPER,
                           NOISE_ADC, NOISE_FRACTION, corrected_template)
from train_nn import (ROSTERS, SIGMA_RANGE_UM, LABEL_CENTRE, LABEL_SCALE,
                      normalize, build_model, load_real, robust_spread)

ROSTER = "centre_12_27"


def parse_args():
    parser = argparse.ArgumentParser(prog="study_baseline_donors", description=__doc__.strip().splitlines()[0])
    parser.add_argument("--n-sweeps", type=int, default=40, help="Calibration sweeps to use.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=200000)
    parser.add_argument("--n-val", type=int, default=40000)
    return parser.parse_args()


def robust_var(values):
    values = values[np.isfinite(values)]
    return (1.4826 * np.median(np.abs(values - np.median(values)))) ** 2


def load_sweep(sweep, gen, pedestal):
    """Per bunch: fit parameters, firmware deviation and baseline - pedestal (good bunches only)."""
    amp = np.load(AMPLITUDE_DIR / f"{sweep}.npz")
    run = np.load(xrm_raw.REAL_DIR / "runs" / f"{sweep}.npz")

    # The amplitude file holds a time-sorted subset of the run file's bunches.
    order = np.argsort(run["bunch_times_ns"])
    run_times = run["bunch_times_ns"][order]
    idx = np.searchsorted(run_times, amp["bunch_times_ns"])
    assert np.array_equal(run_times[idx], amp["bunch_times_ns"]), sweep
    baseline = run["baseline"][order][idx] - pedestal

    params, firmware = amp["fit_params"], amp["firmware"]
    ok = (np.all(np.isfinite(params), axis=1) & np.all(np.isfinite(firmware), axis=1)
          & np.all(np.isfinite(baseline), axis=1))
    ok &= ~np.any((np.abs(params - FIT_LOWER) < 1e-3) | (np.abs(params - FIT_UPPER) < 1e-3), axis=1)

    deviation = np.array([f - (gen.fw_gain * corrected_template(gen.grid, p, gen.factors, gen.channels) + gen.fw_offset)
                          for f, p in zip(firmware[ok], params[ok])])
    return params[ok], deviation, baseline[ok]


def fit_lines(deviation, baseline):
    """Per channel, least-squares a, b in deviation = a + b * baseline."""
    n_ch = deviation.shape[1]
    a, b = np.zeros(n_ch), np.zeros(n_ch)
    for ch in range(n_ch):
        b[ch], a[ch] = np.polyfit(baseline[:, ch], deviation[:, ch], 1)
    return a, b


def make_toys(gen, pool_params, pool_term, sigma_y, rng):
    """Like XrmGenerator.sample(), from the given donor pool, plus each donor's baseline term."""
    donors = rng.integers(len(pool_params), size=len(sigma_y))
    params = pool_params[donors].copy()
    params[:, 1] = sigma_y
    x = np.empty((len(sigma_y), len(gen.channels)))
    for i, (p, d) in enumerate(zip(params, donors)):
        firmware = gen.fw_gain * corrected_template(gen.grid, p, gen.factors, gen.channels) + gen.fw_offset
        firmware = firmware + pool_term[d]
        noise_size = NOISE_ADC + NOISE_FRACTION * np.clip(firmware, 0, None)
        x[i] = firmware + noise_size * rng.standard_normal(firmware.shape)
    return x, params[:, 0], sigma_y


def main():
    args = parse_args()
    gen = XrmGenerator(DEFAULT_CALIBRATION)
    pedestal = np.load(DATA_DIR / "amplitudes_pedestal.npz")["pedestal"]
    cal_sweeps = np.load(DEFAULT_CALIBRATION)["calibration_sweeps"]
    sweeps = cal_sweeps[np.linspace(0, len(cal_sweeps) - 1, args.n_sweeps).astype(int)]

    # -----------------------------------------------------------------------
    # 1-2. How much of the deviation follows the baseline?
    # -----------------------------------------------------------------------
    print(f"Loading {len(sweeps)} calibration sweeps...", flush=True)
    data = [load_sweep(s, gen, pedestal) for s in sweeps]
    half = {"fit": data[0::2], "test": data[1::2]}
    dev = {k: np.concatenate([d[1] for d in v]) for k, v in half.items()}
    base = {k: np.concatenate([d[2] for d in v]) for k, v in half.items()}

    a, b = fit_lines(dev["fit"], base["fit"])
    pred = a + b * base["test"]
    roster = ROSTERS[ROSTER]
    print(f"\nPer channel, on the test half ({len(dev['test'])} bunches):")
    print("  deviation = firmware - (gain * corrected template + offset); baseline in raw ADC above pedestal")
    print(f"{'ch':>4s} {'baseline':>9s} {'dev rms':>8s} {'slope b':>8s} {'corr':>6s} {'explained':>9s}")
    explained = np.empty(len(gen.channels))
    for ch in range(len(gen.channels)):
        explained[ch] = 1 - robust_var(dev["test"][:, ch] - pred[:, ch]) / robust_var(dev["test"][:, ch])
        corr = np.corrcoef(base["test"][:, ch], dev["test"][:, ch])[0, 1]
        if ch in roster:
            print(f"{ch:>4d} {np.median(base['test'][:, ch]):9.1f} {np.sqrt(robust_var(dev['test'][:, ch])):8.1f} "
                  f"{b[ch]:+8.3f} {corr:+6.2f} {100 * explained[ch]:8.0f}%")
    print(f"Median over channels {roster[0]}-{roster[-1]}: {100 * np.median(explained[roster]):.0f}% of the "
          f"deviation's variance explained by the baseline")

    # Refit on all sweeps for the toy.
    all_dev = np.concatenate([d[1] for d in data])
    all_base = np.concatenate([d[2] for d in data])
    a, b = fit_lines(all_dev, all_base)
    pool_params = np.concatenate([d[0] for d in data])
    pool_term = {"control": np.zeros_like(all_base), "baseline": a + b * all_base}
    term_rms = np.sqrt(np.mean(pool_term["baseline"][:, roster] ** 2))
    dev_rms = np.sqrt(np.mean(all_dev[:, roster] ** 2))
    print(f"\nDonor pool: {len(pool_params)} bunches. Baseline term rms {term_rms:.1f} ADC "
          f"vs deviation rms {dev_rms:.1f} ADC (channels {roster[0]}-{roster[-1]})")

    train_and_compare(pool_versions(gen, pool_params, pool_term), args.seeds, [ROSTER], args.n_train, args.n_val)


def train_and_compare(versions, seeds, rosters, n_train=200000, n_val=40000):
    """
    versions: {name: make(sigma_y, rng) -> (amplitudes, mu, sigma_y)}, one toy recipe each.
    Per seed, every version gets the same random state; one network per (seed, version, roster)
    is trained and tested on the real held-out sweeps.
    """
    held_out = list(np.load(DEFAULT_CALIBRATION)["held_out_sweeps"])
    amps, fit_sigma, _, sweep_idx, cmos = load_real(held_out[::2])
    test_sweeps = np.unique(sweep_idx)
    results = {(r, v): [] for r in rosters for v in versions}

    for seed in seeds:
        print(f"\n--- seed {seed} ---", flush=True)
        for version, make in versions.items():
            rng = np.random.default_rng(seed)
            x_tr, mu_tr, s_tr = make(rng.uniform(*SIGMA_RANGE_UM, n_train), rng)
            x_va, mu_va, s_va = make(rng.uniform(*SIGMA_RANGE_UM, n_val), rng)
            y_tr = (np.column_stack([s_tr, mu_tr]) - LABEL_CENTRE) / LABEL_SCALE
            y_va = (np.column_stack([s_va, mu_va]) - LABEL_CENTRE) / LABEL_SCALE

            for roster_name in rosters:
                roster = ROSTERS[roster_name]
                keras.utils.set_random_seed(seed)
                xt = normalize(x_tr, roster)
                mean, std = xt.mean(axis=0), xt.std(axis=0) + 1e-9
                model = build_model(len(roster))
                model.fit((xt - mean) / std, y_tr, validation_data=((normalize(x_va, roster) - mean) / std, y_va),
                          epochs=100, batch_size=256, verbose=0,
                          callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                                   restore_best_weights=True)])

                pred = (model.predict((normalize(amps, roster) - mean) / std, verbose=0, batch_size=4096)
                        * LABEL_SCALE + LABEL_CENTRE)[:, 0]
                diff = pred - fit_sigma
                scatter = np.median([robust_spread(pred[sweep_idx == s]) for s in test_sweeps])
                per_sweep = np.array([np.median(pred[sweep_idx == s]) for s in test_sweeps]) - cmos[test_sweeps]
                per_sweep = per_sweep[np.isfinite(per_sweep)]
                m = [np.median(diff), robust_spread(diff), scatter, np.median(per_sweep), np.std(per_sweep)]
                results[(roster_name, version)].append(m)
                print(f"  {roster_name:>16s} {version:>12s}: real nn-fit {m[0]:+.2f} um, spread {m[1]:.2f}, "
                      f"within-sweep {m[2]:.2f}, vs CMOS {m[3]:+.2f} +- {m[4]:.2f}", flush=True)

    labels = ["nn-fit median", "nn-fit spread", "within-sweep scatter", "vs CMOS median", "vs CMOS spread"]
    for roster_name in rosters:
        print(f"\n=== Real held-out sweeps ({len(test_sweeps)}), {roster_name}: "
              f"mean over {len(seeds)} seeds (min..max) ===")
        print(f"{'':>22s} " + " ".join(f"{v:>20s}" for v in versions))
        for i, label in enumerate(labels):
            cells = []
            for version in versions:
                v = np.array(results[(roster_name, version)])[:, i]
                cells.append(f"{v.mean():+6.2f} ({v.min():+.2f}..{v.max():+.2f})")
            print(f"{label:>22s} " + " ".join(f"{c:>20s}" for c in cells))


def pool_versions(gen, pool_params, pool_terms):
    """Toy recipes from one donor pool, one per baseline term (see make_toys)."""
    def recipe(term):
        def make(sigma_y, rng):
            gen.rng = rng
            return make_toys(gen, pool_params, term, sigma_y, rng)
        return make
    return {name: recipe(term) for name, term in pool_terms.items()}

if __name__ == "__main__":
    main()
