#!/usr/bin/env python3

"""
Walk through xrm_generator.py for ONE fake bunch.

Every step of XrmGenerator.sample() is written out here line by line, with
the intermediate numbers printed and drawn:

  1. Parameters: a random real bunch's fit parameters, with the requested sigma_y.
  2. Template:   ideal profile for those parameters, x a factor per channel.
  3. Firmware:   gain * amplitude + offset + noise.

At the end the same bunch is made again with XrmGenerator.sample() from
the same random state, to check that this walkthrough does exactly what
the generator does.

    python explain_one_bunch.py --sigma 120
"""

import argparse
import copy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xrm_generator import (XrmGenerator, AMPLITUDE_DIR, DEFAULT_CALIBRATION, NOISE_ADC, NOISE_FRACTION,
                           corrected_template)

PARAM_NAMES = ["mu (um)", "sigma_y (um)", "v_offset (ADC)", "x0", "norm", "scale"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, GREEN, GREY = "#2a78d6", "#1baf7a", "#a9a7a0"


def parse_args():
    parser = argparse.ArgumentParser(prog="explain_one_bunch", description="Make one fake bunch, step by step.")
    parser.add_argument("--sigma", type=float, default=120.0, help="Requested beam size, um.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (changes the real bunch and the noise).")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), help="Calibration file.")
    parser.add_argument("-o", "--output", default=str(AMPLITUDE_DIR.parent / "explain_one_bunch.png"))
    return parser.parse_args()


def heading(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main():
    args = parse_args()
    gen = XrmGenerator(args.calibration, np.random.default_rng(args.seed))
    start_state = copy.deepcopy(gen.rng.bit_generator.state)
    channels = gen.channels
    measured = gen.factor_measured

    # -----------------------------------------------------------------------
    heading(f"Step 1: fit parameters (requested sigma_y = {args.sigma} um)")
    # -----------------------------------------------------------------------
    donor = gen.rng.integers(len(gen.donor_params), size=1)[0]
    donor_params = gen.donor_params[donor]
    params = donor_params.copy()
    params[1] = args.sigma

    sweep = gen.donor_sweep[donor]
    lo, hi = gen.donor_params[:, 1].min(), gen.donor_params[:, 1].max()
    print(f"The calibration file holds the fit parameters of {len(gen.donor_params)} real bunches "
          f"(sigma_y {lo:.1f}..{hi:.1f} um).")
    print(f"Picked at random: entry {donor}, from sweep {sweep}. Only sigma_y is replaced:\n")
    print(f"{'fit parameter':>16s} {'real bunch':>12s} {'fake bunch':>12s}")
    for name, a, b in zip(PARAM_NAMES, donor_params, params):
        print(f"{name:>16s} {a:12.4g} {b:12.4g}" + ("   <- replaced" if a != b else ""))
    if not lo <= args.sigma <= hi:
        print(f"\n{args.sigma} um is outside the range of real beam sizes, so this kind of bunch")
        print("can't be checked against real data (generate flags it: in_real_sigma_range = False).")

    # -----------------------------------------------------------------------
    heading("Step 2: corrected template for these parameters")
    # -----------------------------------------------------------------------
    template = gen.grid.model_profile_for_channels_index(channels, *params)
    v_offset = params[2]
    reference = v_offset + gen.factors * (template - v_offset)
    print("template  = ideal profile from the template library (includes the constant v_offset)")
    print(f"corrected = v_offset + factor * (template - v_offset),  v_offset = {v_offset:.1f} ADC")
    print(f"The factors (0.67..1.26) are measured on channels {channels[measured].min():.0f}-"
          f"{channels[measured].max():.0f}; the others get no signal and keep factor 1.")
    print("The corrected template stands in for the reference amplitudes (the best offline measurement).")

    # -----------------------------------------------------------------------
    heading("Step 3: what the firmware would measure")
    # -----------------------------------------------------------------------
    clean = gen.fw_gain * reference + gen.fw_offset
    noise_size = NOISE_ADC + NOISE_FRACTION * np.clip(clean, 0, None)
    noise = noise_size * gen.rng.standard_normal(clean.shape)
    firmware = clean + noise
    print("firmware = gain * reference + offset + noise")
    print("  gain and offset: per channel, a straight-line fit of real firmware vs real reference amplitudes")
    print(f"  noise: random, standard deviation {NOISE_ADC:g} ADC + {100 * NOISE_FRACTION:g}% of the amplitude")

    # -----------------------------------------------------------------------
    heading("All channels (ADC; * = no measured factor)")
    # -----------------------------------------------------------------------
    print(f"{'ch':>4s} {'factor':>6s} {'template':>8s} {'=ref':>8s} "
          f"{'gain':>5s} {'offset':>6s} {'noise':>6s} {'=firmw':>7s}")
    for c in range(len(channels)):
        print(f"{c:>3d}{' ' if measured[c] else '*'} {gen.factors[c]:6.3f} {template[c]:8.1f} {reference[c]:8.1f} "
              f"{gen.fw_gain[c]:5.3f} {gen.fw_offset[c]:+6.1f} {noise[c]:+6.1f} {firmware[c]:7.1f}")

    # -----------------------------------------------------------------------
    heading("Check against XrmGenerator.sample()")
    # -----------------------------------------------------------------------
    gen.rng.bit_generator.state = start_state
    x, mu, _, sample_sweep = gen.sample([args.sigma])
    same = np.array_equal(x[0], firmware) and mu[0] == params[0] and sample_sweep[0] == sweep
    print(f"Same random state, same bunch: {'IDENTICAL' if same else 'DIFFERENT'} "
          f"(max difference {np.max(np.abs(x[0] - firmware)):.1e} ADC)")
    print(f"Labels stored for training: true_sig_y = {args.sigma}, true_mu = {params[0]:.2f}")

    # The real bunch whose parameters were borrowed, for comparison in the plot.
    d = np.load(AMPLITUDE_DIR / f"{sweep}.npz")
    row = np.nonzero(np.all(d["fit_params"] == donor_params, axis=1))[0][0]

    # What the generator would make for that real bunch itself (without noise).
    donor_model = gen.fw_gain * corrected_template(gen.grid, donor_params, gen.factors, channels) + gen.fw_offset

    plot(args, channels, measured, donor_params, params, d["firmware"][row], donor_model,
         template, reference, clean, noise_size, firmware)


def style(ax, title, ylabel="amplitude (ADC)", legend="best"):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("channel", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc=legend)


def plot(args, ch, measured, donor_params, params, donor_firmware, donor_model, template, reference, clean, noise_size, firmware):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))

    a = ax[0]
    a.plot(ch, donor_model, color=GREEN, lw=1.5, label="generator's version of it (same parameters, no noise)")
    a.plot(ch, donor_firmware, "o", color=INK, ms=4, label="its real firmware amplitudes")
    style(a, f"1. The real bunch whose parameters are borrowed\n    (its sigma_y = {donor_params[1]:.1f} um)", legend="upper left")

    a = ax[1]
    a.plot(ch, template, "--", color=BLUE, lw=1, label="template")
    a.plot(ch, reference, color=BLUE, lw=1.5, label="corrected template (x factor)")
    a.axvspan(-0.5, ch[measured].min() - 0.5, color=GRID, alpha=0.4, lw=0)
    a.axvspan(ch[measured].max() + 0.5, ch.max() + 0.5, color=GRID, alpha=0.4, lw=0, label="no measured factor")
    style(a, f"2. Template for the fake bunch\n    (sigma_y = {params[1]:.1f} um)")

    a = ax[2]
    a.plot(ch, reference, color=GREY, lw=1, label="corrected template")
    a.plot(ch, clean, color=GREEN, lw=1.5, label="gain x template + offset")
    a.fill_between(ch, clean - noise_size, clean + noise_size, color=GREEN, alpha=0.2, lw=0, label="+- noise size")
    a.plot(ch, firmware, "o", color=INK, ms=4, label="+ noise = fake firmware amplitudes (the output)")
    style(a, "3. What the firmware would measure", legend="upper left")

    fig.suptitle(f"One fake bunch, step by step (sigma_y = {args.sigma} um, seed {args.seed})",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(args.output, dpi=110, facecolor="white")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
