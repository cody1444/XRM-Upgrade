#!/usr/bin/env python3

"""
Walk through xrm_generator.py for ONE fake bunch.

Every step of XrmGenerator.sample() is written out here line by line, with
the intermediate numbers printed and drawn:

  1. Donor:     pick a real bunch with a similar beam size; the fake bunch
                gets the donor's fit parameters, except sigma_y.
  2. Template:  ideal profile for the fake parameters, x a factor per channel.
  3. Deviation: + the donor's real deviation from its own corrected template.
  4. Firmware:  gain * reference + offset + scatter.

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

from xrm_generator import XrmGenerator, AMPLITUDE_DIR, DEFAULT_CALIBRATION, corrected_template

PARAM_NAMES = ["mu (um)", "sigma_y (um)", "v_offset (ADC)", "x0", "norm", "scale"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a9a7a0"


def parse_args():
    parser = argparse.ArgumentParser(prog="explain_one_bunch", description="Make one fake bunch, step by step.")
    parser.add_argument("--sigma", type=float, default=120.0, help="Requested beam size, um.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (changes the donor and the scatter).")
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
    heading(f"Step 1: donor (requested sigma_y = {args.sigma} um)")
    # -----------------------------------------------------------------------
    donor = gen.donor_for_sigma(args.sigma)
    donor_params = gen.donor_params[donor]
    params = donor_params.copy()
    params[1] = args.sigma

    sweep = gen.donor_sweep[donor]
    lo, hi = gen.donor_params[:, 1].min(), gen.donor_params[:, 1].max()
    print(f"The library has {len(gen.donor_params)} donors with sigma_y {lo:.1f}..{hi:.1f} um.")
    if lo <= args.sigma <= hi:
        print("The requested beam size is inside that range: the donor has a similar beam size.")
    else:
        print("The requested beam size is OUTSIDE that range: the donor comes from the nearest beam size,")
        print("so its deviation was measured on a different beam size (unverified, see GENERATOR.md).")
    print(f"Chosen donor: library entry {donor}, from sweep {sweep}.\n")
    print(f"{'fit parameter':>16s} {'donor':>12s} {'fake bunch':>12s}")
    for name, a, b in zip(PARAM_NAMES, donor_params, params):
        print(f"{name:>16s} {a:12.4g} {b:12.4g}" + ("   <- replaced" if a != b else ""))

    # The donor is a real bunch: find it in its sweep file.
    d = np.load(AMPLITUDE_DIR / f"{sweep}.npz")
    row = np.nonzero(np.all(d["fit_params"] == donor_params, axis=1))[0][0]
    donor_reference = d["reference"][row]
    donor_corrected = corrected_template(gen.grid, donor_params, gen.factors, channels)
    deviation = gen.donor_deviation[donor]
    print(f"\nThe donor is bunch {row} of that sweep. Its deviation was stored at calibration as")
    print("  deviation = donor's real reference amplitudes - donor's own corrected template")
    print("Recomputed now from the sweep file, the two agree to "
          f"{np.max(np.abs(donor_reference - donor_corrected - deviation)):.1e} ADC.")

    # -----------------------------------------------------------------------
    heading("Step 2: corrected template for the fake bunch's parameters")
    # -----------------------------------------------------------------------
    template = gen.grid.model_profile_for_channels_index(channels, *params)
    v_offset = params[2]
    corrected = v_offset + gen.factors * (template - v_offset)
    print("template  = ideal profile from the template library (includes the constant v_offset)")
    print(f"corrected = v_offset + factor * (template - v_offset),  v_offset = {v_offset:.1f} ADC")
    print(f"The factors (0.67..1.26) are measured on channels {channels[measured].min():.0f}-"
          f"{channels[measured].max():.0f}; the others get no signal and keep factor 1.")

    # -----------------------------------------------------------------------
    heading("Step 3: add the donor's deviation")
    # -----------------------------------------------------------------------
    brightness = params[4] / donor_params[4]
    reference = corrected + brightness * deviation
    print("reference = corrected + brightness * deviation")
    print(f"brightness = fake norm / donor norm = {brightness:.3f}")
    print("  (always 1 here: the fake bunch keeps the donor's norm; it differs from 1 only in validate's twins)")
    signal = corrected[measured] - v_offset
    print(f"Size of the deviation on measured channels: {np.sqrt(np.mean(deviation[measured] ** 2)):.0f} ADC rms, "
          f"against a typical signal of {np.median(signal):.0f} ADC.")

    # -----------------------------------------------------------------------
    heading("Step 4: what the firmware would measure")
    # -----------------------------------------------------------------------
    clean = gen.fw_gain * reference + gen.fw_offset
    size = np.interp(reference, gen.fw_scatter_amplitude, gen.fw_scatter)
    shared_draw = gen.rng.standard_normal()
    own_draw = gen.rng.standard_normal(reference.shape)
    s = gen.fw_shared
    scatter = size * (np.sqrt(s) * shared_draw + np.sqrt(1 - s) * own_draw)
    firmware = clean + scatter
    print("firmware = gain * reference + offset + scatter")
    print("scatter  = size * (sqrt(shared) * one draw for the whole bunch + sqrt(1 - shared) * one draw per channel)")
    print(f"shared = {s:.2f}; this bunch's common draw = {shared_draw:+.2f}")
    print("size is read from the scatter table (amplitude -> typical scatter):")
    print("  " + ", ".join(f"{a:.0f}: {t:.1f}" for a, t in zip(gen.fw_scatter_amplitude, gen.fw_scatter)))

    # -----------------------------------------------------------------------
    heading("All channels (ADC; * = no measured factor)")
    # -----------------------------------------------------------------------
    print(f"{'ch':>4s} {'factor':>6s} {'template':>8s} {'correct':>8s} {'+dev':>7s} {'=ref':>7s} "
          f"{'gain':>5s} {'offset':>6s} {'size':>5s} {'scatter':>7s} {'=firmw':>7s}")
    for c in range(len(channels)):
        print(f"{c:>3d}{' ' if measured[c] else '*'} {gen.factors[c]:6.3f} {template[c]:8.1f} {corrected[c]:8.1f} "
              f"{brightness * deviation[c]:+7.1f} {reference[c]:7.1f} {gen.fw_gain[c]:5.3f} "
              f"{gen.fw_offset[c]:+6.1f} {size[c]:5.1f} {scatter[c]:+7.1f} {firmware[c]:7.1f}")

    # -----------------------------------------------------------------------
    heading("Check against XrmGenerator.sample()")
    # -----------------------------------------------------------------------
    gen.rng.bit_generator.state = start_state
    x, mu, _, sample_sweep = gen.sample([args.sigma])
    same = np.array_equal(x[0], firmware) and mu[0] == params[0] and sample_sweep[0] == sweep
    print(f"Same random state, same bunch: {'IDENTICAL' if same else 'DIFFERENT'} "
          f"(max difference {np.max(np.abs(x[0] - firmware)):.1e} ADC)")
    print(f"Labels stored for training: true_sig_y = {args.sigma}, true_mu = {params[0]:.2f}")

    plot(args, channels, measured, donor_params, params, donor_reference, donor_corrected, deviation,
         template, corrected, reference, clean, size, firmware)


def style(ax, title, ylabel="amplitude (ADC)"):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("channel", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)


def plot(args, ch, measured, donor_params, params, donor_reference, donor_corrected, deviation,
         template, corrected, reference, clean, size, firmware):
    fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
    ax = ax.ravel()

    a = ax[0]
    a.plot(ch, donor_corrected, color=BLUE, lw=1.5, label="its corrected template")
    a.plot(ch, donor_reference, "o", color=INK, ms=4, label="donor's real reference amplitudes")
    a.bar(ch, deviation, color=ORANGE, alpha=0.6, label="deviation = real - template")
    style(a, f"1. Donor: a real bunch, sigma_y = {donor_params[1]:.1f} um")

    a = ax[1]
    a.plot(ch, donor_corrected, color=GREY, lw=1, label=f"donor's, sigma_y = {donor_params[1]:.1f} um (for comparison)")
    a.plot(ch, template, "--", color=BLUE, lw=1, label="template")
    a.plot(ch, corrected, color=BLUE, lw=1.5, label="corrected template (x factor)")
    a.axvspan(-0.5, ch[measured].min() - 0.5, color=GRID, alpha=0.4, lw=0)
    a.axvspan(ch[measured].max() + 0.5, ch.max() + 0.5, color=GRID, alpha=0.4, lw=0, label="no measured factor")
    style(a, f"2. Template for the fake bunch, sigma_y = {params[1]:.1f} um")

    a = ax[2]
    a.plot(ch, corrected, color=BLUE, lw=1.5, label="corrected template")
    a.bar(ch, reference - corrected, color=ORANGE, alpha=0.6, label="+ donor's deviation")
    a.plot(ch, reference, "o", color=INK, ms=4, label="= fake reference amplitudes")
    style(a, "3. Add the donor's real deviation")

    a = ax[3]
    a.plot(ch, reference, color=GREY, lw=1, label="fake reference")
    a.plot(ch, clean, color=GREEN, lw=1.5, label="gain x reference + offset")
    a.fill_between(ch, clean - size, clean + size, color=GREEN, alpha=0.2, lw=0, label="+- typical scatter")
    a.plot(ch, firmware, "o", color=INK, ms=4, label="= fake firmware amplitudes (the output)")
    style(a, "4. What the firmware would measure")

    fig.suptitle(f"One fake bunch, step by step (sigma_y = {args.sigma} um, seed {args.seed})",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(args.output, dpi=110, facecolor="white")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
