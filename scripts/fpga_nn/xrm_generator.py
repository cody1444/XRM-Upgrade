#!/usr/bin/env python3

"""
XRM data generator: fake firmware amplitudes that look like real ones.

Each fake bunch is built in three steps:
  1. Parameters: take the fit parameters of a random real bunch (the
                 donor: beam position, brightness, image placement) and
                 replace its beam size with the requested one.
  2. Template:   the ideal profile for those parameters, from the template
                 library.
  3. Firmware:   turn these amplitudes into what the FPGA would measure:
                 about 0.93x per channel plus an offset, plus random noise
                 of 1 ADC + 2% of the amplitude.

Two optional extras, both off by default (the default is the benchmark
in GENERATOR.md):
  factors     in step 2, multiply the template by a correction factor per
              channel (real channels read consistently higher or lower
              than the templates)
  deviations  after step 2, add the donor's real deviation: its real
              amplitudes minus what step 2 makes for its own parameters
              (real bunches vary ~11% per channel from bunch to bunch)
The noise matters even though it is small: networks trained without it
learn details that real bunches don't have.

Commands:
  calibrate  measure factors and firmware gain/offset, collect donors (fit
             parameters and real amplitudes) from real data (data/amplitudes,
             stable period) -> one calibration file, for every combination of extras
  generate   make fake bunches -> .npz with x_data, true_mu, true_sig_y
  validate   compare fakes of real bunches with real sweeps not used for calibration
generate and validate take --factors and --deviations to switch the extras on.

Inputs come from build_amplitude_dataset.py (data/amplitudes/<sweep>.npz):
per bunch and channel, the reference amplitude (careful offline
processing), the firmware amplitude (firmware_model.py) and the bunch's
template fit. Fit parameters are always in the order
(mu, sigma_y, v_offset, x0, norm, scale).
"""

import argparse

import numpy as np
from scipy.spatial import cKDTree

import xrm_raw
from xrm_raw import ffr

DATA_DIR = xrm_raw.REAL_DIR.parent
AMPLITUDE_DIR = DATA_DIR / "amplitudes"
TEMPLATE_GRID = xrm_raw.REPO_DIR / "beam_profiles" / "smeared_grid_CMOS_upto200um.npz"
DEFAULT_CALIBRATION = DATA_DIR / "generator_calibration.npz"

# Sweeps from this day on share one stable detector setup.
STABLE_FROM = "2025-12-04"

# Stable-period sweeps excluded from calibration: normal current per bunch
# but 3-10x less detector signal and the image shifted (mu ~65-75 um).
# They come in short bursts next to "beam on, no signal" sweeps, so the
# detector or beamline was probably being changed. See plot_condition_correlations.py.
EXCLUDED_SWEEPS = {
    "2025-12-04.150436", "2025-12-04.150736", "2025-12-04.151321", "2025-12-04.151519",
    "2025-12-04.151703", "2025-12-04.151839", "2025-12-04.161835",
    "2025-12-05.211448", "2025-12-05.211715", "2025-12-05.211939",
    "2025-12-08.121909", "2025-12-08.122320", "2025-12-08.180242", "2025-12-08.180429",
}

# Channel factors: only where the template signal is clearly above zero,
# and only for channels with enough such bunches (the others keep factor 1).
MIN_SIGNAL_ADC = 100.0
MIN_FACTOR_ENTRIES = 2000

# Firmware noise per channel: standard deviation = 1 ADC + 2% of the amplitude
# (roughly the measured scatter of firmware around gain * reference + offset).
NOISE_ADC = 1.0
NOISE_FRACTION = 0.02

# Donors: drop fits that hit a bound of the template
# fit (bounds copied from fit_from_rootfile.py), and the 1% most extreme beam
# sizes at each end so the "range covered by real data" isn't set by outliers.
FIT_LOWER = np.array([0.0, 10.0, -150.0, 465.0, 0.0, 13.0])
FIT_UPPER = np.array([150.0, 200.0, 50.0, 545.0, np.inf, 17.0])
DONOR_SIGMA_PERCENTILES = (1, 99)
DONOR_EVERY = 5

# validate with deviations: each real bunch gets a random donor among the 20
# most similar in all six fit parameters (an unmatched donor's deviation can
# make some channels several times too variable).
TWIN_NEIGHBOURS = 20

# Validation uses the channels that get X-rays.
VALIDATION_CHANNELS = np.arange(4, 40)


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

def stable_sweeps():
    """Good sweeps of the stable period, split alternately into calibration and held-out halves."""
    paths = [p for p in sorted(AMPLITUDE_DIR.glob("*.npz"))
             if p.stem >= STABLE_FROM and p.stem not in EXCLUDED_SWEEPS
             and str(np.load(p)["category"]) == "good"]
    return paths[0::2], paths[1::2]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def measure_channel_factors(paths):
    """Per channel: median of (reference - v_offset) / (template - v_offset)."""
    ratios = []
    for path in paths:
        d = np.load(path)
        v_offset = d["fit_params"][::2, 2:3]
        signal = d["model"][::2] - v_offset
        signal[~(signal > MIN_SIGNAL_ADC)] = np.nan
        ratios.append((d["reference"][::2] - v_offset) / signal)
    ratios = np.concatenate(ratios)

    measured = np.sum(np.isfinite(ratios), axis=0) >= MIN_FACTOR_ENTRIES
    factors = np.ones(ratios.shape[1])
    factors[measured] = np.nanmedian(ratios[:, measured], axis=0)
    return factors, measured


def measure_firmware_response(paths):
    """
    Per channel, a straight-line fit: firmware = gain * reference + offset.
    (The generator applies it to the template, which stands in for the reference.)
    """
    reference = np.concatenate([np.load(p)["reference"][::10] for p in paths])
    firmware = np.concatenate([np.load(p)["firmware"][::10] for p in paths])
    n_ch = reference.shape[1]

    gain, offset = np.ones(n_ch), np.zeros(n_ch)
    for ch in range(n_ch):
        ok = np.isfinite(reference[:, ch]) & np.isfinite(firmware[:, ch])
        gain[ch], offset[ch] = np.polyfit(reference[ok, ch], firmware[ok, ch], 1)
    return gain, offset


def collect_donors(paths):
    """Every DONOR_EVERY-th real bunch without a bad fit: fit parameters, real amplitudes, sweep."""
    params, reference, sweep = [], [], []
    for path in paths:
        d = np.load(path)
        for ref, p in zip(d["reference"][::DONOR_EVERY], d["fit_params"][::DONOR_EVERY]):
            if np.all(np.isfinite(ref)) and np.all(np.isfinite(p)):
                params.append(p)
                reference.append(ref)
                sweep.append(path.stem)
    params, reference, sweep = np.array(params), np.array(reference), np.array(sweep)

    at_bound = np.any((np.abs(params - FIT_LOWER) < 1e-3) | (np.abs(params - FIT_UPPER) < 1e-3), axis=1)
    sigma_lo, sigma_hi = np.percentile(params[~at_bound, 1], DONOR_SIGMA_PERCENTILES)
    keep = ~at_bound & (params[:, 1] >= sigma_lo) & (params[:, 1] <= sigma_hi)
    return params[keep], reference[keep], sweep[keep]


def calibrate(output):
    calibration_paths, held_out = stable_sweeps()
    print(f"Stable period from {STABLE_FROM}: {len(calibration_paths)} calibration sweeps, "
          f"{len(held_out)} held out for validation ({len(EXCLUDED_SWEEPS)} sweeps excluded)")

    factors, measured = measure_channel_factors(calibration_paths)
    print(f"Channel factors: {measured.sum()} channels measured, "
          f"range {factors[measured].min():.2f}..{factors[measured].max():.2f}")

    gain, offset = measure_firmware_response(calibration_paths)
    print(f"Firmware response: gain {np.median(gain[VALIDATION_CHANNELS]):.3f} "
          f"(median over channels {VALIDATION_CHANNELS[0]}-{VALIDATION_CHANNELS[-1]})")

    donor_params, donor_reference, donor_sweep = collect_donors(calibration_paths)
    print(f"Donors: {len(donor_params)} real bunches, sigma_y "
          f"{donor_params[:, 1].min():.1f}..{donor_params[:, 1].max():.1f} um")

    np.savez(
        output,
        factors=factors,
        factor_measured=measured,
        fw_gain=gain,
        fw_offset=offset,
        donor_params=donor_params,
        donor_reference=donor_reference,
        donor_sweep=donor_sweep,
        calibration_sweeps=np.array([p.stem for p in calibration_paths]),
        held_out_sweeps=np.array([p.stem for p in held_out]),
    )
    print(f"Saved {output}")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class XrmGenerator:

    def __init__(self, calibration=DEFAULT_CALIBRATION, rng=None, factors=False, deviations=False):
        self.rng = np.random.default_rng() if rng is None else rng
        self.grid = ffr.SmearedGridModel.from_npz(TEMPLATE_GRID)
        self.use_factors = factors
        self.use_deviations = deviations

        cal = np.load(calibration)
        self.factors = cal["factors"]
        self.fw_gain = cal["fw_gain"]
        self.channels = np.arange(len(self.fw_gain), dtype=float)
        self.fw_offset = cal["fw_offset"]
        self.donor_params = cal["donor_params"]
        self.donor_reference = cal["donor_reference"]
        self.donor_sweep = cal["donor_sweep"]
        self.held_out_sweeps = cal["held_out_sweeps"]

    # Step 2
    def template(self, params):
        """Ideal profile for these fit parameters; with factors: v_offset + factor * (template - v_offset)."""
        template = self.grid.model_profile_for_channels_index(self.channels, *params)
        if not self.use_factors:
            return template
        v_offset = params[2]
        return v_offset + self.factors * (template - v_offset)

    def deviation(self, donor):
        """The donor's real amplitudes minus what template() makes for the donor's own parameters."""
        return self.donor_reference[donor] - self.template(self.donor_params[donor])

    # Steps 2 and 3 (and the deviation)
    def bunch(self, params, donor=None):
        """Firmware amplitudes for one bunch with the given fit parameters (and donor, for deviations)."""
        reference = self.template(params)
        if self.use_deviations:
            # Scaled to this bunch's brightness; 1 in sample(), where the bunch has the donor's norm.
            brightness = params[4] / self.donor_params[donor, 4]
            reference = reference + brightness * self.deviation(donor)
        return self.firmware_from_reference(reference)

    def firmware_from_reference(self, reference):
        """firmware = gain * reference + offset + noise, per channel."""
        firmware = self.fw_gain * reference + self.fw_offset
        noise_size = NOISE_ADC + NOISE_FRACTION * np.clip(firmware, 0, None)
        return firmware + noise_size * self.rng.standard_normal(firmware.shape)

    def sample(self, sigma_y):
        """Fake bunches for the requested beam sizes -> (amplitudes (n, 42), mu, sigma_y, donor sweep)."""
        sigma_y = np.asarray(sigma_y, dtype=float)

        # Step 1: a random real bunch's fit parameters, with the requested beam size.
        donors = self.rng.integers(len(self.donor_params), size=len(sigma_y))
        params = self.donor_params[donors].copy()
        params[:, 1] = sigma_y

        x = np.array([self.bunch(p, d) for p, d in zip(params, donors)])
        return x, params[:, 0], sigma_y, self.donor_sweep[donors]


def generate(calibration, count, sigma_min, sigma_max, seed, output, factors=False, deviations=False):
    rng = np.random.default_rng(seed)
    gen = XrmGenerator(calibration, rng, factors, deviations)

    sigma_y = rng.uniform(sigma_min, sigma_max, count)
    x, mu, sigma_y, donor_sweep = gen.sample(sigma_y)

    # Real bunches only exist inside this beam-size range, so only there can
    # the generator be checked against real data.
    lo, hi = gen.donor_params[:, 1].min(), gen.donor_params[:, 1].max()
    in_real_range = (sigma_y >= lo) & (sigma_y <= hi)

    np.savez(
        output,
        x_data=x.astype(np.float32),
        true_mu=mu.astype(np.float32),
        true_sig_y=sigma_y.astype(np.float32),
        in_real_sigma_range=in_real_range,
        real_sigma_range_um=np.array([lo, hi]),
        donor_sweep=donor_sweep,
        factors=factors,
        deviations=deviations,
    )

    print(f"Generated {count} bunches (factors {'on' if factors else 'off'}, "
          f"deviations {'on' if deviations else 'off'}), sigma_y {sigma_min}..{sigma_max} um -> {output}")
    print(f"Note: real data covers sigma_y {lo:.0f}..{hi:.0f} um; "
          f"{100 * np.mean(~in_real_range):.0f}% of these bunches lie outside it.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def normalized(amplitudes):
    """Amplitudes of the validation channels divided by their sum, as the network sees them."""
    sub = amplitudes[:, VALIDATION_CHANNELS]
    return sub / sub.sum(axis=1, keepdims=True)


def fine_structure(shape):
    """Sum of |second difference| of the normalized shape: shrinks as sigma_y grows."""
    return np.abs(np.diff(shape, 2, axis=1)).sum(axis=1)


def within_sweep(shape, sweep):
    """Each bunch's difference from its own sweep's average shape."""
    out = np.empty_like(shape)
    for s in np.unique(sweep):
        out[sweep == s] = shape[sweep == s] - shape[sweep == s].mean(axis=0)
    return out


def neighbour_corr(shape, sweep):
    """Average bunch-to-bunch correlation of channels 1, 2, 3 and >5 apart."""
    corr = np.corrcoef(within_sweep(shape, sweep).T)
    i, j = np.triu_indices(corr.shape[0], 1)
    groups = [j - i == 1, j - i == 2, j - i == 3, j - i > 5]
    return [np.nanmean(corr[i[g], j[g]]) for g in groups]


def per_sweep_spread(shape, sweep):
    """Per channel: median over sweeps of the bunch-to-bunch standard deviation."""
    return np.median([shape[sweep == s].std(axis=0) for s in np.unique(sweep)], axis=0)


def validate(calibration, n_sweeps, every, seed, factors=False, deviations=False):
    """For real held-out bunches: generate a fake from each one's own fit parameters and compare."""
    gen = XrmGenerator(calibration, np.random.default_rng(seed), factors, deviations)
    picks = np.linspace(0, len(gen.held_out_sweeps) - 1, n_sweeps).astype(int)
    print(f"Generator: factors {'on' if factors else 'off'}, deviations {'on' if deviations else 'off'}")

    # With deviations, each real bunch needs a donor like it in all six fit
    # parameters (each scaled by its spread so all count equally).
    scale = np.std(gen.donor_params, axis=0)
    tree = cKDTree(gen.donor_params / scale) if deviations else None

    def similar_donor(params):
        if tree is None:
            return None
        _, nearest = tree.query(params / scale, k=TWIN_NEIGHBOURS)
        return gen.rng.choice(nearest)

    real, fake, sweep_id = [], [], []
    for k, sweep in enumerate(gen.held_out_sweeps[picks]):
        d = np.load(AMPLITUDE_DIR / f"{sweep}.npz")
        for fw, params in zip(d["firmware"][::every], d["fit_params"][::every]):
            if np.all(np.isfinite(fw)) and np.all(np.isfinite(params)):
                real.append(fw)
                fake.append(gen.bunch(params, similar_donor(params)))
                sweep_id.append(k)

    sweep_id = np.array(sweep_id)
    shapes = {"real": normalized(np.array(real)), "generator": normalized(np.array(fake))}
    print(f"{len(sweep_id)} real bunches from {n_sweeps} held-out sweeps; "
          f"shapes over channels {VALIDATION_CHANNELS[0]}-{VALIDATION_CHANNELS[-1]}")

    off = 100 * np.abs(shapes["generator"].mean(axis=0) / shapes["real"].mean(axis=0) - 1)
    ratio = per_sweep_spread(shapes["generator"], sweep_id) / per_sweep_spread(shapes["real"], sweep_id)
    print(f"\nAverage shape: generator off by {np.median(off):.1f}% on a typical channel "
          f"(worst channel {off.max():.1f}%)")
    print(f"Bunch-to-bunch spread per channel, generator / real: {np.median(ratio):.2f} "
          f"({ratio.min():.2f}-{ratio.max():.2f})")

    print("\nNeighbouring channels varying together (1 / 2 / 3 / >5 channels apart):")
    for name, shape in shapes.items():
        print(f"  {name:>9s}: " + " / ".join(f"{c:+.2f}" for c in neighbour_corr(shape, sweep_id)))

    print("\nFine structure of the profile (shrinks as sigma_y grows):")
    for name, shape in shapes.items():
        fs = fine_structure(shape)
        spread = np.median([fs[sweep_id == s].std() for s in np.unique(sweep_id)])
        print(f"  {name:>9s}: average {fs.mean():.4f}, bunch-to-bunch spread {spread:.4f}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
            prog="xrm_generator",
            description="Realistic fake XRM firmware amplitudes.",
            )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("calibrate", help="Measure the generator's inputs from real data.")
    p.add_argument("-o", "--output", default=str(DEFAULT_CALIBRATION), help="Calibration file to write.")

    p = sub.add_parser("generate", help="Make fake bunches.")
    p.add_argument("-n", "--count", type=int, default=100000, help="Number of bunches.")
    p.add_argument("--sigma-min", type=float, default=10.0, help="Smallest beam size, um.")
    p.add_argument("--sigma-max", type=float, default=200.0, help="Largest beam size, um.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), help="Calibration file.")
    p.add_argument("-o", "--output", required=True, help="Output .npz file.")
    add_extras(p)

    p = sub.add_parser("validate", help="Compare fakes of real bunches with held-out real sweeps.")
    p.add_argument("--n-sweeps", type=int, default=60, help="Held-out sweeps to use.")
    p.add_argument("--every", type=int, default=10, help="Use every n-th bunch.")
    p.add_argument("--seed", type=int, default=1, help="Random seed.")
    p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), help="Calibration file.")
    add_extras(p)

    return parser.parse_args()


def add_extras(parser):
    parser.add_argument("--factors", action="store_true", help="Multiply templates by the channel factors.")
    parser.add_argument("--deviations", action="store_true", help="Add the donors' real deviations.")


def main():
    args = parse_args()

    if args.command == "calibrate":
        calibrate(args.output)
    elif args.command == "generate":
        generate(args.calibration, args.count, args.sigma_min, args.sigma_max, args.seed, args.output,
                 args.factors, args.deviations)
    else:
        validate(args.calibration, args.n_sweeps, args.every, args.seed, args.factors, args.deviations)


if __name__ == "__main__":
    main()
