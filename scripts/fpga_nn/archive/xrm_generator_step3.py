#!/usr/bin/env python3

"""
XRM data generator: fake firmware amplitudes that look like real ones.

Each fake bunch is built in four steps:
  1. Donor:     pick a real bunch with a similar beam size; borrow its beam
                position, brightness and image placement.
  2. Template:  the ideal profile for the requested beam size, from the
                template library, multiplied by a correction factor per
                channel (real channels read consistently higher or lower
                than the templates).
  3. Deviation: add the donor's real deviation from its own corrected
                template (how a real bunch differs from an ideal one).
  4. Firmware:  turn these best-quality amplitudes into what the FPGA would
                measure: about 0.93x per channel plus a little scatter.

Commands:
  calibrate  measure factors, firmware response and donor deviations from
             real data (data/amplitudes, stable period) -> one calibration file
  generate   make fake bunches -> .npz with x_data, true_mu, true_sig_y
  validate   compare fake twins with real sweeps not used for calibration

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

# Firmware scatter is measured in these amplitude bins (ADC).
AMPLITUDE_BINS = np.array([-50, 0, 25, 50, 100, 200, 400, 800, 1600, 3200])

# Donors: drop fits that hit a bound of the template fit (bounds copied
# from fit_from_rootfile.py) and the 1% most extreme beam sizes at each end.
FIT_LOWER = np.array([0.0, 10.0, -150.0, 465.0, 0.0, 13.0])
FIT_UPPER = np.array([150.0, 200.0, 50.0, 545.0, np.inf, 17.0])
DONOR_SIGMA_PERCENTILES = (1, 99)
DONOR_EVERY = 5
SIGMA_BIN_UM = 2.0
TWIN_NEIGHBOURS = 20

# Validation uses the channels with measured factors.
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


def corrected_template(grid, params, factors, channels):
    """v_offset + factor * (template - v_offset): the ideal profile, corrected per channel."""
    template = grid.model_profile_for_channels_index(channels, *params)
    v_offset = params[2]
    return v_offset + factors * (template - v_offset)


def robust_std(values):
    """Standard deviation estimated from the median absolute deviation (ignores outliers)."""
    values = values[np.isfinite(values)]
    if len(values) < 30:
        return np.nan
    return 1.4826 * np.median(np.abs(values - np.median(values)))


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
    Per channel, firmware = gain * reference + offset + scatter.
    The scatter is described by a table (typical size per amplitude bin)
    and the fraction of it that all channels share in a bunch.
    """
    reference = np.concatenate([np.load(p)["reference"][::10] for p in paths])
    firmware = np.concatenate([np.load(p)["firmware"][::10] for p in paths])
    n_ch = reference.shape[1]

    gain, offset = np.ones(n_ch), np.zeros(n_ch)
    for ch in range(n_ch):
        ok = np.isfinite(reference[:, ch]) & np.isfinite(firmware[:, ch])
        gain[ch], offset[ch] = np.polyfit(reference[ok, ch], firmware[ok, ch], 1)
    residual = firmware - (gain * reference + offset)

    # Scatter table: in each amplitude bin, the median amplitude and the spread of the residual.
    amplitude, res = reference.ravel(), residual.ravel()
    scatter_amplitude, scatter = [], []
    for lo, hi in zip(AMPLITUDE_BINS[:-1], AMPLITUDE_BINS[1:]):
        in_bin = (amplitude >= lo) & (amplitude < hi) & np.isfinite(res)
        if in_bin.sum() >= 30:
            scatter_amplitude.append(np.median(amplitude[in_bin]))
            scatter.append(robust_std(res[in_bin]))

    # Shared fraction: the average correlation of residuals between channels
    # more than 5 apart (they share nothing else). Outliers are clipped at 5 sigma.
    ok = np.all(np.isfinite(residual), axis=1)
    z = np.clip(residual[ok] / residual[ok].std(axis=0), -5, 5)
    corr = np.corrcoef(z.T)
    i, j = np.triu_indices(n_ch, 1)
    far = j - i > 5
    shared = float(np.clip(np.nanmean(corr[i[far], j[far]]), 0, 1))

    return gain, offset, np.array(scatter_amplitude), np.array(scatter), shared


def build_donor_library(paths, grid, factors):
    """Each donor's fit parameters and its deviation from its own corrected template."""
    channels = np.arange(len(factors), dtype=float)
    params, deviation, sweep = [], [], []
    for path in paths:
        d = np.load(path)
        for reference, p in zip(d["reference"][::DONOR_EVERY], d["fit_params"][::DONOR_EVERY]):
            if np.all(np.isfinite(reference)) and np.all(np.isfinite(p)):
                params.append(p)
                deviation.append(reference - corrected_template(grid, p, factors, channels))
                sweep.append(path.stem)
    params, deviation, sweep = np.array(params), np.array(deviation), np.array(sweep)

    at_bound = np.any((np.abs(params - FIT_LOWER) < 1e-3) | (np.abs(params - FIT_UPPER) < 1e-3), axis=1)
    sigma_lo, sigma_hi = np.percentile(params[~at_bound, 1], DONOR_SIGMA_PERCENTILES)
    keep = ~at_bound & (params[:, 1] >= sigma_lo) & (params[:, 1] <= sigma_hi)
    return params[keep], deviation[keep], sweep[keep]


def calibrate(output):
    calibration_paths, held_out = stable_sweeps()
    print(f"Stable period from {STABLE_FROM}: {len(calibration_paths)} calibration sweeps, "
          f"{len(held_out)} held out for validation ({len(EXCLUDED_SWEEPS)} sweeps excluded)")

    factors, measured = measure_channel_factors(calibration_paths)
    print(f"Channel factors: {measured.sum()} channels measured, "
          f"range {factors[measured].min():.2f}..{factors[measured].max():.2f}")

    gain, offset, scatter_amplitude, scatter, shared = measure_firmware_response(calibration_paths)
    print(f"Firmware response: gain {np.median(gain[measured]):.3f} (median over measured channels), "
          f"shared scatter fraction {shared:.2f}")
    print("  scatter: " + ", ".join(f"{s:.1f} ADC at {a:.0f}" for a, s in zip(scatter_amplitude, scatter)))

    grid = ffr.SmearedGridModel.from_npz(TEMPLATE_GRID)
    donor_params, donor_deviation, donor_sweep = build_donor_library(calibration_paths, grid, factors)
    print(f"Donor library: {len(donor_params)} bunches, sigma_y "
          f"{donor_params[:, 1].min():.1f}..{donor_params[:, 1].max():.1f} um")

    np.savez(
        output,
        factors=factors,
        factor_measured=measured,
        fw_gain=gain,
        fw_offset=offset,
        fw_scatter_amplitude=scatter_amplitude,
        fw_scatter=scatter,
        fw_shared=shared,
        donor_params=donor_params,
        donor_deviation=donor_deviation,
        donor_sweep=donor_sweep,
        calibration_sweeps=np.array([p.stem for p in calibration_paths]),
        held_out_sweeps=np.array([p.stem for p in held_out]),
    )
    print(f"Saved {output}")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class XrmGenerator:

    def __init__(self, calibration=DEFAULT_CALIBRATION, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.grid = ffr.SmearedGridModel.from_npz(TEMPLATE_GRID)

        cal = np.load(calibration)
        self.factors = cal["factors"]
        self.factor_measured = cal["factor_measured"]
        self.channels = np.arange(len(self.factors), dtype=float)
        self.fw_gain = cal["fw_gain"]
        self.fw_offset = cal["fw_offset"]
        self.fw_scatter_amplitude = cal["fw_scatter_amplitude"]
        self.fw_scatter = cal["fw_scatter"]
        self.fw_shared = float(cal["fw_shared"])
        self.donor_params = cal["donor_params"]
        self.donor_deviation = cal["donor_deviation"]
        self.donor_sweep = cal["donor_sweep"]
        self.held_out_sweeps = cal["held_out_sweeps"]

        # For picking donors by beam size: each donor's 2 um sigma_y bin.
        self.donor_bin = np.floor(self.donor_params[:, 1] / SIGMA_BIN_UM).astype(int)
        self.filled_bins = np.unique(self.donor_bin)

        # For picking donors by all six fit parameters: a nearest-neighbour
        # search, with each parameter scaled by its spread so all count equally.
        self.param_scale = np.std(self.donor_params, axis=0)
        self.param_tree = cKDTree(self.donor_params / self.param_scale)

    # Step 1: donor
    def donor_for_sigma(self, sigma_y):
        """A random donor from the 2 um sigma_y bin closest to sigma_y that has donors."""
        wanted = int(np.floor(sigma_y / SIGMA_BIN_UM))
        nearest = self.filled_bins[np.argmin(np.abs(self.filled_bins - wanted))]
        return self.rng.choice(np.nonzero(self.donor_bin == nearest)[0])

    def donor_for_params(self, params):
        """A random donor among the 20 most similar in all six fit parameters."""
        _, nearest = self.param_tree.query(np.asarray(params) / self.param_scale, k=TWIN_NEIGHBOURS)
        return self.rng.choice(nearest)

    # Step 4: firmware
    def firmware_from_reference(self, reference):
        """firmware = gain * reference + offset + scatter (partly shared by all channels)."""
        size = np.interp(reference, self.fw_scatter_amplitude, self.fw_scatter)
        shared = self.rng.standard_normal()
        own = self.rng.standard_normal(reference.shape)
        scatter = size * (np.sqrt(self.fw_shared) * shared + np.sqrt(1 - self.fw_shared) * own)
        return self.fw_gain * reference + self.fw_offset + scatter

    def bunch(self, params, donor):
        """Steps 2-4 for one bunch with fit parameters params and the given donor."""
        reference = corrected_template(self.grid, params, self.factors, self.channels)
        brightness = params[4] / self.donor_params[donor, 4]
        reference = reference + brightness * self.donor_deviation[donor]
        return self.firmware_from_reference(reference)

    def sample(self, sigma_y):
        """Fake bunches for the requested beam sizes -> (amplitudes (n, 42), mu, sigma_y, donor sweep)."""
        sigma_y = np.asarray(sigma_y, dtype=float)
        x = np.empty((len(sigma_y), len(self.channels)))
        mu = np.empty(len(sigma_y))
        donor_sweep = np.empty(len(sigma_y), dtype=self.donor_sweep.dtype)

        for i, s in enumerate(sigma_y):
            donor = self.donor_for_sigma(s)
            params = self.donor_params[donor].copy()
            params[1] = s
            x[i] = self.bunch(params, donor)
            mu[i] = params[0]
            donor_sweep[i] = self.donor_sweep[donor]

        return x, mu, sigma_y, donor_sweep

    def twin(self, params):
        """Fake version of a real bunch: its own fit parameters, a similar donor's deviation.
        Donors come from calibration sweeps only, so a held-out bunch never gets its own deviation."""
        params = np.asarray(params, dtype=float)
        return self.bunch(params, self.donor_for_params(params))


def generate(calibration, count, sigma_min, sigma_max, seed, output):
    rng = np.random.default_rng(seed)
    gen = XrmGenerator(calibration, rng)

    sigma_y = rng.uniform(sigma_min, sigma_max, count)
    x, mu, sigma_y, donor_sweep = gen.sample(sigma_y)

    # Real deviations only exist inside the donors' beam-size range; outside
    # it they are borrowed from the nearest beam size and are unverified.
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
        factor_measured=gen.factor_measured,
    )

    print(f"Generated {count} bunches, sigma_y {sigma_min}..{sigma_max} um -> {output}")
    print(f"Note: real deviations exist for sigma_y {lo:.0f}..{hi:.0f} um; "
          f"{100 * np.mean(~in_real_range):.0f}% of these bunches borrow from the nearest beam size.")


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


def validate(calibration, n_sweeps, every, seed):
    gen = XrmGenerator(calibration, np.random.default_rng(seed))
    picks = np.linspace(0, len(gen.held_out_sweeps) - 1, n_sweeps).astype(int)

    real, fake, plain, sweep_id = [], [], [], []
    for k, sweep in enumerate(gen.held_out_sweeps[picks]):
        d = np.load(AMPLITUDE_DIR / f"{sweep}.npz")
        for fw, params, template in zip(d["firmware"][::every], d["fit_params"][::every], d["model"][::every]):
            if np.all(np.isfinite(fw)) and np.all(np.isfinite(params)) and np.all(np.isfinite(template)):
                real.append(fw)
                fake.append(gen.twin(params))
                plain.append(gen.firmware_from_reference(template))
                sweep_id.append(k)

    sweep_id = np.array(sweep_id)
    shapes = {"real": normalized(np.array(real)),
              "generator": normalized(np.array(fake)),
              "template-only": normalized(np.array(plain))}
    print(f"{len(sweep_id)} real bunches from {n_sweeps} held-out sweeps; "
          f"shapes over channels {VALIDATION_CHANNELS[0]}-{VALIDATION_CHANNELS[-1]}")

    mean_real = shapes["real"].mean(axis=0)
    spread_real = per_sweep_spread(shapes["real"], sweep_id)
    print("\nCompared with real bunches, per channel:")
    for name in ("generator", "template-only"):
        off = 100 * np.abs(shapes[name].mean(axis=0) / mean_real - 1)
        ratio = per_sweep_spread(shapes[name], sweep_id) / spread_real
        print(f"  {name:>14s}: average shape off by {np.median(off):.1f}% (worst channel {off.max():.1f}%); "
              f"bunch-to-bunch spread fake/real {np.median(ratio):.2f} ({ratio.min():.2f}-{ratio.max():.2f})")

    print("\nNeighbouring channels varying together (1 / 2 / 3 / >5 channels apart):")
    for name, shape in shapes.items():
        print(f"  {name:>14s}: " + " / ".join(f"{c:+.2f}" for c in neighbour_corr(shape, sweep_id)))

    print("\nFine structure of the profile (shrinks as sigma_y grows):")
    for name, shape in shapes.items():
        fs = fine_structure(shape)
        spread = np.median([fs[sweep_id == s].std() for s in np.unique(sweep_id)])
        print(f"  {name:>14s}: average {fs.mean():.4f}, bunch-to-bunch spread {spread:.4f}")


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

    p = sub.add_parser("validate", help="Compare fake twins with held-out real sweeps.")
    p.add_argument("--n-sweeps", type=int, default=60, help="Held-out sweeps to use.")
    p.add_argument("--every", type=int, default=10, help="Use every n-th bunch.")
    p.add_argument("--seed", type=int, default=1, help="Random seed.")
    p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), help="Calibration file.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "calibrate":
        calibrate(args.output)
    elif args.command == "generate":
        generate(args.calibration, args.count, args.sigma_min, args.sigma_max, args.seed, args.output)
    else:
        validate(args.calibration, args.n_sweeps, args.every, args.seed)


if __name__ == "__main__":
    main()
