#!/usr/bin/env python3

import argparse
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
            prog='generate_fake_histograms',
            description='create fake histograms to train neural network',
            epilog='Run with an .npz containing beam profiles'
            )

    parser.add_argument(
            'filename',
            help="Input .npz file containing beam profiles",
            )

    parser.add_argument(
            'config',
            help='Input .config file',
            )

    parser.add_argument(
            '-c',
            '--count',
            default=1000,
            type=int,
            help='Number of histograms to generate',
            )

    parser.add_argument(
            '--seed',
            type=int,
            default=None,
            help='Random seed for reproducible fake histograms.',
            )

    parser.add_argument(
            '--CS',
            type=int,
            default=None,
            help='Fixed channel shift for validation data',
            )

    parser.add_argument(
            '-o',
            '--output-dir',
            default='fake_histograms',
            help='Folder to store fake histograms',
            )

    return parser.parse_args()

def load_config(config_path):
    namespace = {}

    with open(config_path, "r") as f:
        exec(f.read(), namespace)

    return namespace

def unpack_npz(filename):
    data = np.load(filename)

    smeprf = data['smeprf']
    mu_axis = data['mu_axis']
    sig_axis = data['sig_axis']
    y_positions = data['y_positions']

    return smeprf, mu_axis, sig_axis, y_positions

def require_config_keys(config, required_keys):
    missing_keys = []

    for key in required_keys:
        if key not in config:
            missing_keys.append(key)

    if missing_keys:
        raise ValueError(
            "Missing required config parameters: "
            + ", ".join(missing_keys)
        )

def validate_config(config, y_positions, mu_axis, sig_axis):
    required_keys = [
        "channel_indices",
        "expected_num_channels",
        "mu_mean",
        "mu_std",
        "sig_y_primary_mean",
        "sig_y_primary_std",
        "sig_y_secondary_mean",
        "sig_y_secondary_std",
        "sig_y_primary_weight",
        "noise_fraction",
        "channel_shift_distribution",
        "channel_shift_mean",
        "channel_shift_std",
        "channel_shift_min",
        "channel_shift_max",
    ]
    require_config_keys(config, required_keys)

    channel_indices = np.asarray(config["channel_indices"], dtype=int)
    expected_num_channels = int(config["expected_num_channels"])

    mu_mean = float(config["mu_mean"])
    mu_std = float(config["mu_std"])

    sig_y_primary_mean = float(config["sig_y_primary_mean"])
    sig_y_primary_std = float(config["sig_y_primary_std"])

    sig_y_secondary_mean = float(config["sig_y_secondary_mean"])
    sig_y_secondary_std = float(config["sig_y_secondary_std"])

    sig_y_primary_weight = float(config["sig_y_primary_weight"])

    noise_fraction = float(config["noise_fraction"])

    channel_shift_distribution = str(config["channel_shift_distribution"]).strip().lower()
    channel_shift_mean = float(config["channel_shift_mean"])
    channel_shift_std = float(config["channel_shift_std"])
    channel_shift_min = int(config["channel_shift_min"])
    channel_shift_max = int(config["channel_shift_max"])

    if channel_indices.ndim != 1:
        raise ValueError("channel_indices must be one-dimensional.")

    if len(channel_indices) != expected_num_channels:
        raise ValueError(
            f"Expected {expected_num_channels} channel indices, "
            f"got {len(channel_indices)}."
        )

    if np.any(channel_indices < 0) or np.any(channel_indices >= len(y_positions)):
        raise ValueError(
            f"channel_indices must be between 0 and {len(y_positions) - 1}."
        )

    if len(np.unique(channel_indices)) != len(channel_indices):
        raise ValueError("channel_indices contains duplicate values.")

    if not (mu_axis.min() <= mu_mean <= mu_axis.max()):
        raise ValueError(
            f"mu_mean={mu_mean} is outside available mu_axis range "
            f"{mu_axis.min()} to {mu_axis.max()}."
        )

    if not (sig_axis.min() <= sig_y_primary_mean <= sig_axis.max()):
        raise ValueError(
            f"sig_y_primary_mean={sig_y_primary_mean} is outside available "
            f"sig_axis range {sig_axis.min()} to {sig_axis.max()}."
        )

    if not (sig_axis.min() <= sig_y_secondary_mean <= sig_axis.max()):
        raise ValueError(
            f"sig_y_secondary_mean={sig_y_secondary_mean} is outside available "
            f"sig_axis range {sig_axis.min()} to {sig_axis.max()}."
        ) 

    if mu_std <= 0:
        raise ValueError("mu_std must be positive.")

    if sig_y_primary_std <= 0:
        raise ValueError("sig_y_primary_std must be positive.")

    if sig_y_secondary_std <= 0:
        raise ValueError("sig_y_secondary_std must be positive.")

    if not (0.0 <= sig_y_primary_weight <= 1.0):
        raise ValueError("sig_y_primary_weight must be between 0 and 1.")

    if noise_fraction < 0:
        raise ValueError("noise_fraction must be non-negative")

    if channel_shift_distribution not in ['gaussian', 'uniform']:
        raise ValueError(
                f"channel_shift_distribution must be set to gaussian or uniform, "
                f"got {channel_shift_distribution}"
        )

    if channel_shift_std < 0:
        raise ValueError("channel_shift_std must be non-negative.")
    
    if channel_shift_min > channel_shift_max:
        raise ValueError("channel_shift_min is greater than channel_shift_max")

    return {
        "channel_indices": channel_indices,
        "expected_num_channels": expected_num_channels,
        "mu_mean": mu_mean,
        "mu_std": mu_std,
        "sig_y_primary_mean": sig_y_primary_mean,
        "sig_y_primary_std": sig_y_primary_std,
        "sig_y_secondary_mean": sig_y_secondary_mean,
        "sig_y_secondary_std": sig_y_secondary_std,
        "sig_y_primary_weight": sig_y_primary_weight,
        "noise_fraction": noise_fraction,
        "channel_shift_distribution": channel_shift_distribution,
        "channel_shift_mean": channel_shift_mean,
        "channel_shift_std": channel_shift_std,
        "channel_shift_min": channel_shift_min,
        "channel_shift_max": channel_shift_max,
    }

def validate_fixed_channel_shift(fixed_channel_shift):
    if fixed_channel_shift is None:
        return
    
    if abs(fixed_channel_shift) > 5:
        raise ValueError("fixed_channel_shift must be between -5 and 5")

def get_nearest_axis_index(axis, value):
    return np.argmin(np.abs(axis - value))

def sample_truncated_normal(mean, std, min_value, max_value, rng):
    if std < 0.0:
        raise ValueError("Standard deviation must be non-negative")

    if std == 0.0:
        if min_value <= mean <= max_value:
            return mean

        raise ValueError(
            f"Fixed value {mean} is outside allowed range "
            f"[{min_value}, {max_value}]"
        )

    while True:
        value = rng.normal(loc=mean, scale=std)

        if min_value <= value <= max_value:
            return value

def sample_mu_and_sig_y(params, rng, mu_axis, sig_axis):
    mu_sample = sample_truncated_normal(
        mean=params["mu_mean"],
        std=params["mu_std"],
        min_value=mu_axis.min(),
        max_value=mu_axis.max(),
        rng=rng,
    )

    if rng.random() < params["sig_y_primary_weight"]:
        sig_y_sample = sample_truncated_normal(
            mean=params["sig_y_primary_mean"],
            std=params["sig_y_primary_std"],
            min_value=sig_axis.min(),
            max_value=sig_axis.max(),
            rng=rng,
        )
    else:
        sig_y_sample = sample_truncated_normal(
            mean=params["sig_y_secondary_mean"],
            std=params["sig_y_secondary_std"],
            min_value=sig_axis.min(),
            max_value=sig_axis.max(),
            rng=rng,
        )

    return mu_sample, sig_y_sample

def sample_channel_shift(params, rng):
    if params["channel_shift_distribution"] == 'gaussian':
        while True:
            shift = rng.normal(
                    loc=params["channel_shift_mean"],
                    scale=params["channel_shift_std"],
            )

            if params["channel_shift_min"] <= shift <= params["channel_shift_max"]:
                return int(np.round(shift))
 
    elif params["channel_shift_distribution"] == 'uniform':
        return rng.integers(
                low=params["channel_shift_min"],
                high=params["channel_shift_max"]+1,
        )

    else:
        raise ValueError(f"Unexpected channel_shift_distribution: {params['channel_shift_distribution']}")

def shift_profile(profile, shift):
    shifted = np.zeros_like(profile)

    if shift > 0:
        shifted[shift:] = profile[:-shift]
    elif shift < 0:
        shifted[:shift] = profile[-shift:]
    else:
        shifted = profile.copy()

    return shifted

def shift_profile_in_y_position_space(profile, channel_shift):
    # Positive channel_shift means move toward larger y_position.
    # Because y_positions decreases with array index, flip the sign.
    return shift_profile(profile, -channel_shift)

def profile_to_histogram(profile, channel_indices):
    return profile[channel_indices]

def add_gaussian_noise(hist, noise_fraction, rng):
    noise = rng.normal(
            loc=0.0,
            scale=noise_fraction * hist,
            size=hist.shape,
    )

    noisy_hist = hist + noise
    noisy_hist = np.clip(noisy_hist, 0.0, None)

    return noisy_hist

def normalize_area(hist):
    total = np.sum(hist)
    
    if total <= 0:
        raise ValueError("Cannot normalize histogram with non-positive area")

    return hist / total

def generate_training_data(smeprf, mu_axis, sig_axis, params, count, rng):
    num_channels = params["expected_num_channels"]
    fixed_channel_shift = params["fixed_channel_shift"]
    num_sensor_channels = len(smeprf[0, 0, :])

    histograms = np.empty((count, num_channels))
    raw_histograms = np.empty((count, num_sensor_channels))
    template_mu_values = np.empty(count)
    template_sig_y_values = np.empty(count)
    channel_shifts = np.empty(count, dtype=int)

    for i in range(count):
        mu_sample, sig_y_sample = sample_mu_and_sig_y(
            params=params,
            rng=rng,
            mu_axis=mu_axis,
            sig_axis=sig_axis,
        )

        mu_idx = get_nearest_axis_index(mu_axis, mu_sample)
        sig_y_idx = get_nearest_axis_index(sig_axis, sig_y_sample)

        profile = smeprf[mu_idx, sig_y_idx, :]

        if fixed_channel_shift is not None:
            channel_shift = fixed_channel_shift
        else:
            channel_shift = sample_channel_shift(params, rng)
        shifted_profile = shift_profile_in_y_position_space(profile, channel_shift)

        noisy_shifted_profile = add_gaussian_noise(
            hist=shifted_profile,
            noise_fraction=params["noise_fraction"],
            rng=rng,
        )

        hist = profile_to_histogram(
            profile=noisy_shifted_profile,
            channel_indices=params["channel_indices"],
        )

        fake_hist = normalize_area(hist)

        histograms[i, :] = fake_hist
        raw_histograms[i, :] = noisy_shifted_profile
        template_mu_values[i] = mu_axis[mu_idx]
        template_sig_y_values[i] = sig_axis[sig_y_idx]
        channel_shifts[i] = channel_shift

    return {
        "histograms": histograms,
        "raw_histograms": raw_histograms,
        "template_mu": template_mu_values,
        "template_sig_y": template_sig_y_values,
        "channel_shift": channel_shifts,
    }

def main():
    args = parse_args()

    smeprf, mu_axis, sig_axis, y_positions = unpack_npz(args.filename)

    print(f"Found .npz file: {args.filename}...")
    print(f"{args.filename} details:")
    print(f"\tmu_axis range: {mu_axis.min()} to {mu_axis.max()}")
    print(f"\tsig_axis range: {sig_axis.min()} to {sig_axis.max()}")
    print(f"\tsmeprf shape: {smeprf.shape}\n")

    config = load_config(args.config)
    validate_fixed_channel_shift(args.CS)
    params = validate_config(config, y_positions, mu_axis, sig_axis)
    params["fixed_channel_shift"] = args.CS

    channel_indices = params["channel_indices"]
    mu_mean = params["mu_mean"]
    mu_std = params["mu_std"]
    sig_y_primary_mean = params["sig_y_primary_mean"]
    sig_y_primary_std = params["sig_y_primary_std"]
    sig_y_secondary_mean = params["sig_y_secondary_mean"]
    sig_y_secondary_std = params["sig_y_secondary_std"]
    sig_y_primary_weight = params["sig_y_primary_weight"]
    noise_fraction = params["noise_fraction"]
    channel_shift_distribution = params["channel_shift_distribution"]
    channel_shift_mean = params["channel_shift_mean"]
    channel_shift_std = params["channel_shift_std"]
    channel_shift_min = params["channel_shift_min"]
    channel_shift_max = params["channel_shift_max"]
    fixed_channel_shift = params["fixed_channel_shift"]

    print(f"Found .config file: {args.config}...")
    print("Current config settings:")
    print(f"\tmu mean: {mu_mean}\tmu std: {mu_std}")
    print(f"\tprimary sigma_y mean: {sig_y_primary_mean}\tprimary sigma_y std: {sig_y_primary_std}")
    print(f"\tsecondary sigma_y mean: {sig_y_secondary_mean}\tsecondary sigma_y std: {sig_y_secondary_std}")
    print(f"\tprimary sigma_y weight: {sig_y_primary_weight}")
    print(f"\tnoise_fraction: {noise_fraction}")
    print(f"\tchannel shift distribution: {channel_shift_distribution}")
    if channel_shift_distribution == "gaussian":
        print(f"\tchannel_shift_mean: {channel_shift_mean}")
        print(f"\tchannel_shift_std: {channel_shift_std}\n")
        print(f"\tchannel_shift min: {channel_shift_min}")
        print(f"\tchannel_shift max: {channel_shift_max}")
    elif channel_shift_distribution == "uniform":
        print(f"\t channel_shift_min: {channel_shift_min}")
        print(f"\t channel_shift_max: {channel_shift_max}")
    if fixed_channel_shift is None:
        print(f"\tFixed channel shift: OFF")
    else:
        print(f"\tFixed channel shift: {fixed_channel_shift}")
    print(f"\tTotal selected channels: {len(channel_indices)}")
    print(f"\tSelected channels:\n\t{channel_indices}\n")

    rng = np.random.default_rng(args.seed)

    print(f"Generating {args.count} histograms...")
    training_data = generate_training_data(
        smeprf=smeprf,
        mu_axis=mu_axis,
        sig_axis=sig_axis,
        params=params,
        count=args.count,
        rng=rng,
    )

    print(f"Storing histograms in {args.output_dir}...\n")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fixed_channel_shift == None:
        output_file = output_dir / "fake_training_data.npz"
    else:
        output_file = output_dir / "fake_validation_data.npz"
    np.savez(
        output_file,
        histograms=training_data["histograms"],
        raw_histograms=training_data["raw_histograms"],
        template_mu=training_data["template_mu"],
        template_sig_y=training_data["template_sig_y"],
        channel_shift=training_data["channel_shift"],
    )

    print(f"Saved training data to {output_file}")

if __name__ == "__main__":
    main()
