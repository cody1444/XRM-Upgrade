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

def validate_config(config):
    required_keys = [
        "seed",
        "mu_distribution",
        "sig_y_distribution",
        "channel_shift_distribution",
        "noise_fraction",
    ]

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config value: {key}")

    validate_distribution(config["mu_distribution"], name="mu_distribution")
    validate_distribution(config["sig_y_distribution"], name="sig_y_distribution")
    validate_distribution(config["channel_shift_distribution"], name="channel_shift_distribution")

    if not 0 <= config["noise_fraction"] <= 1:
        raise ValueError("noise_fraction must be between 0 and 1")

def validate_distribution(dist, name):
    if "components" not in dist:
        raise KeyError(f"{name} is missing required key: components")

    components = dist["components"]

    if len(components) == 0:
        raise ValueError(f"{name}: components must contain at least one component")

    for i, component in enumerate(components):
        component_name = f"{name}.components[{i}]"

        if "type" not in component:
            raise KeyError(f"{component_name} is missing required key: type")

        dist_type = component["type"]

        if dist_type == "uniform":
            required = ["min", "max", "weight"]

        elif dist_type == "gaussian":
            required = ["mean", "std", "min", "max", "weight"]

        else:
            raise ValueError(
                f"{component_name} has unknown distribution type: {dist_type}"
            )

        for key in required:
            if key not in component:
                raise KeyError(f"{component_name} is missing required key: {key}")

        if component["min"] >= component["max"]:
            raise ValueError(f"{component_name}: min must be less than max")

        if component["weight"] < 0:
            raise ValueError(f"{component_name}: weight must be non-negative")

        if dist_type == "gaussian" and component["std"] <= 0:
            raise ValueError(f"{component_name}: std must be positive")

    total_weight = sum(component["weight"] for component in components)

    if total_weight <= 0:
        raise ValueError(f"{name}: total component weight must be positive")

def validate_fixed_channel_shift(fixed_channel_shift):
    if fixed_channel_shift is None:
        return
    
    if abs(fixed_channel_shift) > 5:
        raise ValueError("fixed_channel_shift must be between -5 and 5")

def get_axis_index(axis, value):
    matches = np.where(axis == value)[0]

    if len(matches) != 1:
        raise ValueError(f"Expected one match for {value}, got {len(matches)}")

    return matches[0]

def sample_distribution(rng, dist_config, size):
    """Sample from a possibly multi-component distribution."""
    components = dist_config["components"]

    weights = np.array(
        [component["weight"] for component in components],
        dtype=float,
    )
    weights = weights / weights.sum()

    component_indices = rng.choice(
        len(components),
        size=size,
        p=weights,
    )

    samples = np.empty(size, dtype=float)

    for i, component in enumerate(components):
        mask = component_indices == i
        n_samples = mask.sum()

        if n_samples == 0:
            continue

        dist_type = component["type"]

        if dist_type == "uniform":
            samples[mask] = rng.integers(
                low=int(component["min"]),
                high=int(component["max"]) + 1,
                size=n_samples,
            )

        elif dist_type == "gaussian":
            samples[mask] = sample_truncated_gaussian(
                rng=rng,
                mean=component["mean"],
                std=component["std"],
                low=component["min"],
                high=component["max"],
                size=n_samples,
            )

        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    return samples

def sample_truncated_gaussian(rng, mean, std, low, high, size):
    samples = np.empty(size, dtype=float)
    filled = 0

    while filled < size:
        needed = size - filled

        proposal = rng.normal(
            loc=mean,
            scale=std,
            size=needed * 2,
        )

        accepted = proposal[(proposal >= low) & (proposal <= high)]
        accepted = np.rint(accepted)
        accepted = np.clip(accepted, low, high)

        n_accept = min(len(accepted), needed)
        samples[filled:filled + n_accept] = accepted[:n_accept]
        filled += n_accept

    return samples

def apply_channel_shift(profile, shift):
    shifted = np.zeros_like(profile)

    if shift > 0:
        shifted[:-shift] = profile[shift:]
    elif shift < 0:
        shifted[-shift:] = profile[:shift]
    else:
        shifted = profile.copy()

    return shifted

def add_fractional_noise(hist, noise_fraction, rng):
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

def generate_training_data(
    profiles,
    mu_axis,
    sig_axis,
    count,
    config,
    rng,
    fixed_channel_shift=None,
):
    mu_values = sample_distribution(
        rng=rng,
        dist_config=config["mu_distribution"],
        size=count,
    )

    sig_y_values = sample_distribution(
        rng=rng,
        dist_config=config["sig_y_distribution"],
        size=count,
    )

    if fixed_channel_shift is None:
        channel_shifts = sample_distribution(
            rng=rng,
            dist_config=config["channel_shift_distribution"],
            size=count,
        )
    else:
        channel_shifts = np.full(count, fixed_channel_shift, dtype=int)

    noise_fraction = config["noise_fraction"]
    num_channels = profiles.shape[2]

    x_data = np.empty((count, num_channels), dtype=np.float32)
    y_data = np.empty((count, 2), dtype=np.float32)

    for event_idx in range(count):
        mu_true = float(mu_values[event_idx])
        sig_y_true = float(sig_y_values[event_idx])
        channel_shift = int(channel_shifts[event_idx])

        mu_idx = get_axis_index(mu_axis, mu_true)
        sig_y_idx = get_axis_index(sig_axis, sig_y_true)

        full_profile = profiles[mu_idx, sig_y_idx, :]

        shifted_profile = apply_channel_shift(
            full_profile,
            channel_shift,
        )

        noisy_profile = add_fractional_noise(
            shifted_profile,
            noise_fraction,
            rng,
        )

        normalized_profile = normalize_area(noisy_profile)

        x_data[event_idx, :] = normalized_profile
        y_data[event_idx, :] = [mu_true, sig_y_true]

    return x_data, y_data, channel_shifts

def main():
    args = parse_args()

    profiles, mu_axis, sig_axis, y_positions = unpack_npz(args.filename)

    config = load_config(args.config)
    validate_config(config)
    validate_fixed_channel_shift(args.CS)

    seed = config["seed"]
    rng = np.random.default_rng(seed)

    x_data, y_data, channel_shifts = generate_training_data(
        profiles=profiles,
        mu_axis=mu_axis,
        sig_axis=sig_axis,
        count=args.count,
        config=config,
        rng=rng,
        fixed_channel_shift=args.CS,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "fake_histograms.npz"

    np.savez_compressed(
        output_file,
        x_data=x_data,
        y_data=y_data,
        true_mu=y_data[:, 0],
        true_sig_y=y_data[:, 1],
        channel_shift=channel_shifts,
        y_positions=y_positions,
        seed=np.array(seed, dtype=np.int64),
    )
if __name__ == "__main__":
    main()
