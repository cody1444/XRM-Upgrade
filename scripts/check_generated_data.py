#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from generate_fake_histograms import (
    apply_channel_shift,
    normalize_area,
    get_axis_index,
    load_config,
)

def parse_args():
    parser = argparse.ArgumentParser(
            prog='check_generated_data',
            description='plots distributions for the generated data',
            epilog='Run with an .npz containing the events'
            )

    parser.add_argument(
            'filename',
            help="Input .npz file containing the events",
            )

    parser.add_argument(
            'beam_profiles',
            help="Input .npz file containing the beam profiles",
            )

    parser.add_argument(
            'config',
            help="Input .config file"
    )

    parser.add_argument(
            'output_dir',
            help='Folder to store plots',
            )

    return parser.parse_args()

def get_min_and_max(axis):
    return  np.min(axis), np.max(axis)

def get_bins(min, max):
    bins = np.arange(min - 0.5, max + 1, 1)
    return bins

def plot_histogram(true_vals, axis, output_dir, type):
    min, max = get_min_and_max(axis)
    bins = get_bins(min, max)
    print(f"{type} bins:")
    print(bins)
    num_events = len(true_vals)

    plt.figure()
    plt.hist(true_vals, bins=bins, label=f"N = {num_events}")
    if type == "channel shift":
        plt.xlabel(f"{type} [n]")
    else:
        plt.xlabel(f"{type} [um]")
    plt.ylabel("Count")
    plt.title(f"Distribution of generated {type} values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"fake_{type}_distribution.png", dpi=300)
    plt.close()

def get_axis_index(axis, value):
    matches = np.where(axis == value)[0]

    if len(matches) != 1:
        raise ValueError(f"Expected one match for {value}, got {len(matches)}")

    return matches[0]

def get_random_event_index(training_data, rng):
    num_events = len(training_data["x_data"])
    return rng.integers(low=0, high=num_events)

def get_random_targets(training_data, random_event_index):
    random_template_mu = training_data['true_mu'][random_event_index]
    random_template_sig_y = training_data['true_sig_y'][random_event_index]
    random_channel_shift = training_data['channel_shift'][random_event_index]
    return random_template_mu, random_template_sig_y, random_channel_shift

def get_shifted_template_for_event(
    event_idx,
    training_data,
    profiles,
    mu_axis,
    sig_axis,
):
    true_mu = training_data["true_mu"][event_idx]
    true_sig_y = training_data["true_sig_y"][event_idx]
    channel_shift = int(training_data["channel_shift"][event_idx])

    mu_idx = get_axis_index(mu_axis, true_mu)
    sig_y_idx = get_axis_index(sig_axis, true_sig_y)

    template = profiles[mu_idx, sig_y_idx, :]
    shifted_template = apply_channel_shift(template, channel_shift)
    normalized_template = normalize_area(shifted_template)

    return normalized_template

def plot_random_hist(training_data, beam_profiles, rng, output_dir):
    random_event_index = get_random_event_index(training_data, rng)

    noisy_points = training_data['x_data'][random_event_index]
    
    random_template_mu, random_template_sig_y, random_channel_shift = get_random_targets(
        training_data, 
        random_event_index,
    )
    
    mu_i = get_axis_index(
        beam_profiles['mu_axis'],
        random_template_mu,
    )
    
    sig_y_i = get_axis_index(
        beam_profiles['sig_axis'],
        random_template_sig_y,
    )

    true_profile = beam_profiles['smeprf'][mu_i, sig_y_i, :]
    true_profile = normalize_area(true_profile)
    
    shifted_profile = get_shifted_template_for_event(
        random_event_index, 
        training_data,
        beam_profiles["smeprf"], 
        beam_profiles["mu_axis"],
        beam_profiles["sig_axis"],
    )

    y_positions = beam_profiles["y_positions"]
    y_positions_mm = y_positions * 1e3

    plt.figure()

    plt.fill_between(
        y_positions_mm[::-1],
        true_profile[::-1],
        alpha=0.2,
        label="true profile",
    )

    plt.fill_between(
        y_positions_mm[::-1],
        shifted_profile[::-1],
        alpha=0.3,
        label="shifted profile"
    )

    plt.scatter(
        y_positions_mm[::-1],
        noisy_points[::-1],
        marker="o",
        s=1,
        label="training data",
    )

    plt.xlabel("y position [mm]")
    plt.ylabel("Amplitude")
    plt.title(
        f"Original profile shadow\n"
        f"event={random_event_index}, "
        f"mu={random_template_mu:.1f} um, "
        f"sig_y={random_template_sig_y:.1f} um, "
        f"shift={random_channel_shift}"
    )
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"event_{random_event_index:05d}_profile_shadow.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_data = np.load(args.filename)
    beam_profiles = np.load(args.beam_profiles)
    config = load_config(args.config)

    template_mu = training_data['true_mu']
    mu_axis = beam_profiles['mu_axis']
    plot_histogram(template_mu, mu_axis, output_dir, "mu")

    template_sig_y = training_data['true_sig_y']
    sig_y_axis = beam_profiles['sig_axis']
    plot_histogram(template_sig_y, sig_y_axis, output_dir, "sig_y")

    template_channel_shift = training_data['channel_shift']
    shift_component = config["channel_shift_distribution"]["components"][0]
    channel_shift_min = int(shift_component['min'])
    channel_shift_max = int(shift_component['max'])
    channel_shift_axis = np.arange(channel_shift_min, channel_shift_max + 1, 1)
    plot_histogram(template_channel_shift, channel_shift_axis, output_dir, "channel shift")

    rng = np.random.default_rng()
    plot_random_hist(training_data, beam_profiles, rng, output_dir)


if __name__ == "__main__":
    main()
