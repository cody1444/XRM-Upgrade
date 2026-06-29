#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
            help="Input .config file containing event statistics"
            )

    parser.add_argument(
            'output_dir',
            help='Folder to store plots',
            )

    return parser.parse_args()

def load_config(filename):
    namespace = {}
    with open(filename, "r") as f:
        exec(f.read(), namespace)
    return namespace

def get_min_and_max(axis):
    return  np.min(axis), np.max(axis)

def get_bins(min, max):
    bins = np.arange(min - 0.5, max + 1.5, 1)
    return bins

def plot_histogram(template, axis, output_dir, type):
    min, max = get_min_and_max(axis)
    bins = get_bins(min, max)
    num_events = len(template)

    plt.figure()
    plt.hist(template, bins=bins, label=f"N = {num_events}")
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
    num_events = len(training_data["histograms"])
    return rng.integers(low=0, high=num_events)

def get_random_targets(training_data, random_event_index):
    random_template_mu = training_data['template_mu'][random_event_index]
    random_template_sig_y = training_data['template_sig_y'][random_event_index]
    random_channel_shift = training_data['channel_shift'][random_event_index]
    return random_template_mu, random_template_sig_y, random_channel_shift

def shift_profile(profile, shift):
    shifted = np.zeros_like(profile)

    if shift > 0:
        shifted[shift:] = profile[:-shift]
    elif shift < 0:
        shifted[:shift] = profile[-shift:]
    else:
        shifted = profile.copy()

    return shifted

def plot_random_hist(training_data, beam_profiles, rng, output_dir, config):
    random_event_index = get_random_event_index(training_data, rng)
    channel_indices = np.asarray(config['channel_indices'], dtype = int)
    noisy_points = training_data['raw_histograms'][random_event_index]
    
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
    shifted_profile = shift_profile(true_profile, random_channel_shift)
    y_positions = beam_profiles['y_positions']

    plt.figure()

    plt.fill_between(
        y_positions,
        true_profile,
        alpha=0.2,
        label="original noiseless profile"
    )

    plt.plot(
        y_positions,
        shifted_profile, 
        label=f"shifted noiseless profile (shift = {random_channel_shift})", 
        linewidth=0.5,
        markersize=2
    )

    plt.scatter(
        y_positions[channel_indices],
        noisy_points,
        marker="x",
        s=25,
        label="shifted noisy data",
    )

    plt.xlabel("y position")
    plt.ylabel("Amplitude")
    plt.title(
        f"Original profile shadow\n"
        f"event={random_event_index}, "
        f"mu={random_template_mu:.1f} um, "
        f"sig_y={random_template_sig_y:.1f} um"
    )
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )
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

    template_mu = training_data['template_mu']
    mu_axis = beam_profiles['mu_axis']
    plot_histogram(template_mu, mu_axis, output_dir, "mu")

    template_sig_y = training_data['template_sig_y']
    sig_y_axis = beam_profiles['sig_axis']
    plot_histogram(template_sig_y, sig_y_axis, output_dir, "sig_y")

    config = load_config(args.config)
    template_channel_shift = training_data['channel_shift']
    channel_shift_min = float(config['channel_shift_min'])
    channel_shift_max = float(config['channel_shift_max'])
    channel_shift_axis = np.arange(channel_shift_min, channel_shift_max + 1, 1)
    plot_histogram(template_channel_shift, channel_shift_axis, output_dir, "channel shift")

    rng = np.random.default_rng()
    plot_random_hist(training_data, beam_profiles, rng, output_dir, config)


if __name__ == "__main__":
    main()
