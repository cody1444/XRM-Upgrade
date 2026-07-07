from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
            prog='channel_selection',
            description='provides channel indices from a beam profile library',
            epilog='Run with an .npz containing the beam profiles'
            )

    parser.add_argument(
            'filename',
            help="Input .npz file containing the beam profiles",
            )

    parser.add_argument(
            '--test_data',
            action='store_true',
            help="Plot score for dummy, sawtooth data",
            )

    parser.add_argument(
            'output_dir',
            help='Folder to store plots',
            )

    return parser.parse_args()

def validate_channels_and_positions(n_channels, n_positions):
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")

    if n_positions <= 0:
        raise ValueError("n_positions must be positive")

    if n_channels > n_positions:
        raise ValueError("n_channels cannot exceed n_positions")

def validate_width(width):
    if width <= 0:
        raise ValueError("width must be positive")
    
def uniform_channels(n_channels, n_positions=1024):
    validate_channels_and_positions(n_channels, n_positions)

    return np.linspace(0, n_positions - 1, n_channels).round().astype(int).tolist()

def window_channels(center, width, n_channels=16, n_positions=1024):
    validate_channels_and_positions(n_channels, n_positions)
    validate_width(width)

    start = center - width / 2
    stop = center + width / 2

    indices = np.linspace(start, stop, n_channels)
    indices = np.round(indices).astype(int)
    indices = np.clip(indices, 0, n_positions - 1)
    indices = np.unique(indices)

    if len(indices) != n_channels:
        raise ValueError(
            f"Requested {n_channels} channels, but only got "
            f"{len(indices)} unique indices. Increase width or adjust center."
        )

    return indices.tolist()

def get_sig_y_sdev(smeprf):
    profiles = smeprf['smeprf']

    y_positions = smeprf['y_positions']
    mu_axis = smeprf['mu_axis']
    sig_y_stdev = np.empty((len(mu_axis), len(y_positions)))

    for mu_idx in range(len(mu_axis)):
        for y_idx in range(len(y_positions)):
            amplitudes_over_sig_y = profiles[mu_idx,:,y_idx]
            sig_y_stdev[mu_idx, y_idx] = np.std(amplitudes_over_sig_y)

    return sig_y_stdev

def rank_channels(sig_y_stdev):
    num_of_channels = sig_y_stdev.shape[1]
    channel_scores = np.empty(num_of_channels)

    for y_idx in range(num_of_channels):
        channel_scores[y_idx] = np.mean(sig_y_stdev[:,y_idx])

    ranked_indices = np.argsort(channel_scores)[::-1]

    return channel_scores, ranked_indices

def plot_channel_scores(y_positions, channel_scores, ranked_indices=None, top_n=16):
    plt.figure(figsize=(10, 5))

    plt.plot(
        y_positions,
        channel_scores,
        linewidth=1.5,
        label=r"Channel score"
    )

    if ranked_indices is not None:
        top_indices = ranked_indices[:top_n]

        plt.scatter(
            y_positions[top_indices],
            channel_scores[top_indices],
            s=30,
            label=f"Top {top_n} channels",
            zorder=3,
        )

    plt.xlabel("y position")
    plt.ylabel(r"Mean std. dev. over $\sigma_y$")
    plt.title(r"Channel score versus y position")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_heatmap(sig_y_stdev, bins):
    pass

def make_test_smeprf_for_channel_score_plot(smeprf):
    profiles = smeprf["smeprf"]

    num_mu, num_sigma, num_channels = profiles.shape

    raw = np.arange(num_sigma)
    template_col = (raw - raw.mean()) / raw.std()

    period = num_channels // 2
    scalars = 1e-6 * (np.arange(num_channels) % period)

    one_mu_slice = template_col[:, None] * scalars[None, :]
    test_profiles = np.repeat(one_mu_slice[None, :, :], repeats=num_mu, axis=0)

    test_smeprf = dict(smeprf)
    test_smeprf["smeprf"] = test_profiles

    return test_smeprf

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    beam_profiles = np.load(args.filename)
    y_positions = beam_profiles['y_positions']

    if args.test_data is True:
        test_data = make_test_smeprf_for_channel_score_plot(beam_profiles)
        sig_y_sdev = get_sig_y_sdev(test_data)

    else:
        sig_y_sdev = get_sig_y_sdev(beam_profiles)
    channel_scores, ranked_indices = rank_channels(sig_y_sdev)

    plot_channel_scores(y_positions, channel_scores, ranked_indices)

if __name__ == "__main__":
    main()
