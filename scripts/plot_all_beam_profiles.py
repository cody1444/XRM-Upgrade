import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        prog="beam_profiles",
        description="plots beam profiles",
        epilog="run with .npz file"
    )

    parser.add_argument(
        "filename",
        help="insert beam profile library",
    )

    parser.add_argument(
        "-i", "--mu_index",
        type=int,
        default=None,
        help="insert desired mu index for the beam profile plot"
    )

    parser.add_argument(
        "-o", "--output-dir",
        default="beam_profile_plots",
        help="folder where plots will be saved"
    )

    return parser.parse_args()

def get_color_map(sig_axis):
    highlight_indices = [
        i for i, sig in enumerate(sig_axis)
        if i == 0 or i == len(sig_axis) - 1 or np.isclose(sig % 10, 0)
    ]

    cmap = plt.colormaps["tab20"]
    colors = cmap(np.linspace(0, 1, len(highlight_indices)))
    color_map = dict(zip(highlight_indices, colors))

    return color_map

def plot_all_beam_profiles(filename, output_dir):
    data = np.load(filename)
    mu_axis = data["mu_axis"]
    sig_axis = data["sig_axis"]

    color_map = get_color_map(sig_axis)

    for mu_index in range(len(mu_axis)):
        plt.figure(figsize=(10, 6))
        plot_beam_profile_for_single_mu(filename, mu_index, color_map)

        output_filename = output_dir / f"beam_profiles_mu_index_{mu_index:03d}.png"
        plt.savefig(output_filename, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved {output_filename}")

def plot_beam_profile_for_single_mu(filename, mu_idx, color_map):
    data = np.load(filename)

    profiles = data["smeprf"]
    mu_axis = data["mu_axis"]
    sig_axis = data["sig_axis"]
    y_positions = 1e3 * data["y_positions"]

    highlight_indices = set(color_map.keys())

    for sig_index in range(len(sig_axis)):
        profile = profiles[mu_idx, sig_index, :]

        if sig_index in highlight_indices:
            label = rf"$\sigma_y = {sig_axis[sig_index]:.3g}\,\mu\mathrm{{m}}$"
            linewidth = 1.3
            plt.plot(
                y_positions,
                profile,
                linewidth=linewidth,
                color=color_map[sig_index],
                label=label,
            )

        else:
            plt.plot(
                y_positions,
                profile,
                alpha=0.50,
                color="gray",
                linewidth=0.8,
            )

    plt.title(rf"Beam profiles for $\mu = {mu_axis[mu_idx]:.3g}\,\mu\mathrm{{m}}$")
    plt.xlabel("position [mm]")
    plt.ylabel("profile amplitude [arb. units]")
    plt.legend(loc="upper right")

def plot_beam_profile_library_mosaic(
    filename,
    output_path="beam_profile_library_mosaic.png",
    ncols=14,
    figsize=(18, 14),
    dpi=300,
):
    """
    Make a tiled image showing one subplot per mu value.
    Each subplot overlays all sig_y profiles for that mu.

    Assumes an npz file with:
        smeprf: shape (num_mu, num_sig_y, num_channels)
        mu_axis
        sig_axis
        y_positions
    """

    data = np.load(filename)

    profiles = data["smeprf"]
    mu_axis = data["mu_axis"]
    y_positions = data["y_positions"]

    num_mu, num_sig_y, _ = profiles.shape

    nrows = int(np.ceil(num_mu / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()

    for mu_idx in range(num_mu):
        ax = axes[mu_idx]

        # Plot every sig_y profile for this mu
        for sig_idx in range(num_sig_y):
            ax.plot(
                y_positions,
                profiles[mu_idx, sig_idx, :],
                linewidth=0.25,
                alpha=0.35,
            )

        # Tiny title just to indicate each tile is a different mu.
        # You can remove this if it clutters the slide.
        ax.set_title(f"$\\mu$={mu_axis[mu_idx]:.0f}", fontsize=5)

        # Remove axis clutter
        ax.set_xticks([])
        ax.set_yticks([])

    # Turn off unused axes
    for ax in axes[num_mu:]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = Path(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved mosaic plot to {output_path}")

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_beam_profile_library_mosaic(
        args.filename,
        output_path=output_dir / "beam_profile_library_mosaic.png",
    )


if __name__ == "__main__":
    main()
