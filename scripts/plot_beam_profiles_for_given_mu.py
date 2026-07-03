import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
        prog='beam_profiles',
        description='plots beam profiles',
        epilog='run with .npz file'
        )

parser.add_argument('filename')

parser.add_argument('mu_index', type=int, default=0)

args = parser.parse_args()

data = np.load(args.filename)

# data structure:
#   smeprf: shape=(151, 91, 1024), dtype=float32
#   mu_axis: shape=(151,), dtype=float64
#   sig_axis: shape=(91,), dtype=float64
#   y_positions: shape=(1024,), dtype=float64

smeprf = data['smeprf']
mu_axis = data['mu_axis']
sig_axis = data['sig_axis']
y_positions = data['y_positions']

highlight_indices = [
    i for i, sig in enumerate(sig_axis)
    if i == 0 or i == len(sig_axis) - 1 or sig % 10 == 0
]

colors = plt.cm.tab20(np.linspace(0, 1, len(highlight_indices)))
color_map = dict(zip(highlight_indices, colors))

for sig_index in range(len(sig_axis)):
    profile = smeprf[args.mu_index, sig_index, :]

    if sig_index in highlight_indices:
        if sig_index == 0:
            label = rf"$\sigma_y = {sig_axis[sig_index]:.3g}\,\mu\mathrm{{m}}$"
            linewidth = 2.0
        elif sig_index == len(sig_axis) - 1:
            label = rf"$\sigma_y = {sig_axis[sig_index]:.3g}\,\mu\mathrm{{m}}$"
            linewidth = 2.0
        else:
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
plt.title(rf"Beam profiles for $\mu = {mu_axis[args.mu_index]:.3g}\,\mu\mathrm{{m}}$")
plt.xlabel("profile index")
plt.ylabel("profile amplitude")
plt.legend(loc="upper right")
plt.show()
