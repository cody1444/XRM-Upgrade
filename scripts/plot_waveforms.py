#!/usr/bin/env python3

from pathlib import Path
import argparse

import uproot
import numpy as np
import matplotlib.pyplot as plt


TRACE_LENGTH = 64


def plot_channels_stitched(
    root_file,
    outdir,
    lin_ch=None,
    max_channels=None,
    min_linch=0,
    max_linch=127,
):
    root_file = Path(root_file)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with uproot.open(root_file) as f:
        tree = f["waveforms"]

        branches = tree.arrays(
            [
                "event_number",
                "carrier",
                "asic",
                "channel",
                "lin_ch",
                "logical_window",
                "physical_window",
                "starting_sample",
                "waveform",
            ],
            library="np",
        )

    event_number = branches["event_number"]
    carrier = branches["carrier"]
    asic = branches["asic"]
    channel = branches["channel"]
    lin_ch_arr = branches["lin_ch"]
    logical_window = branches["logical_window"]
    physical_window = branches["physical_window"]
    starting_sample = branches["starting_sample"]
    waveform = branches["waveform"]

    # Remove clearly invalid channels, like lin_ch=48080.
    valid_mask = (
        (lin_ch_arr >= min_linch)
        & (lin_ch_arr <= max_linch)
        & ((starting_sample == 0) | (starting_sample == 32))
    )

    if lin_ch is not None:
        valid_mask &= lin_ch_arr == lin_ch

    event_number = event_number[valid_mask]
    carrier = carrier[valid_mask]
    asic = asic[valid_mask]
    channel = channel[valid_mask]
    lin_ch_arr = lin_ch_arr[valid_mask]
    logical_window = logical_window[valid_mask]
    physical_window = physical_window[valid_mask]
    starting_sample = starting_sample[valid_mask]
    waveform = waveform[valid_mask]

    unique_channels = sorted(set(int(ch) for ch in lin_ch_arr))
    print(f"Unique valid lin_ch values: {unique_channels}")
    print(f"Number of valid channels to plot: {len(unique_channels)}")

    # Group rows by lin_ch.
    groups = {}
    for i in range(len(event_number)):
        ch = int(lin_ch_arr[i])
        groups.setdefault(ch, []).append(i)

    n_saved = 0

    for linch, indices in sorted(groups.items()):
        # Get hardware mapping info for this lin_ch.
        # Usually this should be constant for a given lin_ch.
        carrier_vals = sorted(set(int(carrier[i]) for i in indices))
        asic_vals = sorted(set(int(asic[i]) for i in indices))
        channel_vals = sorted(set(int(channel[i]) for i in indices))

        carrier_text = ",".join(str(x) for x in carrier_vals)
        asic_text = ",".join(str(x) for x in asic_vals)
        channel_text = ",".join(str(x) for x in channel_vals)

        # Each trace is one event/window for this lin_ch.
        trace_keys = sorted(
            {
                (
                    int(event_number[i]),
                    int(logical_window[i]),
                    int(physical_window[i]),
                )
                for i in indices
            }
        )

        trace_index_map = {key: j for j, key in enumerate(trace_keys)}

        x_all = []
        y_all = []

        indices = sorted(
            indices,
            key=lambda i: (
                int(event_number[i]),
                int(logical_window[i]),
                int(physical_window[i]),
                int(starting_sample[i]),
            ),
        )

        for i in indices:
            ev = int(event_number[i])
            lw = int(logical_window[i])
            pw = int(physical_window[i])
            start = int(starting_sample[i])

            trace_key = (ev, lw, pw)
            trace_index = trace_index_map[trace_key]

            samples = np.asarray(waveform[i])
            local_sample_index = start + np.arange(len(samples))

            x = TRACE_LENGTH * trace_index + local_sample_index
            y = samples

            x_all.append(x)
            y_all.append(y)

        x_all = np.concatenate(x_all)
        y_all = np.concatenate(y_all)

        order = np.argsort(x_all)
        x_all = x_all[order]
        y_all = y_all[order]

        plt.figure(figsize=(14, 4))
        plt.plot(x_all, y_all, linewidth=0.8)
        plt.xlabel("Stitched sample index")
        plt.ylabel("ADC counts")
        plt.title(
            f"lin_ch={linch}, carrier={carrier_text}, asic={asic_text}, "
            f"channel={channel_text}, all windows/events stitched"
        )
        plt.grid(True)

        outfile = outdir / f"linch{linch:03d}_stitched.png"
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()

        n_saved += 1

        if max_channels is not None and n_saved >= max_channels:
            break

    print(f"Saved {n_saved} plots to {outdir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot stitched time-ordered waveforms from a ROOT file."
    )

    parser.add_argument("root_file", help="Input ROOT file")
    parser.add_argument(
        "-o",
        "--outdir",
        default="stitched_waveform_plots",
        help="Output directory",
    )
    parser.add_argument(
        "--lin-ch",
        type=int,
        default=None,
        help="Only plot one lin_ch, e.g. --lin-ch 60",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=None,
        help="Maximum number of channels to plot, useful for testing",
    )
    parser.add_argument(
        "--min-linch",
        type=int,
        default=0,
        help="Minimum allowed lin_ch",
    )
    parser.add_argument(
        "--max-linch",
        type=int,
        default=127,
        help="Maximum allowed lin_ch",
    )

    args = parser.parse_args()

    plot_channels_stitched(
        root_file=args.root_file,
        outdir=args.outdir,
        lin_ch=args.lin_ch,
        max_channels=args.max_channels,
        min_linch=args.min_linch,
        max_linch=args.max_linch,
    )


if __name__ == "__main__":
    main()
