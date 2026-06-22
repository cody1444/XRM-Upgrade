import numpy as np

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
