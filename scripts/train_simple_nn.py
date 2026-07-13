#!/usr/bin/env python3

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

def parse_args():
    parser = argparse.ArgumentParser(
            prog="train_simple_nn",
            description="Train a simple Keras regression model on fake XRM histograms.",
            )

    parser.add_argument(
            "training_data",
            help="Input .npz file containing fake training data."
            )

    parser.add_argument(
            "validation_data",
            help="Input .npz file containing fake validation data."
            )

    parser.add_argument(
            "channel_selections",
            help="Input .npz file containing the channel selections"
    )

    parser.add_argument(
            "--epochs",
            type=int,
            default=50,
            help="Number of training epochs."
            )

    parser.add_argument(
            "--batch-size",
            type=int,
            default=128,
            help="Training batch size."
            )

    return parser.parse_args()

def load_data(filename):
    data = np.load(filename)

    X = data["x_data"].astype("float32")

    y = np.column_stack([
        data["true_mu"],
        data["true_sig_y"],
        ]).astype("float32")

    return X, y

def load_channel_selections(filename):
    data = np.load(filename)
    channel_selections = data["channel_selections"]

    return channel_selections

def build_model(input_dim, output_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim, )),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(output_dim)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )

    return model

def mask_x_data_by_channel_selection(x_data, channel_indices):
    channel_indices = np.asarray(channel_indices, dtype=int)

    if x_data.ndim != 2:
        raise ValueError(f"x_data must be 2D, got shape {x_data.shape}")

    if channel_indices.ndim != 1:
        raise ValueError(
            f"channel_indices must be 1D, got shape {channel_indices.shape}"
        )

    if np.any(channel_indices < 0):
        raise ValueError("channel_indices contains negative indices")

    if np.any(channel_indices >= x_data.shape[1]):
        raise ValueError(
            "channel_indices contains values outside the x_data channel range. "
            f"x_data has {x_data.shape[1]} channels, "
            f"max channel index is {channel_indices.max()}."
        )

    return x_data[:, channel_indices]

def train_one_model(X_train, X_val, y_train, y_val, seed, epochs, batch_size):
    keras.utils.set_random_seed(seed)

    model = build_model(
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0,
    )

    y_pred = model.predict(X_val, verbose=0)

    sig_y_mae = np.mean(
        np.abs(y_pred[:, 1] - y_val[:, 1])
    )

    return {
        "sig_y_mae": sig_y_mae,
    }

def summarize_runs(results):
    value = results[0]["sig_y_mae"]

    print(f"Roster sig_y MAE: {value:.4f}")

def main():
    args = parse_args()

    X_train, y_train = load_data(args.training_data)
    X_val, y_val = load_data(args.validation_data)

    channel_selection = load_channel_selections(args.channel_selections)

    #seeds = [101, 102, 103, 104, 105]
    seeds = [101]

    for roster_idx, channel_roster in enumerate(channel_selection):
        results = []

        X_train_masked = mask_x_data_by_channel_selection(X_train, channel_roster)
        X_val_masked = mask_x_data_by_channel_selection(X_val, channel_roster)

        print(f"\nChannel roster {roster_idx}")
        print(f"channels: {channel_roster}")

        for i,seed in enumerate(seeds):

            print(f"\nTraining run with seed {seed}")
            result = train_one_model(
                X_train=X_train_masked,
                X_val=X_val_masked,
                y_train=y_train,
                y_val=y_val,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

            results.append(result)
            
            print(f"sig_y MAE: {result['sig_y_mae']:.4f}")

        summarize_runs(results)

if __name__ == "__main__":
    main()

