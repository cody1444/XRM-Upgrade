#!/usr/bin/env python3

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

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
    HISTOGRAMS_KEY = "histograms"
    MU_KEY = "template_mu"
    SIG_Y_KEY = "template_sig_y"
    #SHIFT_KEY = "channel_shift"

    data = np.load(filename)

    X = data[HISTOGRAMS_KEY].astype("float32")

    y = np.column_stack([
        data[MU_KEY],
        data[SIG_Y_KEY],
        ]).astype("float32")

    return X, y

def train_val_split(X, y, val_fraction=0.2, seed=123):
    rng = np.random.default_rng(seed)

    n_samples = len(X)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_val = int(val_fraction * n_samples)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    return X_train, X_val, y_train, y_val

def build_model(input_dim, output_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim, )),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(output_dim)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )

    return model

def train_one_model(X_train, X_val, y_train, y_val, seed, epochs, batch_size, make_plots=False):
    keras.utils.set_random_seed(seed)

    model = build_model(
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)

    y_pred = model.predict(X_val, verbose=1)

    if make_plots:
        plot_training_history(history, f"training_convergence.png")

        make_diagnostic_plots(
            y_pred,
            y_val,
            "prediction_diagnostics.png",
        )
    diagnostic_results = get_regression_diagnostics(
        y_pred,
        y_val, 
        should_print=False,
    )

    return {
            "val_loss": val_loss,
            "val_mae": val_mae,
            "mu_mae": diagnostic_results["mu_mae"],
            "sig_y_mae": diagnostic_results["sig_y_mae"],
            "mu_rmse": diagnostic_results["mu_rmse"],
            "sig_y_rmse": diagnostic_results["sig_y_rmse"],
    }

def get_regression_diagnostics(y_pred, y_true, should_print=False):
    error = y_pred - y_true

    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error**2, axis=0))

    results = {
        "mu_mae": mae[0], 
        "sig_y_mae": mae[1],
        "mu_rmse": rmse[0],
        "sig_y_rmse": rmse[1],
    }

    if should_print:
        print(f"mu MAE: {results['mu_mae']:.4f}")
        print(f"sig_y MAE: {results['sig_y_mae']:.4f}")
        print(f"mu RMSE: {results['mu_rmse']:.4f}")
        print(f"sig_y RMSE: {results['sig_y_rmse']:.4f}")

    return results

def summarize_runs(results):
    keys = [
        "mu_mae",
        "sig_y_mae",
        "mu_rmse",
        "sig_y_rmse",
    ]

    print("\nMulti-run summary:")

    for key in keys:
        values = np.array([result[key] for result in results])

        print(
            f"{key}: "
            f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
        )

def make_diagnostic_plots(y_pred, y_true, output_filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(y_true[:, 0], y_pred[:, 0], s=5)
    axes[0].set_xlabel("True mu [um]")
    axes[0].set_ylabel("Predicted mu [um]")
    axes[0].set_title("mu prediction")

    axes[1].scatter(y_true[:, 1], y_pred[:, 1], s=5)
    axes[1].set_xlabel("True sig_y [um]")
    axes[1].set_ylabel("Predicted sig_y [um]")
    axes[1].set_title("sig_y prediction")

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def plot_training_history(history, output_filename):
    epochs = np.arange(1, len(history.history["loss"]) + 1)

    plt.figure()
    plt.scatter(epochs, history.history["loss"], label="training loss")
    plt.scatter(epochs, history.history["val_loss"], label="validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def summarize_array(name, array):
    print(f"{name} shape: {array.shape}")
    print(f"{name} dtype: {array.dtype}")

def main():
    args = parse_args()

    X_train, y_train = load_data(args.training_data)
    X_val, y_val = load_data(args.validation_data)

    summarize_array("X_train", X_train)
    summarize_array("y_train", y_train)
    summarize_array("X_val", X_val)
    summarize_array("y_val", y_val)

    seeds = [101, 102, 103, 104, 105]
    results = []

    for i,seed in enumerate(seeds):
        print(f"\nTraining run with seed {seed}")

        result = train_one_model(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            make_plots=(i==0),
        )

        results.append(result)
        
        print(f"mu MAE: {result['mu_mae']:.4f}")
        print(f"sig_y MAE: {result['sig_y_mae']:.4f}")
        print(f"mu RMSE: {result['mu_rmse']:.4f}")
        print(f"sig_y RMSE: {result['sig_y_rmse']:.4f}")

    summarize_runs(results)

if __name__ == "__main__":
    main()

