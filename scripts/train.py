"""Train the original two-layer linear architecture on real FFT targets."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=72)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/trained"))
    args = parser.parse_args()
    if not (0 < args.k <= args.n and args.samples >= 10 and args.epochs > 0):
        parser.error("Require 0 < K <= N, samples >= 10, and epochs > 0")
    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    signals = np.random.default_rng(args.seed).standard_normal((args.samples, args.n)).astype("float32")
    targets = np.fft.fft(signals, axis=-1).real.astype("float32")
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(args.n,)),
        tf.keras.layers.Dense(args.k, use_bias=False, activation="linear"),
        tf.keras.layers.Dense(args.n, use_bias=False, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(signals, targets, epochs=args.epochs, batch_size=args.samples,
              validation_split=0.2, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / f"1D_FFT_{args.n}_reduce_{args.k}.h5")


if __name__ == "__main__":
    main()
