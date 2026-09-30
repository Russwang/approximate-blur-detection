"""Lightweight inference for the project's linear, bias-free Fourier networks."""
from pathlib import Path
import json

import h5py
import numpy as np


class FourierNetwork:
    """Read two Dense kernels without requiring TensorFlow or optimizer state."""

    def __init__(self, path):
        with h5py.File(path, "r") as handle:
            layers = json.loads(handle.attrs["model_config"])["config"]["layers"]
            dense = [layer for layer in layers if layer["class_name"] == "Dense"]
            if len(dense) != 2 or any(
                layer["config"]["activation"] != "linear"
                or layer["config"]["use_bias"] for layer in dense
            ):
                raise ValueError("Expected exactly two linear Dense layers without bias")
            kernels = []
            for layer in dense:
                group = handle["model_weights"][layer["config"]["name"]]
                values = []
                group.visititems(lambda name, value: values.append(value[...])
                                 if isinstance(value, h5py.Dataset)
                                 and name.rsplit("/", 1)[-1] in ("kernel", "kernel:0")
                                 else None)
                if len(values) != 1:
                    raise ValueError("Expected one kernel per Dense layer")
                kernels.append(values[0])
        self.first, self.second = kernels
        self.n, self.k = self.first.shape
        if self.second.shape != (self.k, self.n):
            raise ValueError("Incompatible network kernel shapes")

    def transform(self, signals):
        signals = np.asarray(signals, dtype=np.float32)
        if signals.ndim == 0 or signals.shape[-1] != self.n:
            raise ValueError(f"Signals must have final dimension {self.n}")
        return (signals @ self.first) @ self.second

    def transform2d(self, blocks):
        if blocks.shape[-2:] != (self.n, self.n):
            raise ValueError(f"Blocks must end in shape ({self.n}, {self.n})")
        rows = self.transform(blocks)
        return self.transform(rows.swapaxes(-1, -2)).swapaxes(-1, -2)


def operation_counts(n, k):
    """Paper Eq. (6), per 1D ANN call; excludes masking/inverse FFT/I/O."""
    if not (isinstance(n, int) and isinstance(k, int) and 0 < k <= n):
        raise ValueError("Expected integer widths 0 < K <= N")
    return {"additions": k * (n - 1) + n * (k - 1), "multiplications": 2 * k * n}


def block_scores(image, n=128, network=None, mask_radius=60):
    """Mean log high-pass reconstruction magnitude, following the notebooks.

    Discard incomplete edge blocks. A positive floor makes flat blocks finite.
    The ANN applies two real transforms; it is not a full complex 2D FFT.
    """
    gray = np.asarray(image, dtype=np.float32)
    if gray.ndim != 2 or not np.isfinite(gray).all():
        raise ValueError("Expected a finite 2D grayscale image")
    if not (isinstance(n, int) and n > 0 and 0 < mask_radius < n // 2):
        raise ValueError("Mask radius must be positive and smaller than half the block width")
    if network is not None and network.n != n:
        raise ValueError("Network width must match block width")
    h, w = gray.shape
    h, w = h // n * n, w // n * n
    if not h or not w:
        raise ValueError(f"Image must contain at least one complete {n} x {n} block")
    blocks = gray[:h, :w].reshape(h // n, n, w // n, n).transpose(0, 2, 1, 3)
    spectrum = (np.fft.fft2(blocks, axes=(-2, -1)) if network is None
                else network.transform2d(blocks))
    shifted = np.fft.fftshift(spectrum, axes=(-2, -1))
    c = n // 2
    shifted[..., c - mask_radius:c + mask_radius, c - mask_radius:c + mask_radius] = 0
    recon = np.fft.ifft2(np.fft.ifftshift(shifted, axes=(-2, -1)), axes=(-2, -1))
    return (20 * np.log(np.maximum(np.abs(recon), 1e-12))).mean(axis=(-2, -1))


def model_path(n, k):
    return Path(__file__).resolve().parent / "model" / f"1D_FFT_{n}_reduce_{k}.h5"
