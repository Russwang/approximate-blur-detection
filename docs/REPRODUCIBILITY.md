# Reproducibility

## Lightweight inference

Use the quick start in the README. The inference module reads the two linear Dense kernels in each HDF5 file and computes `(X @ W1) @ W2`. It preserves the two-layer architecture rather than collapsing the weights into one matrix. Loading does not deserialize a Keras optimizer or execute a model definition.

`scripts/demo.py` uses a fixed random seed (42), a 512 x 768 synthetic texture, Gaussian blur with radius 3, and a horizontal 21-sample motion average with periodic boundaries. Pixel intensities remain on the 0–255 scale. It applies no notebook-style crop or resize, discards incomplete edge blocks, and uses an epsilon floor of `1e-12` before the natural logarithm. The block score is `mean(20 * ln(abs(high-pass reconstruction)))`, not a base-10 dB measurement.

Both transform paths mask the central spectrum with radius 60 and use a complex inverse FFT to reconstruct a high-pass image. The ANN path applies the learned real-valued map along rows and columns. It is not equivalent to a complete complex 2D FFT.

### Thresholds

- `--calibration matched` (default): FFT and ANN each use their own score on the supplied clean image. This is a demo adaptation that makes the score scales explicit.
- `--calibration fft`: both use the clean reference's FFT score, matching the threshold rule in the paper and `BlurDetection_XFFT_Loop.ipynb`. This produces a false positive on the synthetic clean image with both illustrated trimmed configurations.

Every JSON report includes `ann_blurry_with_fft_threshold`, regardless of the chosen calibration. A lower image-level score than the threshold is classified as blurred. Calibration on the clean reference is not an independent validation benchmark; the synthetic example should not be used to claim dataset accuracy.

Reports include per-1D-call arithmetic counts. For an image with B complete N x N blocks, applying two layers separately along every row and column requires `2*N*B` 1D calls. Masking, inverse FFT, memory traffic and I/O are excluded. No runtime speedup is claimed.

## Research environment and training

```bash
python -m pip install -r requirements-research.txt
python scripts/train.py --n 128 --k 72 --samples 10000 --epochs 1000
python -m jupyterlab
```

The new training entry point uses 10,000 Gaussian random signals, FFT real-part targets, a full-dataset batch size, Adam/MSE, and a 20% validation split. It writes to `outputs/trained/` to avoid overwriting supplied models. A seed controls random generation; TensorFlow results may still vary across hardware and kernels.

The paper describes 1,000 epochs, while the supplied training notebooks use 2,000 epochs. The historical weights have no accompanying training manifest, so their exact seed, epoch count and environment cannot be certified. `docs/requirements-original.txt` records the supplied environment snapshot. The lighter dependency files are intended for a fresh installation, not an exact recreation of every original package.

Before running notebooks, launch Jupyter from the repository root, obtain the dataset, create `outtxt/` and `outimg/`, and check N/K sweep settings and filenames. In particular, the N=256 detection loop requests K=48, which is not among the supplied N=256 model files. Adjust the sweep or train that model first. Models are not regenerated automatically.

## Verification performed for this edition

The supported validation commands are:

```bash
python -m unittest discover -s tests -v
python scripts/demo.py --n 128 --k 72
python scripts/demo.py --n 256 --k 144 --output outputs/demo-256
```

The tests check an untrimmed trained network against FFT real parts, batched versus sequential transforms, flat-image handling, invalid inputs, and the paper's operation-count configuration. The TensorFlow training entry point and the original notebooks were not executed end-to-end during portfolio preparation. Original dataset results were not remeasured.
