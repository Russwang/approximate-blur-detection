# Research interpretation and implementation notes

## What the study supports

The paper evaluates approximation MSE across N=64,128,256 and blur detection in selected clean/defocus/motion scenarios. Its blur tables identify K=72 for N=128 and K=144 for N=256 as effective tested widths. Its reported reduction of up to approximately 45% is relative to an untrimmed ANN with K=N.

For one call, `ADD = K*(N-1) + N*(K-1)` and `MUL = 2*K*N`. Summing these counts gives 43.84% and 43.79% reductions for the two table configurations. This computation does not compare hardware runtime with an optimized FFT. The paper itself notes the ANN's quadratic complexity versus FFT's O(N log N).

## Linear approximation

The training target in the notebooks is `np.fft.fft(signals).real`. Both Dense layers are linear and bias-free. Their composition is a linear map with rank at most K. Although the paper discusses universal approximation, this particular implementation is better understood as a factorized linear approximation; it does not use the nonlinear activations of the usual universal approximation theorem.

Taking only the real part removes phase information. Applying a real map independently along rows and columns also differs from taking the real part of a full complex 2D FFT. Frequency scores can therefore shift between the FFT and ANN paths; calibration needs separate scrutiny.

## Historical notebook behavior

The original code is retained as research material, with outputs cleared. Important details for any further experiments:

- Clean/blurred pairing in loop notebooks follows directory traversal order instead of explicit scene identifiers.
- Some loops retain dimensions from before cropping/resizing; extra edge iterations are skipped by full-block checks.
- Some single-run notebooks write the last block's `mean` to their result files rather than the image-level `result`.
- Several paths assume output directories already exist. One detection function accepts a model parameter but refers to the global model variable.
- Radius 60 is unsuitable for small blocks such as N=64; the new CLI restricts its illustrated block sizes to 128 and 256 and validates radius bounds.
- The saved complexity plot uses an N=256 sweep starting at K=140; the paper table uses K=144. The plot's block multipliers are historical settings, not a universal image-wide operation formula.
- The MSE notebook evaluates one newly sampled vector per model, not a fixed shared held-out set. Its saved plot is illustrative; rerunning it will not necessarily reproduce the same curve.

The new inference path adds deterministic inputs, explicit pairing, batched kernel evaluation, finite flat-image scores, shape checks and JSON reporting. Its method-matched threshold is an explicitly documented demo adaptation. It does not silently change the archived notebook code or claim to recreate the paper's tables.

## Further work

Useful next experiments include a fixed held-out signal set, scene-based image splits, calibration on separate training data, comparison with full FFT and simple spatial baselines, memory measurements and hardware latency. These are potential extensions, not results claimed by this repository.
