# ADR 0029: Separable Transposed FFT for 2D MRI Reconstruction

## Status
Accepted (supersedes the cubic row-column loops in [ADR 0015](0015-precompute-mri-twiddle-factors.md))

## Context
In `crates/domain_physics/src/physics/mri/reconstruction.rs`, [ADR 0015](0015-precompute-mri-twiddle-factors.md) precomputed 1D twiddle factor lookup tables for `separable_transform` while retaining triple-nested row and column Discrete Fourier Transform (DFT) loops with $O(R \cdot C^2 + C \cdot R^2)$ time complexity ($O(N^3)$ for $N \times N$ slices). In addition to cubic arithmetic growth on clinical matrix sizes ($256 \times 256$ and $512 \times 512$), the row pass indexed across `nalgebra::DMatrix<Complex<f64>>` in non-contiguous column-major strides (`matrix[(r, n)]`), incurring cache misses and pushing $512 \times 512$ round-trip simulation and reconstruction latency past 11 seconds in release builds.

## Decision
We replaced the nested row and column summation loops in `separable_transform` with transposed unit-stride 1D `rustfft` passes:
1. **Early Empty-Dimension Guard**: Return `DMatrix::zeros(rows, cols)` immediately whenever `rows == 0 || cols == 0` so `FftPlanner::plan_fft` is never invoked with zero length, including degenerate `0 x C` and `R x 0` inputs.
2. **Axis-Specific Unnormalized 1D FFT Plans**: Build `row_fft = planner.plan_fft(cols, direction)` and `col_fft = planner.plan_fft(rows, direction)` using `FftDirection::Forward` (`-2\pi` phase) for `simulate_signal_2d` and `FftDirection::Inverse` (`+2\pi` phase) for `inverse_dft_2d`, preserving exact unnormalized scaling (`rows * cols` round-trip factor).
3. **Shared Scratch Allocation**: Allocate a single reusable scratch buffer sized to `row_fft.get_inplace_scratch_len().max(col_fft.get_inplace_scratch_len())` so prime and mixed-radix dimensions (such as Bluestein convolution plans) execute without per-line allocations.
4. **Transposed Unit-Stride Batched Execution**: Transpose `matrix` ($R \times C$) into a $C \times R$ column-major matrix so each input row occupies a contiguous column slice, execute `row_fft.process_with_scratch` across the full slice, transpose back to $R \times C$, and execute `col_fft.process_with_scratch` across all columns in place.

## Consequences
- **Positive**: Reduces 2D MRI k-space simulation and inverse reconstruction complexity from $O(R \cdot C^2 + C \cdot R^2)$ to $O(R \cdot C \log(R \cdot C))$ with unit-stride memory access on both passes.
- **Positive**: Preserves exact unnormalized phase definitions, `#[verified_engine::verified]` statement bounds, and the public signatures of `simulate_signal_2d` and `inverse_dft_2d` in `crates/unified_verification/api_baseline.json`.
- **Verification**: Validated in `crates/domain_physics/tests/measure_mri.rs` against a direct 2D DFT reference across prime, rectangular, single-line, and empty matrices, alongside a $256 \times 256$ round-trip scaling regression guard.
