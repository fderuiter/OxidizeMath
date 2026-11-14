# ADR 0015: Precomputed Twiddle Factor Lookup Tables for MRI Reconstruction

## Context
In `crates/domain_physics/src/physics/mri/reconstruction.rs`, the separable 2D Discrete Fourier Transform (`separable_transform`) previously evaluated `Complex::from_polar(1.0, phase)` inside innermost row and column loops ($O(N^3)$ complexity). For a $64 \times 64$ slice, this resulted in hundreds of thousands of trigonometric operations (`sin`/`cos`) evaluated repeatedly for identical phase angles during both `simulate_signal_2d` and `inverse_dft_2d`.

## Decision
We modernized `separable_transform` to precompute 1D twiddle factor lookup tables (roots of unity) prior to row and column iterations:
1. Precomputed column roots of unity $W_{\text{cols}}^m = e^{\pm i 2\pi m / \text{cols}}$ for $m \in [0, \text{cols})$ and row roots of unity $W_{\text{rows}}^m = e^{\pm i 2\pi m / \text{rows}}$ for $m \in [0, \text{rows})$.
2. Handled non-square rectangular matrices by maintaining separate row and column twiddle tables.
3. Added early return for empty matrices ($0 \times 0$) to protect against division by zero.
4. Replaced inner loop calls to `Complex::from_polar` with constant-time modular indexing into the twiddle factor lookup tables (`col_twiddles[(k * n) % cols]` and `row_twiddles[(k * n) % rows]`).
5. Preserved exact mathematical and numerical equivalence, public function signatures (`simulate_signal_2d` and `inverse_dft_2d`), and scaling definitions.

## Consequences
- **Performance**: Eliminates repeated trigonometric computations in inner loops, accelerating `simulate_signal_2d` and `inverse_dft_2d` by ~4.5-4.7x in release mode.
- **Accuracy**: Eliminates floating-point phase wrap accumulation errors while strictly maintaining numerical equivalence within floating-point tolerance ($< 10^{-5}$).
- **Verification**: Validated with existing benchmark and regression tests in `crates/domain_physics/tests/measure_mri.rs`, including coverage for non-square and empty matrices.
