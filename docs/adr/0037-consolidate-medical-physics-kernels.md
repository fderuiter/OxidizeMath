# ADR 0037: Consolidate Duplicated Medical Physics Kernels

## Status

Accepted (extends internal kernel delegation established in [ADR 0016](0016-shared-elemental-activation-kernels.md))

## Context

Within [`crates/domain_physics/src/physics/medical/`](../../crates/domain_physics/src/physics/medical/mod.rs), four core respiratory motion and calibration kernels were duplicated across separate submodules:

1. **Pearson correlation (`pearson_correlation`)**: [`evaluation.rs`](../../crates/domain_physics/src/physics/medical/evaluation.rs) computed single-pass uncentered sums ($n \sum X_i Y_i - \sum X_i \sum Y_i$), which suffer catastrophic floating-point cancellation on respiratory signals with large DC offsets, whereas [`optical_motion/validation.rs`](../../crates/domain_physics/src/physics/medical/optical_motion/validation.rs) implemented numerically stable two-pass mean-centered covariance with slice truncation.
2. **Root-mean-square error (`root_mean_square_error`)**: Implemented separately in `evaluation.rs` (returning `Option<f64>` for equal non-empty slices) and `optical_motion/validation.rs` (truncating unequal slices and returning `0.0` on empty inputs).
3. **Relative percentage error (`spatial_accuracy_error` and `percentage_error`)**: Implemented separately in `evaluation.rs` (guarding `|measured| < TOLERANCE_STANDARD` with `0.0`) and `optical_motion/validation.rs` (returning `f64::INFINITY` when `measured == 0.0`).
4. **Weighted average height (`weighted_average_height`)**: Duplicated verbatim across [`optical_motion/processing.rs`](../../crates/domain_physics/src/physics/medical/optical_motion/processing.rs) and [`radar_gating/surface.rs`](../../crates/domain_physics/src/physics/medical/radar_gating/surface.rs).

All eight public functions and their parameter names are locked by [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) under `check-api-drift`.

## Decision

Following the internal kernel consolidation pattern in [ADR 0016](0016-shared-elemental-activation-kernels.md), we unified the four medical physics kernels behind `pub(super)` helpers inside `crates/domain_physics/src/physics/medical/` without altering any public API signatures:

1. **Mean-centered `evaluation::pearson_correlation` and `relative_percentage_error`**:
   - Updated `evaluation::pearson_correlation` to use two-pass mean-centered covariance with `#[verified_engine::verified]`, returning `None` on unequal slice lengths, `len < 2`, or zero variance.
   - Added `#[verified_engine::verified] pub(super) fn relative_percentage_error` in `evaluation.rs` to handle exact-zero and tolerance-bounded denominator guards, and delegated `spatial_accuracy_error` to it with `TOLERANCE_STANDARD` and `0.0`.
2. **Thin delegation wrappers in `optical_motion::validation`**:
   - Updated `optical_motion::validation::{root_mean_square_error, pearson_correlation}` to slice inputs to `n = min(len_a, len_b)` and delegate to `evaluation` with `.unwrap_or(0.0)`.
   - Updated `optical_motion::validation::percentage_error` to delegate to `evaluation::relative_percentage_error(measured, reference, 0.0, f64::INFINITY)`.
3. **Shared `signal::mean_vertex_height` kernel**:
   - Added `#[verified_engine::verified] pub(super) fn mean_vertex_height(vertices: &[Point3<f64>]) -> f64` in [`signal.rs`](../../crates/domain_physics/src/physics/medical/signal.rs) and delegated both `optical_motion::processing::weighted_average_height` and `radar_gating::surface::weighted_average_height` to it.
4. **Parity, DC-offset, and zero-allocation verification**:
   - Added unit and integration tests across `evaluation.rs`, [`test_optical_motion.rs`](../../crates/domain_physics/tests/test_optical_motion.rs), and [`test_radar_gating.rs`](../../crates/domain_physics/tests/test_radar_gating.rs) verifying cross-module parity, large DC-offset stability, edge-case fallbacks, and zero dynamic heap allocations under `VerifiedAllocator` and `VerifiedEngine::run_verified`.

## Consequences

- **Numerical Stability & Consistency**: Eliminates kernel drift across optical and radar motion pipelines and prevents catastrophic cancellation in `evaluation::pearson_correlation` on DC-shifted signals.
- **Zero-Allocation Safety**: Slice views and iterator reductions perform zero heap allocations inside `VerifiedEngine::run_verified` while staying well within the 60-statement and 60-line verification ceilings.
- **API Compatibility**: Preserves 100% of public function signatures and parameter names in `crates/unified_verification/api_baseline.json`.
