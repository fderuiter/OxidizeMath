# ADR 0021: In-Place Row Softmax Allocation Elimination

## Context
In `crates/domain_ai/src/ai/activations.rs`, `softmax_row_wise` previously executed `row.map` to compute exponentiated values and `exps.map` to normalize probabilities across each row of the matrix. In `nalgebra`, `DVector::map` allocates a new heap buffer on each invocation. For an $M \times N$ matrix, this resulted in $2 \times M$ transient dynamic heap allocations during evaluation. In iterative multi-head attention and transformer layers, these per-row allocations caused substantial memory fragmentation and heap allocation churn in inner loops.

## Decision
We refactored `softmax_row_wise` to eliminate transient per-row dynamic allocations by performing in-place mutations directly in the preallocated output buffer:
1. Preallocated a single destination `DMatrix::zeros(nrows, ncols)` buffer.
2. Replaced `row.map` and `exps.map` vector allocations with indexed scalar loops across row elements.
3. Computed the row maximum directly across the input row to maintain numerical stability without dynamic slices.
4. Wrote exponentiated values `(matrix[(r, c)] - max_val).exp()` directly into `result[(r, c)]` while accumulating `sum_exps` in the same pass.
5. Normalized rows in-place via `result[(r, c)] /= sum_exps` when `sum_exps > 0.0`.
6. Preserved 0.0 defaults for empty matrices or non-positive sums (e.g. all-NaN inputs), maintaining deterministic safety guarantees.
7. Maintained identical public API signatures (`pub fn softmax_row_wise(matrix: &DMatrix<f64>) -> DMatrix<f64>`) and exact numerical equivalence.

## Consequences
- **Performance**: Completely eliminates $2 \times M$ transient heap allocations per matrix softmax operation, lowering memory churn in neural network evaluation loops.
- **Maintainability & Safety**: Preserves pure function signature, `#[verified_engine::verified]` annotation, and numerical stability for single-element rows and large negative values.
- **Verification**: Passes all domain test suites and satisfies the `verify-records` architectural verification gate.
