# ADR 0037: Consolidate Optimizer, Softmax, and Multi-Head Attention Kernels

## Status

Accepted (extends [ADR 0014](0014-fused-multi-head-attention-projections.md), [ADR 0016](0016-shared-elemental-activation-kernels.md), and [ADR 0021](0021-inplace-row-softmax-allocation-elimination.md))

## Context

Five modules across `pure_math` and `domain_ai` maintained parallel copies of core numerical kernels:

1. **Optimizer Slice Updates & State Churn**: [`crates/pure_math/src/pure_math/analysis/optimization/mod.rs`](../../crates/pure_math/src/pure_math/analysis/optimization/mod.rs) duplicated `Adam` moment accumulation and bias-correction logic across `DMatrix` and `DVector` `EvolutionEngine` implementations, while `Adam::update_matrix`, `Adam::update_vector`, `SGD::update_matrix`, and `SGD::update_vector` cloned incoming gradient buffers and removed/re-inserted `AdamState` entries on each step.
2. **Bespoke SDS Adam Implementation**: [`crates/domain_ai/src/ai/sds/training.rs`](../../crates/domain_ai/src/ai/sds/training.rs) re-implemented `Adam` moment updates inside `AdamOptimizer::step` and duplicated matrix shape checks across `AdamOptimizer::step` and `SgdOptimizer::step`.
3. **Unguarded Vector Softmax Copies**: [`crates/domain_ai/src/ai/deep_learning_theory/probability.rs`](../../crates/domain_ai/src/ai/deep_learning_theory/probability.rs) and a private `softmax` helper in [`crates/domain_ai/src/ai/optimization.rs`](../../crates/domain_ai/src/ai/optimization.rs) computed vector softmax without the empty-input and `sum_exps > 0.0` (`NaN` / zero-sum) guards established for `softmax_row_wise` in [`crates/domain_ai/src/ai/activations.rs`](../../crates/domain_ai/src/ai/activations.rs) under [ADR 0021](0021-inplace-row-softmax-allocation-elimination.md).
4. **Duplicated Fused Attention Loops**: [`crates/domain_ai/src/ai/transformer/attention.rs`](../../crates/domain_ai/src/ai/transformer/attention.rs) duplicated fused `Q`/`K`/`V` GEMM projections and strided `DMatrixView` head loops across `MultiHeadAttention::forward_impl` and `MultiHeadAttention::forward_with_weights`.

## Decision

We consolidated all four kernel families behind existing public facades without changing any exported types, struct fields, or method signatures in [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json):

1. **Shared Zero-Allocation `Adam` and `SGD` Slice Kernels**:
   - Extracted private `sgd_step_slice` and `adam_step_slices` helpers in [`crates/pure_math/src/pure_math/analysis/optimization/mod.rs`](../../crates/pure_math/src/pure_math/analysis/optimization/mod.rs) to mutate `DMatrix` and `DVector` slices in place.
   - Updated `Adam::update_matrix` and `Adam::update_vector` to mutate `AdamState` in place via `HashMap::entry` without cloning gradients, and preserved the `0.01` step rate in `SGD::update_matrix` and `SGD::update_vector`.
2. **SDS Optimizer Delegation & Shared Shape Checks**:
   - Extracted `validate_param_grad_shapes` in [`crates/domain_ai/src/ai/sds/training.rs`](../../crates/domain_ai/src/ai/sds/training.rs) for both `AdamOptimizer::step` and `SgdOptimizer::step`, and delegated `AdamOptimizer::step` to `pure_math::pure_math::analysis::optimization::Adam` while preserving all public fields on `AdamOptimizer`.
3. **Shared Guarded Softmax Kernel**:
   - Added `pub(crate)` generic `softmax_into` and `softmax` helpers in [`crates/domain_ai/src/ai/activation_kernels.rs`](../../crates/domain_ai/src/ai/activation_kernels.rs) enforcing empty-input and `sum_exps > 0.0` zero-default guards across any `nalgebra` matrix row view or vector.
   - Routed `activations::softmax_row_wise` (via `nalgebra` row iterators), `probability::softmax`, and `optimization::cross_entropy_softmax_prime` through the shared kernel and deleted private `optimization::softmax`.
4. **Shared Fused Multi-Head Attention Helper**:
   - Extracted private `MultiHeadAttention::forward_fused` with an optional `&mut Vec<DMatrix<f64>>` sink in [`crates/domain_ai/src/ai/transformer/attention.rs`](../../crates/domain_ai/src/ai/transformer/attention.rs), routing `forward_impl` (`None`, skipping `Vec` allocation) and `forward_with_weights` (`Some(&mut all_weights)`) through the single fused implementation.

## Consequences

- **Zero Post-Init Heap Allocations**: Pre-initialized `Adam` and `SGD` matrix and vector updates execute with zero dynamic heap allocations under `VerifiedAllocator` and `VerifiedEngine::run_verified` per the [Zero-Allocation Runtime Guide](../architecture/zero_allocation_runtime.md).
- **Uniform Numerical Stability**: Every vector and matrix `softmax` path returns deterministic zero outputs for empty or `NaN` inputs and matches within `TOLERANCE_FAST`.
- **Zero Public API Drift**: Preserves 100% of public signatures in [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json).
