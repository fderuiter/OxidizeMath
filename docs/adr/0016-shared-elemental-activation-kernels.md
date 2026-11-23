# ADR 0016: Shared Elemental Activation Kernel Module

## Status

Accepted

## Context
In `crates/domain_ai`, mathematical activation functions were previously duplicated across `crates/domain_ai/src/ai/activations.rs` (which applied ReLU in place on dynamic matrices via `x.max(0.0)`) and `crates/domain_ai/src/ai/deep_learning_theory/calculus.rs` (which applied ReLU out of place on vectors via `if v > 0.0 { v } else { 0.0 }`, along with derivatives and other activations). This duplication caused separate unit tests, redundancy in core mathematical logic, and potential risk of subtle mathematical divergence.

## Decision
We consolidated activation logic across matrix and vector representations into a shared elemental scalar kernel:
1. Created `crates/domain_ai/src/ai/activation_kernels.rs` providing elemental scalar activation functions (`relu`, `relu_prime`, `sigmoid`, `sigmoid_prime`, `tanh`, `tanh_prime`, `gelu`, `gelu_prime`) with `#[inline]` annotations for SIMD optimization and zero dynamic memory allocations.
2. Kept the elemental scalar kernels internal with `pub(crate)` visibility to strictly preserve public API baseline contracts in `crates/unified_verification/api_baseline.json`.
3. Updated `crates/domain_ai/src/ai/activations.rs` to delegate element mutation directly to `kernel::relu` in place via `matrix.apply(|x| *x = kernel::relu(*x))` without dynamic memory allocations.
4. Updated `crates/domain_ai/src/ai/deep_learning_theory/calculus.rs` to delegate element mappings (`relu`, `relu_prime`, `sigmoid`, `sigmoid_prime`, `tanh`, `tanh_prime`, `gelu`, `gelu_prime`) to the shared scalar activation kernels while preserving all public signatures.
5. Added unit tests in `domain_ai` verifying exact numerical equivalence between matrix in-place and vector out-of-place activation outputs.

## Consequences
- **Consistency**: Eliminates duplicate activation logic and ensures identical numerical behavior across matrix and vector layers.
- **Performance**: Zero runtime allocation overhead for matrix operations and inlined elemental kernels enabling compiler auto-vectorization.
- **Compatibility**: Preserves 100% of public API signatures in `crates/unified_verification/api_baseline.json` and passes all verified suite checks.
