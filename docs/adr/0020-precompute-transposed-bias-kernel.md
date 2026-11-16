# 20. Precompute Transposed Bias and Kernel in Tensor Operations

## Status
Accepted

## Context
In `crates/domain_climate/src/climate/tensor_ops.rs`, row iterators in `conv1d` and `conv_transpose1d` repeatedly called `bias.transpose()` inside `output.row_iter_mut().for_each(...)`. In `nalgebra`, every vector transposition allocates a dynamic row vector on the heap, causing $O(N)$ redundant heap allocations per convolution call where $N$ is the batch sample count. Additionally, `conv1d` evaluated `input * kernel.transpose()` inline rather than precomputing and borrowing a transposed kernel matrix reference.

During multi-pass training loops (`CeraTrainer`) and sequential convolutional inference (`Autoencoder`), these unhoisted allocations caused unnecessary heap churn and performance degradation in inner loops.

## Decision
1. Precomputed the transposed bias row vector (`let bias_t = bias.transpose();`) once outside the row iterator in both `conv1d` and `conv_transpose1d`.
2. Passed a borrowed reference to the precomputed transposed vector (`&bias_t`) inside `row += &bias_t` in each row iteration, eliminating per-row dynamic allocations.
3. Precomputed the transposed kernel matrix (`let kernel_t = kernel.transpose();`) into a local variable in `conv1d` prior to matrix multiplication, computing `input * &kernel_t`.
4. Maintained identical public API signatures and exact numerical equivalence for all tensor convolution operations.
5. Added unit tests covering multi-row batches, edge cases (zero rows, zero input/output channels), and dimension mismatch panic assertions.

## Consequences
- Eliminates $N$ heap allocations per forward invocation in `conv1d` and `conv_transpose1d`, substantially reducing dynamic allocation overhead during batch inference and training.
- Retains numerical stability and exact floating-point equivalence across all callers in `domain_climate` (`Autoencoder`, `Predictor`, and `CeraTrainer`).
- All unit, integration, and verification gate tests pass cleanly.
