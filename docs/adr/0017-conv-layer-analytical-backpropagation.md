# 17. Analytical Backpropagation in ConvLayer and CeraTrainer Gradient Wiring

## Status
Accepted

## Context
The `ConvLayer::update_weights` method in `crates/domain_climate/src/climate/autoencoder.rs` aborted execution with an `unimplemented!` panic macro call. In addition, `CeraTrainer::optimizer_step` in `crates/domain_climate/src/climate/training.rs` discarded computed batch losses and updated model parameters using constant dummy vectors (`1e-4`). This contradicted the verified training and representation learning requirements specified in the CERA framework architecture.

## Decision
1. Implemented analytical backpropagation on `ConvLayer` via `backward_update` and `backward_update_vector`:
   - Computes weight gradients $\nabla_W = \frac{1}{N} d_Z^T X$ and bias gradients $\nabla_b = \frac{1}{N} \sum d_Z$.
   - Computes input gradient $\nabla_X = d_Z W$ using the layer's un-mutated kernel before weight update.
   - Handles empty batches ($N = 0$) gracefully to avoid divide-by-zero errors during gradient averaging.
   - Checks that all gradient elements are finite (`f32::is_finite`) before performing parameter updates with the optimizer.
2. Removed the `unimplemented!` panic macro from `ConvLayer::update_weights`, delegating parameter updates cleanly without panics.
3. Computed analytical Mean Squared Error (MSE) loss gradients for reconstruction and prediction batches in `CeraTrainer::train`:
   - Extracted helper functions `compute_reconstruction_gradients` and `compute_prediction_gradients` to satisfy NASA Power of 10 statement limits.
   - Replaced constant dummy vectors in `optimizer_step` with the calculated analytical loss gradients and input vectors.
   - Stabilized gradient norms via clipping and verified finiteness before parameter updates.
   - Updated `train` to record and return epoch loss history.
4. Added unit and edge case tests for `ConvLayer` and updated `test_cera_integration` in `tests/cera.rs` to verify that `CeraTrainer::train` decreases loss values across successive training epochs.

## Consequences
- `ConvLayer` provides working analytical backpropagation and gradient updates without panicking.
- `CeraTrainer` optimizes model parameters using real MSE loss gradients, driving loss reduction across training epochs.
- All domain climate tests and verification suites pass cleanly.
