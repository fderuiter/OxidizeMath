# 13. Trainable Climate Autoencoder Backpropagation and Training Alignment

## Status
Accepted

## Context
The `Autoencoder` struct in `crates/domain_climate` panicked inside its `Trainable<f32>::backward_update` implementation because gradient calculations were omitted. Furthermore, `CeraTrainer::optimizer_step` passed dummy gradient vectors of dimension 1 to both `Autoencoder` and `Predictor`, causing dimension mismatch panics against multi-channel layer kernels. Consequently, end-to-end model training routines panicked at runtime, and integration tests, unit tests, and doctests were guarded by `#[should_panic]` attributes.

## Decision
1. Implemented standard reverse-mode backpropagation for `Autoencoder` under the `Trainable<f32>` interface across all six convolutional layers (three encoder layers and three decoder layers):
   - Cached activations and pre-activations across the full model during the forward pass.
   - Handled linear activations (identity) at layer boundaries (latent space layer 2 and reconstruction output layer 5), while propagating LeakyReLU (alpha = 0.01) derivatives across hidden layers (0, 1, 3, 4).
   - Calculated weight gradients (`d_w = d_z * a_prev^T`) and bias gradients (`d_b = d_z`) and propagated error upstream using the un-mutated forward kernels before calling `optimizer.update_matrix` and `optimizer.update_vector`.
2. Updated `CeraTrainer::optimizer_step` to pass valid dimensional vectors matching `config.in_channels` for the autoencoder, and matching `config.num_levels * config.aligned_channels` (input) and `config.output_size` (gradient) for the predictor.
3. Removed `#[should_panic]` attributes from `test_cera_integration` in `tests/cera.rs`, `test_cera_training_and_prediction` in `src/climate/cera.rs`, and the doctest in `src/climate/mod.rs`.
4. Added unit and finite-difference gradient tests verifying analytical derivative correctness across layers.

## Consequences
- `Autoencoder` supports standard reverse-mode gradient descent and model optimization under the `Trainable<f32>` trait.
- `CeraTrainer` executes training loops without dimension mismatch or runtime panics.
- Unit, integration, and documentation tests execute actively and assert valid parameter learning and non-zero predictions.
