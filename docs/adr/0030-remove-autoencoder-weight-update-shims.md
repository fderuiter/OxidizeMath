# ADR 0030: Remove Obsolete Climate Autoencoder Weight-Update Shims

## Status
Accepted (extends [ADR 0004](0004-remove-deprecated-domain-apis.md), [ADR 0013](0013-trainable-climate-autoencoder-backpropagation.md), [ADR 0017](0017-conv-layer-analytical-backpropagation.md), and [ADR 0019](0019-purge-dead-code-bypasses.md))

## Context
[ADR 0013](0013-trainable-climate-autoencoder-backpropagation.md) and [ADR 0017](0017-conv-layer-analytical-backpropagation.md) migrated CERA model optimization in `crates/domain_climate/src/climate/training.rs` (`CeraTrainer::optimizer_step`) to analytical backpropagation via `Trainable<f32>::backward_update` on `Autoencoder` and `Predictor`, as well as `ConvLayer::backward_update`. However, `crates/domain_climate/src/climate/autoencoder.rs` retained four obsolete inherent methods:
1. **`ConvLayer::update_weights`**: Passed zero-filled gradient matrices and vectors to the optimizer instead of analytical gradients, called only by a single legacy assertion in `crates/domain_climate/tests/autoencoder.rs` and the two uncalled wrappers below.
2. **`ConvLayer::backward_update_vector`**: Allocated temporary single-row matrices to wrap `ConvLayer::backward_update` and had zero callers across the workspace.
3. **`Encoder::update_weights` and `Decoder::update_weights`**: Iterated over `self.layers` calling the zero-gradient `ConvLayer::update_weights` shim and had zero callers across the workspace.

## Decision
1. **Delete Obsolete Shims and Uncalled Wrappers**: Remove `ConvLayer::backward_update_vector`, `ConvLayer::update_weights`, `Encoder::update_weights`, and `Decoder::update_weights` from `crates/domain_climate/src/climate/autoencoder.rs`, while keeping `ConvLayer::backward_update` and `impl Trainable<f32> for Autoencoder` intact.
2. **Align Edge-Case Integration Test**: Remove the sole `layer.update_weights(&mut opt, 0)` assertion from `test_conv_layer_edge_cases` in `crates/domain_climate/tests/autoencoder.rs`, retaining all analytical gradient, finite-difference, empty-batch, and `NaN` finiteness assertions.
3. **Regenerate Public API Baseline**: Regenerate `crates/unified_verification/api_baseline.json` via `unified_verification -- regenerate-baseline` to remove the four retired inherent method signatures.

## Consequences
- **Positive**: Eliminates misleading zero-gradient weight-update shims and dead wrapper methods in `domain_climate`, preventing silent no-op optimizer updates and reducing maintenance overhead.
- **Neutral**: Analytical backpropagation (`ConvLayer::backward_update` and `Trainable<f32>` implementations on `Autoencoder` and `Predictor`) and all active CERA training workflows remain unchanged.
