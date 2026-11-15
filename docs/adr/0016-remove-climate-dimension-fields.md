# ADR 0016: Remove Unread Dimension Fields from Climate Models

## Status
Accepted

## Context
`ConvLayer` in `crates/domain_climate/src/climate/autoencoder.rs` and `Predictor` in `crates/domain_climate/src/climate/predictor.rs` maintained unread private dimension fields (`in_channels`, `out_channels`, `input_size`, and `output_size`). These unused fields required suppression attributes (`#[allow(dead_code)]`) and four dead code whitelist bypass entries in `verification_whitelist.toml`.

## Decision
1. Removed `in_channels` and `out_channels` fields and assignments from `ConvLayer`. The constructor `ConvLayer::new` retains its arguments to size allocated tensor matrices without storing redundant fields.
2. Removed `input_size` and `output_size` fields and annotations from `Predictor`.
3. Updated `Predictor::new_from_layers(layers: [ConvLayer; 5])` to accept only the layers array, consistent with `autoencoder::Encoder::new_from_layers` and `Decoder::new_from_layers`.
4. Pruned the four `[[dead_code]]` entries from `verification_whitelist.toml`.

## Consequences
- **Positive:** Reduces the memory footprint of `ConvLayer` and `Predictor`, cleans up obsolete dead code whitelist exceptions, and aligns `Predictor` constructor interfaces with existing domain patterns.
- **Negative:** Callers of `Predictor::new_from_layers` must pass only the layers array.
