//! Integration tests for test transformer architecture.
use domain_ai::ai::transformer::traits::NormalizationLayer;
use domain_ai::ai::transformer::{EncoderLayer, FeedForward, MultiHeadAttention};
use nalgebra::DMatrix;

/// A mock normalization layer that does nothing (Identity).
struct IdentityNorm;

impl NormalizationLayer for IdentityNorm {
    #[verified_engine::verified]
    fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        x.clone()
    }
}

#[test]
#[verified_engine::verified]
fn test_transformer_dependency_injection() {
    let d_model = 64;
    let h = 4;
    let d_ff = 256;

    // Create standard components
    let attn = MultiHeadAttention::new(d_model, h);
    let ff = FeedForward::new(d_model, d_ff);

    // Create our custom component
    let identity_norm1 = IdentityNorm;
    let identity_norm2 = IdentityNorm;

    // Inject them into the EncoderLayer
    // We explicitly specify the types to prove the generic parameters work
    let layer: EncoderLayer<MultiHeadAttention, FeedForward, IdentityNorm> =
        EncoderLayer::new_with_components(attn, ff, identity_norm1, identity_norm2);

    // Create a dummy input
    let x = DMatrix::from_element(5, d_model, 1.0);

    // Run forward pass
    let output = layer.forward(&x, None);

    // Verify dimensions
    assert_eq!(output.nrows(), 5);
    assert_eq!(output.ncols(), d_model);

    // Since we used IdentityNorm, the values will be different than standard LayerNorm,
    // but the fact that it compiled and ran proves the architectural flexibility.
}
