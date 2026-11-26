use super::attention::MultiHeadAttention;
use super::feed_forward::FeedForward;
use super::layer_norm::LayerNorm;
use crate::ai::transformer::traits::{AttentionMechanism, FeedForwardNetwork, NormalizationLayer};
use nalgebra::DMatrix;

/// A single Decoder layer.
pub struct DecoderLayer<
    A: AttentionMechanism = MultiHeadAttention,
    F: FeedForwardNetwork = FeedForward,
    N: NormalizationLayer = LayerNorm,
> {
    /// The masked multi-head self-attention mechanism.
    pub self_attn: A,
    /// The multi-head cross-attention mechanism.
    pub cross_attn: A,
    /// The position-wise feed-forward network.
    pub feed_forward: F,
    /// Layer normalization applied after the self-attention mechanism.
    pub norm1: N,
    /// Layer normalization applied after the cross-attention mechanism.
    pub norm2: N,
    /// Layer normalization applied after the feed-forward network.
    pub norm3: N,
}

impl DecoderLayer<MultiHeadAttention, FeedForward, LayerNorm> {
    /// Creates a new `DecoderLayer` instance.
    ///
    /// # Arguments
    ///
    /// * `d_model`: The dimension of the model.
    /// * `h`: The number of attention heads.
    /// * `d_ff`: The dimension of the feed-forward network.
    ///
    /// # Returns
    ///
    /// A new `DecoderLayer` instance.
    #[verified_engine::verified]
    pub fn new(d_model: usize, h: usize, d_ff: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, h),
            cross_attn: MultiHeadAttention::new(d_model, h),
            feed_forward: FeedForward::new(d_model, d_ff),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
            norm3: LayerNorm::new(d_model),
        }
    }
}

impl<A: AttentionMechanism, F: FeedForwardNetwork, N: NormalizationLayer> DecoderLayer<A, F, N> {
    /// Creates a new `DecoderLayer` with injected components.
    #[verified_engine::verified]
    pub fn new_with_components(
        self_attn: A,
        cross_attn: A,
        feed_forward: F,
        norm1: N,
        norm2: N,
        norm3: N,
    ) -> Self {
        Self {
            self_attn,
            cross_attn,
            feed_forward,
            norm1,
            norm2,
            norm3,
        }
    }

    /// Performs the forward pass of the decoder layer.
    ///
    /// # Arguments
    ///
    /// * `x`: The input matrix (e.g., target sequence embeddings).
    /// * `enc_output`: The output matrix from the encoder.
    /// * `self_attn_mask`: Optional mask for the self-attention mechanism.
    /// * `cross_attn_mask`: Optional mask for the cross-attention mechanism.
    ///
    /// # Returns
    ///
    /// The output matrix.
    #[verified_engine::verified]
    pub fn forward(
        &self,
        x: &DMatrix<f64>,
        enc_output: &DMatrix<f64>,
        self_attn_mask: Option<&DMatrix<f64>>,
        cross_attn_mask: Option<&DMatrix<f64>>,
    ) -> DMatrix<f64> {
        // 1. Masked self-attention
        let self_attn_output = self.self_attn.forward(x, x, x, self_attn_mask);
        let x_plus_self_attn = x + self_attn_output;
        let normed_self_attn = self.norm1.forward(&x_plus_self_attn);

        // 2. Cross-attention (Encoder-Decoder attention)
        let cross_attn_output =
            self.cross_attn
                .forward(&normed_self_attn, enc_output, enc_output, cross_attn_mask);
        let normed_self_attn_plus_cross = &normed_self_attn + cross_attn_output;
        let normed_cross_attn = self.norm2.forward(&normed_self_attn_plus_cross);

        // 3. Feed-forward network
        let ff_output = self.feed_forward.forward(&normed_cross_attn);
        let normed_cross_attn_plus_ff = &normed_cross_attn + ff_output;
        self.norm3.forward(&normed_cross_attn_plus_ff)
    }
}

/// The full Decoder, composed of a stack of identical DecoderLayers.
pub struct Decoder<
    A: AttentionMechanism = MultiHeadAttention,
    F: FeedForwardNetwork = FeedForward,
    N: NormalizationLayer = LayerNorm,
> {
    /// A vector of `DecoderLayer` instances.
    pub layers: Vec<DecoderLayer<A, F, N>>,
}

impl Decoder<MultiHeadAttention, FeedForward, LayerNorm> {
    /// Creates a new `Decoder` instance.
    ///
    /// # Arguments
    ///
    /// * `num_layers`: The number of decoder layers.
    /// * `d_model`: The dimension of the model.
    /// * `h`: The number of attention heads.
    /// * `d_ff`: The dimension of the feed-forward network.
    ///
    /// # Returns
    ///
    /// A new `Decoder` instance.
    #[verified_engine::verified]
    pub fn new(num_layers: usize, d_model: usize, h: usize, d_ff: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| DecoderLayer::new(d_model, h, d_ff))
                .collect(),
        }
    }
}

impl<A: AttentionMechanism, F: FeedForwardNetwork, N: NormalizationLayer> Decoder<A, F, N> {
    /// Creates a new `Decoder` with injected layers.
    #[verified_engine::verified]
    pub fn new_with_layers(layers: Vec<DecoderLayer<A, F, N>>) -> Self {
        Self { layers }
    }

    /// Performs the forward pass through all decoder layers.
    ///
    /// # Arguments
    ///
    /// * `x`: The input matrix.
    /// * `enc_output`: The output from the encoder.
    /// * `self_attn_mask`: Optional mask for self-attention.
    /// * `cross_attn_mask`: Optional mask for cross-attention.
    ///
    /// # Returns
    ///
    /// The output matrix.
    #[verified_engine::verified]
    pub fn forward(
        &self,
        mut x: DMatrix<f64>,
        enc_output: &DMatrix<f64>,
        self_attn_mask: Option<&DMatrix<f64>>,
        cross_attn_mask: Option<&DMatrix<f64>>,
    ) -> DMatrix<f64> {
        for layer in &self.layers {
            x = layer.forward(&x, enc_output, self_attn_mask, cross_attn_mask);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_decoder_layer_dims() {
        let seq_len = 10;
        let d_model = 512;
        let h = 8;
        let d_ff = 2048;

        let x = DMatrix::zeros(seq_len, d_model);
        let enc_output = DMatrix::zeros(seq_len, d_model);
        let decoder_layer = DecoderLayer::new(d_model, h, d_ff);
        let output = decoder_layer.forward(&x, &enc_output, None, None);

        assert_eq!(output.nrows(), seq_len);
        assert_eq!(output.ncols(), d_model);
    }
}
