use super::attention::MultiHeadAttention;
use super::decoder::Decoder;
use super::encoder::Encoder;
use super::feed_forward::FeedForward;
use super::layer_norm::LayerNorm;
use crate::ai::transformer::traits::{AttentionMechanism, FeedForwardNetwork, NormalizationLayer};

use verified_engine::Theory;

/// The full Transformer model container.
///
/// #  Architecture
///
/// This struct acts as a **high-level container** for the `Encoder` and `Decoder` stacks.
/// It provides the structural skeleton of a Transformer but leaves specific pipeline details
/// (like embedding generation) to the user or wrapping frameworks.
///
/// ##  Scope & Limitations
///
/// In a production-ready implementation (like BERT or GPT), this struct would typically also manage:
/// *   **Embeddings**: Converting token IDs to vectors (`Input Embeddings`).
/// *   **Positional Encodings**: Injecting sequence order information (e.g., Sinusoidal or Learned).
/// *   **Head**: The final Linear Projection and Softmax layer for token prediction.
///
/// This implementation focuses strictly on the **Attention & Feed-Forward** backbone.
///
/// #  Usage
///
/// Since this struct does not currently coordinate a unified `forward` pass (as that depends on
/// whether you are doing Seq2Seq, Causal LM, or Masked LM), you should instantiate components manually:
///
/// ```rust
/// # use domain_ai::__doctest_facade as math_explorer;
/// use math_explorer::ai::transformer::{Transformer, Encoder, Decoder};
///
/// // 1. Configure Layers
/// let encoder = Encoder::new(6, 512, 8, 2048);
/// let decoder = Decoder::new(6, 512, 8, 2048);
///
/// // 2. Assemble Model
/// let model = Transformer {
///     encoder,
///     decoder,
/// };
///
/// // 3. (User Responsibility) Add Embeddings & Positional Encoding here...
/// ```
#[derive(Theory)]
#[theory(
    description = "The Transformer architecture relies entirely on attention mechanisms, dispensing with recurrence and convolutions entirely. The encoder maps an input sequence to a continuous representation, which the decoder uses to generate an output sequence.",
    citation = "Attention Is All You Need (Vaswani et al., 2017)"
)]
pub struct Transformer<
    A: AttentionMechanism = MultiHeadAttention,
    F: FeedForwardNetwork = FeedForward,
    N: NormalizationLayer = LayerNorm,
> {
    /// The stack of Encoder layers (Self-Attention + Feed-Forward).
    pub encoder: Encoder<A, F, N>,
    /// The stack of Decoder layers (Masked Self-Attention + Cross-Attention + Feed-Forward).
    pub decoder: Decoder<A, F, N>,
}
