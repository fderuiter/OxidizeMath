use super::attention::MultiHeadAttention;
use super::decoder::Decoder;
use super::encoder::Encoder;
use super::feed_forward::FeedForward;
use super::layer_norm::LayerNorm;
use crate::ai::transformer::traits::{AttentionMechanism, FeedForwardNetwork, NormalizationLayer};

use verified_engine::Theory;

/// The full sequence-to-sequence Transformer encoder-decoder container.
///
/// Composes stacked multi-head scaled dot-product attention and position-wise feed-forward sublayers:
///
/// $$\mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}, \qquad \mathrm{FFN}(\mathbf{x}) = \max(\mathbf{0}, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$
///
/// Reference: Vaswani et al. (2017), arXiv:1706.03762; see
/// [`docs/math_specs/domain_ai.md`](../../../../../docs/math_specs/domain_ai.md).
///
/// # Stability
/// Scaling logits by $1/\sqrt{d_k}$ keeps attention dot-product variance $O(1)$, preventing
/// softmax saturation, while residual connections and [`LayerNorm`] bound hidden-state norms across depth.
///
/// # Domain of Validity
/// Non-empty 2D token embedding matrices `(seq_len, d_model)` with finite entries and positive
/// embedding dimension `d_model` divisible by the configured attention head count $h \ge 1$.
///
/// # Architecture
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
