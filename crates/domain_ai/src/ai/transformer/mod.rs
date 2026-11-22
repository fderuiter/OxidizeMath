//! # Transformer Architecture
//!
//! This module implements the Transformer architecture as described in the paper
//! "Attention Is All You Need".
//!
//! ## Architecture Overview
//!
//! The Transformer follows an Encoder-Decoder structure using stacked self-attention
//! and point-wise, fully connected layers.
//!
//! ```mermaid
//! graph TD
//!     subgraph Encoder
//!     Input[Input] --> EncLayer1[Encoder Layer 1]
//!     EncLayer1 --> EncLayer2[Encoder Layer 2]
//!     EncLayer2 --> EncOutput[Encoder Output]
//!     end
//!
//!     subgraph Decoder
//!     Target[Target] --> DecLayer1[Decoder Layer 1]
//!     EncOutput --> DecLayer1
//!     DecLayer1 --> DecLayer2[Decoder Layer 2]
//!     EncOutput --> DecLayer2
//!     DecLayer2 --> Output[Output]
//!     end
//! ```
//!
//! ### Multi-Head Attention Tensor Flow
//! ```mermaid
//! sequenceDiagram
//!     participant Input
//!     participant QKV as Q, K, V Projections
//!     participant SDPA as Scaled Dot-Product
//!     participant OutProj as Output Projection
//!
//!     Input->>QKV: X (seq_len, d_model)
//!     QKV->>SDPA: Queries, Keys, Values (heads, seq_len, d_k)
//!     Note over SDPA: Softmax(Q * K^T / sqrt(d_k)) * V
//!     SDPA->>OutProj: Concatenated Heads (seq_len, d_model)
//!     OutProj->>Input: Contextualized Output (seq_len, d_model)
//! ```
//!
//! ## Components
//!
//! - **Attention**: Scaled Dot-Product Attention and Multi-Head Attention.
//! - **Feed Forward**: Position-wise Feed-Forward Networks.
//! - **Layer Norm**: Layer Normalization.
//! - **Encoder**: Stack of `EncoderLayer`s.
//! - **Decoder**: Stack of `DecoderLayer`s.
//!
//! ##  Quick Start: End-to-End Example
//!
//! Constructing and running a full Transformer model (Encoder + Decoder).
//!
//! ```rust
//! # use domain_ai::__doctest_facade as math_explorer;
//! use math_explorer::ai::transformer::{Encoder, Decoder, Transformer};
//! use math_explorer::nalgebra::DMatrix;
//!
//! // 1. Configure Model Parameters
//! let d_model = 64;  // Embedding dimension
//! let heads = 4;     // Number of attention heads
//! let d_ff = 128;    // Feed-forward dimension
//! let layers = 2;    // Number of layers
//!
//! // 2. Instantiate Components
//! let encoder = Encoder::new(layers, d_model, heads, d_ff);
//! let decoder = Decoder::new(layers, d_model, heads, d_ff);
//!
//! // 3. Assemble Transformer
//! let transformer = Transformer {
//!     encoder,
//!     decoder,
//! };
//!
//! // 4. Create Dummy Input (Sequence Length = 10)
//! let seq_len = 10;
//! // Using constant values to simulate embeddings
//! let input_src = DMatrix::from_element(seq_len, d_model, 0.5);
//! let input_tgt = DMatrix::from_element(seq_len, d_model, 0.5);
//!
//! // 5. Run Forward Pass
//! // A. Encoder Pass
//! let enc_output = transformer.encoder.forward(input_src, None);
//! assert_eq!(enc_output.shape(), (seq_len, d_model));
//!
//! // B. Decoder Pass
//! // Note: In a real Seq2Seq model, you would pass the encoder output here.
//! let dec_output = transformer.decoder.forward(input_tgt, &enc_output, None, None);
//! assert_eq!(dec_output.shape(), (seq_len, d_model));
//! ```

#[allow(missing_docs)]
pub mod attention;
#[allow(missing_docs)]
pub mod decoder;
#[allow(missing_docs)]
pub mod encoder;
#[allow(missing_docs)]
pub mod feed_forward;
#[allow(missing_docs)]
pub mod layer_norm;
#[allow(missing_docs)]
pub mod model;
#[allow(missing_docs)]
pub mod positional_encoding;
#[allow(missing_docs)]
pub mod tokenization;
#[allow(missing_docs)]
pub mod traits;

pub use attention::MultiHeadAttention;
pub use decoder::{Decoder, DecoderLayer};
pub use encoder::{Encoder, EncoderLayer};
pub use feed_forward::FeedForward;
pub use layer_norm::LayerNorm;
pub use model::Transformer;
pub use traits::{AttentionMechanism, FeedForwardNetwork, NormalizationLayer};

// [cite:algorithmic_information_rust]
