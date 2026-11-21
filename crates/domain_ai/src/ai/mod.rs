//! # Deep Learning & AI
//!
//! > **"What I cannot create, I cannot understand."** — Richard Feynman
//!
//! This module implements state-of-the-art Artificial Intelligence architectures **from scratch** using pure Rust and linear algebra.
//! We avoid "black box" APIs to provide a transparent view of the mathematical machinery under the hood.
//!
//! ##  The Ecosystem
//!
//! The AI module is organized into four foundational pillars:
//!
//! ```mermaid
//! graph TD
//!     AI[Artificial Intelligence]
//!
//!     subgraph "Foundations"
//!     Theory[Deep Learning Theory]
//!     Activations[Activations]
//!     end
//!
//!     subgraph "Architectures"
//!     Trans[Transformers]
//!     Attention[Attention Mechanisms]
//!     end
//!
//!     subgraph "Generative Models"
//!     NeRF[NeRF & 3DGS]
//!     SDS[Score Distillation Sampling]
//!     Diff[Diffusion Models]
//!     end
//!
//!     subgraph "Decision Making"
//!     RL[Reinforcement Learning]
//!     Calibration[Self-Calibration]
//!     end
//!
//!     AI --> Theory
//!     AI --> Trans
//!     AI --> NeRF
//!     AI --> RL
//!
//!     Trans --> Attention
//!     NeRF --> SDS --> Diff
//!     Theory --> Activations
//! ```
//!
//! ##  Quick Start: Transformer Encoder
//!
//! Create a standard Transformer Encoder stack to process sequential data.
//!
//! ```rust
//! # use domain_ai::__doctest_facade as math_explorer;
//! use math_explorer::ai::transformer::Encoder;
//! use math_explorer::nalgebra::DMatrix;
//!
//! // 1. Configure the Architecture
//! let num_layers = 2;
//! let d_model = 512;      // Embedding dimension
//! let num_heads = 8;      // Attention heads
//! let d_ff = 2048;        // Feed-forward hidden dimension
//!
//! let encoder = Encoder::new(num_layers, d_model, num_heads, d_ff);
//!
//! // 2. Prepare Input (Batch Size x Sequence Length x Embedding Dim)
//! // Note: This implementation currently handles 2D matrices (SeqLen x Dim)
//! let seq_len = 10;
//! // Create dummy data using a deterministic function
//! let input_data = DMatrix::<f64>::from_fn(seq_len, d_model, |r, c| (r + c) as f64 * 0.01);
//!
//! // 3. Forward Pass
//! let output = encoder.forward(input_data, None);
//!
//! println!("Output shape: ({}, {})", output.nrows(), output.ncols());
//! ```
//!
//! ##  Deep Dive
//!
//! ### 1. [Transformers](transformer)
//! The backbone of modern NLP. We implement:
//! *   **Multi-Head Attention**: Splitting the embedding space to attend to different parts of the sequence.
//! *   **Positional Encoding**: Injecting sequence order information via sine/cosine functions.
//! *   **Layer Normalization**: Stabilizing training dynamics.
//!
//! ### 2. [Generative AI (SDS & NeRF)](sds)
//! We explore **Neural Radiance Fields (NeRF)** and **3D Gaussian Splatting**, optimized via **Score Distillation Sampling (SDS)**.
//! This allows generating 3D assets from 2D text-to-image diffusion models.
//!
//! ### 3. [Reinforcement Learning](reinforcement_learning)
//! Agents that learn from interaction. Includes:
//! *   **Q-Learning**: Off-policy control.
//! *   **Policy Gradients**: Optimizing the policy directly.
//! *   **Bellman Equations**: The fundamental recursive relationship for value functions.

#[allow(missing_docs)]
pub(crate) mod activation_kernels;
#[allow(missing_docs)]
pub mod activations;
pub mod deep_learning_theory;
pub mod gaussian_splatting;
#[allow(missing_docs)]
pub mod optimization;
pub mod reinforcement_learning;
pub mod sds;
pub mod self_calibration;
pub mod transformer;
#[allow(missing_docs)]
pub mod utils;

// Re-export transformer components for backward compatibility.
pub use transformer::attention;
pub use transformer::feed_forward;
pub use transformer::positional_encoding;

/// Error types for AI modules.
pub use crate::error;

// [cite:graph_parameters_rust]

use pure_math::stochastic_signature_verification;
stochastic_signature_verification!(
    module = "ai",
    external_dependencies = ["LibTorch", "CUDA"],
    signatures = {
        output => (10, 512);
    },
    statistical_bounds = {
        mean in [-0.1, 0.1];
    },
    stochastic_test = {
        let seq_len = 10;
        let d_model = 512;
        let x = nalgebra::DMatrix::<f64>::from_fn(seq_len, d_model, |r, c| (r + c) as f64 * 0.1);
        let ffn = transformer::FeedForward::new(d_model, 2048);
        let output = ffn.forward(&x);

        let mean = output.iter().sum::<f64>() / (seq_len * d_model) as f64;
    }
);
