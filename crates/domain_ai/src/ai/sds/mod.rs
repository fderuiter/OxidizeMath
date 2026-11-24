//! # Score Distillation Sampling (SDS)
//!
//! A method for optimization using a pre-trained Diffusion Model as a critic.
//!
//! SDS (introduced in *DreamFusion*) allows us to optimize a differentiable 3D representation (like NeRF or Gaussian Splatting)
//! such that its 2D renderings look like "high probability" images according to a text-to-image diffusion model.
//!
//! ## The Loop
//!
//! Instead of training on a dataset of 3D objects, we "distill" knowledge from a 2D model.
//!
//! ```mermaid
//! graph LR
//!     subgraph "Forward Pass"
//!     Params[3D Parameters] --> Render[Differentiable Renderer]
//!     Render --> Image[2D Image]
//!     end
//!
//!     subgraph "Score Distillation"
//!     Image --> Noise[Add Noise]
//!     Noise --> UNet[Diffusion U-Net]
//!     Text[Text Prompt] --> UNet
//!     UNet --> NoisePred[Predicted Noise]
//!     NoisePred --> Grad[Calculate Gradient]
//!     end
//!
//!     subgraph "Backward Pass"
//!     Grad -->|Backpropagate| Params
//!     end
//!
//!     style UNet fill:#f9f,stroke:#333
//!     style Render fill:#9cf,stroke:#333
//! ```
//!
//! ## Components
//!
//! *   **[Diffusion](diffusion)**: The probabilistic model that estimates the score function $\nabla \log p(x)$.
//! *   **[Rendering](rendering)**: Differentiable rendering logic (volumetric or rasterization).
//! *   **[Score](score)**: Calculation of the SDS gradient update.
//! *   **[Gradient](gradient)**: Utilities for backpropagating the score through the renderer.

/// Forward diffusion variance schedules and latent noise injection.
pub mod diffusion;
/// Score Distillation Sampling residual gradient computation.
pub mod gradient;
/// Concrete zero-allocation volumetric NeRF and analytical score-distillation models.
pub mod models;
/// Analytical prompt prototype encoder and Tweedie Gaussian diffusion score prior.
pub mod priors;
/// Volumetric NeRF ray-bundle generation, stratified sampling, and quadrature.
pub mod rendering;
/// Text-conditioned diffusion score estimation and classifier-free guidance.
pub mod score;
/// Differentiable NeRF backpropagation and parameter optimizers (`AdamOptimizer`, `SgdOptimizer`).
pub mod training;

pub use models::{
    AnalyticalScorePrior, AnalyticalVoxelNeRF, PromptPrototypeEncoder, VoxelPointSample,
    rgb_image_to_latent,
};

// [cite:graph_parameters_rust]
