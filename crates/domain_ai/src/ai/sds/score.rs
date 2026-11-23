use nalgebra::DMatrix;

/// Module 3.1: Text Encoding
/// Input: Text Prompt y.
/// Operation: Pass text through CLIP Text Encoder.
/// Output: Text Embedding Vectors e_text.
pub trait TextEncoder {
    /// Encodes a conditioning prompt `text` into a sequence-by-embedding matrix $\mathbf{e}_{\mathrm{text}}$.
    #[verified_engine::verified]
    fn encode(&self, text: &str) -> DMatrix<f64>;
}

/// Denoising diffusion noise-prediction interface $\hat{\boldsymbol{\epsilon}}_\phi(\mathbf{z}_t; y, t)$ for Score Distillation Sampling.
///
/// Approximates the scaled score of the perturbed latent $\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\mathbf{x} + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$:
///
/// $$\nabla_\theta \mathcal{L}_{\mathrm{SDS}}(\phi, \mathbf{x} = g(\theta)) = \mathbb{E}_{t, \boldsymbol{\epsilon}}\!\left[w(t)\left(\hat{\boldsymbol{\epsilon}}_\phi(\mathbf{z}_t; y, t) - \boldsymbol{\epsilon}\right)\frac{\partial \mathbf{x}}{\partial \theta}\right]$$
///
/// Reference: Ho, Jain & Abbeel (2020), arXiv:2006.11239; Poole et al. (2022), arXiv:2209.14988;
/// see [`docs/math_specs/domain_ai.md`](../../../../../docs/math_specs/domain_ai.md).
///
/// # Stability
/// Omitting the U-Net Jacobian $\partial \hat{\boldsymbol{\epsilon}}_\phi / \partial \mathbf{z}_t$ yields a
/// low-variance mode-seeking score residual well-conditioned for `timestep` $t$ away from $\bar{\alpha}_t \in \{0, 1\}$.
///
/// # Domain of Validity
/// Finite noisy latent matrices `latent` matching the diffusion spatial resolution, positive
/// integer `timestep` $t \in \{1, \dots, T\}$, and finite conditioning matrix `text_embeddings`.
pub trait DiffusionModel {
    /// Predicts noise given noisy latent, timestep, and text embeddings.
    /// Returns predicted noise epsilon.
    #[verified_engine::verified]
    fn predict_noise(
        &self,
        latent: &DMatrix<f64>,
        timestep: usize,
        text_embeddings: &DMatrix<f64>,
    ) -> DMatrix<f64>;
}

/// Module 3.3: Classifier-Free Guidance (CFG)
/// Input: Predicted Noise with text (cond), Predicted Noise without text (uncond).
/// Operation: Amplify the signal that aligns with the text.
/// s is the guidance scale.
/// Output: Guided Noise Prediction.
#[verified_engine::verified]
pub fn classifier_free_guidance(
    noise_cond: &DMatrix<f64>,
    noise_uncond: &DMatrix<f64>,
    guidance_scale: f64,
) -> DMatrix<f64> {
    noise_uncond + (noise_cond - noise_uncond) * guidance_scale
}

#[cfg(test)]
#[path = "tests_score.rs"]
mod tests;
