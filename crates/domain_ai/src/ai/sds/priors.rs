//! Analytical prompt encoder and Tweedie Gaussian diffusion score prior for SDS.

use crate::ai::sds::diffusion::NoiseSchedule;
use crate::ai::sds::score::{DiffusionModel, TextEncoder};
use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::{DMatrix, Vector3};

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Deterministic prompt-to-prototype encoder implementing [`TextEncoder`].
///
/// Produces a `(1, 8)` embedding `[cond_flag, w_sphere, w_torus, w_octa, r, g, b, radius]`.
#[derive(Debug, Clone, Default)]
pub struct PromptPrototypeEncoder;

impl PromptPrototypeEncoder {
    /// Creates a new deterministic [`PromptPrototypeEncoder`].
    pub fn new() -> Self {
        Self
    }
}

impl TextEncoder for PromptPrototypeEncoder {
    #[verified_engine::verified]
    fn encode(&self, text: &str) -> DMatrix<f64> {
        let clean = text.trim().to_ascii_lowercase();
        let mut emb = DMatrix::zeros(1, 8);
        if clean.is_empty() {
            emb[(0, 4)] = 0.12;
            emb[(0, 5)] = 0.12;
            emb[(0, 6)] = 0.14;
            emb[(0, 7)] = 0.50;
            assert_eq!(emb.ncols(), 8);
            return emb;
        }
        emb[(0, 0)] = 1.0;
        let is_torus = clean.contains("torus") || clean.contains("ring") || clean.contains("donut");
        let is_octa =
            clean.contains("octa") || clean.contains("diamond") || clean.contains("crystal");
        let is_twin =
            clean.contains("twin") || clean.contains("lobe") || clean.contains("dumbbell");
        if is_torus {
            emb[(0, 2)] = 1.0;
            emb[(0, 4)] = 0.92;
            emb[(0, 5)] = 0.52;
            emb[(0, 6)] = 0.20;
            emb[(0, 7)] = 0.52;
        } else if is_octa {
            emb[(0, 3)] = 1.0;
            emb[(0, 4)] = 0.25;
            emb[(0, 5)] = 0.82;
            emb[(0, 6)] = 0.92;
            emb[(0, 7)] = 0.56;
        } else if is_twin {
            emb[(0, 1)] = 0.55;
            emb[(0, 2)] = 0.45;
            emb[(0, 4)] = 0.86;
            emb[(0, 5)] = 0.34;
            emb[(0, 6)] = 0.78;
            emb[(0, 7)] = 0.54;
        } else {
            let hash: u32 = clean.bytes().fold(2166136261_u32, |acc, b| {
                acc.wrapping_mul(16777619) ^ (b as u32)
            });
            emb[(0, 1)] = 1.0;
            emb[(0, 4)] = if clean.contains("amber") || clean.contains("gold") {
                0.94
            } else {
                0.35 + 0.55 * (((hash & 0xFF) as f64) / 255.0)
            };
            emb[(0, 5)] = 0.35 + 0.50 * ((((hash >> 8) & 0xFF) as f64) / 255.0);
            emb[(0, 6)] = 0.45 + 0.45 * ((((hash >> 16) & 0xFF) as f64) / 255.0);
            emb[(0, 7)] = 0.52;
        }
        assert_eq!(emb.ncols(), 8);
        emb
    }
}

/// Closed-form Tweedie Gaussian diffusion score prior implementing [`DiffusionModel`].
#[derive(Debug, Clone)]
pub struct AnalyticalScorePrior {
    width: usize,
    height: usize,
    schedule: NoiseSchedule,
    cond_variance: f64,
    uncond_variance: f64,
}

impl AnalyticalScorePrior {
    /// Creates an analytical diffusion score critic for `(width, height)` ray grids.
    #[verified_engine::verified]
    pub fn new(width: usize, height: usize, timesteps: usize) -> Self {
        assert!(width >= 1 && height >= 1, "image grid must be non-empty");
        assert!(timesteps >= 2, "timesteps must be at least 2");
        Self {
            width,
            height,
            schedule: NoiseSchedule::new(timesteps),
            cond_variance: 0.08,
            uncond_variance: 4.0,
        }
    }

    /// Borrows the internal [`NoiseSchedule`].
    pub fn schedule(&self) -> &NoiseSchedule {
        &self.schedule
    }

    /// Evaluates the clean 2D target RGB latent `(width * height, 3)` implied by `text_embeddings`.
    #[verified_engine::verified]
    pub fn target_latent(&self, text_embeddings: &DMatrix<f64>) -> DMatrix<f64> {
        assert!(text_embeddings.ncols() >= 8);
        let num_pixels = self.width * self.height;
        let mut target = DMatrix::zeros(num_pixels, 3);
        let cond = text_embeddings[(0, 0)].clamp(0.0, 1.0);
        let w_sphere = text_embeddings[(0, 1)];
        let w_torus = text_embeddings[(0, 2)];
        let w_octa = text_embeddings[(0, 3)];
        let tint = Vector3::new(
            text_embeddings[(0, 4)],
            text_embeddings[(0, 5)],
            text_embeddings[(0, 6)],
        );
        let radius = text_embeddings[(0, 7)].clamp(0.25, 0.85);

        for x in 0..self.width {
            for y in 0..self.height {
                let u = 2.0 * ((x as f64 + 0.5) / (self.width as f64)) - 1.0;
                let v = 1.0 - 2.0 * ((y as f64 + 0.5) / (self.height as f64));
                let r = (u * u + v * v).sqrt();
                let occ_sphere = sigmoid(10.0 * (radius - r));
                let ring_dist = (r - 0.72 * radius).abs();
                let occ_torus = sigmoid(14.0 * (0.24 * radius - ring_dist));
                let l1_norm = u.abs() + v.abs();
                let occ_octa = sigmoid(10.0 * (1.12 * radius - l1_norm));
                let w_sum = (w_sphere + w_torus + w_octa).max(1e-6);
                let mask = cond
                    * ((w_sphere * occ_sphere + w_torus * occ_torus + w_octa * occ_octa) / w_sum);
                let rim = (1.0 - 0.35 * r * r).clamp(0.45, 1.0);
                let bg = Vector3::new(0.04, 0.05, 0.07);
                let px = bg * (1.0 - mask) + tint * (mask * rim);
                let idx = flatten_2d_index(y, x, self.height);
                target[(idx, 0)] = px.x;
                target[(idx, 1)] = px.y;
                target[(idx, 2)] = px.z;
            }
        }
        target
    }
}

impl DiffusionModel for AnalyticalScorePrior {
    #[verified_engine::verified]
    fn predict_noise(
        &self,
        latent: &DMatrix<f64>,
        timestep: usize,
        text_embeddings: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        assert_eq!(latent.ncols(), 3);
        let (signal_scale, noise_scale) = self.schedule.get_scales(timestep);
        let alpha_bar = signal_scale * signal_scale;
        let one_minus_alpha_bar = noise_scale * noise_scale;
        let is_cond = text_embeddings.ncols() >= 8 && text_embeddings[(0, 0)] > 0.5;
        let prior_var = if is_cond {
            self.cond_variance
        } else {
            self.uncond_variance
        };
        let denom = (alpha_bar * prior_var + one_minus_alpha_bar).max(1e-8);
        let coeff = noise_scale / denom;
        let target = if latent.nrows() == self.width * self.height && text_embeddings.ncols() >= 8 {
            self.target_latent(text_embeddings)
        } else {
            DMatrix::from_element(latent.nrows(), 3, 0.05)
        };
        (latent - target * signal_scale) * coeff
    }
}
