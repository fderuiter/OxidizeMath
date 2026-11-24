use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::{Distribution, Normal};

const DEFAULT_BETA_START: f64 = 0.0001;
const DEFAULT_BETA_END: f64 = 0.02;

/// Module 2.1: Time Sampling
/// Input: None.
/// Operation: Select a random integer timestep t.
/// Output: Timestep scalar t.
#[verified_engine::verified]
pub fn sample_timestep(max_timesteps: usize) -> usize {
    let mut rng = oxidize_core::rng::OxidizeRng::default();
    sample_timestep_with_rng(max_timesteps, &mut rng)
}

/// Module 2.1: Time Sampling using an injected RNG.
#[verified_engine::verified]
pub fn sample_timestep_with_rng<R: Rng + ?Sized>(max_timesteps: usize, rng: &mut R) -> usize {
    rng.gen_range(1..=max_timesteps)
}

/// Module 2.2: Noise Schedule Lookup
/// Input: Timestep t.
/// Operation: Retrieve pre-defined noise variance coefficients beta_t and calculate cumulative product alpha_bar_t.
/// Output: Signal scale sqrt(alpha_bar_t) and Noise scale sqrt(1 - alpha_bar_t).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct NoiseSchedule {
    #[allow(missing_docs)]
    pub betas: Vec<f64>,
    #[allow(missing_docs)]
    pub alphas: Vec<f64>,
    #[allow(missing_docs)]
    pub alpha_bars: Vec<f64>,
}

impl NoiseSchedule {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(timesteps: usize) -> Self {
        // Linear schedule for betas as an example (e.g. 1e-4 to 0.02)
        let beta_start = DEFAULT_BETA_START;
        let beta_end = DEFAULT_BETA_END;
        let mut betas = Vec::with_capacity(timesteps);
        let mut alphas = Vec::with_capacity(timesteps);
        let mut alpha_bars = Vec::with_capacity(timesteps);
        let mut alpha_bar_acc = 1.0;

        for i in 0..timesteps {
            let t_ratio = i as f64 / (timesteps - 1) as f64;
            let beta = beta_start + t_ratio * (beta_end - beta_start);
            let alpha = 1.0 - beta;
            alpha_bar_acc *= alpha;

            betas.push(beta);
            alphas.push(alpha);
            alpha_bars.push(alpha_bar_acc);
        }

        Self {
            betas,
            alphas,
            alpha_bars,
        }
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn get_scales(&self, t: usize) -> (f64, f64) {
        // t is 1-indexed in the prompt, let's assume 0-indexed internally or handle offset.
        // Prompt: t ~ U{1...T}. Let's say index t-1.
        let idx = if t == 0 { 0 } else { t - 1 };
        let idx = idx.min(self.alpha_bars.len() - 1);

        let alpha_bar = self.alpha_bars[idx];
        let signal_scale = alpha_bar.sqrt();
        let noise_scale = (1.0 - alpha_bar).sqrt();

        (signal_scale, noise_scale)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for NoiseSchedule {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawNoiseSchedule {
            betas: Vec<f64>,
            alphas: Vec<f64>,
            alpha_bars: Vec<f64>,
        }
        let raw = RawNoiseSchedule::deserialize(deserializer)?;
        if raw.betas.is_empty()
            || raw.betas.len() != raw.alphas.len()
            || raw.betas.len() != raw.alpha_bars.len()
        {
            return Err(serde::de::Error::custom(
                "NoiseSchedule vectors must be non-empty and equal length",
            ));
        }
        if raw.betas.iter().any(|x| !x.is_finite())
            || raw.alphas.iter().any(|x| !x.is_finite())
            || raw
                .alpha_bars
                .iter()
                .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
        {
            return Err(serde::de::Error::custom(
                "NoiseSchedule entries must be finite with alpha_bars in [0, 1]",
            ));
        }
        Ok(Self {
            betas: raw.betas,
            alphas: raw.alphas,
            alpha_bars: raw.alpha_bars,
        })
    }
}

/// Module 2.3: Noise Injection
/// Input: Rendered Image x_render, Random Gaussian Noise epsilon ~ N(0, I).
/// Operation: Linear combination (interpolation).
/// Output: Noisy Latent Image z_t (scaled to diffusion model resolution).
#[verified_engine::verified]
pub fn inject_noise(
    image: &DMatrix<f64>,
    noise: &DMatrix<f64>,
    signal_scale: f64,
    noise_scale: f64,
) -> DMatrix<f64> {
    image * signal_scale + noise * noise_scale
}

/// Helper to generate Gaussian noise matching image dimensions
#[verified_engine::verified]
pub fn generate_noise(rows: usize, cols: usize) -> Result<DMatrix<f64>, crate::error::AIError> {
    let mut rng = oxidize_core::rng::OxidizeRng::default();
    generate_noise_with_rng(rows, cols, &mut rng)
}

/// Helper to generate Gaussian noise matching image dimensions using an injected RNG.
#[verified_engine::verified]
pub fn generate_noise_with_rng<R: Rng + ?Sized>(
    rows: usize,
    cols: usize,
    rng: &mut R,
) -> Result<DMatrix<f64>, crate::error::AIError> {
    let normal =
        Normal::new(0.0, 1.0).map_err(|e| math_commons::error::MathError::ConversionError {
            reason: format!("Invalid normal distribution: {}", e),
        })?;
    Ok(DMatrix::from_fn(rows, cols, |_, _| normal.sample(rng)))
}

#[cfg(test)]
#[path = "tests_diffusion.rs"]
mod tests;
