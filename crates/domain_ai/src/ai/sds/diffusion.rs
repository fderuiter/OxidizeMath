use engine_diffeq::{EulerMaruyamaSolver, SdeSystem, StochasticCalculus};
use math_commons::math_kernel::types::{SolverWorkspace, fill_standard_normal};
use nalgebra::DMatrix;
use rand::Rng;

struct VariancePreservingPixelSde {
    beta: f64,
}

impl SdeSystem<f64> for VariancePreservingPixelSde {
    fn state_dim(&self) -> usize {
        1
    }
    fn wiener_dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out_drift: &mut [f64]) {
        out_drift[0] = -0.5 * self.beta * x[0];
    }
    fn diffusion(&self, _t: f64, _x: &[f64], out_diffusion: &mut [f64]) {
        out_diffusion[0] = self.beta.max(0.0).sqrt();
    }
}

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

/// Helper to generate Gaussian noise matching image dimensions using Tier 0 Box-Muller over `RngCore`.
#[verified_engine::verified]
pub fn generate_noise_with_rng<R: Rng + ?Sized>(
    rows: usize,
    cols: usize,
    rng: &mut R,
) -> Result<DMatrix<f64>, crate::error::AIError> {
    let mut mat = DMatrix::zeros(rows, cols);
    fill_standard_normal(mat.as_mut_slice(), rng);
    debug_assert_eq!(mat.nrows(), rows);
    Ok(mat)
}

/// Advances latent pixels in-place by one variance-preserving SDE step
/// $dx = -\frac{1}{2}\beta x\,dt + \sqrt{\beta}\,dW_t$ via [`engine_diffeq::EulerMaruyamaSolver`].
#[verified_engine::verified]
pub fn step_vp_sde_in_place<R: Rng + ?Sized>(
    latent: &mut DMatrix<f64>,
    beta: f64,
    dt: f64,
    rng: &mut R,
) {
    assert!(beta >= 0.0 && dt >= 0.0);
    let sde = VariancePreservingPixelSde { beta };
    let solver = EulerMaruyamaSolver::new(StochasticCalculus::Ito);
    let mut scratch = [0.0_f64; 4];
    let mut ws = SolverWorkspace::new(&mut scratch);
    for px in latent.as_mut_slice() {
        let mut state = [*px];
        solver.step(&sde, 0.0, dt, &mut state, &mut ws, rng);
        *px = state[0];
    }
}

#[cfg(test)]
#[path = "tests_diffusion.rs"]
mod tests;
