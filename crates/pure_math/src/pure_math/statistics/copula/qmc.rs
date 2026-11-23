//! Zero-allocation Genz (1992) separation-of-variables quasi-Monte Carlo integration
//! for multivariate Gaussian copulas.

use super::core::{CorrelationMatrix, Probability};
use super::transforms::{fast_norm_cdf, fast_norm_inv};
use crate::error::CopulaError;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

/// Maximum copula dimension handled purely on the stack with zero heap allocations.
pub const MAX_INLINE_COPULA_LEGS: usize = 12;

/// Default number of antithetic Halton QMC pairs (`2,048` total integrand evaluations).
pub const DEFAULT_COPULA_QMC_PAIRS: usize = 1_024;

const HALTON_PRIMES: [usize; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
const PROB_CLAMP_EPS: f64 = 1e-15;

/// Preallocated workspace for arbitrary-dimension (`d > 12`) zero-allocation Gaussian copula QMC.
#[derive(Debug, Clone)]
pub struct CopulaQmcWorkspace {
    z_scores: Vec<f64>,
    y_state: Vec<f64>,
    shifts: Vec<f64>,
}

impl CopulaQmcWorkspace {
    /// Preallocates scratch buffers for a copula with up to `max_legs` dimensions.
    #[verified_engine::verified]
    pub fn with_capacity(max_legs: usize) -> Self {
        let cap = max_legs.max(1);
        Self {
            z_scores: vec![0.0; cap],
            y_state: vec![0.0; cap],
            shifts: vec![0.0; cap],
        }
    }

    /// Ensures the workspace can hold at least `dim` legs (allocates only if growing).
    #[verified_engine::verified]
    pub fn ensure_capacity(&mut self, dim: usize) {
        if self.z_scores.len() < dim {
            self.z_scores.resize(dim, 0.0);
            self.y_state.resize(dim, 0.0);
            self.shifts.resize(dim, 0.0);
        }
    }
}

#[inline]
fn halton_radical_inverse(mut index: usize, base: usize) -> f64 {
    let inv_base = 1.0 / (base as f64);
    let mut factor = inv_base;
    let mut result = 0.0;
    while index > 0 {
        result += (index % base) as f64 * factor;
        index /= base;
        factor *= inv_base;
    }
    result
}

#[inline]
fn quasi_uniform_coord(sample_idx: usize, dim_idx: usize, shift: f64) -> f64 {
    let base = if dim_idx < HALTON_PRIMES.len() {
        HALTON_PRIMES[dim_idx]
    } else {
        // Golden-ratio Kronecker fallback for dimensions > 17
        let alpha = ((dim_idx as f64 + 1.0) * 0.618_033_988_749_894_9).fract();
        return ((sample_idx as f64) * alpha + shift).fract();
    };
    (halton_radical_inverse(sample_idx, base) + shift).fract()
}

fn fill_cranley_patterson_shifts(shifts: &mut [f64], seed: u64) {
    let mut rng = OxidizeRng::new(seed);
    let scale = 1.0 / ((u64::MAX as f64) + 1.0);
    for slot in shifts.iter_mut() {
        *slot = (rng.next_u64() as f64) * scale;
    }
}

fn eval_single_genz_point(
    chol_l: &nalgebra::DMatrix<f64>,
    z_scores: &[f64],
    shifts: &[f64],
    y_state: &mut [f64],
    sample_idx: usize,
    antithetic: bool,
    e0: f64,
) -> f64 {
    let dim = z_scores.len();
    let mut prod = e0;
    for i in 0..(dim - 1) {
        let mut u_q = quasi_uniform_coord(sample_idx, i, shifts[i]);
        if antithetic {
            u_q = 1.0 - u_q;
        }
        let ei = if i == 0 {
            e0
        } else {
            let mut dot = 0.0;
            for (j, &yj) in y_state.iter().enumerate().take(i) {
                dot += chol_l[(i, j)] * yj;
            }
            let diag = chol_l[(i, i)];
            let val = if diag > 0.0 {
                fast_norm_cdf((z_scores[i] - dot) / diag)
            } else if dot <= z_scores[i] {
                1.0
            } else {
                0.0
            };
            prod *= val;
            val
        };
        if prod <= 0.0 {
            return 0.0;
        }
        if chol_l[(i, i)] > 0.0 {
            let p = (u_q * ei).clamp(PROB_CLAMP_EPS, 1.0 - PROB_CLAMP_EPS);
            y_state[i] = fast_norm_inv(p);
        } else {
            y_state[i] = 0.0;
        }
    }

    let last = dim - 1;
    let mut last_dot = 0.0;
    for (j, &yj) in y_state.iter().enumerate().take(last) {
        last_dot += chol_l[(last, j)] * yj;
    }
    let last_diag = chol_l[(last, last)];
    let last_prob = if last_diag > 0.0 {
        fast_norm_cdf((z_scores[last] - last_dot) / last_diag)
    } else if last_dot <= z_scores[last] {
        1.0
    } else {
        0.0
    };
    prod * last_prob
}

fn integrate_genz_slices(
    corr: &CorrelationMatrix,
    u: &[Probability],
    seed: u64,
    n_pairs: usize,
    z_scores: &mut [f64],
    y_state: &mut [f64],
    shifts: &mut [f64],
) -> Result<Probability, CopulaError> {
    let dim = u.len();
    if dim == 0 {
        return Probability::new(0.0);
    }
    if u.iter().any(|p| p.value() <= 0.0) {
        return Probability::new(0.0);
    }
    if u.iter().all(|p| p.value() >= 1.0) {
        return Probability::new(1.0);
    }
    if dim == 1 {
        return Ok(u[0]);
    }

    for (dst, &ui) in z_scores.iter_mut().zip(u.iter()) {
        *dst = if ui.value() >= 1.0 - PROB_CLAMP_EPS {
            8.5
        } else {
            fast_norm_inv(ui.value())
        };
    }

    let pairs = n_pairs.max(1);
    fill_cranley_patterson_shifts(&mut shifts[..dim - 1], seed);
    let chol_l = corr.cholesky_factor();
    let e0 = if chol_l[(0, 0)] > 0.0 {
        fast_norm_cdf(z_scores[0] / chol_l[(0, 0)])
    } else if z_scores[0] >= 0.0 {
        1.0
    } else {
        0.0
    };

    let mut total = 0.0;
    for m in 1..=pairs {
        total += eval_single_genz_point(chol_l, z_scores, shifts, y_state, m, false, e0);
        total += eval_single_genz_point(chol_l, z_scores, shifts, y_state, m, true, e0);
    }
    let prob = (total / (2.0 * pairs as f64)).clamp(0.0, 1.0);
    Probability::new(prob)
}

/// Evaluates the multivariate Gaussian copula CDF using stack buffers for `d <= 12`
/// (zero heap allocations) and deterministic `BASE_SEED` Cranley-Patterson Halton QMC.
pub(crate) fn multivariate_cdf_qmc(
    corr: &CorrelationMatrix,
    u: &[Probability],
    seed: u64,
    n_pairs: usize,
) -> Result<Probability, CopulaError> {
    let dim = u.len();
    if dim <= MAX_INLINE_COPULA_LEGS {
        let mut z_buf = [0.0_f64; MAX_INLINE_COPULA_LEGS];
        let mut y_buf = [0.0_f64; MAX_INLINE_COPULA_LEGS];
        let mut shift_buf = [0.0_f64; MAX_INLINE_COPULA_LEGS];
        integrate_genz_slices(
            corr,
            u,
            seed,
            n_pairs,
            &mut z_buf[..dim],
            &mut y_buf[..dim],
            &mut shift_buf[..dim],
        )
    } else {
        let mut ws = CopulaQmcWorkspace::with_capacity(dim);
        multivariate_cdf_with_workspace(corr, u, seed, n_pairs, &mut ws)
    }
}

/// Evaluates the multivariate Gaussian copula CDF using a caller-supplied preallocated workspace.
pub(crate) fn multivariate_cdf_with_workspace(
    corr: &CorrelationMatrix,
    u: &[Probability],
    seed: u64,
    n_pairs: usize,
    workspace: &mut CopulaQmcWorkspace,
) -> Result<Probability, CopulaError> {
    let dim = u.len();
    if workspace.z_scores.len() < dim {
        return Err(CopulaError::Math(
            math_commons::error::MathError::DimensionMismatch {
                expected: math_commons::math_kernel::types::Dimension(dim),
                actual: math_commons::math_kernel::types::Dimension(workspace.z_scores.len()),
            },
        ));
    }
    integrate_genz_slices(
        corr,
        u,
        seed,
        n_pairs,
        &mut workspace.z_scores[..dim],
        &mut workspace.y_state[..dim],
        &mut workspace.shifts[..dim],
    )
}

// [cite:clinical_trials_statistics]
