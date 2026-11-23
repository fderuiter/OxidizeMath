//! Preallocated zero-allocation Monte Carlo and variance-reduction workspace.

use super::error::MonteCarloError;
use super::stats::{MonteCarloEstimate, effective_sample_size, two_sided_normal_z};
use crate::pure_math::statistics::qmc::{
    HaltonSequence, MAX_HALTON_DIM, MAX_SOBOL_DIM, SobolSequence,
};
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

/// Guard margin keeping uniform samples strictly inside `(0.0, 1.0)`.
const OPEN_EPS: f64 = 1e-15;

/// Scale factor converting a 53-bit integer draw into `[0.0, 1.0)`.
const INV_TWO_POW_53: f64 = 1.0 / ((1_u64 << 53) as f64);

/// Number of Cranley-Patterson replicates used for randomized QMC confidence intervals.
pub(super) const DEFAULT_QMC_REPLICATES: usize = 8;

/// Draws a uniform value strictly inside `(0.0, 1.0)` from `rng` without heap allocation.
#[inline]
#[verified_engine::verified]
fn draw_open_unit(rng: &mut OxidizeRng) -> f64 {
    let u = ((rng.next_u64() >> 11) as f64) * INV_TWO_POW_53;
    u.clamp(OPEN_EPS, 1.0 - OPEN_EPS)
}

/// Draws a uniform integer in `0..upper` without modulo bias or heap allocation.
#[inline]
#[verified_engine::verified]
pub(crate) fn draw_bounded_index(rng: &mut OxidizeRng, upper: usize) -> usize {
    if upper <= 1 {
        return 0;
    }
    let bound = upper as u64;
    let zone = u64::MAX - (u64::MAX % bound);
    loop {
        let candidate = rng.next_u64();
        if candidate < zone {
            return (candidate % bound) as usize;
        }
    }
}

/// Variance-reduction and sampling strategy selector for [`MonteCarloWorkspace`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingMethod {
    /// Standard independent and identically distributed uniform sampling.
    Iid,
    /// Paired $(u, 1 - u)$ antithetic variates.
    Antithetic,
    /// Stratified Latin Hypercube sampling across every coordinate axis.
    LatinHypercube,
    /// Cranley-Patterson randomized Halton quasi-Monte Carlo.
    RandomizedHalton,
    /// Cranley-Patterson randomized Joe-Kuo Sobol quasi-Monte Carlo.
    RandomizedSobol,
}

/// Preallocated workspace executing IID, antithetic, control-variate, importance-sampling,
/// Latin Hypercube, and randomized QMC estimators with zero hot-loop heap allocations.
#[derive(Debug, Clone)]
pub struct MonteCarloWorkspace {
    pub(super) max_samples: usize,
    pub(super) dimension: usize,
    pub(super) confidence_level: f64,
    pub(super) point_buf: Vec<f64>,
    pub(super) anti_point_buf: Vec<f64>,
    pub(super) values_buf: Vec<f64>,
    pub(super) control_buf: Vec<f64>,
    pub(super) weights_buf: Vec<f64>,
    pub(super) lhs_grid: Vec<f64>,
    pub(super) perm_buf: Vec<usize>,
    pub(super) batch_means_buf: Vec<f64>,
    pub(super) halton: Option<HaltonSequence>,
    pub(super) sobol: Option<SobolSequence>,
}

impl MonteCarloWorkspace {
    /// Preallocates all sample, weight, permutation, and QMC buffers for `max_samples` and `dimension`.
    #[verified_engine::verified]
    pub fn new(max_samples: usize, dimension: usize) -> Result<Self, MonteCarloError> {
        if max_samples == 0 {
            return Err(MonteCarloError::InvalidSampleCount { samples: 0 });
        }
        if dimension == 0 || dimension > MAX_HALTON_DIM {
            return Err(MonteCarloError::InvalidDimension {
                dimension,
                max_dimension: MAX_HALTON_DIM,
            });
        }
        let grid_len =
            max_samples
                .checked_mul(dimension)
                .ok_or(MonteCarloError::InvalidSampleCount {
                    samples: max_samples,
                })?;
        let halton = HaltonSequence::new(dimension).ok();
        let sobol = if dimension <= MAX_SOBOL_DIM {
            SobolSequence::new(dimension).ok()
        } else {
            None
        };
        Ok(Self {
            max_samples,
            dimension,
            confidence_level: 0.95,
            point_buf: vec![0.0; dimension],
            anti_point_buf: vec![0.0; dimension],
            values_buf: vec![0.0; max_samples],
            control_buf: vec![0.0; max_samples],
            weights_buf: vec![0.0; max_samples],
            lhs_grid: vec![0.0; grid_len],
            perm_buf: (0..max_samples).collect(),
            batch_means_buf: vec![0.0; DEFAULT_QMC_REPLICATES],
            halton,
            sobol,
        })
    }

    /// Configures the nominal confidence level in `(0.0, 1.0)` via builder pattern.
    #[verified_engine::verified]
    pub fn with_confidence_level(mut self, confidence_level: f64) -> Result<Self, MonteCarloError> {
        self.set_confidence_level(confidence_level)?;
        Ok(self)
    }

    /// Updates the nominal confidence level in `(0.0, 1.0)`.
    #[verified_engine::verified]
    pub fn set_confidence_level(&mut self, confidence_level: f64) -> Result<(), MonteCarloError> {
        let _ = two_sided_normal_z(confidence_level)?;
        self.confidence_level = confidence_level;
        Ok(())
    }

    /// Maximum sample count preallocated in this workspace.
    #[inline]
    #[verified_engine::verified]
    pub fn max_samples(&self) -> usize {
        self.max_samples
    }

    /// Coordinate dimensionality of each sample point.
    #[inline]
    #[verified_engine::verified]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    #[inline]
    #[verified_engine::verified]
    pub(super) fn check_budget(&self, n_samples: usize) -> Result<(), MonteCarloError> {
        if n_samples == 0 || n_samples > self.max_samples {
            return Err(MonteCarloError::InvalidSampleCount { samples: n_samples });
        }
        Ok(())
    }

    /// Dispatches an integrand over `(0, 1)^d` using the selected [`SamplingMethod`].
    #[verified_engine::verified]
    pub fn estimate_with_method<F>(
        &mut self,
        method: SamplingMethod,
        n_samples: usize,
        rng: &mut OxidizeRng,
        integrand: F,
    ) -> Result<MonteCarloEstimate, MonteCarloError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        match method {
            SamplingMethod::Iid => self.estimate_iid(n_samples, rng, integrand),
            SamplingMethod::Antithetic => self.estimate_antithetic(n_samples, rng, integrand),
            SamplingMethod::LatinHypercube => {
                self.estimate_latin_hypercube(n_samples, rng, integrand)
            }
            SamplingMethod::RandomizedHalton => self.estimate_qmc_halton(n_samples, rng, integrand),
            SamplingMethod::RandomizedSobol => self.estimate_qmc_sobol(n_samples, rng, integrand),
        }
    }

    /// Standard IID Monte Carlo expectation over `(0, 1)^d`.
    #[verified_engine::verified]
    pub fn estimate_iid<F>(
        &mut self,
        n_samples: usize,
        rng: &mut OxidizeRng,
        mut integrand: F,
    ) -> Result<MonteCarloEstimate, MonteCarloError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        self.check_budget(n_samples)?;
        for idx in 0..n_samples {
            for coord in &mut self.point_buf {
                *coord = draw_open_unit(rng);
            }
            let val = integrand(&self.point_buf);
            if !val.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            self.values_buf[idx] = val;
        }
        MonteCarloEstimate::from_samples(&self.values_buf[..n_samples], self.confidence_level)
    }

    /// Paired antithetic-variates estimator evaluating `(u, 1 - u)` pairs within `n_samples` budget.
    #[verified_engine::verified]
    pub fn estimate_antithetic<F>(
        &mut self,
        n_samples: usize,
        rng: &mut OxidizeRng,
        mut integrand: F,
    ) -> Result<MonteCarloEstimate, MonteCarloError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        self.check_budget(n_samples)?;
        if n_samples < 2 {
            return Err(MonteCarloError::InvalidSampleCount { samples: n_samples });
        }
        let n_pairs = n_samples / 2;
        for pair_idx in 0..n_pairs {
            for (u_slot, anti_slot) in self.point_buf.iter_mut().zip(&mut self.anti_point_buf) {
                let u = draw_open_unit(rng);
                *u_slot = u;
                *anti_slot = (1.0 - u).clamp(OPEN_EPS, 1.0 - OPEN_EPS);
            }
            let y1 = integrand(&self.point_buf);
            let y2 = integrand(&self.anti_point_buf);
            if !y1.is_finite() || !y2.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            self.values_buf[pair_idx] = 0.5 * (y1 + y2);
        }
        let mut est =
            MonteCarloEstimate::from_samples(&self.values_buf[..n_pairs], self.confidence_level)?;
        est.samples = 2 * n_pairs;
        est.effective_sample_size = (2 * n_pairs) as f64;
        Ok(est)
    }

    /// Optimal-$\hat{\beta}^*$ control-variate estimator given `(f(u), g(u))` and `control_mean` $= \mathbb{E}[g(U)]$.
    #[verified_engine::verified]
    pub fn estimate_control_variate<F>(
        &mut self,
        n_samples: usize,
        control_mean: f64,
        rng: &mut OxidizeRng,
        mut eval_pair: F,
    ) -> Result<MonteCarloEstimate, MonteCarloError>
    where
        F: FnMut(&[f64]) -> (f64, f64),
    {
        self.check_budget(n_samples)?;
        if !control_mean.is_finite() {
            return Err(MonteCarloError::NonFiniteEvaluation);
        }
        for idx in 0..n_samples {
            for coord in &mut self.point_buf {
                *coord = draw_open_unit(rng);
            }
            let (f_val, g_val) = eval_pair(&self.point_buf);
            if !f_val.is_finite() || !g_val.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            self.values_buf[idx] = f_val;
            self.control_buf[idx] = g_val;
        }
        let beta = optimal_control_beta(
            &self.values_buf[..n_samples],
            &self.control_buf[..n_samples],
        );
        for (f_slot, &g_val) in self.values_buf[..n_samples]
            .iter_mut()
            .zip(&self.control_buf[..n_samples])
        {
            *f_slot -= beta * (g_val - control_mean);
        }
        MonteCarloEstimate::from_samples(&self.values_buf[..n_samples], self.confidence_level)
    }

    /// Self-normalized importance sampling returning calibrated SE and Kong's exact `ESS`.
    ///
    /// `sampler` maps a uniform draw $u \in (0, 1)^d$ to `(f(x), unnormalized_weight w(x))`.
    #[verified_engine::verified]
    pub fn estimate_importance_sampling<S>(
        &mut self,
        n_samples: usize,
        rng: &mut OxidizeRng,
        mut sampler: S,
    ) -> Result<MonteCarloEstimate, MonteCarloError>
    where
        S: FnMut(&[f64]) -> (f64, f64),
    {
        self.check_budget(n_samples)?;
        let z = two_sided_normal_z(self.confidence_level)?;
        for idx in 0..n_samples {
            for coord in &mut self.point_buf {
                *coord = draw_open_unit(rng);
            }
            let (val, weight) = sampler(&self.point_buf);
            if !val.is_finite() || !weight.is_finite() || weight < 0.0 {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            self.values_buf[idx] = val;
            self.weights_buf[idx] = weight;
        }
        let weights = &self.weights_buf[..n_samples];
        let values = &self.values_buf[..n_samples];
        let ess = effective_sample_size(weights)?;
        let sum_w: f64 = weights.iter().sum();
        let mean: f64 = values
            .iter()
            .zip(weights)
            .map(|(&v, &w)| w * v)
            .sum::<f64>()
            / sum_w;
        let se = snis_standard_error(values, weights, sum_w, mean, ess);
        let half = z * se;
        Ok(MonteCarloEstimate {
            mean,
            variance: se * se * ess,
            standard_error: se,
            ci_lower: mean - half,
            ci_upper: mean + half,
            confidence_level: self.confidence_level,
            samples: n_samples,
            effective_sample_size: ess,
        })
    }

    /// Stratified Latin Hypercube sampling across `(0, 1)^d` using preallocated Fisher-Yates permutations.
    #[verified_engine::verified]
    pub fn estimate_latin_hypercube<F>(
        &mut self,
        n_samples: usize,
        rng: &mut OxidizeRng,
        mut integrand: F,
    ) -> Result<MonteCarloEstimate, MonteCarloError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        self.check_budget(n_samples)?;
        let dim = self.dimension;
        let inv_n = 1.0 / (n_samples as f64);
        let active_grid = &mut self.lhs_grid[..n_samples * dim];

        for axis in 0..dim {
            for idx in 0..n_samples {
                self.perm_buf[idx] = idx;
            }
            for idx in (1..n_samples).rev() {
                let swap_idx = draw_bounded_index(rng, idx + 1);
                self.perm_buf.swap(idx, swap_idx);
            }
            for (row_idx, point_row) in active_grid.chunks_exact_mut(dim).enumerate() {
                let stratum = self.perm_buf[row_idx] as f64;
                let jitter = draw_open_unit(rng);
                point_row[axis] = ((stratum + jitter) * inv_n).clamp(OPEN_EPS, 1.0 - OPEN_EPS);
            }
        }

        for (idx, point_row) in active_grid.chunks_exact(dim).enumerate() {
            let val = integrand(point_row);
            if !val.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            self.values_buf[idx] = val;
        }
        MonteCarloEstimate::from_samples(&self.values_buf[..n_samples], self.confidence_level)
    }
}

/// Computes the empirical optimal control-variate coefficient $\hat{\beta}^* = \text{Cov}(f, g) / \text{Var}(g)$.
#[verified_engine::verified]
fn optimal_control_beta(f_vals: &[f64], g_vals: &[f64]) -> f64 {
    let n = f_vals.len();
    if n < 2 {
        return 0.0;
    }
    let inv_n = 1.0 / (n as f64);
    let mean_f = f_vals.iter().sum::<f64>() * inv_n;
    let mean_g = g_vals.iter().sum::<f64>() * inv_n;
    let mut cov = 0.0;
    let mut var_g = 0.0;
    for (&fv, &gv) in f_vals.iter().zip(g_vals) {
        let df = fv - mean_f;
        let dg = gv - mean_g;
        cov += df * dg;
        var_g += dg * dg;
    }
    if var_g > 1e-30 { cov / var_g } else { 0.0 }
}

/// Computes Bessel-corrected delta-method SE for self-normalized importance sampling.
#[verified_engine::verified]
fn snis_standard_error(values: &[f64], weights: &[f64], sum_w: f64, mean: f64, ess: f64) -> f64 {
    if values.len() < 2 || ess <= 1.0 {
        return 0.0;
    }
    let inv_sum_w = 1.0 / sum_w;
    let mut weighted_sq = 0.0;
    for (&v, &w) in values.iter().zip(weights) {
        let nw = w * inv_sum_w;
        let diff = v - mean;
        weighted_sq += nw * nw * diff * diff;
    }
    ((ess / (ess - 1.0)) * weighted_sq).sqrt()
}

// [cite:clinical_trials_statistics]
