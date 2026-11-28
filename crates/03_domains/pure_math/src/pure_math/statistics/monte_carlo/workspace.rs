//! Preallocated zero-allocation Monte Carlo and variance-reduction workspace.

use super::error::MonteCarloError;
use super::stats::{MonteCarloEstimate, effective_sample_size, two_sided_normal_z};
use crate::pure_math::statistics::qmc::{
    HaltonSequence, MAX_HALTON_DIM, MAX_NIEDERREITER_DIM, MAX_SOBOL_DIM, NiederreiterSequence,
    SobolSequence,
};
use math_memory::{ArenaLayout, ArenaRegion, OwnedByteArena};
use math_rng::{sample_bounded_usize, sample_open};
use oxidize_core::rng::OxidizeRng;

/// Guard margin keeping uniform samples strictly inside `(0.0, 1.0)`.
const OPEN_EPS: f64 = 1e-15;

/// Number of Cranley-Patterson replicates used for randomized QMC confidence intervals.
pub(super) const DEFAULT_QMC_REPLICATES: usize = 8;

/// Draws a uniform value strictly inside `(0.0, 1.0)` from `rng` without heap allocation.
#[inline]
#[verified_engine::verified]
fn draw_open_unit(rng: &mut OxidizeRng) -> f64 {
    sample_open(rng)
}

/// Draws a uniform integer in `0..upper` without modulo bias or heap allocation.
#[inline]
#[verified_engine::verified]
pub(crate) fn draw_bounded_index(rng: &mut OxidizeRng, upper: usize) -> usize {
    if upper <= 1 {
        return 0;
    }
    sample_bounded_usize(rng, upper)
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
    /// Scrambled + Cranley-Patterson randomized 64D Halton quasi-Monte Carlo.
    RandomizedHalton,
    /// Cranley-Patterson randomized 64D 53-bit Joe-Kuo Sobol quasi-Monte Carlo.
    RandomizedSobol,
    /// Scrambled + Cranley-Patterson randomized 64D 53-bit base-2 Niederreiter quasi-Monte Carlo.
    RandomizedNiederreiter,
}

/// Preallocated workspace executing IID, antithetic, control-variate, importance-sampling,
/// Latin Hypercube, and randomized QMC estimators with zero hot-loop heap allocations.
#[derive(Debug, Clone)]
pub struct MonteCarloWorkspace {
    pub(super) max_samples: usize,
    pub(super) dimension: usize,
    pub(super) confidence_level: f64,
    pub(super) arena: OwnedByteArena,
    pub(super) point_buf: ArenaRegion<f64>,
    pub(super) anti_point_buf: ArenaRegion<f64>,
    pub(super) values_buf: ArenaRegion<f64>,
    pub(super) control_buf: ArenaRegion<f64>,
    pub(super) weights_buf: ArenaRegion<f64>,
    pub(super) lhs_grid: ArenaRegion<f64>,
    pub(super) perm_buf: ArenaRegion<usize>,
    pub(super) batch_means_buf: ArenaRegion<f64>,
    pub(super) halton: Option<HaltonSequence>,
    pub(super) sobol: Option<SobolSequence>,
    pub(super) niederreiter: Option<NiederreiterSequence>,
}

impl MonteCarloWorkspace {
    /// Preallocates all sample, weight, permutation, and QMC buffers in a single [`OwnedByteArena`].
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
        let map_layout = |_| MonteCarloError::InvalidSampleCount {
            samples: max_samples,
        };
        let grid_len =
            max_samples
                .checked_mul(dimension)
                .ok_or(MonteCarloError::InvalidSampleCount {
                    samples: max_samples,
                })?;
        let mut layout = ArenaLayout::new();
        let point_buf = layout.reserve::<f64>(dimension).map_err(map_layout)?;
        let anti_point_buf = layout.reserve::<f64>(dimension).map_err(map_layout)?;
        let values_buf = layout.reserve::<f64>(max_samples).map_err(map_layout)?;
        let control_buf = layout.reserve::<f64>(max_samples).map_err(map_layout)?;
        let weights_buf = layout.reserve::<f64>(max_samples).map_err(map_layout)?;
        let lhs_grid = layout.reserve::<f64>(grid_len).map_err(map_layout)?;
        let perm_buf = layout.reserve::<usize>(max_samples).map_err(map_layout)?;
        let batch_means_buf = layout
            .reserve::<f64>(DEFAULT_QMC_REPLICATES)
            .map_err(map_layout)?;
        let mut arena = OwnedByteArena::try_from_layout(&layout).map_err(map_layout)?;
        for (i, slot) in arena.region_slice_mut(perm_buf).iter_mut().enumerate() {
            *slot = i;
        }
        let halton = HaltonSequence::new(dimension).ok();
        let sobol = (dimension <= MAX_SOBOL_DIM)
            .then(|| SobolSequence::new(dimension).ok())
            .flatten();
        let niederreiter = (dimension <= MAX_NIEDERREITER_DIM)
            .then(|| NiederreiterSequence::new(dimension).ok())
            .flatten();
        Ok(Self {
            max_samples,
            dimension,
            confidence_level: 0.95,
            arena,
            point_buf,
            anti_point_buf,
            values_buf,
            control_buf,
            weights_buf,
            lhs_grid,
            perm_buf,
            batch_means_buf,
            halton,
            sobol,
            niederreiter,
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
            SamplingMethod::RandomizedNiederreiter => {
                self.estimate_qmc_niederreiter(n_samples, rng, integrand)
            }
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
        let conf = self.confidence_level;
        let pt_reg = self.point_buf;
        let val_reg = self.values_buf;
        let mut cur = self.arena.raw_cursor();
        let point_buf = cur.region_mut(pt_reg);
        let values_buf = cur.region_mut(val_reg);
        for slot in values_buf.iter_mut().take(n_samples) {
            for coord in point_buf.iter_mut() {
                *coord = draw_open_unit(rng);
            }
            let val = integrand(point_buf);
            if !val.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            *slot = val;
        }
        MonteCarloEstimate::from_samples(&values_buf[..n_samples], conf)
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
        let conf = self.confidence_level;
        let pt_reg = self.point_buf;
        let anti_reg = self.anti_point_buf;
        let val_reg = self.values_buf;
        let mut cur = self.arena.raw_cursor();
        let point_buf = cur.region_mut(pt_reg);
        let anti_point_buf = cur.region_mut(anti_reg);
        let values_buf = cur.region_mut(val_reg);
        for slot in values_buf.iter_mut().take(n_pairs) {
            for (u_slot, anti_slot) in point_buf.iter_mut().zip(anti_point_buf.iter_mut()) {
                let u = draw_open_unit(rng);
                *u_slot = u;
                *anti_slot = (1.0 - u).clamp(OPEN_EPS, 1.0 - OPEN_EPS);
            }
            let y1 = integrand(point_buf);
            let y2 = integrand(anti_point_buf);
            if !y1.is_finite() || !y2.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            *slot = 0.5 * (y1 + y2);
        }
        let mut est = MonteCarloEstimate::from_samples(&values_buf[..n_pairs], conf)?;
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
        let conf = self.confidence_level;
        let pt_reg = self.point_buf;
        let val_reg = self.values_buf;
        let ctrl_reg = self.control_buf;
        let mut cur = self.arena.raw_cursor();
        let point_buf = cur.region_mut(pt_reg);
        let values_buf = cur.region_mut(val_reg);
        let control_buf = cur.region_mut(ctrl_reg);
        for (val_slot, ctrl_slot) in values_buf
            .iter_mut()
            .zip(control_buf.iter_mut())
            .take(n_samples)
        {
            for coord in point_buf.iter_mut() {
                *coord = draw_open_unit(rng);
            }
            let (f_val, g_val) = eval_pair(point_buf);
            if !f_val.is_finite() || !g_val.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            *val_slot = f_val;
            *ctrl_slot = g_val;
        }
        let beta = optimal_control_beta(&values_buf[..n_samples], &control_buf[..n_samples]);
        for (f_slot, &g_val) in values_buf[..n_samples]
            .iter_mut()
            .zip(&control_buf[..n_samples])
        {
            *f_slot -= beta * (g_val - control_mean);
        }
        MonteCarloEstimate::from_samples(&values_buf[..n_samples], conf)
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
        let conf = self.confidence_level;
        let z = two_sided_normal_z(conf)?;
        let pt_reg = self.point_buf;
        let val_reg = self.values_buf;
        let wt_reg = self.weights_buf;
        let mut cur = self.arena.raw_cursor();
        let point_buf = cur.region_mut(pt_reg);
        let values_buf = cur.region_mut(val_reg);
        let weights_buf = cur.region_mut(wt_reg);
        for (val_slot, wt_slot) in values_buf
            .iter_mut()
            .zip(weights_buf.iter_mut())
            .take(n_samples)
        {
            for coord in point_buf.iter_mut() {
                *coord = draw_open_unit(rng);
            }
            let (val, weight) = sampler(point_buf);
            if !val.is_finite() || !weight.is_finite() || weight < 0.0 {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            *val_slot = val;
            *wt_slot = weight;
        }
        let weights = &weights_buf[..n_samples];
        let values = &values_buf[..n_samples];
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
            confidence_level: conf,
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
        let conf = self.confidence_level;
        let inv_n = 1.0 / (n_samples as f64);
        let val_reg = self.values_buf;
        let grid_reg = self.lhs_grid;
        let perm_reg = self.perm_buf;
        let mut cur = self.arena.raw_cursor();
        let values_buf = cur.region_mut(val_reg);
        let lhs_grid = cur.region_mut(grid_reg);
        let perm_buf = cur.region_mut(perm_reg);
        let active_grid = &mut lhs_grid[..n_samples * dim];

        for axis in 0..dim {
            for (idx, slot) in perm_buf.iter_mut().enumerate().take(n_samples) {
                *slot = idx;
            }
            for idx in (1..n_samples).rev() {
                let swap_idx = draw_bounded_index(rng, idx + 1);
                perm_buf.swap(idx, swap_idx);
            }
            for (row_idx, point_row) in active_grid.chunks_exact_mut(dim).enumerate() {
                let stratum = perm_buf[row_idx] as f64;
                let jitter = draw_open_unit(rng);
                point_row[axis] = ((stratum + jitter) * inv_n).clamp(OPEN_EPS, 1.0 - OPEN_EPS);
            }
        }

        for (val_slot, point_row) in values_buf
            .iter_mut()
            .zip(active_grid.chunks_exact(dim))
            .take(n_samples)
        {
            let val = integrand(point_row);
            if !val.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            *val_slot = val;
        }
        MonteCarloEstimate::from_samples(&values_buf[..n_samples], conf)
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
