//! Cranley-Patterson randomized Halton and Joe-Kuo Sobol estimators for [`MonteCarloWorkspace`].

use super::error::MonteCarloError;
use super::stats::{MonteCarloEstimate, two_sided_normal_z, welford_mean_m2};
use super::workspace::{DEFAULT_QMC_REPLICATES, MonteCarloWorkspace};
use crate::pure_math::statistics::qmc::{MAX_HALTON_DIM, MAX_NIEDERREITER_DIM, MAX_SOBOL_DIM};
use oxidize_core::rng::OxidizeRng;

impl MonteCarloWorkspace {
    /// Cranley-Patterson randomized Halton quasi-Monte Carlo estimator over `(0, 1)^d`.
    #[verified_engine::verified]
    pub fn estimate_qmc_halton<F>(
        &mut self,
        n_samples: usize,
        rng: &mut OxidizeRng,
        mut integrand: F,
    ) -> Result<MonteCarloEstimate, MonteCarloError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        self.check_budget(n_samples)?;
        let z = two_sided_normal_z(self.confidence_level)?;
        if n_samples < DEFAULT_QMC_REPLICATES * 2 {
            return self.estimate_single_halton_run(n_samples, rng, &mut integrand);
        }
        let reps = DEFAULT_QMC_REPLICATES;
        let base_pts = n_samples / reps;
        let rem = n_samples % reps;
        let conf = self.confidence_level;
        let dim = self.dimension;
        let pt_reg = self.point_buf;
        let bm_reg = self.batch_means_buf;
        let halton = self
            .halton
            .as_mut()
            .ok_or(MonteCarloError::InvalidDimension {
                dimension: dim,
                max_dimension: MAX_HALTON_DIM,
            })?;
        let mut cur = self.arena.raw_cursor();
        let point_buf = cur.region_mut(pt_reg);
        let batch_means_buf = cur.region_mut(bm_reg);
        for (r, bm_slot) in batch_means_buf.iter_mut().enumerate().take(reps) {
            let pts = base_pts + usize::from(r < rem);
            halton.apply_cranley_patterson_shift(rng);
            let mut batch_sum = 0.0;
            for _ in 0..pts {
                halton.next_point(point_buf)?;
                let val = integrand(point_buf);
                if !val.is_finite() {
                    return Err(MonteCarloError::NonFiniteEvaluation);
                }
                batch_sum += val;
            }
            *bm_slot = batch_sum / (pts as f64);
        }
        estimate_from_qmc_batches(&batch_means_buf[..reps], n_samples, conf, z)
    }

    #[verified_engine::verified]
    fn estimate_single_halton_run<F>(
        &mut self,
        n_samples: usize,
        rng: &mut OxidizeRng,
        integrand: &mut F,
    ) -> Result<MonteCarloEstimate, MonteCarloError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        let conf = self.confidence_level;
        let dim = self.dimension;
        let pt_reg = self.point_buf;
        let val_reg = self.values_buf;
        let halton = self
            .halton
            .as_mut()
            .ok_or(MonteCarloError::InvalidDimension {
                dimension: dim,
                max_dimension: MAX_HALTON_DIM,
            })?;
        let mut cur = self.arena.raw_cursor();
        let point_buf = cur.region_mut(pt_reg);
        let values_buf = cur.region_mut(val_reg);
        halton.apply_cranley_patterson_shift(rng);
        for slot in values_buf.iter_mut().take(n_samples) {
            halton.next_point(point_buf)?;
            let val = integrand(point_buf);
            if !val.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            *slot = val;
        }
        MonteCarloEstimate::from_samples(&values_buf[..n_samples], conf)
    }

    /// Cranley-Patterson randomized Joe-Kuo Sobol quasi-Monte Carlo estimator over `(0, 1)^d`.
    #[verified_engine::verified]
    pub fn estimate_qmc_sobol<F>(
        &mut self,
        n_samples: usize,
        rng: &mut OxidizeRng,
        mut integrand: F,
    ) -> Result<MonteCarloEstimate, MonteCarloError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        self.check_budget(n_samples)?;
        if self.dimension > MAX_SOBOL_DIM {
            return Err(MonteCarloError::InvalidDimension {
                dimension: self.dimension,
                max_dimension: MAX_SOBOL_DIM,
            });
        }
        let z = two_sided_normal_z(self.confidence_level)?;
        if n_samples < DEFAULT_QMC_REPLICATES * 2 {
            return self.estimate_single_sobol_run(n_samples, rng, &mut integrand);
        }
        let reps = DEFAULT_QMC_REPLICATES;
        let base_pts = n_samples / reps;
        let rem = n_samples % reps;
        let conf = self.confidence_level;
        let dim = self.dimension;
        let pt_reg = self.point_buf;
        let bm_reg = self.batch_means_buf;
        let sobol = self
            .sobol
            .as_mut()
            .ok_or(MonteCarloError::InvalidDimension {
                dimension: dim,
                max_dimension: MAX_SOBOL_DIM,
            })?;
        let mut cur = self.arena.raw_cursor();
        let point_buf = cur.region_mut(pt_reg);
        let batch_means_buf = cur.region_mut(bm_reg);
        for (r, bm_slot) in batch_means_buf.iter_mut().enumerate().take(reps) {
            let pts = base_pts + usize::from(r < rem);
            sobol.apply_cranley_patterson_shift(rng);
            let mut batch_sum = 0.0;
            for _ in 0..pts {
                sobol.next_point(point_buf)?;
                let val = integrand(point_buf);
                if !val.is_finite() {
                    return Err(MonteCarloError::NonFiniteEvaluation);
                }
                batch_sum += val;
            }
            *bm_slot = batch_sum / (pts as f64);
        }
        estimate_from_qmc_batches(&batch_means_buf[..reps], n_samples, conf, z)
    }

    #[verified_engine::verified]
    fn estimate_single_sobol_run<F>(
        &mut self,
        n_samples: usize,
        rng: &mut OxidizeRng,
        integrand: &mut F,
    ) -> Result<MonteCarloEstimate, MonteCarloError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        let conf = self.confidence_level;
        let dim = self.dimension;
        let pt_reg = self.point_buf;
        let val_reg = self.values_buf;
        let sobol = self
            .sobol
            .as_mut()
            .ok_or(MonteCarloError::InvalidDimension {
                dimension: dim,
                max_dimension: MAX_SOBOL_DIM,
            })?;
        let mut cur = self.arena.raw_cursor();
        let point_buf = cur.region_mut(pt_reg);
        let values_buf = cur.region_mut(val_reg);
        sobol.apply_cranley_patterson_shift(rng);
        for slot in values_buf.iter_mut().take(n_samples) {
            sobol.next_point(point_buf)?;
            let val = integrand(point_buf);
            if !val.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            *slot = val;
        }
        MonteCarloEstimate::from_samples(&values_buf[..n_samples], conf)
    }

    /// Scrambled + Cranley-Patterson randomized 64D base-2 Niederreiter QMC estimator over `(0, 1)^d`.
    #[verified_engine::verified]
    pub fn estimate_qmc_niederreiter<F>(
        &mut self,
        n_samples: usize,
        rng: &mut OxidizeRng,
        mut integrand: F,
    ) -> Result<MonteCarloEstimate, MonteCarloError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        self.check_budget(n_samples)?;
        if self.dimension > MAX_NIEDERREITER_DIM {
            return Err(MonteCarloError::InvalidDimension {
                dimension: self.dimension,
                max_dimension: MAX_NIEDERREITER_DIM,
            });
        }
        let z = two_sided_normal_z(self.confidence_level)?;
        if n_samples < DEFAULT_QMC_REPLICATES * 2 {
            return self.estimate_single_niederreiter_run(n_samples, rng, &mut integrand);
        }
        let reps = DEFAULT_QMC_REPLICATES;
        let base_pts = n_samples / reps;
        let rem = n_samples % reps;
        let conf = self.confidence_level;
        let dim = self.dimension;
        let pt_reg = self.point_buf;
        let bm_reg = self.batch_means_buf;
        let nied = self
            .niederreiter
            .as_mut()
            .ok_or(MonteCarloError::InvalidDimension {
                dimension: dim,
                max_dimension: MAX_NIEDERREITER_DIM,
            })?;
        let mut cur = self.arena.raw_cursor();
        let point_buf = cur.region_mut(pt_reg);
        let batch_means_buf = cur.region_mut(bm_reg);
        for (r, bm_slot) in batch_means_buf.iter_mut().enumerate().take(reps) {
            let pts = base_pts + usize::from(r < rem);
            nied.apply_cranley_patterson_shift(rng);
            let mut batch_sum = 0.0;
            for _ in 0..pts {
                nied.next_point(point_buf)?;
                let val = integrand(point_buf);
                if !val.is_finite() {
                    return Err(MonteCarloError::NonFiniteEvaluation);
                }
                batch_sum += val;
            }
            *bm_slot = batch_sum / (pts as f64);
        }
        estimate_from_qmc_batches(&batch_means_buf[..reps], n_samples, conf, z)
    }

    #[verified_engine::verified]
    fn estimate_single_niederreiter_run<F>(
        &mut self,
        n_samples: usize,
        rng: &mut OxidizeRng,
        integrand: &mut F,
    ) -> Result<MonteCarloEstimate, MonteCarloError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        let conf = self.confidence_level;
        let dim = self.dimension;
        let pt_reg = self.point_buf;
        let val_reg = self.values_buf;
        let nied = self
            .niederreiter
            .as_mut()
            .ok_or(MonteCarloError::InvalidDimension {
                dimension: dim,
                max_dimension: MAX_NIEDERREITER_DIM,
            })?;
        let mut cur = self.arena.raw_cursor();
        let point_buf = cur.region_mut(pt_reg);
        let values_buf = cur.region_mut(val_reg);
        nied.apply_cranley_patterson_shift(rng);
        for slot in values_buf.iter_mut().take(n_samples) {
            nied.next_point(point_buf)?;
            let val = integrand(point_buf);
            if !val.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            *slot = val;
        }
        MonteCarloEstimate::from_samples(&values_buf[..n_samples], conf)
    }
}

/// Aggregates independent Cranley-Patterson QMC replicate batch means into a [`MonteCarloEstimate`].
#[verified_engine::verified]
fn estimate_from_qmc_batches(
    batches: &[f64],
    total_samples: usize,
    confidence_level: f64,
    z: f64,
) -> Result<MonteCarloEstimate, MonteCarloError> {
    let (mean, m2) = welford_mean_m2(batches)?;
    let reps = batches.len();
    let batch_var = if reps > 1 {
        m2 / ((reps - 1) as f64)
    } else {
        0.0
    };
    let se = if reps > 1 {
        (batch_var / (reps as f64)).sqrt()
    } else {
        0.0
    };
    let half = z * se;
    Ok(MonteCarloEstimate {
        mean,
        variance: batch_var,
        standard_error: se,
        ci_lower: mean - half,
        ci_upper: mean + half,
        confidence_level,
        samples: total_samples,
        effective_sample_size: total_samples as f64,
    })
}

// [cite:clinical_trials_statistics]
