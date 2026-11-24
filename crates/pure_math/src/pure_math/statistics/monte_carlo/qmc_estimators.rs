//! Cranley-Patterson randomized Halton and Joe-Kuo Sobol estimators for [`MonteCarloWorkspace`].

use super::error::MonteCarloError;
use super::stats::{MonteCarloEstimate, two_sided_normal_z, welford_mean_m2};
use super::workspace::{DEFAULT_QMC_REPLICATES, MonteCarloWorkspace};
use crate::pure_math::statistics::qmc::{MAX_HALTON_DIM, MAX_SOBOL_DIM};
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
        let halton = self
            .halton
            .as_mut()
            .ok_or(MonteCarloError::InvalidDimension {
                dimension: self.dimension,
                max_dimension: MAX_HALTON_DIM,
            })?;
        for r in 0..reps {
            let pts = base_pts + usize::from(r < rem);
            halton.apply_cranley_patterson_shift(rng);
            let mut batch_sum = 0.0;
            for _ in 0..pts {
                halton.next_point(&mut self.point_buf)?;
                let val = integrand(&self.point_buf);
                if !val.is_finite() {
                    return Err(MonteCarloError::NonFiniteEvaluation);
                }
                batch_sum += val;
            }
            self.batch_means_buf[r] = batch_sum / (pts as f64);
        }
        estimate_from_qmc_batches(
            &self.batch_means_buf[..reps],
            n_samples,
            self.confidence_level,
            z,
        )
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
        let halton = self
            .halton
            .as_mut()
            .ok_or(MonteCarloError::InvalidDimension {
                dimension: self.dimension,
                max_dimension: MAX_HALTON_DIM,
            })?;
        halton.apply_cranley_patterson_shift(rng);
        for idx in 0..n_samples {
            halton.next_point(&mut self.point_buf)?;
            let val = integrand(&self.point_buf);
            if !val.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            self.values_buf[idx] = val;
        }
        MonteCarloEstimate::from_samples(&self.values_buf[..n_samples], self.confidence_level)
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
        let sobol = self
            .sobol
            .as_mut()
            .ok_or(MonteCarloError::InvalidDimension {
                dimension: self.dimension,
                max_dimension: MAX_SOBOL_DIM,
            })?;
        for r in 0..reps {
            let pts = base_pts + usize::from(r < rem);
            sobol.apply_cranley_patterson_shift(rng);
            let mut batch_sum = 0.0;
            for _ in 0..pts {
                sobol.next_point(&mut self.point_buf)?;
                let val = integrand(&self.point_buf);
                if !val.is_finite() {
                    return Err(MonteCarloError::NonFiniteEvaluation);
                }
                batch_sum += val;
            }
            self.batch_means_buf[r] = batch_sum / (pts as f64);
        }
        estimate_from_qmc_batches(
            &self.batch_means_buf[..reps],
            n_samples,
            self.confidence_level,
            z,
        )
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
        let sobol = self
            .sobol
            .as_mut()
            .ok_or(MonteCarloError::InvalidDimension {
                dimension: self.dimension,
                max_dimension: MAX_SOBOL_DIM,
            })?;
        sobol.apply_cranley_patterson_shift(rng);
        for idx in 0..n_samples {
            sobol.next_point(&mut self.point_buf)?;
            let val = integrand(&self.point_buf);
            if !val.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            self.values_buf[idx] = val;
        }
        MonteCarloEstimate::from_samples(&self.values_buf[..n_samples], self.confidence_level)
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
