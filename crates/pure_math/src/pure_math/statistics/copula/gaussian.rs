//! Gaussian copula implementation with bivariate and multivariate Genz-Sobol evaluation.

use super::core::{Correlation, CorrelationMatrix, Probability};
use super::transforms::{inverse_standard_normal, standard_normal_cdf};
use crate::error::CopulaError;
use crate::pure_math::analysis::integration::MultivariateIntegrator;
use crate::pure_math::statistics::monte_carlo::{normal_cdf_fast, normal_inv_cdf_fast};
use crate::pure_math::statistics::qmc::MAX_SOBOL_DIM;
use nalgebra::DMatrix;

/// Default sample budget for multivariate Genz-Sobol Gaussian copula integration.
const DEFAULT_GENZ_SOBOL_SAMPLES: usize = 2048;

/// Default deterministic Cranley-Patterson seed for multivariate copula evaluation.
const DEFAULT_GENZ_SEED: u64 = 0x4745_4E5A_534F_424F;

/// Gaussian copula over arbitrary positive-definite correlation matrices.
///
/// Models the joint cumulative distribution of $d$ uniform marginals $(u_1, \dots, u_d)$ via
///
/// ```text
/// C(u₁, ..., u_d; Σ) = Φ_Σ(Φ⁻¹(u₁), ..., Φ⁻¹(u_d))
/// ```
///
/// using exact Cholesky separation of variables (Genz, 1992) integrated over randomized
/// Joe-Kuo [`SobolSequence`](crate::pure_math::statistics::qmc::SobolSequence) points.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::copula::{
///     GaussianCopula, Probability, Correlation
/// };
///
/// let rho = Correlation::new(-0.3).unwrap();
/// let copula = GaussianCopula::bivariate(rho).unwrap();
///
/// let u1 = Probability::new(0.99).unwrap();
/// let u2 = Probability::new(0.60).unwrap();
///
/// let joint_prob = copula.cdf(&[u1, u2]).unwrap();
/// assert!(joint_prob.value() > 0.0 && joint_prob.value() < 1.0);
/// ```
pub struct GaussianCopula {
    correlation: CorrelationMatrix,
}

impl GaussianCopula {
    /// Creates a new Gaussian copula from a validated correlation matrix.
    #[verified_engine::verified]
    pub fn new(correlation: CorrelationMatrix) -> Self {
        Self { correlation }
    }

    /// Creates a bivariate Gaussian copula with correlation `rho`.
    #[verified_engine::verified]
    pub fn bivariate(rho: Correlation) -> Result<Self, CopulaError> {
        let correlation = CorrelationMatrix::bivariate(rho)?;
        Ok(Self::new(correlation))
    }

    /// Computes the copula CDF $C(u_1, \dots, u_d)$.
    #[verified_engine::verified]
    pub fn cdf(&self, u: &[Probability]) -> Result<Probability, CopulaError> {
        if u.len() != self.correlation.dimension() {
            return Err(CopulaError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(
                        self.correlation.dimension(),
                    ),
                    actual: math_commons::math_kernel::types::Dimension(u.len()),
                },
            ));
        }

        if u.len() == 2 {
            return self.bivariate_cdf(u[0], u[1]);
        }

        self.multivariate_cdf(u)
    }

    /// Computes the bivariate Gaussian copula CDF.
    #[verified_engine::verified]
    fn bivariate_cdf(&self, u1: Probability, u2: Probability) -> Result<Probability, CopulaError> {
        if u1.value() == 0.0 || u2.value() == 0.0 {
            return Probability::new(0.0);
        }
        if u1.value() == 1.0 {
            return Ok(u2);
        }
        if u2.value() == 1.0 {
            return Ok(u1);
        }
        let z1 = inverse_standard_normal(u1)?;
        let z2 = inverse_standard_normal(u2)?;
        let rho = self
            .correlation
            .get_correlation(0, 1)
            .ok_or(CopulaError::Math(
                math_commons::error::MathError::NumericalError {
                    reason: "Failed to get correlation".to_string(),
                },
            ))?;
        let prob = self.bivariate_normal_cdf(z1, z2, rho)?;
        Probability::new(prob)
    }

    /// Evaluates the bivariate normal CDF with degenerate correlation handling.
    #[verified_engine::verified]
    fn bivariate_normal_cdf(&self, z1: f64, z2: f64, rho: f64) -> Result<f64, CopulaError> {
        if !z1.is_finite() || !z2.is_finite() {
            return Err(CopulaError::Math(
                math_commons::error::MathError::NumericalError {
                    reason: "Non-finite z-scores".to_string(),
                },
            ));
        }
        if (rho.abs() - 1.0).abs() < math_commons::registry::TOLERANCE_HIGH {
            if rho > 0.0 {
                let prob1 = standard_normal_cdf(z1)?.value();
                let prob2 = standard_normal_cdf(z2)?.value();
                return Ok(prob1.min(prob2));
            }
            let prob1 = standard_normal_cdf(z1)?.value();
            let prob2 = standard_normal_cdf(-z2)?.value();
            return Ok((prob1 + prob2 - 1.0).max(0.0));
        }
        self.integrate_bivariate_normal(z1, z2, rho)
    }

    /// Midpoint quadrature for bivariate normal CDF.
    #[verified_engine::verified]
    fn integrate_bivariate_normal(&self, z1: f64, z2: f64, rho: f64) -> Result<f64, CopulaError> {
        let n_points = 100;
        let x_min = -10.0;
        let y_min = -10.0;
        if z1 < x_min || z2 < y_min {
            return Ok(0.0);
        }
        let dx = (z1 - x_min) / (n_points as f64);
        let dy = (z2 - y_min) / (n_points as f64);
        let rho_sq = rho * rho;
        let denom = 2.0 * std::f64::consts::PI * (1.0 - rho_sq).sqrt();
        let factor = -1.0 / (2.0 * (1.0 - rho_sq));

        let mut sum = 0.0;
        for i in 0..n_points {
            let x = x_min + ((i as f64) + 0.5) * dx;
            for j in 0..n_points {
                let y = y_min + ((j as f64) + 0.5) * dy;
                let exponent = factor * (x * x - 2.0 * rho * x * y + y * y);
                sum += (1.0 / denom) * exponent.exp() * dx * dy;
            }
        }
        Ok(sum.clamp(0.0, 1.0))
    }

    /// Evaluates the multivariate Gaussian copula CDF via Cholesky factorization and Genz
    /// separation of variables over a Joe-Kuo [`SobolSequence`](crate::pure_math::statistics::qmc::SobolSequence).
    #[verified_engine::verified]
    pub fn multivariate_cdf(&self, u: &[Probability]) -> Result<Probability, CopulaError> {
        let (prob, _) =
            self.multivariate_cdf_with_error(u, DEFAULT_GENZ_SOBOL_SAMPLES, DEFAULT_GENZ_SEED)?;
        Ok(prob)
    }

    /// Evaluates the multivariate Gaussian copula CDF and returns `(probability, cranley_patterson_se)`.
    #[verified_engine::verified]
    pub fn multivariate_cdf_with_error(
        &self,
        u: &[Probability],
        n_samples: usize,
        seed: u64,
    ) -> Result<(Probability, f64), CopulaError> {
        let dim = self.correlation.dimension();
        if u.len() != dim {
            return Err(CopulaError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(dim),
                    actual: math_commons::math_kernel::types::Dimension(u.len()),
                },
            ));
        }

        // Verify the full correlation matrix is SPD even if marginals sit on 0/1 boundaries.
        let full_cov = self.correlation.matrix();
        if full_cov.clone().cholesky().is_none() {
            return Err(CopulaError::NotPositiveDefinite);
        }

        if u.iter().any(|p| p.value() == 0.0) {
            return Ok((Probability::new(0.0)?, 0.0));
        }

        // Filter out marginals equal to 1.0 (upper bound +inf) via marginal consistency.
        let active: Vec<usize> = (0..dim).filter(|&i| u[i].value() < 1.0).collect();
        if active.is_empty() {
            return Ok((Probability::new(1.0)?, 0.0));
        }
        if active.len() == 1 {
            return Ok((u[active[0]], 0.0));
        }

        let (chol_l, ordered_bounds) = build_ordered_cholesky(full_cov, u, &active)?;
        integrate_genz_sobol(&chol_l, &ordered_bounds, n_samples, seed)
    }

    /// Returns the correlation matrix.
    #[verified_engine::verified]
    pub fn correlation(&self) -> &CorrelationMatrix {
        &self.correlation
    }
}

/// Orders active variables by ascending upper normal quantile $b_i$ and computes Cholesky factor $L$.
#[verified_engine::verified]
fn build_ordered_cholesky(
    cov: &DMatrix<f64>,
    u: &[Probability],
    active: &[usize],
) -> Result<(DMatrix<f64>, Vec<f64>), CopulaError> {
    let mut pairs: Vec<(usize, f64)> = Vec::with_capacity(active.len());
    for &idx in active {
        pairs.push((idx, normal_inv_cdf_fast(u[idx].value())));
    }
    pairs.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let m = pairs.len();
    let mut sub_cov = DMatrix::zeros(m, m);
    let mut bounds = Vec::with_capacity(m);
    for (r, &(orig_r, b_val)) in pairs.iter().enumerate() {
        bounds.push(b_val);
        for (c, &(orig_c, _)) in pairs.iter().enumerate() {
            sub_cov[(r, c)] = cov[(orig_r, orig_c)];
        }
    }
    let chol = sub_cov.cholesky().ok_or(CopulaError::NotPositiveDefinite)?;
    Ok((chol.l(), bounds))
}

/// Executes Genz's $(m - 1)$-dimensional separation-of-variables integrand on `MultivariateIntegrator`.
#[verified_engine::verified]
fn integrate_genz_sobol(
    chol_l: &DMatrix<f64>,
    bounds: &[f64],
    n_samples: usize,
    seed: u64,
) -> Result<(Probability, f64), CopulaError> {
    let m = bounds.len();
    let qmc_dim = m - 1;
    let budget = n_samples.max(32);
    let mut integrator = if qmc_dim <= MAX_SOBOL_DIM {
        MultivariateIntegrator::new_sobol(qmc_dim, budget)
    } else {
        MultivariateIntegrator::new_halton(qmc_dim, budget)
    }
    .map_err(|e| {
        CopulaError::Math(math_commons::error::MathError::NumericalError {
            reason: e.to_string(),
        })
    })?
    .with_seed(seed);

    let e0 = normal_cdf_fast(bounds[0] / chol_l[(0, 0)]);
    let mut y_buf = vec![0.0_f64; qmc_dim];
    let res = integrator
        .integrate_unit_cube(|w| {
            let mut prod = e0;
            let mut e_prev = e0;
            for k in 1..m {
                let u_star = (w[k - 1] * e_prev).clamp(1e-15, 1.0 - 1e-15);
                y_buf[k - 1] = normal_inv_cdf_fast(u_star);
                let mut dot = 0.0;
                for (j, &yj) in y_buf[..k].iter().enumerate() {
                    dot += chol_l[(k, j)] * yj;
                }
                let e_k = normal_cdf_fast((bounds[k] - dot) / chol_l[(k, k)]);
                prod *= e_k;
                e_prev = e_k;
            }
            prod
        })
        .map_err(|e| {
            CopulaError::Math(math_commons::error::MathError::NumericalError {
                reason: e.to_string(),
            })
        })?;

    let clamped = res.value.clamp(0.0, 1.0);
    Ok((Probability::new(clamped)?, res.error))
}

/// Computes the joint probability for a Same Game Parlay (SGP).
#[verified_engine::verified]
pub fn sgp_joint_probability(
    marginals: &[Probability],
    correlation_matrix: &CorrelationMatrix,
) -> Result<Probability, CopulaError> {
    if marginals.len() != correlation_matrix.dimension() {
        return Err(CopulaError::Math(
            math_commons::error::MathError::DimensionMismatch {
                expected: math_commons::math_kernel::types::Dimension(
                    correlation_matrix.dimension(),
                ),
                actual: math_commons::math_kernel::types::Dimension(marginals.len()),
            },
        ));
    }

    let copula = GaussianCopula::new(correlation_matrix.clone());
    copula.cdf(marginals)
}

#[cfg(test)]
mod tests;

// [cite:clinical_trials_statistics]
