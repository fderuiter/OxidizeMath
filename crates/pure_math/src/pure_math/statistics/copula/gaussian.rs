//! Gaussian copula implementation with stack-buffered, caller-scratch, and randomized Genz-Sobol evaluation.

use super::bivariate::bivariate_copula_cdf;
use super::core::{Correlation, CorrelationMatrix, Probability};
use super::multivariate::{evaluate_copula_cdf_with_scratch, required_scratch_len};
use super::qmc::{CopulaQmcWorkspace, multivariate_cdf_qmc, multivariate_cdf_with_workspace};
use crate::error::CopulaError;
use crate::pure_math::analysis::integration::MultivariateIntegrator;
use crate::pure_math::statistics::monte_carlo::{normal_cdf_fast, normal_inv_cdf_fast};
use crate::pure_math::statistics::qmc::MAX_SOBOL_DIM;
use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;
use nalgebra::DMatrix;

/// Maximum number of parlay legs evaluated entirely on the stack by [`GaussianCopula::cdf`]
/// and [`sgp_joint_probability`].
pub const MAX_STACK_LEGS: usize = 16;

/// Stack scratch length corresponding to [`MAX_STACK_LEGS`].
const MAX_STACK_SCRATCH: usize = required_scratch_len(MAX_STACK_LEGS);

/// Default sample budget for multivariate Genz-Sobol Gaussian copula integration.
const DEFAULT_GENZ_SOBOL_SAMPLES: usize = 2048;

/// Default deterministic Cranley-Patterson seed for multivariate copula evaluation.
const DEFAULT_GENZ_SEED: u64 = 0x4745_4E5A_534F_424F;

/// Evaluates `C(u; corr)` using stack scratch up to [`MAX_STACK_LEGS`] and heap fallback above.
#[verified_engine::verified]
pub(crate) fn evaluate_copula_cdf(
    u: &[Probability],
    corr: &CorrelationMatrix,
) -> Result<Probability, CopulaError> {
    let d = corr.dimension();
    if u.len() != d {
        return Err(CopulaError::Math(MathError::DimensionMismatch {
            expected: Dimension(d),
            actual: Dimension(u.len()),
        }));
    }
    if d <= MAX_STACK_LEGS {
        let mut stack_scratch = [0.0_f64; MAX_STACK_SCRATCH];
        return evaluate_copula_cdf_with_scratch(
            u,
            corr,
            &mut stack_scratch[..required_scratch_len(d)],
        );
    }
    let mut heap_scratch = vec![0.0_f64; required_scratch_len(d)];
    evaluate_copula_cdf_with_scratch(u, corr, &mut heap_scratch)
}

/// Multivariate Gaussian copula `C(u_1, ..., u_d; Sigma) = Phi_Sigma(Phi^-1(u_1), ..., Phi^-1(u_d))`.
///
/// Uses exact copula boundary reduction, 20-point Genz (2004) Gauss-Legendre quadrature for two
/// active legs, and Genz (1992) separation-of-variables over a deterministic Richtmyer QMC
/// sequence for three or more active legs in [`Self::cdf`], plus replicated Cranley-Patterson
/// Joe-Kuo [`SobolSequence`](crate::pure_math::statistics::qmc::SobolSequence) error estimation
/// in [`Self::multivariate_cdf_with_error`].
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::copula::{
///     Correlation, GaussianCopula, Probability,
/// };
///
/// let rho = Correlation::new(-0.3).unwrap();
/// let copula = GaussianCopula::bivariate(rho).unwrap();
///
/// let u1 = Probability::new(0.01).unwrap();
/// let u2 = Probability::new(0.60).unwrap();
/// let joint_prob = copula.cdf(&[u1, u2]).unwrap();
/// assert!(joint_prob.value() < 0.01 * 0.60);
/// ```
pub struct GaussianCopula {
    correlation: CorrelationMatrix,
}

impl GaussianCopula {
    /// Creates a new Gaussian copula from a validated [`CorrelationMatrix`].
    #[verified_engine::verified]
    pub fn new(correlation: CorrelationMatrix) -> Self {
        Self { correlation }
    }

    /// Creates a bivariate Gaussian copula from a single correlation coefficient.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::copula::{Correlation, GaussianCopula};
    ///
    /// let rho = Correlation::new(0.5).unwrap();
    /// let copula = GaussianCopula::bivariate(rho).unwrap();
    /// assert_eq!(copula.correlation().dimension(), 2);
    /// ```
    #[verified_engine::verified]
    pub fn bivariate(rho: Correlation) -> Result<Self, CopulaError> {
        let correlation = CorrelationMatrix::bivariate(rho)?;
        Ok(Self::new(correlation))
    }

    /// Returns the scratch slice length in `f64` elements required by [`Self::cdf_with_scratch`]
    /// for a copula of `dimension` legs.
    #[verified_engine::verified]
    pub const fn scratch_len(dimension: usize) -> usize {
        required_scratch_len(dimension)
    }

    /// Computes the copula CDF `C(u_1, ..., u_d)` using a stack buffer for up to
    /// [`MAX_STACK_LEGS`] (`16`) legs without heap allocation.
    #[verified_engine::verified]
    pub fn cdf(&self, u: &[Probability]) -> Result<Probability, CopulaError> {
        evaluate_copula_cdf(u, &self.correlation)
    }

    /// Computes the copula CDF `C(u_1, ..., u_d)` for arbitrary leg counts with zero heap
    /// allocations using caller-supplied `scratch` (`scratch.len() >= Self::scratch_len(d)`).
    #[verified_engine::verified]
    pub fn cdf_with_scratch(
        &self,
        u: &[Probability],
        scratch: &mut [f64],
    ) -> Result<Probability, CopulaError> {
        evaluate_copula_cdf_with_scratch(u, &self.correlation, scratch)
    }

    /// Evaluates the Gaussian copula CDF with an explicit Cranley-Patterson `seed` and antithetic
    /// Halton pair count `n_pairs` (zero heap allocations for `d <= 12`).
    #[verified_engine::verified]
    pub fn cdf_with_seed(
        &self,
        u: &[Probability],
        seed: u64,
        n_pairs: usize,
    ) -> Result<Probability, CopulaError> {
        let dim = self.correlation.dimension();
        if u.len() != dim {
            return Err(CopulaError::Math(MathError::DimensionMismatch {
                expected: Dimension(dim),
                actual: Dimension(u.len()),
            }));
        }
        if dim == 2 {
            let rho = Correlation::new(self.correlation.matrix()[(0, 1)].clamp(-1.0, 1.0))?;
            return bivariate_copula_cdf(u[0], u[1], rho);
        }
        multivariate_cdf_qmc(&self.correlation, u, seed, n_pairs)
    }

    /// Evaluates the Gaussian copula CDF using a caller-supplied [`CopulaQmcWorkspace`]
    /// with zero heap allocations across arbitrary dimensions.
    #[verified_engine::verified]
    pub fn cdf_with_workspace(
        &self,
        u: &[Probability],
        seed: u64,
        n_pairs: usize,
        workspace: &mut CopulaQmcWorkspace,
    ) -> Result<Probability, CopulaError> {
        let dim = self.correlation.dimension();
        if u.len() != dim {
            return Err(CopulaError::Math(MathError::DimensionMismatch {
                expected: Dimension(dim),
                actual: Dimension(u.len()),
            }));
        }
        if dim == 2 {
            let rho = Correlation::new(self.correlation.matrix()[(0, 1)].clamp(-1.0, 1.0))?;
            return bivariate_copula_cdf(u[0], u[1], rho);
        }
        multivariate_cdf_with_workspace(&self.correlation, u, seed, n_pairs, workspace)
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
            return Err(CopulaError::Math(MathError::DimensionMismatch {
                expected: Dimension(dim),
                actual: Dimension(u.len()),
            }));
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

        let full_cov = self.correlation.matrix();
        if active.len() == 2 {
            let rho_raw = full_cov[(active[0], active[1])].clamp(-1.0, 1.0);
            if (rho_raw.abs() - 1.0).abs() <= 1e-12 {
                let prob =
                    bivariate_copula_cdf(u[active[0]], u[active[1]], Correlation::new(rho_raw)?)?;
                return Ok((prob, 0.0));
            }
        }

        let (chol_l, ordered_bounds) = build_ordered_cholesky(full_cov, u, &active)?;
        integrate_genz_sobol(&chol_l, &ordered_bounds, n_samples, seed)
    }

    /// Returns the underlying correlation matrix.
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
        CopulaError::Math(MathError::NumericalError {
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
            CopulaError::Math(MathError::NumericalError {
                reason: e.to_string(),
            })
        })?;

    let clamped = res.value.clamp(0.0, 1.0);
    Ok((Probability::new(clamped)?, res.error))
}

/// Computes the correlated joint win probability for a Same Game Parlay (SGP) without cloning
/// `correlation_matrix` (zero heap allocation up to [`MAX_STACK_LEGS`] legs).
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::copula::{
///     Correlation, CorrelationMatrix, Probability, sgp_joint_probability,
/// };
///
/// let p_a = Probability::new(0.01).unwrap();
/// let p_b = Probability::new(0.60).unwrap();
/// let corr_matrix = CorrelationMatrix::bivariate(Correlation::new(-0.15).unwrap()).unwrap();
/// let joint_prob = sgp_joint_probability(&[p_a, p_b], &corr_matrix).unwrap();
/// assert!(joint_prob.value() < 0.006);
/// ```
#[verified_engine::verified]
pub fn sgp_joint_probability(
    marginals: &[Probability],
    correlation_matrix: &CorrelationMatrix,
) -> Result<Probability, CopulaError> {
    evaluate_copula_cdf(marginals, correlation_matrix)
}

/// Computes the correlated joint win probability for an arbitrary-leg Same Game Parlay (SGP)
/// with zero heap allocations using caller-supplied `scratch`.
#[verified_engine::verified]
pub fn sgp_joint_probability_with_scratch(
    marginals: &[Probability],
    correlation_matrix: &CorrelationMatrix,
    scratch: &mut [f64],
) -> Result<Probability, CopulaError> {
    evaluate_copula_cdf_with_scratch(marginals, correlation_matrix, scratch)
}

#[cfg(test)]
mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod proptest_suite {
    use super::*;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

    #[test]
    fn proptest_gaussian_copula_frechet_and_two_increasing_invariants() {
        let mut runner = TestRunner::new_with_rng(
            Config::with_cases(64),
            TestRng::deterministic_rng(RngAlgorithm::ChaCha),
        );
        let strategy = (
            -0.95f64..0.95f64,
            0.02f64..0.96f64,
            0.02f64..0.96f64,
            0.01f64..0.45f64,
        );
        runner
            .run(&strategy, |(rho, u1, v1, delta)| {
                let u2 = (u1 + delta).min(0.99);
                let v2 = (v1 + delta).min(0.99);
                let copula = GaussianCopula::bivariate(Correlation::new(rho).unwrap()).unwrap();
                let pu1 = Probability::new(u1).unwrap();
                let pv1 = Probability::new(v1).unwrap();
                let pu2 = Probability::new(u2).unwrap();
                let pv2 = Probability::new(v2).unwrap();
                let p_one = Probability::new(1.0).unwrap();
                let p_zero = Probability::new(0.0).unwrap();

                let c11 = copula.cdf(&[pu1, pv1]).unwrap().value();
                let c12 = copula.cdf(&[pu1, pv2]).unwrap().value();
                let c21 = copula.cdf(&[pu2, pv1]).unwrap().value();
                let c22 = copula.cdf(&[pu2, pv2]).unwrap().value();
                let c_sym = copula.cdf(&[pv1, pu1]).unwrap().value();

                let frechet_lo = (u1 + v1 - 1.0).max(0.0);
                let frechet_hi = u1.min(v1);
                prop_assert!(c11 + 1e-9 >= frechet_lo && c11 <= frechet_hi + 1e-9);
                prop_assert!((c11 - c_sym).abs() < 1e-12);
                prop_assert!(c22 - c12 - c21 + c11 >= -1e-9);
                prop_assert!((copula.cdf(&[pu1, p_one]).unwrap().value() - u1).abs() < 1e-12);
                prop_assert!(copula.cdf(&[pu1, p_zero]).unwrap().value().abs() < 1e-12);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn proptest_trivariate_qmc_copula_frechet_and_marginal_consistency() {
        let mut runner = TestRunner::new_with_rng(
            Config::with_cases(24),
            TestRng::deterministic_rng(RngAlgorithm::ChaCha),
        );
        let strategy = (
            -0.42f64..0.88f64,
            0.05f64..0.95f64,
            0.05f64..0.95f64,
            0.05f64..0.95f64,
        );
        runner
            .run(&strategy, |(rho, u1, u2, u3)| {
                let corr =
                    CorrelationMatrix::equicorrelated(3, Correlation::new(rho).unwrap()).unwrap();
                let copula = GaussianCopula::new(corr);
                let p1 = Probability::new(u1).unwrap();
                let p2 = Probability::new(u2).unwrap();
                let p3 = Probability::new(u3).unwrap();
                let p_one = Probability::new(1.0).unwrap();
                let p_zero = Probability::new(0.0).unwrap();

                let (prob, se) = copula
                    .multivariate_cdf_with_error(&[p1, p2, p3], 512, 0x51A7_C0DE)
                    .unwrap();
                let c = prob.value();
                let frechet_lo = (u1 + u2 + u3 - 2.0).max(0.0);
                let frechet_hi = u1.min(u2).min(u3);
                let tol = (4.0 * se).max(5e-3);
                prop_assert!(c + tol >= frechet_lo && c <= frechet_hi + tol);
                let (m1, se1) = copula
                    .multivariate_cdf_with_error(&[p1, p_one, p_one], 256, 0x51A7_C0DE)
                    .unwrap();
                prop_assert!((m1.value() - u1).abs() < 1e-12 && se1 == 0.0);
                let (m0, se0) = copula
                    .multivariate_cdf_with_error(&[p1, p2, p_zero], 256, 0x51A7_C0DE)
                    .unwrap();
                prop_assert!(m0.value().abs() < 1e-12 && se0 == 0.0);
                Ok(())
            })
            .unwrap();
    }
}

// [cite:clinical_trials_statistics]
