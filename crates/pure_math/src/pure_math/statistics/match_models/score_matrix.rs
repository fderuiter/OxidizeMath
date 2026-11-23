//! Zero-allocation score-grid kernels for Independent Poisson, Dixon-Coles (1997),
//! and Karlis-Ntzoufras (2003) Bivariate Poisson match models.

use crate::error::{CopulaError, ZipError};
use crate::pure_math::statistics::zip_regression::PoissonRate;
use math_commons::error::MathError;
use math_commons::math_kernel::types::{Dimension, flatten_2d_index};
use serde::{Deserialize, Serialize};

/// Low-score dependence parameter $\rho \in [-0.99, 0.99]$ in the Dixon-Coles (1997) model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DixonColesRho(f64);

impl DixonColesRho {
    /// Validates a Dixon-Coles dependence parameter $\rho \in [-0.99, 0.99]$.
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, CopulaError> {
        if !(-0.99..=0.99).contains(&value) || !value.is_finite() {
            return Err(CopulaError::InvalidCorrelation { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw $\rho$ value.
    #[verified_engine::verified]
    pub fn value(self) -> f64 {
        self.0
    }
}

impl Default for DixonColesRho {
    #[verified_engine::verified]
    fn default() -> Self {
        Self(-0.10)
    }
}

/// Shared covariance goal rate $\lambda_3 \ge 0$ in the Karlis-Ntzoufras (2003) bivariate Poisson model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SharedGoalRate(f64);

impl SharedGoalRate {
    /// Creates a non-negative finite shared scoring intensity $\lambda_3 \ge 0$.
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, ZipError> {
        if value < 0.0 || !value.is_finite() {
            return Err(ZipError::InvalidRate { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw $\lambda_3$ value.
    #[verified_engine::verified]
    pub fn value(self) -> f64 {
        self.0
    }
}

impl Default for SharedGoalRate {
    #[verified_engine::verified]
    fn default() -> Self {
        Self(0.15)
    }
}

#[verified_engine::verified]
pub(super) fn validate_matrix_slice(
    matrix_len: usize,
    max_goals: usize,
) -> Result<usize, MathError> {
    let cols = max_goals + 1;
    let expected = cols * cols;
    if matrix_len != expected {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(expected),
            actual: Dimension(matrix_len),
        });
    }
    Ok(cols)
}

#[verified_engine::verified]
fn poisson_pmf_exact(lambda: f64, k: usize) -> f64 {
    if lambda <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    let mut term = (-lambda).exp();
    for step in 1..=k {
        term *= lambda / (step as f64);
    }
    term
}

#[verified_engine::verified]
fn normalize_grid(out_matrix: &mut [f64]) {
    let mut sum = 0.0;
    for val in out_matrix.iter() {
        sum += *val;
    }
    if sum > 0.0 && sum.is_finite() {
        let inv = 1.0 / sum;
        for val in out_matrix.iter_mut() {
            *val *= inv;
        }
    }
}

/// Evaluates the Dixon-Coles (1997) low-score adjustment factor $\tau_{\lambda,\mu}(x, y)$.
#[verified_engine::verified]
pub fn dixon_coles_tau(
    home_goals: usize,
    away_goals: usize,
    lambda_home: f64,
    lambda_away: f64,
    rho: f64,
) -> f64 {
    match (home_goals, away_goals) {
        (0, 0) => (1.0 - lambda_home * lambda_away * rho).max(0.0),
        (0, 1) => (1.0 + lambda_home * rho).max(0.0),
        (1, 0) => (1.0 + lambda_away * rho).max(0.0),
        (1, 1) => (1.0 - rho).max(0.0),
        _ => 1.0,
    }
}

/// Evaluates the Karlis-Ntzoufras (2003) bivariate Poisson joint PMF $P(X = x, Y = y)$:
///
/// $$P(X = x, Y = y) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)} \sum_{k=0}^{\min(x, y)} \frac{\lambda_1^{x-k}}{(x-k)!} \frac{\lambda_2^{y-k}}{(y-k)!} \frac{\lambda_3^k}{k!}$$
#[verified_engine::verified]
pub fn bivariate_poisson_pmf(
    home_goals: usize,
    away_goals: usize,
    lambda_1: f64,
    lambda_2: f64,
    lambda_3: f64,
) -> f64 {
    if lambda_3 <= 0.0 {
        return poisson_pmf_exact(lambda_1, home_goals) * poisson_pmf_exact(lambda_2, away_goals);
    }
    let min_xy = home_goals.min(away_goals);
    let mut sum = 0.0;
    let base_exp = (-lambda_3).exp();
    let mut cov_term = base_exp;
    for k in 0..=min_xy {
        if k > 0 {
            cov_term *= lambda_3 / (k as f64);
        }
        let p1 = poisson_pmf_exact(lambda_1, home_goals - k);
        let p2 = poisson_pmf_exact(lambda_2, away_goals - k);
        sum += p1 * p2 * cov_term;
    }
    sum
}

/// Populates a normalized `(max_goals + 1) x (max_goals + 1)` row-major score matrix under
/// independent Poisson goal processes without heap allocation.
#[verified_engine::verified]
pub fn independent_poisson_matrix_into(
    lambda_home: PoissonRate,
    lambda_away: PoissonRate,
    max_goals: usize,
    out_matrix: &mut [f64],
) -> Result<(), MathError> {
    let cols = validate_matrix_slice(out_matrix.len(), max_goals)?;
    let lh = lambda_home.value();
    let la = lambda_away.value();
    for home in 0..cols {
        let ph = poisson_pmf_exact(lh, home);
        for away in 0..cols {
            let pa = poisson_pmf_exact(la, away);
            out_matrix[flatten_2d_index(away, home, cols)] = ph * pa;
        }
    }
    normalize_grid(out_matrix);
    Ok(())
}

/// Populates a normalized `(max_goals + 1) x (max_goals + 1)` row-major score matrix under the
/// Dixon-Coles (1997) low-score correlation model without heap allocation.
#[verified_engine::verified]
pub fn dixon_coles_matrix_into(
    lambda_home: PoissonRate,
    lambda_away: PoissonRate,
    rho: DixonColesRho,
    max_goals: usize,
    out_matrix: &mut [f64],
) -> Result<(), MathError> {
    let cols = validate_matrix_slice(out_matrix.len(), max_goals)?;
    let lh = lambda_home.value();
    let la = lambda_away.value();
    let r = rho.value();
    for home in 0..cols {
        let ph = poisson_pmf_exact(lh, home);
        for away in 0..cols {
            let pa = poisson_pmf_exact(la, away);
            let tau = dixon_coles_tau(home, away, lh, la, r);
            out_matrix[flatten_2d_index(away, home, cols)] = ph * pa * tau;
        }
    }
    normalize_grid(out_matrix);
    Ok(())
}

/// Populates a normalized `(max_goals + 1) x (max_goals + 1)` row-major score matrix under the
/// Karlis-Ntzoufras (2003) bivariate Poisson model without heap allocation.
#[verified_engine::verified]
pub fn bivariate_poisson_matrix_into(
    lambda_1: PoissonRate,
    lambda_2: PoissonRate,
    lambda_3: SharedGoalRate,
    max_goals: usize,
    out_matrix: &mut [f64],
) -> Result<(), MathError> {
    let cols = validate_matrix_slice(out_matrix.len(), max_goals)?;
    let l1 = lambda_1.value();
    let l2 = lambda_2.value();
    let l3 = lambda_3.value();
    for home in 0..cols {
        for away in 0..cols {
            out_matrix[flatten_2d_index(away, home, cols)] =
                bivariate_poisson_pmf(home, away, l1, l2, l3);
        }
    }
    normalize_grid(out_matrix);
    Ok(())
}
