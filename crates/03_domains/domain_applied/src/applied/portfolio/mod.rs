//! # Markowitz Mean-Variance Portfolio Optimization
//!
//! Zero-allocation unconstrained Lagrangian and constrained long-only active-set
//! quadratic-programming solvers for multi-asset capital allocation across up to
//! `MAX_PORTFOLIO_ASSETS` (`16`) correlated assets:
//!
//! - **Unconstrained Analytical Frontier**: Exact Merton two-fund Lagrangian solutions
//!   for global minimum-variance (`GMV`), target-return, efficient-frontier, and
//!   maximum-Sharpe tangency portfolios.
//! - **Constrained Long-Only & Weight-Capped Active-Set QP**: Primal-dual active-set
//!   solver with Cholesky Schur-complement equality projections enforcing $\mathbf{1}^T w = 1$
//!   and $0 \le w_i \le u_i$, locating constrained `GMV`, efficient frontiers, and
//!   maximum-Sharpe tangency portfolios with zero dynamic heap allocations.
//!
//! ## Example
//!
//! ```rust
//! # use domain_applied::__doctest_facade as math_explorer;
//! use math_explorer::applied::portfolio::{
//!     PortfolioSpec, long_only_min_variance, long_only_tangency, unconstrained_tangency,
//! };
//!
//! let returns = [0.12, 0.08, 0.05];
//! let vols = [0.20, 0.12, 0.06];
//! let upper_corrs = [0.25, -0.10, 0.15];
//! let spec = PortfolioSpec::from_upper_correlations(&returns, &vols, &upper_corrs).unwrap();
//!
//! let gmv = long_only_min_variance(&spec, 0.03).unwrap();
//! let tan = long_only_tangency(&spec, 0.03).unwrap();
//! let unc = unconstrained_tangency(&spec, 0.03).unwrap();
//!
//! assert!((gmv.weights_slice().iter().sum::<f64>() - 1.0).abs() < 1e-10);
//! assert!(tan.sharpe_ratio >= gmv.sharpe_ratio);
//! assert!(unc.sharpe_ratio + 1e-10 >= tan.sharpe_ratio);
//! ```

mod active_set;
mod frontier;
mod linalg;
mod unconstrained;

#[cfg(test)]
mod tests;

pub use frontier::{
    constrained_frontier_in_place, constrained_min_variance, constrained_tangency,
    constrained_target_return, long_only_frontier_in_place, long_only_min_variance,
    long_only_tangency, long_only_target_return,
};
pub use unconstrained::{
    unconstrained_frontier_in_place, unconstrained_min_variance, unconstrained_tangency,
    unconstrained_target_return,
};

use linalg::{build_allocation_from_slice, cholesky_16};
use math_commons::math_kernel::types::flatten_2d_index;

/// Maximum number of correlated assets supported by stack-allocated Markowitz solvers.
pub const MAX_PORTFOLIO_ASSETS: usize = 16;

const SYMMETRY_TOL: f64 = 1e-10;

/// Stack-allocated error domain for Markowitz portfolio optimization.
///
/// Kept `Copy` (zero heap fields) so error paths inside `VerifiedAllocator` never allocate.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum PortfolioError {
    /// Fewer than 2 assets were supplied.
    #[error("portfolio requires at least 2 assets, got {count}")]
    TooFewAssets {
        /// Supplied asset count.
        count: usize,
    },
    /// Asset count exceeds `MAX_PORTFOLIO_ASSETS` (`16`).
    #[error("asset count {count} exceeds maximum supported {max}")]
    TooManyAssets {
        /// Supplied asset count.
        count: usize,
        /// Maximum supported asset count.
        max: usize,
    },
    /// Vector or flattened matrix slice length does not match `asset_count`.
    #[error("dimension mismatch: expected length {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected slice length.
        expected: usize,
        /// Actual slice length.
        actual: usize,
    },
    /// Input contains `NaN` or infinite values.
    #[error("portfolio input contains non-finite (NaN or infinite) values")]
    NonFiniteInput,
    /// Asset volatility is non-positive (`<= 0.0`).
    #[error("asset volatility {value} must be strictly positive")]
    InvalidVolatility {
        /// Invalid volatility parameter.
        value: f64,
    },
    /// Correlation entry lies outside `(-1.0, 1.0)` off-diagonal or differs from `1.0` on diagonal.
    #[error("correlation {value} violates valid correlation bounds")]
    InvalidCorrelation {
        /// Invalid correlation entry.
        value: f64,
    },
    /// Per-asset upper weight bounds are invalid or sum to less than `1.0`.
    #[error("upper weight bounds must lie in [0, 1] and sum to at least 1.0")]
    InvalidWeightBounds,
    /// Covariance or correlation matrix is not symmetric within tolerance.
    #[error("covariance or correlation matrix is not symmetric")]
    NotSymmetric,
    /// Covariance matrix is not strictly symmetric positive-definite.
    #[error("covariance matrix must be strictly positive-definite")]
    NotPositiveDefinite,
    /// Expected return vector is collinear with the unit vector, collapsing the frontier.
    #[error("expected returns are identical across all assets; frontier span is singular")]
    SingularReturnSpan,
    /// Target return lies outside the attainable range `[min_return, max_return]`.
    #[error("target return {target} is outside feasible interval [{min_return}, {max_return}]")]
    InfeasibleTargetReturn {
        /// Requested target return.
        target: f64,
        /// Minimum feasible portfolio return.
        min_return: f64,
        /// Maximum feasible portfolio return.
        max_return: f64,
    },
    /// No feasible portfolio achieves positive excess return above `risk_free_rate`.
    #[error("no feasible tangency portfolio achieves positive excess return over risk-free rate")]
    NoExcessReturnTangency,
    /// Iterative active-set or root solver exceeded its iteration cap.
    #[error("portfolio solver failed to converge within {iterations} iterations")]
    ConvergenceFailed {
        /// Iteration budget exhausted.
        iterations: usize,
    },
}

impl math_diagnostics::Diagnostic for PortfolioError {
    #[verified_engine::verified]
    fn severity(&self) -> math_diagnostics::Severity {
        math_diagnostics::Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn math_diagnostics::DiagnosticVisitor) {
        visitor.record_str("error_type", "PortfolioError");
        visitor.record_display("description", self);
    }
}

/// Optimal or evaluated capital allocation across up to [`MAX_PORTFOLIO_ASSETS`] assets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortfolioAllocation {
    /// Active number of assets (`2..=MAX_PORTFOLIO_ASSETS`).
    pub asset_count: usize,
    /// Portfolio weights stored in `weights[..asset_count]`.
    pub weights: [f64; MAX_PORTFOLIO_ASSETS],
    /// Annualized expected portfolio return $\mu^T w$.
    pub expected_return: f64,
    /// Portfolio return variance $w^T \Sigma w$.
    pub variance: f64,
    /// Annualized portfolio volatility $\sqrt{w^T \Sigma w}$.
    pub volatility: f64,
    /// Excess-return Sharpe ratio $(\mu^T w - r_f) / \sigma_p$.
    pub sharpe_ratio: f64,
}

impl PortfolioAllocation {
    /// Returns the active slice of `asset_count` portfolio weights.
    #[verified_engine::verified]
    pub fn weights_slice(&self) -> &[f64] {
        &self.weights[..self.asset_count]
    }
}

/// Validated symmetric positive-definite multi-asset expected-return and covariance specification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortfolioSpec {
    pub(crate) asset_count: usize,
    pub(crate) expected_returns: [f64; MAX_PORTFOLIO_ASSETS],
    pub(crate) covariance: [[f64; MAX_PORTFOLIO_ASSETS]; MAX_PORTFOLIO_ASSETS],
    pub(crate) cholesky_l: [[f64; MAX_PORTFOLIO_ASSETS]; MAX_PORTFOLIO_ASSETS],
}

fn validate_asset_returns(
    expected_returns: &[f64],
) -> Result<(usize, [f64; MAX_PORTFOLIO_ASSETS]), PortfolioError> {
    let n = expected_returns.len();
    if n < 2 {
        return Err(PortfolioError::TooFewAssets { count: n });
    }
    if n > MAX_PORTFOLIO_ASSETS {
        return Err(PortfolioError::TooManyAssets {
            count: n,
            max: MAX_PORTFOLIO_ASSETS,
        });
    }
    let mut mu = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    for (dst, &val) in mu.iter_mut().zip(expected_returns.iter()) {
        if !val.is_finite() {
            return Err(PortfolioError::NonFiniteInput);
        }
        *dst = val;
    }
    Ok((n, mu))
}

impl PortfolioSpec {
    /// Constructs a validated [`PortfolioSpec`] from `expected_returns` and row-major `covariance_flat`.
    #[verified_engine::verified]
    pub fn from_covariance(
        expected_returns: &[f64],
        covariance_flat: &[f64],
    ) -> Result<Self, PortfolioError> {
        let (n, mu) = validate_asset_returns(expected_returns)?;
        if covariance_flat.len() != n * n {
            return Err(PortfolioError::DimensionMismatch {
                expected: n * n,
                actual: covariance_flat.len(),
            });
        }
        let mut cov = [[0.0_f64; MAX_PORTFOLIO_ASSETS]; MAX_PORTFOLIO_ASSETS];
        for r in 0..n {
            for c in 0..n {
                let v_rc = covariance_flat[flatten_2d_index(c, r, n)];
                let v_cr = covariance_flat[flatten_2d_index(r, c, n)];
                if !v_rc.is_finite() || !v_cr.is_finite() {
                    return Err(PortfolioError::NonFiniteInput);
                }
                let scale = 1.0_f64.max(v_rc.abs()).max(v_cr.abs());
                if (v_rc - v_cr).abs() > SYMMETRY_TOL * scale {
                    return Err(PortfolioError::NotSymmetric);
                }
                cov[r][c] = 0.5 * (v_rc + v_cr);
            }
        }
        let cholesky_l = cholesky_16(&cov, n)?;
        Ok(Self {
            asset_count: n,
            expected_returns: mu,
            covariance: cov,
            cholesky_l,
        })
    }

    /// Constructs a validated [`PortfolioSpec`] from `expected_returns`, `volatilities`,
    /// and a full row-major `correlations_flat` ($n \times n$) matrix.
    #[verified_engine::verified]
    pub fn from_volatilities_and_correlations(
        expected_returns: &[f64],
        volatilities: &[f64],
        correlations_flat: &[f64],
    ) -> Result<Self, PortfolioError> {
        let (n, mu) = validate_asset_returns(expected_returns)?;
        if volatilities.len() != n {
            return Err(PortfolioError::DimensionMismatch {
                expected: n,
                actual: volatilities.len(),
            });
        }
        if correlations_flat.len() != n * n {
            return Err(PortfolioError::DimensionMismatch {
                expected: n * n,
                actual: correlations_flat.len(),
            });
        }
        for &vol in volatilities {
            if !vol.is_finite() {
                return Err(PortfolioError::NonFiniteInput);
            }
            if vol <= 0.0 {
                return Err(PortfolioError::InvalidVolatility { value: vol });
            }
        }
        let mut cov = [[0.0_f64; MAX_PORTFOLIO_ASSETS]; MAX_PORTFOLIO_ASSETS];
        for r in 0..n {
            for c in 0..n {
                let rho_rc = correlations_flat[flatten_2d_index(c, r, n)];
                let rho_cr = correlations_flat[flatten_2d_index(r, c, n)];
                if !rho_rc.is_finite() || !rho_cr.is_finite() {
                    return Err(PortfolioError::NonFiniteInput);
                }
                if (rho_rc - rho_cr).abs() > SYMMETRY_TOL {
                    return Err(PortfolioError::NotSymmetric);
                }
                let rho = 0.5 * (rho_rc + rho_cr);
                if r == c {
                    if (rho - 1.0).abs() > SYMMETRY_TOL {
                        return Err(PortfolioError::InvalidCorrelation { value: rho });
                    }
                    cov[r][c] = volatilities[r] * volatilities[r];
                } else {
                    if rho <= -1.0 || rho >= 1.0 {
                        return Err(PortfolioError::InvalidCorrelation { value: rho });
                    }
                    cov[r][c] = rho * volatilities[r] * volatilities[c];
                }
            }
        }
        let cholesky_l = cholesky_16(&cov, n)?;
        Ok(Self {
            asset_count: n,
            expected_returns: mu,
            covariance: cov,
            cholesky_l,
        })
    }

    /// Constructs a validated [`PortfolioSpec`] from `expected_returns`, `volatilities`,
    /// and strictly upper-triangular pairwise correlations (`n * (n - 1) / 2` entries).
    #[verified_engine::verified]
    pub fn from_upper_correlations(
        expected_returns: &[f64],
        volatilities: &[f64],
        upper_correlations: &[f64],
    ) -> Result<Self, PortfolioError> {
        let (n, mu) = validate_asset_returns(expected_returns)?;
        if volatilities.len() != n {
            return Err(PortfolioError::DimensionMismatch {
                expected: n,
                actual: volatilities.len(),
            });
        }
        let expected_pairs = n * (n - 1) / 2;
        if upper_correlations.len() != expected_pairs {
            return Err(PortfolioError::DimensionMismatch {
                expected: expected_pairs,
                actual: upper_correlations.len(),
            });
        }
        for &vol in volatilities {
            if !vol.is_finite() {
                return Err(PortfolioError::NonFiniteInput);
            }
            if vol <= 0.0 {
                return Err(PortfolioError::InvalidVolatility { value: vol });
            }
        }
        let mut cov = [[0.0_f64; MAX_PORTFOLIO_ASSETS]; MAX_PORTFOLIO_ASSETS];
        let mut pair_idx = 0_usize;
        for r in 0..n {
            cov[r][r] = volatilities[r] * volatilities[r];
            for c in (r + 1)..n {
                let rho = upper_correlations[pair_idx];
                pair_idx += 1;
                if !rho.is_finite() {
                    return Err(PortfolioError::NonFiniteInput);
                }
                if rho <= -1.0 || rho >= 1.0 {
                    return Err(PortfolioError::InvalidCorrelation { value: rho });
                }
                let cov_rc = rho * volatilities[r] * volatilities[c];
                cov[r][c] = cov_rc;
                cov[c][r] = cov_rc;
            }
        }
        let cholesky_l = cholesky_16(&cov, n)?;
        Ok(Self {
            asset_count: n,
            expected_returns: mu,
            covariance: cov,
            cholesky_l,
        })
    }

    /// Returns the number of assets $n \in [2, 16]$.
    #[verified_engine::verified]
    pub fn asset_count(&self) -> usize {
        self.asset_count
    }

    /// Returns the expected return slice $\mu \in \mathbb{R}^n$.
    #[verified_engine::verified]
    pub fn expected_returns(&self) -> &[f64] {
        &self.expected_returns[..self.asset_count]
    }

    /// Returns covariance matrix entry $\Sigma_{rc}$.
    #[verified_engine::verified]
    pub fn covariance_at(&self, row: usize, col: usize) -> Result<f64, PortfolioError> {
        if row >= self.asset_count || col >= self.asset_count {
            return Err(PortfolioError::DimensionMismatch {
                expected: self.asset_count,
                actual: row.max(col) + 1,
            });
        }
        Ok(self.covariance[row][col])
    }

    /// Evaluates expected return, volatility, variance, and Sharpe ratio for arbitrary `weights`.
    #[verified_engine::verified]
    pub fn evaluate_weights(
        &self,
        weights: &[f64],
        risk_free_rate: f64,
    ) -> Result<PortfolioAllocation, PortfolioError> {
        build_allocation_from_slice(self, weights, risk_free_rate)
    }
}

// [cite:clinical_trials_statistics]
