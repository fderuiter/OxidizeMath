//! Analytical Merton two-fund Lagrangian Markowitz frontier and tangency solvers.

use super::linalg::{build_allocation, cholesky_solve_16};
use super::{MAX_PORTFOLIO_ASSETS, PortfolioAllocation, PortfolioError, PortfolioSpec};

struct MertonInvariants {
    inv_ones: [f64; MAX_PORTFOLIO_ASSETS],
    inv_mu: [f64; MAX_PORTFOLIO_ASSETS],
    a: f64,
    b: f64,
    c: f64,
    discriminant: f64,
}

fn compute_merton_invariants(spec: &PortfolioSpec) -> MertonInvariants {
    let n = spec.asset_count;
    let mut ones = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    for item in ones.iter_mut().take(n) {
        *item = 1.0;
    }
    let inv_ones = cholesky_solve_16(&spec.cholesky_l, &ones, n);
    let inv_mu = cholesky_solve_16(&spec.cholesky_l, &spec.expected_returns, n);
    let mut a = 0.0_f64;
    let mut b = 0.0_f64;
    let mut c = 0.0_f64;
    for i in 0..n {
        a += spec.expected_returns[i] * inv_ones[i];
        b += spec.expected_returns[i] * inv_mu[i];
        c += inv_ones[i];
    }
    MertonInvariants {
        inv_ones,
        inv_mu,
        a,
        b,
        c,
        discriminant: b * c - a * a,
    }
}

/// Solves for the unconstrained global minimum-variance (`GMV`) portfolio
/// $w_{\text{GMV}} = \Sigma^{-1}\mathbf{1} / (\mathbf{1}^T \Sigma^{-1}\mathbf{1})$.
#[verified_engine::verified]
pub fn unconstrained_min_variance(
    spec: &PortfolioSpec,
    risk_free_rate: f64,
) -> Result<PortfolioAllocation, PortfolioError> {
    if !risk_free_rate.is_finite() {
        return Err(PortfolioError::NonFiniteInput);
    }
    let inv = compute_merton_invariants(spec);
    if inv.c <= 1e-16 {
        return Err(PortfolioError::NotPositiveDefinite);
    }
    let mut weights = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    for i in 0..spec.asset_count {
        weights[i] = inv.inv_ones[i] / inv.c;
    }
    Ok(build_allocation(spec, &weights, risk_free_rate))
}

/// Solves for the unconstrained minimum-variance portfolio achieving `target_return`
/// via the analytical two-fund Lagrangian system.
#[verified_engine::verified]
pub fn unconstrained_target_return(
    spec: &PortfolioSpec,
    target_return: f64,
    risk_free_rate: f64,
) -> Result<PortfolioAllocation, PortfolioError> {
    if !target_return.is_finite() || !risk_free_rate.is_finite() {
        return Err(PortfolioError::NonFiniteInput);
    }
    let inv = compute_merton_invariants(spec);
    if inv.c <= 1e-16 {
        return Err(PortfolioError::NotPositiveDefinite);
    }
    let gmv_return = inv.a / inv.c;
    if inv.discriminant <= 1e-18 * (inv.b * inv.c).max(1.0) {
        if (target_return - gmv_return).abs() <= 1e-11 {
            return unconstrained_min_variance(spec, risk_free_rate);
        }
        return Err(PortfolioError::SingularReturnSpan);
    }
    let lambda_mu = (inv.c * target_return - inv.a) / inv.discriminant;
    let lambda_one = (inv.b - inv.a * target_return) / inv.discriminant;
    let mut weights = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    for i in 0..spec.asset_count {
        weights[i] = lambda_mu * inv.inv_mu[i] + lambda_one * inv.inv_ones[i];
    }
    Ok(build_allocation(spec, &weights, risk_free_rate))
}

/// Solves for the unconstrained maximum-Sharpe tangency portfolio
/// $w_{\text{tan}} = \Sigma^{-1}(\mu - r_f \mathbf{1}) / (\mathbf{1}^T \Sigma^{-1}(\mu - r_f \mathbf{1}))$
/// when $r_f < \mu_{\text{GMV}}$ (upward-sloping capital market line).
#[verified_engine::verified]
pub fn unconstrained_tangency(
    spec: &PortfolioSpec,
    risk_free_rate: f64,
) -> Result<PortfolioAllocation, PortfolioError> {
    if !risk_free_rate.is_finite() {
        return Err(PortfolioError::NonFiniteInput);
    }
    let inv = compute_merton_invariants(spec);
    let denom = inv.a - risk_free_rate * inv.c;
    if denom <= 1e-12 {
        return Err(PortfolioError::NoExcessReturnTangency);
    }
    let mut weights = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    for i in 0..spec.asset_count {
        let z_i = inv.inv_mu[i] - risk_free_rate * inv.inv_ones[i];
        weights[i] = z_i / denom;
    }
    Ok(build_allocation(spec, &weights, risk_free_rate))
}

/// Evaluates `out_points.len()` evenly spaced target-return portfolios along the
/// unconstrained efficient frontier over `[min_return, max_return]` in place.
#[verified_engine::verified]
pub fn unconstrained_frontier_in_place(
    spec: &PortfolioSpec,
    min_return: f64,
    max_return: f64,
    risk_free_rate: f64,
    out_points: &mut [PortfolioAllocation],
) -> Result<usize, PortfolioError> {
    if !min_return.is_finite() || !max_return.is_finite() || !risk_free_rate.is_finite() {
        return Err(PortfolioError::NonFiniteInput);
    }
    if min_return > max_return {
        return Err(PortfolioError::InfeasibleTargetReturn {
            target: min_return,
            min_return: max_return,
            max_return: min_return,
        });
    }
    if out_points.is_empty() {
        return Ok(0);
    }
    let count = out_points.len();
    if count == 1 {
        out_points[0] =
            unconstrained_target_return(spec, 0.5 * (min_return + max_return), risk_free_rate)?;
        return Ok(1);
    }
    let span = max_return - min_return;
    for (idx, slot) in out_points.iter_mut().enumerate() {
        let frac = (idx as f64) / ((count - 1) as f64);
        let target = min_return + frac * span;
        *slot = unconstrained_target_return(spec, target, risk_free_rate)?;
    }
    Ok(count)
}

// [cite:clinical_trials_statistics]
