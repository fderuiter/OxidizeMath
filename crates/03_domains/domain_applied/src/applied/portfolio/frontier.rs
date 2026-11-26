//! Constrained and long-only Markowitz minimum-variance, target-return, tangency, and frontier APIs.

use super::active_set::{
    BOUND_TOL, KKT_TOL, extreme_return_bounds, root_find_target_return, solve_box_simplex_qp,
    validate_upper_bounds,
};
use super::linalg::build_allocation;
use super::{MAX_PORTFOLIO_ASSETS, PortfolioAllocation, PortfolioError, PortfolioSpec};

/// Solves for the minimum-variance portfolio under $0 \le w_i \le u_i$ and $\sum w_i = 1$.
#[verified_engine::verified]
pub fn constrained_min_variance(
    spec: &PortfolioSpec,
    risk_free_rate: f64,
    upper_bounds: &[f64],
) -> Result<PortfolioAllocation, PortfolioError> {
    if !risk_free_rate.is_finite() {
        return Err(PortfolioError::NonFiniteInput);
    }
    let ub = validate_upper_bounds(upper_bounds, spec.asset_count)?;
    let zero_c = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    let w = solve_box_simplex_qp(spec, &zero_c, &ub)?;
    Ok(build_allocation(spec, &w, risk_free_rate))
}

/// Solves for the long-only ($w_i \ge 0$, $\sum w_i = 1$) global minimum-variance portfolio.
#[verified_engine::verified]
pub fn long_only_min_variance(
    spec: &PortfolioSpec,
    risk_free_rate: f64,
) -> Result<PortfolioAllocation, PortfolioError> {
    let ones = [1.0_f64; MAX_PORTFOLIO_ASSETS];
    constrained_min_variance(spec, risk_free_rate, &ones[..spec.asset_count])
}

/// Solves for the minimum-variance portfolio achieving `target_return` under
/// $0 \le w_i \le u_i$ and $\sum w_i = 1$.
#[verified_engine::verified]
pub fn constrained_target_return(
    spec: &PortfolioSpec,
    target_return: f64,
    risk_free_rate: f64,
    upper_bounds: &[f64],
) -> Result<PortfolioAllocation, PortfolioError> {
    if !target_return.is_finite() || !risk_free_rate.is_finite() {
        return Err(PortfolioError::NonFiniteInput);
    }
    let ub = validate_upper_bounds(upper_bounds, spec.asset_count)?;
    let (r_min, r_max) = extreme_return_bounds(spec, &ub);
    if target_return < r_min - KKT_TOL || target_return > r_max + KKT_TOL {
        return Err(PortfolioError::InfeasibleTargetReturn {
            target: target_return,
            min_return: r_min,
            max_return: r_max,
        });
    }
    if r_max - r_min <= BOUND_TOL {
        return constrained_min_variance(spec, risk_free_rate, upper_bounds);
    }
    let clamped_target = target_return.clamp(r_min, r_max);
    let w = root_find_target_return(spec, clamped_target, &ub)?;
    Ok(build_allocation(spec, &w, risk_free_rate))
}

/// Solves for the long-only ($w_i \ge 0$, $\sum w_i = 1$) minimum-variance portfolio
/// achieving `target_return`.
#[verified_engine::verified]
pub fn long_only_target_return(
    spec: &PortfolioSpec,
    target_return: f64,
    risk_free_rate: f64,
) -> Result<PortfolioAllocation, PortfolioError> {
    let ones = [1.0_f64; MAX_PORTFOLIO_ASSETS];
    constrained_target_return(
        spec,
        target_return,
        risk_free_rate,
        &ones[..spec.asset_count],
    )
}

fn golden_section_tangency(
    spec: &PortfolioSpec,
    risk_free_rate: f64,
    upper_bounds: &[f64],
    mut left: f64,
    mut right: f64,
    mut best: PortfolioAllocation,
) -> Result<PortfolioAllocation, PortfolioError> {
    let inv_phi = 0.5 * (5.0_f64.sqrt() - 1.0);
    let mut c_pt = right - inv_phi * (right - left);
    let mut d_pt = left + inv_phi * (right - left);
    let mut alloc_c = constrained_target_return(spec, c_pt, risk_free_rate, upper_bounds)?;
    let mut alloc_d = constrained_target_return(spec, d_pt, risk_free_rate, upper_bounds)?;
    for _ in 0..52 {
        if alloc_c.sharpe_ratio >= alloc_d.sharpe_ratio {
            if alloc_c.sharpe_ratio > best.sharpe_ratio {
                best = alloc_c;
            }
            right = d_pt;
            d_pt = c_pt;
            alloc_d = alloc_c;
            c_pt = right - inv_phi * (right - left);
            alloc_c = constrained_target_return(spec, c_pt, risk_free_rate, upper_bounds)?;
        } else {
            if alloc_d.sharpe_ratio > best.sharpe_ratio {
                best = alloc_d;
            }
            left = c_pt;
            c_pt = d_pt;
            alloc_c = alloc_d;
            d_pt = left + inv_phi * (right - left);
            alloc_d = constrained_target_return(spec, d_pt, risk_free_rate, upper_bounds)?;
        }
    }
    Ok(best)
}

/// Solves for the constrained maximum-Sharpe tangency portfolio under
/// $0 \le w_i \le u_i$ and $\sum w_i = 1$ via unimodal golden-section search along the efficient frontier.
#[verified_engine::verified]
pub fn constrained_tangency(
    spec: &PortfolioSpec,
    risk_free_rate: f64,
    upper_bounds: &[f64],
) -> Result<PortfolioAllocation, PortfolioError> {
    if !risk_free_rate.is_finite() {
        return Err(PortfolioError::NonFiniteInput);
    }
    let ub = validate_upper_bounds(upper_bounds, spec.asset_count)?;
    let (_, r_max) = extreme_return_bounds(spec, &ub);
    if r_max <= risk_free_rate + BOUND_TOL {
        return Err(PortfolioError::NoExcessReturnTangency);
    }
    let gmv = constrained_min_variance(spec, risk_free_rate, upper_bounds)?;
    let top = constrained_target_return(spec, r_max, risk_free_rate, upper_bounds)?;
    let best = if gmv.sharpe_ratio >= top.sharpe_ratio {
        gmv
    } else {
        top
    };
    let left = gmv.expected_return.max(risk_free_rate);
    if r_max - left <= BOUND_TOL {
        return Ok(best);
    }
    golden_section_tangency(spec, risk_free_rate, upper_bounds, left, r_max, best)
}

/// Solves for the long-only ($w_i \ge 0$, $\sum w_i = 1$) maximum-Sharpe tangency portfolio.
#[verified_engine::verified]
pub fn long_only_tangency(
    spec: &PortfolioSpec,
    risk_free_rate: f64,
) -> Result<PortfolioAllocation, PortfolioError> {
    let ones = [1.0_f64; MAX_PORTFOLIO_ASSETS];
    constrained_tangency(spec, risk_free_rate, &ones[..spec.asset_count])
}

/// Populates `out_points` in place along the constrained efficient frontier from
/// $\mu_{\text{GMV}}$ to $R_{\max}$ under $0 \le w_i \le u_i$.
#[verified_engine::verified]
pub fn constrained_frontier_in_place(
    spec: &PortfolioSpec,
    risk_free_rate: f64,
    upper_bounds: &[f64],
    out_points: &mut [PortfolioAllocation],
) -> Result<usize, PortfolioError> {
    if !risk_free_rate.is_finite() {
        return Err(PortfolioError::NonFiniteInput);
    }
    let ub = validate_upper_bounds(upper_bounds, spec.asset_count)?;
    if out_points.is_empty() {
        return Ok(0);
    }
    let gmv = constrained_min_variance(spec, risk_free_rate, upper_bounds)?;
    let (_, r_max) = extreme_return_bounds(spec, &ub);
    let count = out_points.len();
    out_points[0] = gmv;
    if count == 1 {
        return Ok(1);
    }
    let r_start = gmv.expected_return;
    let span = (r_max - r_start).max(0.0);
    for (idx, slot) in out_points.iter_mut().enumerate().skip(1) {
        let frac = (idx as f64) / ((count - 1) as f64);
        let target = (r_start + frac * span).min(r_max);
        *slot = constrained_target_return(spec, target, risk_free_rate, upper_bounds)?;
    }
    Ok(count)
}

/// Populates `out_points` in place along the long-only ($w_i \ge 0$, $\sum w_i = 1$)
/// efficient frontier from $\mu_{\text{GMV}}$ to $\max_i \mu_i$.
#[verified_engine::verified]
pub fn long_only_frontier_in_place(
    spec: &PortfolioSpec,
    risk_free_rate: f64,
    out_points: &mut [PortfolioAllocation],
) -> Result<usize, PortfolioError> {
    let ones = [1.0_f64; MAX_PORTFOLIO_ASSETS];
    constrained_frontier_in_place(spec, risk_free_rate, &ones[..spec.asset_count], out_points)
}

// [cite:clinical_trials_statistics]
