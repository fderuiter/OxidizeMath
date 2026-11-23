//! Stack-allocated $16 \times 16$ Cholesky linear-algebra and portfolio-moment helpers.

use super::{MAX_PORTFOLIO_ASSETS, PortfolioAllocation, PortfolioError, PortfolioSpec};

const PIVOT_EPS: f64 = 1e-13;

pub(crate) fn cholesky_16(
    cov: &[[f64; MAX_PORTFOLIO_ASSETS]; MAX_PORTFOLIO_ASSETS],
    n: usize,
) -> Result<[[f64; MAX_PORTFOLIO_ASSETS]; MAX_PORTFOLIO_ASSETS], PortfolioError> {
    let mut l = [[0.0_f64; MAX_PORTFOLIO_ASSETS]; MAX_PORTFOLIO_ASSETS];
    let mut scale = 0.0_f64;
    for (idx, row) in cov.iter().enumerate().take(n) {
        if row[idx].abs() > scale {
            scale = row[idx].abs();
        }
    }
    let diag_tol = PIVOT_EPS * scale.max(1.0);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = cov[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if !sum.is_finite() || sum <= diag_tol {
                    return Err(PortfolioError::NotPositiveDefinite);
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Ok(l)
}

pub(crate) fn cholesky_solve_16(
    l: &[[f64; MAX_PORTFOLIO_ASSETS]; MAX_PORTFOLIO_ASSETS],
    rhs: &[f64; MAX_PORTFOLIO_ASSETS],
    n: usize,
) -> [f64; MAX_PORTFOLIO_ASSETS] {
    let mut y = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    for i in 0..n {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        y[i] = sum / l[i][i];
    }
    let mut x = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    for idx in 0..n {
        let i = n - 1 - idx;
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j][i] * x[j];
        }
        x[i] = sum / l[i][i];
    }
    x
}

pub(crate) fn build_allocation(
    spec: &PortfolioSpec,
    weights: &[f64; MAX_PORTFOLIO_ASSETS],
    risk_free_rate: f64,
) -> PortfolioAllocation {
    let n = spec.asset_count;
    let mut exp_ret = 0.0_f64;
    let mut var = 0.0_f64;
    for i in 0..n {
        let wi = weights[i];
        exp_ret += wi * spec.expected_returns[i];
        let mut row_dot = 0.0_f64;
        for j in 0..n {
            row_dot += spec.covariance[i][j] * weights[j];
        }
        var += wi * row_dot;
    }
    let clean_var = var.max(0.0);
    let vol = clean_var.sqrt();
    let sharpe = if vol > 1e-15 {
        (exp_ret - risk_free_rate) / vol
    } else {
        0.0
    };
    let mut clean_weights = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    clean_weights[..n].copy_from_slice(&weights[..n]);
    PortfolioAllocation {
        asset_count: n,
        weights: clean_weights,
        expected_return: exp_ret,
        variance: clean_var,
        volatility: vol,
        sharpe_ratio: sharpe,
    }
}

pub(crate) fn build_allocation_from_slice(
    spec: &PortfolioSpec,
    weights: &[f64],
    risk_free_rate: f64,
) -> Result<PortfolioAllocation, PortfolioError> {
    if !risk_free_rate.is_finite() {
        return Err(PortfolioError::NonFiniteInput);
    }
    if weights.len() != spec.asset_count {
        return Err(PortfolioError::DimensionMismatch {
            expected: spec.asset_count,
            actual: weights.len(),
        });
    }
    let mut buf = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    for (dst, &w) in buf.iter_mut().zip(weights.iter()) {
        if !w.is_finite() {
            return Err(PortfolioError::NonFiniteInput);
        }
        *dst = w;
    }
    Ok(build_allocation(spec, &buf, risk_free_rate))
}

// [cite:clinical_trials_statistics]
