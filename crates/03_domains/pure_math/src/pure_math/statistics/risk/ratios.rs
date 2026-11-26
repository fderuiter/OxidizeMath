//! Annualized Sharpe, Sortino, and Calmar performance-ratio kernels.

use super::RiskError;
use super::drawdown::{max_drawdown_returns, max_drawdown_wealth};

const DENOM_EPS: f64 = 1e-15;

fn validate_annualization(periods_per_year: f64) -> Result<f64, RiskError> {
    if !periods_per_year.is_finite() || periods_per_year <= 0.0 {
        return Err(RiskError::InvalidAnnualization {
            value: periods_per_year,
        });
    }
    Ok(periods_per_year)
}

/// Computes the annualized Sharpe ratio $\sqrt{N_{\text{yr}}} \cdot \bar{e} / \sigma_e$
/// for periodic `returns` against `risk_free_per_period`.
#[verified_engine::verified]
pub fn sharpe_ratio(
    returns: &[f64],
    risk_free_per_period: f64,
    periods_per_year: f64,
) -> Result<f64, RiskError> {
    let ann = validate_annualization(periods_per_year)?;
    if !risk_free_per_period.is_finite() {
        return Err(RiskError::NonFiniteInput);
    }
    if returns.is_empty() {
        return Err(RiskError::EmptySeries);
    }
    if returns.len() < 2 {
        return Err(RiskError::InsufficientSamples {
            required: 2,
            actual: returns.len(),
        });
    }
    let n = returns.len() as f64;
    let mut excess_sum = 0.0;
    for &r in returns {
        if !r.is_finite() {
            return Err(RiskError::NonFiniteInput);
        }
        excess_sum += r - risk_free_per_period;
    }
    let mean_excess = excess_sum / n;
    let mut var_sum = 0.0;
    for &r in returns {
        let d = (r - risk_free_per_period) - mean_excess;
        var_sum += d * d;
    }
    let std_excess = (var_sum / (n - 1.0)).sqrt();
    if std_excess <= DENOM_EPS {
        return Err(RiskError::ZeroVolatility);
    }
    Ok(ann.sqrt() * mean_excess / std_excess)
}

/// Computes the annualized Sortino ratio $\sqrt{N_{\text{yr}}} \cdot (\bar{r} - \tau) / \sigma_d$
/// using Sortino-Price downside semi-deviation $\sigma_d = \sqrt{\frac{1}{n}\sum \min(0, r_i - \tau)^2}$.
#[verified_engine::verified]
pub fn sortino_ratio(
    returns: &[f64],
    target_return: f64,
    periods_per_year: f64,
) -> Result<f64, RiskError> {
    let ann = validate_annualization(periods_per_year)?;
    if !target_return.is_finite() {
        return Err(RiskError::NonFiniteInput);
    }
    if returns.is_empty() {
        return Err(RiskError::EmptySeries);
    }
    if returns.len() < 2 {
        return Err(RiskError::InsufficientSamples {
            required: 2,
            actual: returns.len(),
        });
    }
    let n = returns.len() as f64;
    let mut excess_sum = 0.0;
    let mut downside_sq_sum = 0.0;
    for &r in returns {
        if !r.is_finite() {
            return Err(RiskError::NonFiniteInput);
        }
        let diff = r - target_return;
        excess_sum += diff;
        if diff < 0.0 {
            downside_sq_sum += diff * diff;
        }
    }
    let downside_dev = (downside_sq_sum / n).sqrt();
    if downside_dev <= DENOM_EPS {
        return Err(RiskError::ZeroVolatility);
    }
    let mean_excess = excess_sum / n;
    Ok(ann.sqrt() * mean_excess / downside_dev)
}

/// Computes the annualized Calmar ratio $\text{CAGR} / \text{MaxDrawdown}$ for periodic `returns`.
#[verified_engine::verified]
pub fn calmar_ratio(returns: &[f64], periods_per_year: f64) -> Result<f64, RiskError> {
    let ann = validate_annualization(periods_per_year)?;
    if returns.is_empty() {
        return Err(RiskError::EmptySeries);
    }
    let mdd = max_drawdown_returns(returns)?;
    if mdd <= DENOM_EPS {
        return Err(RiskError::ZeroMaxDrawdown);
    }
    let mut log_growth_sum = 0.0;
    for &r in returns {
        log_growth_sum += r.ln_1p();
    }
    let cagr = ((ann / (returns.len() as f64)) * log_growth_sum).exp_m1();
    if !cagr.is_finite() {
        return Err(RiskError::NonFiniteInput);
    }
    Ok(cagr / mdd)
}

/// Computes the annualized Calmar ratio $\text{CAGR} / \text{MaxDrawdown}$ directly from a
/// strictly positive `wealth` series (`wealth.len() >= 2`).
#[verified_engine::verified]
pub fn calmar_ratio_wealth(wealth: &[f64], periods_per_year: f64) -> Result<f64, RiskError> {
    let ann = validate_annualization(periods_per_year)?;
    if wealth.is_empty() {
        return Err(RiskError::EmptySeries);
    }
    if wealth.len() < 2 {
        return Err(RiskError::InsufficientSamples {
            required: 2,
            actual: wealth.len(),
        });
    }
    let mdd = max_drawdown_wealth(wealth)?;
    if mdd <= DENOM_EPS {
        return Err(RiskError::ZeroMaxDrawdown);
    }
    let w_start = wealth[0];
    let w_end = wealth[wealth.len() - 1];
    let periods = (wealth.len() - 1) as f64;
    let cagr = ((ann / periods) * (w_end / w_start).ln()).exp_m1();
    if !cagr.is_finite() {
        return Err(RiskError::NonFiniteInput);
    }
    Ok(cagr / mdd)
}

// [cite:clinical_trials_statistics]
