//! Peak-to-trough fractional drawdown series, maximum drawdown, and underwater duration kernels.

use super::RiskError;

fn validate_wealth_point(w: f64) -> Result<f64, RiskError> {
    if !w.is_finite() {
        return Err(RiskError::NonFiniteInput);
    }
    if w <= 0.0 {
        return Err(RiskError::NonPositiveWealth { value: w });
    }
    Ok(w)
}

fn validate_return_step(r: f64) -> Result<f64, RiskError> {
    if !r.is_finite() {
        return Err(RiskError::NonFiniteInput);
    }
    let gross = 1.0 + r;
    if gross <= 0.0 {
        return Err(RiskError::NonPositiveWealth { value: gross });
    }
    Ok(gross)
}

/// Populates `out_drawdowns[..wealth.len()]` with fractional peak-to-trough drawdowns
/// $dd_t = (P_t - W_t) / P_t \in [0, 1)$ and returns the maximum drawdown across `wealth`.
#[verified_engine::verified]
pub fn drawdown_series_from_wealth(
    wealth: &[f64],
    out_drawdowns: &mut [f64],
) -> Result<f64, RiskError> {
    if wealth.is_empty() {
        return Err(RiskError::EmptySeries);
    }
    if out_drawdowns.len() < wealth.len() {
        return Err(RiskError::ScratchBufferTooSmall {
            required: wealth.len(),
            actual: out_drawdowns.len(),
        });
    }
    let mut peak = validate_wealth_point(wealth[0])?;
    let mut max_dd = 0.0_f64;
    for (dst, &raw_w) in out_drawdowns.iter_mut().zip(wealth.iter()) {
        let w = validate_wealth_point(raw_w)?;
        if w > peak {
            peak = w;
        }
        let dd = ((peak - w) / peak).max(0.0);
        *dst = dd;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    Ok(max_dd)
}

/// Compounds unit initial wealth $W_0 = 1$ through periodic `returns`, populates
/// `out_drawdowns[..returns.len()]` with fractional drawdowns, and returns `max_drawdown`.
#[verified_engine::verified]
pub fn drawdown_series_from_returns(
    returns: &[f64],
    out_drawdowns: &mut [f64],
) -> Result<f64, RiskError> {
    if returns.is_empty() {
        return Err(RiskError::EmptySeries);
    }
    if out_drawdowns.len() < returns.len() {
        return Err(RiskError::ScratchBufferTooSmall {
            required: returns.len(),
            actual: out_drawdowns.len(),
        });
    }
    let mut wealth = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut max_dd = 0.0_f64;
    for (dst, &r) in out_drawdowns.iter_mut().zip(returns.iter()) {
        let gross = validate_return_step(r)?;
        wealth *= gross;
        if !wealth.is_finite() {
            return Err(RiskError::NonFiniteInput);
        }
        if wealth > peak {
            peak = wealth;
        }
        let dd = ((peak - wealth) / peak).max(0.0);
        *dst = dd;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    Ok(max_dd)
}

/// Computes the peak-to-trough fractional maximum drawdown $\max_t (P_t - W_t) / P_t$
/// across a strictly positive `wealth` trajectory in $O(1)$ stack memory.
#[verified_engine::verified]
pub fn max_drawdown_wealth(wealth: &[f64]) -> Result<f64, RiskError> {
    if wealth.is_empty() {
        return Err(RiskError::EmptySeries);
    }
    let mut peak = validate_wealth_point(wealth[0])?;
    let mut max_dd = 0.0_f64;
    for &raw_w in wealth {
        let w = validate_wealth_point(raw_w)?;
        if w > peak {
            peak = w;
        }
        let dd = ((peak - w) / peak).max(0.0);
        if dd > max_dd {
            max_dd = dd;
        }
    }
    Ok(max_dd)
}

/// Computes the peak-to-trough fractional maximum drawdown for compounded unit wealth
/// across a periodic `returns` slice in $O(1)$ stack memory.
#[verified_engine::verified]
pub fn max_drawdown_returns(returns: &[f64]) -> Result<f64, RiskError> {
    if returns.is_empty() {
        return Err(RiskError::EmptySeries);
    }
    let mut wealth = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut max_dd = 0.0_f64;
    for &r in returns {
        let gross = validate_return_step(r)?;
        wealth *= gross;
        if !wealth.is_finite() {
            return Err(RiskError::NonFiniteInput);
        }
        if wealth > peak {
            peak = wealth;
        }
        let dd = ((peak - wealth) / peak).max(0.0);
        if dd > max_dd {
            max_dd = dd;
        }
    }
    Ok(max_dd)
}

/// Computes the maximum number of consecutive steps where `wealth[t]` remains strictly
/// below its running high-water mark.
#[verified_engine::verified]
pub fn max_underwater_duration_wealth(wealth: &[f64]) -> Result<usize, RiskError> {
    if wealth.is_empty() {
        return Err(RiskError::EmptySeries);
    }
    let mut peak = validate_wealth_point(wealth[0])?;
    let mut current_span = 0_usize;
    let mut max_span = 0_usize;
    for &raw_w in wealth {
        let w = validate_wealth_point(raw_w)?;
        if w >= peak {
            peak = w;
            current_span = 0;
        } else {
            current_span += 1;
            if current_span > max_span {
                max_span = current_span;
            }
        }
    }
    Ok(max_span)
}

/// Computes the maximum number of consecutive periods where compounded wealth from `returns`
/// remains strictly below its running high-water mark (relative to $W_0 = 1$).
#[verified_engine::verified]
pub fn max_underwater_duration_returns(returns: &[f64]) -> Result<usize, RiskError> {
    if returns.is_empty() {
        return Err(RiskError::EmptySeries);
    }
    let mut wealth = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut current_span = 0_usize;
    let mut max_span = 0_usize;
    for &r in returns {
        let gross = validate_return_step(r)?;
        wealth *= gross;
        if !wealth.is_finite() {
            return Err(RiskError::NonFiniteInput);
        }
        if wealth >= peak - 1e-15 * peak {
            if wealth > peak {
                peak = wealth;
            }
            current_span = 0;
        } else {
            current_span += 1;
            if current_span > max_span {
                max_span = current_span;
            }
        }
    }
    Ok(max_span)
}

// [cite:clinical_trials_statistics]
