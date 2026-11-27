//! # Statistical Tail-Risk, Drawdown, and Performance-Ratio Kernels
//!
//! Zero-allocation statistical risk primitives for capital-allocation trajectories
//! and periodic asset return series:
//!
//! - **Tail Loss Quantiles (`VaR` & `CVaR`)**: Historical empirical (Acerbi-Tasche
//!   coherent expected shortfall), parametric Gaussian, and 4th-order Cornish-Fisher
//!   semi-parametric quantile and expected-shortfall estimators for confidence levels
//!   $\alpha \in (0.5, 1.0)$.
//! - **Drawdown & Underwater Recovery**: Fractional peak-to-trough drawdown series,
//!   maximum drawdown, and maximum consecutive underwater duration for both periodic
//!   return slices and compounded wealth slices.
//! - **Risk-Adjusted Performance Ratios**: Annualized Sharpe, Sortino (downside
//!   semi-deviation), and Calmar (geometric annualized return over maximum drawdown)
//!   ratios.
//!
//! All routines operate strictly on caller-supplied slices and stack accumulators,
//! performing zero dynamic heap allocations under `VerifiedAllocator`.
//!
//! ## Sign Convention
//!
//! Periodic returns $r_t$ are negative during losses ($r_t < 0$), and tail-risk
//! estimators report positive fractional loss magnitudes ($L = -R$): a 95% `VaR`
//! of `0.042` indicates a 4.2% one-period threshold loss exceeded with probability
//! $1 - \alpha = 5\%$, with coherent expected shortfall satisfying
//! $\text{CVaR}_\alpha \ge \text{VaR}_\alpha$.
//!
//! ## Example
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::risk::{
//!     gaussian_cvar, gaussian_var, historical_cvar, historical_var, max_drawdown_returns,
//!     sharpe_ratio,
//! };
//!
//! let returns = [-0.04, -0.02, 0.01, 0.015, 0.02, -0.01, 0.025, 0.012, -0.03, 0.018];
//! let mut scratch = [0.0_f64; 10];
//!
//! let h_var = historical_var(&returns, 0.90, &mut scratch).unwrap();
//! let h_cvar = historical_cvar(&returns, 0.90, &mut scratch).unwrap();
//! assert!(h_cvar >= h_var);
//!
//! let g_var = gaussian_var(&returns, 0.95).unwrap();
//! let g_cvar = gaussian_cvar(&returns, 0.95).unwrap();
//! assert!(g_cvar > g_var);
//!
//! let mdd = max_drawdown_returns(&returns).unwrap();
//! let sharpe = sharpe_ratio(&returns, 0.0001, 252.0).unwrap();
//! assert!(mdd > 0.0 && sharpe.is_finite());
//! ```

mod drawdown;
mod ratios;
mod tail;

#[cfg(test)]
mod tests;

pub use drawdown::{
    drawdown_series_from_returns, drawdown_series_from_wealth, max_drawdown_returns,
    max_drawdown_wealth, max_underwater_duration_returns, max_underwater_duration_wealth,
};
pub use ratios::{calmar_ratio, calmar_ratio_wealth, sharpe_ratio, sortino_ratio};
pub use tail::{
    cornish_fisher_cvar, cornish_fisher_var, gaussian_cvar, gaussian_var, historical_cvar,
    historical_var, validate_confidence,
};

/// Stack-allocated error domain for statistical tail-risk, drawdown, and ratio kernels.
///
/// Kept `Copy` (zero heap fields) so error paths inside `VerifiedAllocator` never allocate.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RiskError {
    /// Input return or wealth slice is empty.
    #[error("input return or wealth series must not be empty")]
    EmptySeries,
    /// Input slice contains fewer observations than required by the statistical estimator.
    #[error("insufficient samples: required at least {required}, got {actual}")]
    InsufficientSamples {
        /// Minimum sample length required.
        required: usize,
        /// Actual slice length provided.
        actual: usize,
    },
    /// Confidence level lies outside the open interval `(0.5, 1.0)` or is non-finite.
    #[error("confidence level {value} must be finite and strictly within (0.5, 1.0)")]
    InvalidConfidence {
        /// Invalid confidence parameter.
        value: f64,
    },
    /// Annualization factor (`periods_per_year`) is non-positive or non-finite.
    #[error("periods_per_year {value} must be finite and strictly positive")]
    InvalidAnnualization {
        /// Invalid annualization frequency.
        value: f64,
    },
    /// Input slice or scalar parameter contains `NaN` or infinite values.
    #[error("input contains non-finite (NaN or infinite) values")]
    NonFiniteInput,
    /// Wealth level or gross return factor `1 + r` is non-positive (`<= 0.0`).
    #[error("wealth or gross return factor {value} must be strictly positive")]
    NonPositiveWealth {
        /// Non-positive wealth or gross return encountered.
        value: f64,
    },
    /// Sample volatility or downside semi-deviation vanishes, preventing normalization.
    #[error("sample volatility or downside deviation is zero")]
    ZeroVolatility,
    /// Maximum drawdown vanishes (`0.0`), leaving the Calmar ratio undefined.
    #[error("maximum drawdown is zero; Calmar ratio denominator vanishes")]
    ZeroMaxDrawdown,
    /// Caller-owned scratch or output buffer is shorter than the input series.
    #[error("scratch buffer too small: required at least {required}, got {actual}")]
    ScratchBufferTooSmall {
        /// Minimum slice capacity required.
        required: usize,
        /// Actual slice length supplied.
        actual: usize,
    },
}

impl math_diagnostics::Diagnostic for RiskError {
    #[verified_engine::verified]
    fn severity(&self) -> math_diagnostics::Severity {
        math_diagnostics::Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn math_diagnostics::DiagnosticVisitor) {
        visitor.record_str("error_type", "RiskError");
        visitor.record_display("description", self);
    }
}

/// Consolidated zero-allocation statistical risk summary for a periodic return series.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RiskSummary {
    /// Empirical historical Value-at-Risk at `confidence`.
    pub historical_var: f64,
    /// Coherent empirical Expected Shortfall (`CVaR`) at `confidence`.
    pub historical_cvar: f64,
    /// Parametric Gaussian Value-at-Risk at `confidence`.
    pub gaussian_var: f64,
    /// Parametric Gaussian Expected Shortfall (`CVaR`) at `confidence`.
    pub gaussian_cvar: f64,
    /// 4th-order Cornish-Fisher Value-at-Risk at `confidence`.
    pub cornish_fisher_var: f64,
    /// 4th-order Cornish-Fisher Expected Shortfall (`CVaR`) at `confidence`.
    pub cornish_fisher_cvar: f64,
    /// Annualized Sharpe ratio.
    pub sharpe: f64,
    /// Annualized Sortino ratio relative to `risk_free_per_period`.
    pub sortino: f64,
    /// Annualized Calmar ratio (`0.0` when `max_drawdown == 0.0`).
    pub calmar: f64,
    /// Peak-to-trough fractional maximum drawdown in `[0.0, 1.0)`.
    pub max_drawdown: f64,
    /// Longest consecutive span strictly below a prior wealth peak.
    pub max_underwater_duration: usize,
}

/// Evaluates tail-risk quantiles, drawdown metrics, and performance ratios in a single pass
/// using caller-owned `scratch` (`scratch.len() >= returns.len()`) with zero heap allocations.
#[verified_engine::verified]
pub fn summarize_return_risk(
    returns: &[f64],
    confidence: f64,
    risk_free_per_period: f64,
    periods_per_year: f64,
    scratch: &mut [f64],
) -> Result<RiskSummary, RiskError> {
    let h_var = historical_var(returns, confidence, scratch)?;
    let h_cvar = historical_cvar(returns, confidence, scratch)?;
    let g_var = gaussian_var(returns, confidence)?;
    let g_cvar = gaussian_cvar(returns, confidence)?;
    let cf_var = cornish_fisher_var(returns, confidence)?;
    let cf_cvar = cornish_fisher_cvar(returns, confidence)?;
    let sharpe = sharpe_ratio(returns, risk_free_per_period, periods_per_year)?;
    let sortino = sortino_ratio(returns, risk_free_per_period, periods_per_year)?;
    let mdd = max_drawdown_returns(returns)?;
    let calmar = match calmar_ratio(returns, periods_per_year) {
        Ok(val) => val,
        Err(RiskError::ZeroMaxDrawdown) => 0.0,
        Err(err) => return Err(err),
    };
    let max_underwater_duration = max_underwater_duration_returns(returns)?;
    Ok(RiskSummary {
        historical_var: h_var,
        historical_cvar: h_cvar,
        gaussian_var: g_var,
        gaussian_cvar: g_cvar,
        cornish_fisher_var: cf_var,
        cornish_fisher_cvar: cf_cvar,
        sharpe,
        sortino,
        calmar,
        max_drawdown: mdd,
        max_underwater_duration,
    })
}

// [cite:clinical_trials_statistics]
