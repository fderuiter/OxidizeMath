//! Value-at-Risk (`VaR`) and Conditional Value-at-Risk / Expected Shortfall (`CVaR`)
//! calculators over empirical loss slices, parametric log-normal wealth, and Cholesky Monte Carlo paths.

use super::portfolio::{
    PortfolioKellyWorkspace, cholesky_into_flat, evaluate_portfolio_kelly_metrics,
};
use crate::error::KellyError;
use crate::pure_math::statistics::copula::transforms::{fast_norm_cdf, fast_norm_inv};
use math_rng::sample_open;
use math_scalars::mat_idx;
use nalgebra::DMatrix;
use oxidize_core::rng::OxidizeRng;

/// Paired Value-at-Risk (`VaR`) and Conditional Value-at-Risk (`CVaR`) at a single confidence level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailRiskEstimate {
    /// Confidence level $\alpha \in (0, 1)$, e.g. `0.95` or `0.99`.
    pub confidence: f64,
    /// Value-at-Risk loss threshold at `confidence` (positive values denote capital loss).
    pub var: f64,
    /// Conditional Value-at-Risk / Expected Shortfall $\mathbb{E}\[L \mid L \ge \text{VaR}_\alpha\]$.
    pub cvar: f64,
}

/// Standard `95%` and `99%` Value-at-Risk and Conditional Value-at-Risk stress report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailRiskMetrics {
    /// `95%` Value-at-Risk loss.
    pub var_95: f64,
    /// `95%` Conditional Value-at-Risk (Expected Shortfall) loss.
    pub cvar_95: f64,
    /// `99%` Value-at-Risk loss.
    pub var_99: f64,
    /// `99%` Conditional Value-at-Risk (Expected Shortfall) loss.
    pub cvar_99: f64,
}

/// Capital, risk-free rate, and holding horizon parameters for portfolio tail-risk valuation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortfolioHorizonSpec {
    /// Continuously compounded risk-free rate $r$.
    pub risk_free_rate: f64,
    /// Initial capital / bankroll $W_0 > 0$.
    pub bankroll: f64,
    /// Evaluation horizon $T > 0$ in years.
    pub horizon: f64,
}

impl PortfolioHorizonSpec {
    /// Creates validated horizon and bankroll parameters for tail-risk valuation.
    #[verified_engine::verified]
    pub fn new(risk_free_rate: f64, bankroll: f64, horizon: f64) -> Result<Self, KellyError> {
        if !risk_free_rate.is_finite() {
            return Err(KellyError::InvalidParameter {
                reason: "risk-free rate must be finite",
            });
        }
        if !bankroll.is_finite() || bankroll <= 0.0 {
            return Err(KellyError::InvalidBankroll { value: bankroll });
        }
        if !horizon.is_finite() || horizon <= 0.0 {
            return Err(KellyError::InvalidParameter {
                reason: "time horizon must be finite and positive",
            });
        }
        Ok(Self {
            risk_free_rate,
            bankroll,
            horizon,
        })
    }
}

fn eval_sorted_var_cvar(sorted_losses: &[f64], confidence: f64) -> TailRiskEstimate {
    let n = sorted_losses.len();
    let raw_idx = ((n as f64) * confidence).floor() as usize;
    let cutoff_idx = raw_idx.min(n - 1);
    let var = sorted_losses[cutoff_idx];
    let tail_slice = &sorted_losses[cutoff_idx..];
    let cvar = tail_slice.iter().sum::<f64>() / (tail_slice.len() as f64);
    TailRiskEstimate {
        confidence,
        var,
        cvar,
    }
}

/// Computes empirical `VaR` and `CVaR` at `confidence` by sorting `losses_scratch` in-place
/// with zero heap allocations (positive entries represent losses; negative entries represent gains).
#[verified_engine::verified]
pub fn empirical_var_cvar_in_place(
    losses_scratch: &mut [f64],
    confidence: f64,
) -> Result<TailRiskEstimate, KellyError> {
    if losses_scratch.is_empty() {
        return Err(KellyError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    if !confidence.is_finite() || confidence <= 0.0 || confidence >= 1.0 {
        return Err(KellyError::InvalidConfidenceLevel { value: confidence });
    }
    if losses_scratch.iter().any(|v| !v.is_finite()) {
        return Err(KellyError::InvalidParameter {
            reason: "loss slice must contain only finite values",
        });
    }
    losses_scratch.sort_unstable_by(f64::total_cmp);
    Ok(eval_sorted_var_cvar(losses_scratch, confidence))
}

/// Computes both `95%` and `99%` empirical `VaR` and `CVaR` in-place with zero heap allocations.
#[verified_engine::verified]
pub fn empirical_tail_risk_in_place(
    losses_scratch: &mut [f64],
) -> Result<TailRiskMetrics, KellyError> {
    if losses_scratch.is_empty() {
        return Err(KellyError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    if losses_scratch.iter().any(|v| !v.is_finite()) {
        return Err(KellyError::InvalidParameter {
            reason: "loss slice must contain only finite values",
        });
    }
    losses_scratch.sort_unstable_by(f64::total_cmp);
    let r95 = eval_sorted_var_cvar(losses_scratch, 0.95);
    let r99 = eval_sorted_var_cvar(losses_scratch, 0.99);
    Ok(TailRiskMetrics {
        var_95: r95.var,
        cvar_95: r95.cvar,
        var_99: r99.var,
        cvar_99: r99.cvar,
    })
}

/// Computes `95%` and `99%` `VaR` and `CVaR` over an immutable loss slice.
#[verified_engine::verified]
pub fn empirical_tail_risk(losses: &[f64]) -> Result<TailRiskMetrics, KellyError> {
    let mut scratch = losses.to_vec();
    empirical_tail_risk_in_place(&mut scratch)
}

fn lognormal_var_cvar_at(
    bankroll: f64,
    drift: f64,
    vol_horizon: f64,
    confidence: f64,
) -> TailRiskEstimate {
    let tail_prob = 1.0 - confidence;
    if vol_horizon <= 1e-15 {
        let loss = bankroll * (1.0 - drift.exp());
        return TailRiskEstimate {
            confidence,
            var: loss,
            cvar: loss,
        };
    }
    let z_alpha = fast_norm_inv(tail_prob);
    let var = bankroll * (1.0 - (drift + vol_horizon * z_alpha).exp());
    let expected_tail_wealth_ratio = (drift + 0.5 * vol_horizon * vol_horizon).exp()
        * fast_norm_cdf(z_alpha - vol_horizon)
        / tail_prob;
    let cvar = bankroll * (1.0 - expected_tail_wealth_ratio);
    TailRiskEstimate {
        confidence,
        var,
        cvar,
    }
}

/// Computes exact closed-form log-normal wealth `95%` and `99%` Value-at-Risk (`VaR`) and
/// Conditional Value-at-Risk (`CVaR`) dollar losses for portfolio weights $\mathbf{w}$ over
/// horizon $T$ (`spec.horizon`) on initial bankroll $W_0$ (`spec.bankroll`):
/// $$\operatorname{VaR}_\alpha = W_0\Bigl(1 - \exp\bigl(\mu_g T + \sigma_p \sqrt{T}\,\Phi^{-1}(1 - \alpha)\bigr)\Bigr), \qquad \operatorname{CVaR}_\alpha = W_0\left(1 - \frac{\exp\bigl(\mu_g T + \tfrac{1}{2}\sigma_p^2 T\bigr)\,\Phi\bigl(\Phi^{-1}(1 - \alpha) - \sigma_p\sqrt{T}\bigr)}{1 - \alpha}\right),$$
/// where $\mu_g = r + \mathbf{w}^\top(\boldsymbol{\mu} - r\mathbf{1}) - \tfrac{1}{2}\mathbf{w}^\top \Sigma \mathbf{w}$ and $\sigma_p = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}$.
///
/// Reference: Rockafellar & Uryasev (2000), *J. Risk* 2(3), DOI: 10.21314/JOR.2000.038;
/// [`docs/math_specs/pure_math.md`](../../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Short-circuits deterministic zero-volatility portfolios ($\sigma_p \sqrt{T} \le 10^{-15}$) to
/// $W_0(1 - e^{\mu_g T})$ to avoid $0/0$ tail-quantile conditioning and clamps quadratic variance
/// $\mathbf{w}^\top \Sigma \mathbf{w} \ge 0$ against floating-point roundoff.
///
/// # Domain of Validity
/// Requires $N \ge 1$ finite weights and expected returns, symmetric positive-definite $N \times N$
/// `covariance`, finite `spec.risk_free_rate`, strictly positive `spec.bankroll > 0`, and strictly
/// positive `spec.horizon > 0`.
#[verified_engine::verified]
pub fn parametric_lognormal_tail_risk(
    weights: &[f64],
    expected_returns: &[f64],
    covariance: &DMatrix<f64>,
    spec: PortfolioHorizonSpec,
) -> Result<TailRiskMetrics, KellyError> {
    let valid = PortfolioHorizonSpec::new(spec.risk_free_rate, spec.bankroll, spec.horizon)?;
    let metrics = evaluate_portfolio_kelly_metrics(
        weights,
        expected_returns,
        covariance,
        valid.risk_free_rate,
    )?;
    let drift = metrics.total_log_growth * valid.horizon;
    let vol_h = metrics.portfolio_volatility * valid.horizon.sqrt();
    let r95 = lognormal_var_cvar_at(valid.bankroll, drift, vol_h, 0.95);
    let r99 = lognormal_var_cvar_at(valid.bankroll, drift, vol_h, 0.99);
    Ok(TailRiskMetrics {
        var_95: r95.var,
        cvar_95: r95.cvar,
        var_99: r99.var,
        cvar_99: r99.cvar,
    })
}

#[inline]
fn next_standard_normal(rng: &mut OxidizeRng) -> f64 {
    let u: f64 = sample_open(rng);
    fast_norm_inv(u)
}

fn sample_antithetic_wealth_losses(
    factor_loadings: &[f64],
    bankroll: f64,
    drift: f64,
    sqrt_h: f64,
    rng: &mut OxidizeRng,
    out_losses: &mut [f64],
) {
    let pairs = out_losses.len() / 2;
    for pair_idx in 0..pairs {
        let mut shock = 0.0;
        for &loading in factor_loadings {
            shock += loading * next_standard_normal(rng);
        }
        out_losses[2 * pair_idx] = bankroll * (1.0 - (drift + sqrt_h * shock).exp());
        out_losses[2 * pair_idx + 1] = bankroll * (1.0 - (drift - sqrt_h * shock).exp());
    }
    if out_losses.len() % 2 == 1 {
        let mut shock = 0.0;
        for &loading in factor_loadings {
            shock += loading * next_standard_normal(rng);
        }
        if let Some(last) = out_losses.last_mut() {
            *last = bankroll * (1.0 - (drift + sqrt_h * shock).exp());
        }
    }
}

/// Simulates correlated multi-asset log-wealth dollar losses into `out_losses` with zero heap
/// allocations and returns empirical `95%` and `99%` `VaR` and `CVaR` over the simulated slice.
#[verified_engine::verified]
pub fn simulate_portfolio_tail_risk_into(
    weights: &[f64],
    expected_returns: &[f64],
    covariance: &DMatrix<f64>,
    spec: PortfolioHorizonSpec,
    rng: &mut OxidizeRng,
    out_losses: &mut [f64],
    workspace: &mut PortfolioKellyWorkspace,
) -> Result<TailRiskMetrics, KellyError> {
    let valid = PortfolioHorizonSpec::new(spec.risk_free_rate, spec.bankroll, spec.horizon)?;
    if out_losses.is_empty() {
        return Err(KellyError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    let metrics = evaluate_portfolio_kelly_metrics(
        weights,
        expected_returns,
        covariance,
        valid.risk_free_rate,
    )?;
    let n = weights.len();
    workspace.check_capacity(n)?;
    cholesky_into_flat(
        covariance,
        n,
        &mut workspace.sym_flat,
        &mut workspace.chol_flat,
    )?;

    for k in 0..n {
        let mut col_dot = 0.0;
        for (i, &wi) in weights.iter().enumerate().skip(k) {
            col_dot += workspace.chol_flat[mat_idx(i, k, n)] * wi;
        }
        workspace.grad[k] = col_dot;
    }

    let drift = metrics.total_log_growth * valid.horizon;
    sample_antithetic_wealth_losses(
        &workspace.grad[..n],
        valid.bankroll,
        drift,
        valid.horizon.sqrt(),
        rng,
        out_losses,
    );
    empirical_tail_risk_in_place(out_losses)
}

/// Simulates `n_paths` correlated portfolio wealth outcomes using `rng`
/// and returns `95%` and `99%` `VaR` and `CVaR`.
#[verified_engine::verified]
pub fn simulated_portfolio_tail_risk_with_rng(
    weights: &[f64],
    expected_returns: &[f64],
    covariance: &DMatrix<f64>,
    spec: PortfolioHorizonSpec,
    n_paths: usize,
    rng: &mut OxidizeRng,
) -> Result<TailRiskMetrics, KellyError> {
    if n_paths == 0 {
        return Err(KellyError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    let mut losses = vec![0.0; n_paths];
    let mut ws = PortfolioKellyWorkspace::with_capacity(weights.len().max(1));
    simulate_portfolio_tail_risk_into(
        weights,
        expected_returns,
        covariance,
        spec,
        rng,
        &mut losses,
        &mut ws,
    )
}

/// Simulates `n_paths` correlated portfolio wealth outcomes from deterministic `seed`
/// and returns `95%` and `99%` `VaR` and `CVaR`.
#[verified_engine::verified]
pub fn simulated_portfolio_tail_risk(
    weights: &[f64],
    expected_returns: &[f64],
    covariance: &DMatrix<f64>,
    spec: PortfolioHorizonSpec,
    n_paths: usize,
    seed: u64,
) -> Result<TailRiskMetrics, KellyError> {
    let mut rng = OxidizeRng::new(seed);
    simulated_portfolio_tail_risk_with_rng(
        weights,
        expected_returns,
        covariance,
        spec,
        n_paths,
        &mut rng,
    )
}

// [cite:clinical_trials_statistics]
