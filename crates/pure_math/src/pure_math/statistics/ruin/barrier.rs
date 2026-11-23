//! Proportional-Kelly drawdown barrier probabilities via diffusion and log-return Lundberg exponents.

use super::analytical::{MAX_SAFE_EXP_ARG, solve_positive_mgf_root};
use crate::error::KellyError;
use crate::pure_math::statistics::kelly::{BankrollFraction, Odds, UnitInterval};

/// Weighted proportional-stake wager entry inside a heterogeneous Kelly portfolio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProportionalPortfolioBet {
    /// True win probability $p_k \in (0, 1)$.
    pub win_prob: UnitInterval,
    /// Decimal payoff odds $O_k = 1 + b_k > 1$.
    pub odds: Odds,
    /// Bankroll fraction wagered ($f_k \in (0, 1)$).
    pub fraction: BankrollFraction,
    /// Relative portfolio sampling frequency ($w_k > 0$).
    pub weight: f64,
}

impl ProportionalPortfolioBet {
    /// Validates and constructs a [`ProportionalPortfolioBet`].
    #[verified_engine::verified]
    pub fn new(
        win_prob: UnitInterval,
        odds: Odds,
        fraction: BankrollFraction,
        weight: f64,
    ) -> Result<Self, KellyError> {
        let p = win_prob.value();
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err(KellyError::InvalidProbability { value: p });
        }
        let f = fraction.value();
        if !f.is_finite() || !(0.0..=1.0).contains(&f) {
            return Err(KellyError::InvalidFraction { value: f });
        }
        if !weight.is_finite() || weight <= 0.0 {
            return Err(KellyError::InvalidStake { value: weight });
        }
        Ok(Self {
            win_prob,
            odds,
            fraction,
            weight,
        })
    }

    /// Constructs a [`ProportionalPortfolioBet`] with unit relative weight (`1.0`).
    #[verified_engine::verified]
    pub fn unit_weight(
        win_prob: UnitInterval,
        odds: Odds,
        fraction: BankrollFraction,
    ) -> Result<Self, KellyError> {
        Self::new(win_prob, odds, fraction, 1.0)
    }
}

/// Proportional-Kelly drawdown barrier probabilities for wealth floor $B_{\text{floor}} / W_0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KellyBarrierRisk {
    /// Wealth floor ratio $\alpha = B_{\text{floor}} / W_0 \in (0, 1)$.
    pub barrier_ratio: f64,
    /// Expected logarithmic return per bet $\mu_{\log} = E[\ln(1 + f R)]$.
    pub log_drift: f64,
    /// Variance of logarithmic return per bet $\sigma^2_{\log} = \operatorname{Var}(\ln(1 + f R))$.
    pub log_variance: f64,
    /// Brownian diffusion exponent $2 \mu_{\log} / \sigma^2_{\log}$.
    pub diffusion_exponent: f64,
    /// Brownian diffusion barrier probability $(B_{\text{floor}} / W_0)^{2 \mu_{\log} / \sigma^2_{\log}}$.
    pub diffusion_barrier_prob: f64,
    /// Exact log-return Lundberg exponent $\theta^*$ solving $E[(1 + f R)^{-\theta}] = 1$.
    pub lundberg_exponent: f64,
    /// Cramér-Lundberg asymptotic barrier bound $(B_{\text{floor}} / W_0)^{\theta^*}$.
    pub lundberg_barrier_bound: f64,
}

/// Computes proportional-Kelly drawdown barrier probabilities for floor ratio $B_{\text{floor}} / W_0$.
///
/// Evaluates both the Brownian diffusion approximation $(B_{\text{floor}} / W_0)^{2 \mu / \sigma^2}$
/// and the exact log-return Lundberg exponent solving $E[(1 + f R)^{-\theta}] = 1$.
#[verified_engine::verified]
pub fn kelly_barrier_risk(
    win_prob: &UnitInterval,
    odds: &Odds,
    fraction: &BankrollFraction,
    barrier_ratio: f64,
) -> Result<KellyBarrierRisk, KellyError> {
    if !barrier_ratio.is_finite() || barrier_ratio <= 0.0 || barrier_ratio >= 1.0 {
        return Err(KellyError::InvalidBarrier {
            floor: barrier_ratio,
            initial_wealth: 1.0,
        });
    }
    let p = win_prob.value();
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return Err(KellyError::InvalidProbability { value: p });
    }
    let f = fraction.value();
    if f == 0.0 {
        return Ok(KellyBarrierRisk {
            barrier_ratio,
            log_drift: 0.0,
            log_variance: 0.0,
            diffusion_exponent: f64::INFINITY,
            diffusion_barrier_prob: 0.0,
            lundberg_exponent: f64::INFINITY,
            lundberg_barrier_bound: 0.0,
        });
    }
    if f >= 1.0 {
        return Ok(KellyBarrierRisk {
            barrier_ratio,
            log_drift: f64::NEG_INFINITY,
            log_variance: f64::INFINITY,
            diffusion_exponent: 0.0,
            diffusion_barrier_prob: 1.0,
            lundberg_exponent: 0.0,
            lundberg_barrier_bound: 1.0,
        });
    }
    let bet = ProportionalPortfolioBet::unit_weight(*win_prob, *odds, *fraction)?;
    portfolio_kelly_barrier_risk(&[bet], barrier_ratio)
}

struct LogMoments {
    total_w: f64,
    max_neg_log: f64,
    log_drift: f64,
    log_variance: f64,
}

fn compute_log_moments(portfolio: &[ProportionalPortfolioBet]) -> Result<LogMoments, KellyError> {
    if portfolio.is_empty() {
        return Err(KellyError::InsufficientBets {
            required: 1,
            actual: 0,
        });
    }
    let mut total_w = 0.0;
    let mut max_neg_log = 0.0_f64;
    for item in portfolio {
        let f = item.fraction.value();
        if !f.is_finite() || f <= 0.0 || f >= 1.0 {
            return Err(KellyError::InvalidFraction { value: f });
        }
        total_w += item.weight;
        max_neg_log = max_neg_log.max(-(-f).ln_1p());
    }
    let mut log_drift = 0.0;
    let mut second_moment = 0.0;
    for item in portfolio {
        let alpha = item.weight / total_w;
        let p = item.win_prob.value();
        let z_win = (item.odds.net_profit_multiplier() * item.fraction.value()).ln_1p();
        let z_loss = (-item.fraction.value()).ln_1p();
        log_drift += alpha * (p * z_win + (1.0 - p) * z_loss);
        second_moment += alpha * (p * z_win * z_win + (1.0 - p) * z_loss * z_loss);
    }
    Ok(LogMoments {
        total_w,
        max_neg_log,
        log_drift,
        log_variance: (second_moment - log_drift * log_drift).max(1e-18),
    })
}

/// Computes proportional-Kelly drawdown barrier probabilities across a heterogeneous portfolio.
#[verified_engine::verified]
pub fn portfolio_kelly_barrier_risk(
    portfolio: &[ProportionalPortfolioBet],
    barrier_ratio: f64,
) -> Result<KellyBarrierRisk, KellyError> {
    if !barrier_ratio.is_finite() || barrier_ratio <= 0.0 || barrier_ratio >= 1.0 {
        return Err(KellyError::InvalidBarrier {
            floor: barrier_ratio,
            initial_wealth: 1.0,
        });
    }
    let m = compute_log_moments(portfolio)?;
    let diffusion_exponent = 2.0 * m.log_drift / m.log_variance;
    if m.log_drift <= 0.0 {
        return Ok(KellyBarrierRisk {
            barrier_ratio,
            log_drift: m.log_drift,
            log_variance: m.log_variance,
            diffusion_exponent,
            diffusion_barrier_prob: 1.0,
            lundberg_exponent: 0.0,
            lundberg_barrier_bound: 1.0,
        });
    }

    let theta_cap = MAX_SAFE_EXP_ARG / m.max_neg_log.max(1e-12);
    let mgf = |theta: f64| {
        let mut acc = 0.0;
        for item in portfolio {
            let alpha = item.weight / m.total_w;
            let p = item.win_prob.value();
            let z_win = (item.odds.net_profit_multiplier() * item.fraction.value()).ln_1p();
            let z_loss = (-item.fraction.value()).ln_1p();
            acc += alpha * (p * (-theta * z_win).exp() + (1.0 - p) * (-theta * z_loss).exp());
        }
        acc - 1.0
    };
    let mgf_prime = |theta: f64| {
        let mut acc = 0.0;
        for item in portfolio {
            let alpha = item.weight / m.total_w;
            let p = item.win_prob.value();
            let z_win = (item.odds.net_profit_multiplier() * item.fraction.value()).ln_1p();
            let z_loss = (-item.fraction.value()).ln_1p();
            acc += alpha
                * (-p * z_win * (-theta * z_win).exp()
                    - (1.0 - p) * z_loss * (-theta * z_loss).exp());
        }
        acc
    };

    let lundberg_exponent =
        solve_positive_mgf_root(mgf, mgf_prime, 0.5 * diffusion_exponent, theta_cap)?;
    Ok(KellyBarrierRisk {
        barrier_ratio,
        log_drift: m.log_drift,
        log_variance: m.log_variance,
        diffusion_exponent,
        diffusion_barrier_prob: barrier_ratio.powf(diffusion_exponent).clamp(0.0, 1.0),
        lundberg_exponent,
        lundberg_barrier_bound: barrier_ratio.powf(lundberg_exponent).clamp(0.0, 1.0),
    })
}

// [cite:clinical_trials_statistics]
