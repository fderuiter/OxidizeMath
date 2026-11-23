//! Exact Gambler's Ruin and fixed-stake Cramér-Lundberg adjustment coefficient solvers.

use crate::error::KellyError;
use crate::pure_math::analysis::roots::{Bisection, NewtonRaphson, RootFinder};
use crate::pure_math::statistics::kelly::{Odds, UnitInterval};

pub(crate) const MAX_SAFE_EXP_ARG: f64 = 500.0;

/// Analytical absorption probabilities for two-barrier fixed-stake Gambler's Ruin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GamblersRuinResult {
    /// Probability of hitting zero before `target_wealth`.
    pub ruin_probability: f64,
    /// Probability of reaching `target_wealth` before ruin ($1 - P_{\text{ruin}}$).
    pub target_hit_probability: f64,
    /// Infinite-horizon ruin probability ($N \to \infty$, `1.0` when $p \le 0.5$).
    pub infinite_horizon_ruin: f64,
    /// Initial bankroll measured in stake units ($W_0 / s$).
    pub initial_units: f64,
    /// Upper wealth target measured in stake units ($W_{\text{target}} / s$).
    pub target_units: f64,
}

/// Weighted fixed-stake wager entry inside a heterogeneous bet portfolio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortfolioBet {
    /// True win probability $p_k \in (0, 1)$.
    pub win_prob: UnitInterval,
    /// Decimal payoff odds $O_k = 1 + b_k > 1$.
    pub odds: Odds,
    /// Fixed capital staked per bet ($s_k > 0$).
    pub stake: f64,
    /// Relative portfolio sampling frequency ($w_k > 0$).
    pub weight: f64,
}

impl PortfolioBet {
    /// Validates and constructs a weighted [`PortfolioBet`].
    #[verified_engine::verified]
    pub fn new(
        win_prob: UnitInterval,
        odds: Odds,
        stake: f64,
        weight: f64,
    ) -> Result<Self, KellyError> {
        let p = win_prob.value();
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err(KellyError::InvalidProbability { value: p });
        }
        if !stake.is_finite() || stake <= 0.0 {
            return Err(KellyError::InvalidStake { value: stake });
        }
        if !weight.is_finite() || weight <= 0.0 {
            return Err(KellyError::InvalidStake { value: weight });
        }
        Ok(Self {
            win_prob,
            odds,
            stake,
            weight,
        })
    }

    /// Constructs a [`PortfolioBet`] with unit relative weight (`1.0`).
    #[verified_engine::verified]
    pub fn unit_weight(win_prob: UnitInterval, odds: Odds, stake: f64) -> Result<Self, KellyError> {
        Self::new(win_prob, odds, stake, 1.0)
    }
}

/// Solution of $E[\exp(-r X)] = 1$ with Cramér-Lundberg and Brownian diffusion ruin bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LundbergSolution {
    /// Exact positive Lundberg adjustment coefficient $r^* > 0$.
    pub adjustment_coefficient: f64,
    /// Cramér-Lundberg upper bound on infinite-horizon ruin $\exp(-r^* U)$.
    pub cramer_lundberg_bound: f64,
    /// Brownian diffusion exponent $2 \mu / \sigma^2$.
    pub diffusion_exponent: f64,
    /// Brownian diffusion approximation $\exp(-(2 \mu / \sigma^2) U)$.
    pub diffusion_approximation: f64,
    /// Portfolio per-bet expected profit $\mu = \mathbb{E}\[X\]$.
    pub drift: f64,
    /// Portfolio per-bet profit variance $\sigma^2 = \operatorname{Var}(X)$.
    pub variance: f64,
}

/// Computes infinite-horizon Gambler's Ruin probability for fixed unit-payoff binary wagers.
#[verified_engine::verified]
pub fn gamblers_ruin_infinite(
    initial_units: f64,
    win_prob: &UnitInterval,
) -> Result<f64, KellyError> {
    if !initial_units.is_finite() || initial_units <= 0.0 {
        return Err(KellyError::InvalidBankroll {
            value: initial_units,
        });
    }
    let p = win_prob.value();
    if p <= 0.5 {
        return Ok(1.0);
    }
    if p >= 1.0 {
        return Ok(0.0);
    }
    Ok(((1.0 - p) / p).powf(initial_units).clamp(0.0, 1.0))
}

fn two_barrier_absorption(i: f64, n: f64, p: f64) -> f64 {
    if p <= 0.0 {
        1.0
    } else if p >= 1.0 {
        0.0
    } else if (p - 0.5).abs() < 1e-12 {
        (n - i) / n
    } else if p > 0.5 {
        let ln_rho = ((1.0 - p) / p).ln();
        if (n * ln_rho).abs() < 1e-5 {
            1.0 - (i * ln_rho).exp_m1() / (n * ln_rho).exp_m1()
        } else {
            ((i * ln_rho).exp() - (n * ln_rho).exp()) / (1.0 - (n * ln_rho).exp())
        }
    } else {
        let ln_r = (p / (1.0 - p)).ln();
        if (n * ln_r).abs() < 1e-5 {
            ((n - i) * ln_r).exp_m1() / (n * ln_r).exp_m1()
        } else {
            (1.0 - ((n - i) * ln_r).exp()) / (1.0 - (n * ln_r).exp())
        }
    }
}

/// Computes exact two-barrier Gambler's Ruin absorption probability with fair-game limit.
///
/// # Example
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::kelly::UnitInterval;
/// use math_explorer::pure_math::statistics::ruin::gamblers_ruin_with_target;
///
/// let fair = UnitInterval::new(0.50).unwrap();
/// let res = gamblers_ruin_with_target(40.0, 100.0, 1.0, &fair).unwrap();
/// assert!((res.ruin_probability - 0.60).abs() < 1e-12);
/// ```
#[verified_engine::verified]
pub fn gamblers_ruin_with_target(
    initial_wealth: f64,
    target_wealth: f64,
    stake: f64,
    win_prob: &UnitInterval,
) -> Result<GamblersRuinResult, KellyError> {
    if !initial_wealth.is_finite() || initial_wealth <= 0.0 {
        return Err(KellyError::InvalidBankroll {
            value: initial_wealth,
        });
    }
    if !stake.is_finite() || stake <= 0.0 {
        return Err(KellyError::InvalidStake { value: stake });
    }
    if !target_wealth.is_finite() || target_wealth <= initial_wealth {
        return Err(KellyError::InvalidBarrier {
            floor: target_wealth,
            initial_wealth,
        });
    }

    let i = initial_wealth / stake;
    let n = target_wealth / stake;
    let inf_ruin = gamblers_ruin_infinite(i, win_prob)?;
    let ruin_probability = two_barrier_absorption(i, n, win_prob.value()).clamp(0.0, 1.0);

    Ok(GamblersRuinResult {
        ruin_probability,
        target_hit_probability: (1.0 - ruin_probability).clamp(0.0, 1.0),
        infinite_horizon_ruin: inf_ruin,
        initial_units: i,
        target_units: n,
    })
}

pub(crate) fn solve_positive_mgf_root<F, D>(
    f: F,
    f_prime: D,
    r_guess: f64,
    r_cap: f64,
) -> Result<f64, KellyError>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    let mut low = r_guess.min(0.25 * r_cap).max(1e-15);
    for _ in 0..64 {
        if f(low) < 0.0 {
            break;
        }
        low *= 0.5;
    }
    let f_low = f(low);
    if !f_low.is_finite() || f_low >= 0.0 {
        return Err(KellyError::ConvergenceFailed {
            reason: "failed to locate negative MGF bracket below root",
        });
    }

    let mut high = (2.0 * low).max(r_guess).min(r_cap);
    for _ in 0..64 {
        if f(high) > 0.0 || high >= r_cap {
            break;
        }
        high = (2.0 * high).min(r_cap);
    }
    let f_high = f(high);
    if !f_high.is_finite() || f_high <= 0.0 {
        return Err(KellyError::ConvergenceFailed {
            reason: "failed to locate positive MGF bracket above root",
        });
    }

    let r_bracketed = Bisection::new(96, 1e-7)
        .find_root(&f, low, high)
        .map_err(|_| KellyError::ConvergenceFailed {
            reason: "bisection failed on MGF bracket",
        })?;

    let start = if f(r_bracketed) > 0.0 {
        r_bracketed
    } else {
        high
    };
    if let Ok(polished) =
        NewtonRaphson::new(32, 1e-13).find_root_with_derivative(&f, &f_prime, start)
        && polished > low
        && polished < high
        && f(polished).abs() <= f(r_bracketed).abs()
    {
        return Ok(polished);
    }

    Bisection::new(128, 1e-12)
        .find_root(&f, low, high)
        .map_err(|_| KellyError::ConvergenceFailed {
            reason: "high-precision bisection safeguard failed",
        })
}

struct FixedPortfolioMoments {
    total_w: f64,
    max_loss: f64,
    drift: f64,
    variance: f64,
}

fn compute_fixed_portfolio_moments(
    portfolio: &[PortfolioBet],
) -> Result<FixedPortfolioMoments, KellyError> {
    if portfolio.is_empty() {
        return Err(KellyError::InsufficientBets {
            required: 1,
            actual: 0,
        });
    }
    let mut total_w = 0.0;
    let mut max_loss = 0.0_f64;
    for item in portfolio {
        let p = item.win_prob.value();
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err(KellyError::InvalidProbability { value: p });
        }
        if !item.stake.is_finite()
            || item.stake <= 0.0
            || !item.weight.is_finite()
            || item.weight <= 0.0
        {
            return Err(KellyError::InvalidStake { value: item.stake });
        }
        total_w += item.weight;
        max_loss = max_loss.max(item.stake);
    }

    let mut drift = 0.0;
    let mut second_moment = 0.0;
    let mut avg_p = 0.0;
    let mut avg_odds = 0.0;
    for item in portfolio {
        let alpha = item.weight / total_w;
        let p = item.win_prob.value();
        let win_x = item.stake * item.odds.net_profit_multiplier();
        drift += alpha * (p * win_x - (1.0 - p) * item.stake);
        second_moment += alpha * (p * win_x * win_x + (1.0 - p) * item.stake * item.stake);
        avg_p += alpha * p;
        avg_odds += alpha * item.odds.value();
    }
    if drift <= 0.0 {
        return Err(KellyError::NoEdge {
            probability: avg_p,
            odds: avg_odds,
        });
    }
    Ok(FixedPortfolioMoments {
        total_w,
        max_loss,
        drift,
        variance: (second_moment - drift * drift).max(1e-18),
    })
}

/// Solves $E[\exp(-r X)] = 1$ across a heterogeneous fixed-stake bet portfolio.
///
/// Uses [`Bisection`] bracketing and safeguarded [`NewtonRaphson`] polishing with zero heap allocations.
#[verified_engine::verified]
pub fn solve_lundberg_ruin(
    portfolio: &[PortfolioBet],
    initial_surplus: f64,
) -> Result<LundbergSolution, KellyError> {
    if !initial_surplus.is_finite() || initial_surplus <= 0.0 {
        return Err(KellyError::InvalidBankroll {
            value: initial_surplus,
        });
    }
    let m = compute_fixed_portfolio_moments(portfolio)?;
    let diffusion_exponent = 2.0 * m.drift / m.variance;
    let r_cap = MAX_SAFE_EXP_ARG / m.max_loss.max(1e-12);

    let mgf_minus_one = |r: f64| {
        let mut acc = 0.0;
        for item in portfolio {
            let alpha = item.weight / m.total_w;
            let p = item.win_prob.value();
            let win_x = item.stake * item.odds.net_profit_multiplier();
            acc += alpha * (p * (-r * win_x).exp() + (1.0 - p) * (r * item.stake).exp());
        }
        acc - 1.0
    };
    let mgf_prime = |r: f64| {
        let mut acc = 0.0;
        for item in portfolio {
            let alpha = item.weight / m.total_w;
            let p = item.win_prob.value();
            let win_x = item.stake * item.odds.net_profit_multiplier();
            acc += alpha
                * (-p * win_x * (-r * win_x).exp()
                    + (1.0 - p) * item.stake * (r * item.stake).exp());
        }
        acc
    };

    let r_star =
        solve_positive_mgf_root(mgf_minus_one, mgf_prime, 0.5 * diffusion_exponent, r_cap)?;
    Ok(LundbergSolution {
        adjustment_coefficient: r_star,
        cramer_lundberg_bound: (-r_star * initial_surplus).exp().clamp(0.0, 1.0),
        diffusion_exponent,
        diffusion_approximation: (-diffusion_exponent * initial_surplus)
            .exp()
            .clamp(0.0, 1.0),
        drift: m.drift,
        variance: m.variance,
    })
}

/// Convenience wrapper solving $E[\exp(-r X)] = 1$ for a single fixed-stake wager.
#[verified_engine::verified]
pub fn solve_single_bet_lundberg(
    win_prob: &UnitInterval,
    odds: &Odds,
    stake: f64,
    initial_surplus: f64,
) -> Result<LundbergSolution, KellyError> {
    let bet = PortfolioBet::unit_weight(*win_prob, *odds, stake)?;
    solve_lundberg_ruin(&[bet], initial_surplus)
}

// [cite:clinical_trials_statistics]
