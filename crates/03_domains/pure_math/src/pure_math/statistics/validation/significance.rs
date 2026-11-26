//! Realized yield hypothesis testing and 95%-significance sample sizes under exact Bernoulli variance.

use super::clv::{SettledBet, normal_tail_p_values};
use crate::error::KellyError;
use crate::pure_math::statistics::kelly::{Odds, UnitInterval};

/// Standard normal 95% one-sided critical quantile $\Phi^{-1}(0.95)$.
pub const Z_95_ONE_SIDED: f64 = 1.644_853_626_951_472_2;
/// Standard normal 95% two-sided critical quantile $\Phi^{-1}(0.975)$.
pub const Z_95_TWO_SIDED: f64 = 1.959_963_984_540_054;

/// Required bet counts to confirm a positive target yield at 95% statistical significance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleSize95 {
    /// Minimum bets required for one-sided 95% significance ($p < 0.05$, $z \ge 1.6449$).
    pub one_sided_bets: usize,
    /// Minimum bets required for two-sided 95% significance ($p < 0.05$, $|z| \ge 1.9600$).
    pub two_sided_bets: usize,
    /// Mean unit-stake Bernoulli return variance $\bar{\sigma}^2 = \frac{1}{n} \sum O_i^2 p_i (1 - p_i)$.
    pub mean_bernoulli_variance: f64,
}

/// Realized yield significance report under exact heterogeneous Bernoulli profit variance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YieldSignificance {
    /// Number of settled bets evaluated.
    pub num_bets: usize,
    /// Total turnover staked $W = \sum s_i$.
    pub total_staked: f64,
    /// Realized net profit $P = \sum s_i (O_i Y_i - 1)$.
    pub net_profit: f64,
    /// Realized yield (ROI on turnover) $Y = P / W$.
    pub realized_yield: f64,
    /// Benchmark/null expected yield $\mathbb{E}_0\[P\] / W$ (`0.0` under fair implied-odds null).
    pub expected_yield: f64,
    /// Exact Bernoulli yield standard error $\sqrt{\sum s_i^2 O_i^2 p_{0,i}(1 - p_{0,i})} / W$.
    pub yield_std_error: f64,
    /// Yield z-score $(Y - \mathbb{E}_0\[Y\]) / \text{SE}(Y)$.
    pub z_score: f64,
    /// One-sided upper-tail p-value $P(Z \ge z)$.
    pub p_value_one_sided: f64,
    /// Two-sided p-value $P(|Z| \ge |z|)$.
    pub p_value_two_sided: f64,
    /// Required sample size for one-sided 95% significance when `realized_yield > 0`.
    pub required_bets_95_one_sided: Option<usize>,
    /// Required sample size for two-sided 95% significance when `realized_yield != 0`.
    pub required_bets_95_two_sided: Option<usize>,
}

fn ceil_sample_count(raw: f64) -> Option<usize> {
    if !raw.is_finite() || raw <= 0.0 {
        return None;
    }
    let ceiled = raw.ceil();
    if ceiled > (usize::MAX as f64) {
        None
    } else {
        Some((ceiled as usize).max(1))
    }
}

struct YieldMoments {
    total_staked: f64,
    net_profit: f64,
    expected_profit: f64,
    profit_variance: f64,
}

fn accumulate_yield_moments(
    bets: &[SettledBet],
    use_closing_null: bool,
) -> Result<YieldMoments, KellyError> {
    if bets.is_empty() {
        return Err(KellyError::InsufficientBets {
            required: 1,
            actual: 0,
        });
    }
    let mut total_staked = 0.0;
    let mut net_profit = 0.0;
    let mut expected_profit = 0.0;
    let mut profit_variance = 0.0;

    for bet in bets {
        if !bet.stake.is_finite() || bet.stake <= 0.0 {
            return Err(KellyError::InvalidStake { value: bet.stake });
        }
        let o = bet.placed_odds.value();
        let p0 = if use_closing_null {
            let pc = bet.closing_prob.value();
            if !pc.is_finite() || pc <= 0.0 || pc >= 1.0 {
                return Err(KellyError::InvalidProbability { value: pc });
            }
            pc
        } else {
            bet.placed_odds.implied_probability()
        };
        total_staked += bet.stake;
        net_profit += if bet.won {
            bet.stake * bet.placed_odds.net_profit_multiplier()
        } else {
            -bet.stake
        };
        if use_closing_null {
            expected_profit += bet.stake * (p0 * o - 1.0);
        }
        let s_o = bet.stake * o;
        profit_variance += s_o * s_o * p0 * (1.0 - p0);
    }
    Ok(YieldMoments {
        total_staked,
        net_profit,
        expected_profit,
        profit_variance,
    })
}

fn evaluate_yield_with_null(
    bets: &[SettledBet],
    use_closing_null: bool,
) -> Result<YieldSignificance, KellyError> {
    let m = accumulate_yield_moments(bets, use_closing_null)?;
    let realized_yield = m.net_profit / m.total_staked;
    let expected_yield = m.expected_profit / m.total_staked;
    let yield_std_error = m.profit_variance.sqrt() / m.total_staked;
    if yield_std_error <= 0.0 || !yield_std_error.is_finite() {
        return Err(KellyError::InvalidProbability { value: 0.0 });
    }

    let z_score = (realized_yield - expected_yield) / yield_std_error;
    let (p_value_one_sided, p_value_two_sided) = normal_tail_p_values(z_score);
    let per_bet_var = (bets.len() as f64) * m.profit_variance / (m.total_staked * m.total_staked);

    let required_bets_95_one_sided = if realized_yield > 0.0 {
        ceil_sample_count(
            (Z_95_ONE_SIDED * Z_95_ONE_SIDED * per_bet_var) / (realized_yield * realized_yield),
        )
    } else {
        None
    };
    let required_bets_95_two_sided = if realized_yield.abs() > 0.0 {
        ceil_sample_count(
            (Z_95_TWO_SIDED * Z_95_TWO_SIDED * per_bet_var) / (realized_yield * realized_yield),
        )
    } else {
        None
    };

    Ok(YieldSignificance {
        num_bets: bets.len(),
        total_staked: m.total_staked,
        net_profit: m.net_profit,
        realized_yield,
        expected_yield,
        yield_std_error,
        z_score,
        p_value_one_sided,
        p_value_two_sided,
        required_bets_95_one_sided,
        required_bets_95_two_sided,
    })
}

/// Tests realized yield against the zero-edge fair-market null ($p_{0,i} = 1 / O_i$, $\operatorname{Var}_0(X_i) = s_i^2(O_i - 1)$).
///
/// # Example
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::kelly::{Odds, UnitInterval};
/// use math_explorer::pure_math::statistics::validation::{
///     SettledBet, evaluate_yield_significance,
/// };
///
/// let even = Odds::new(2.0).unwrap();
/// let close = UnitInterval::new(0.55).unwrap();
/// let bets = [
///     SettledBet::new(even, close, 1.0, true).unwrap(),
///     SettledBet::new(even, close, 1.0, true).unwrap(),
///     SettledBet::new(even, close, 1.0, false).unwrap(),
///     SettledBet::new(even, close, 1.0, true).unwrap(),
/// ];
/// let sig = evaluate_yield_significance(&bets).unwrap();
/// assert!((sig.realized_yield - 0.5).abs() < 1e-12);
/// assert!((sig.z_score - 1.0).abs() < 1e-12);
/// ```
#[verified_engine::verified]
pub fn evaluate_yield_significance(bets: &[SettledBet]) -> Result<YieldSignificance, KellyError> {
    evaluate_yield_with_null(bets, false)
}

/// Tests realized yield against the closing-line expectation ($p_{0,i} = p_{\text{close},i}$) to isolate outcome luck from CLV.
#[verified_engine::verified]
pub fn evaluate_yield_against_closing(
    bets: &[SettledBet],
) -> Result<YieldSignificance, KellyError> {
    evaluate_yield_with_null(bets, true)
}

/// Computes required 95%-significance sample sizes (one-sided and two-sided) across heterogeneous odds.
///
/// When `win_probs` is `None`, uses exact fair-market Bernoulli variance $O_i^2 (1/O_i)(1 - 1/O_i) = O_i - 1$.
/// When `win_probs` is `Some(ps)`, uses exact Bernoulli variance $O_i^2 p_i (1 - p_i)$.
#[verified_engine::verified]
pub fn required_bets_for_95_significance(
    target_yield: f64,
    odds: &[Odds],
    win_probs: Option<&[UnitInterval]>,
) -> Result<SampleSize95, KellyError> {
    if odds.is_empty() {
        return Err(KellyError::InsufficientBets {
            required: 1,
            actual: 0,
        });
    }
    if !target_yield.is_finite() || target_yield <= 0.0 {
        return Err(KellyError::NoEdge {
            probability: 0.0,
            odds: odds[0].value(),
        });
    }
    if let Some(ps) = win_probs
        && ps.len() != odds.len()
    {
        return Err(KellyError::DimensionMismatch {
            expected: odds.len(),
            actual: ps.len(),
        });
    }

    let mut sum_var = 0.0;
    for (idx, o_item) in odds.iter().enumerate() {
        let o = o_item.value();
        let var_i = if let Some(ps) = win_probs {
            let p = ps[idx].value();
            if !p.is_finite() || p <= 0.0 || p >= 1.0 {
                return Err(KellyError::InvalidProbability { value: p });
            }
            o * o * p * (1.0 - p)
        } else {
            o_item.net_profit_multiplier()
        };
        sum_var += var_i;
    }

    let mean_var = sum_var / (odds.len() as f64);
    let y2 = target_yield * target_yield;
    let one_sided_bets = ceil_sample_count((Z_95_ONE_SIDED * Z_95_ONE_SIDED * mean_var) / y2)
        .ok_or(KellyError::ConvergenceFailed {
            reason: "sample size overflowed usize",
        })?;
    let two_sided_bets = ceil_sample_count((Z_95_TWO_SIDED * Z_95_TWO_SIDED * mean_var) / y2)
        .ok_or(KellyError::ConvergenceFailed {
            reason: "sample size overflowed usize",
        })?;

    Ok(SampleSize95 {
        one_sided_bets,
        two_sided_bets,
        mean_bernoulli_variance: mean_var,
    })
}

// [cite:clinical_trials_statistics]
