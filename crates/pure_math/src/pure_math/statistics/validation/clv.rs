//! Closing Line Value (CLV), market devigging, and single-pass ledger summaries.

use crate::error::KellyError;
use crate::pure_math::statistics::kelly::{Odds, UnitInterval};
use statrs::function::erf::erfc;

/// Single-bet Closing Line Value (CLV) metrics per Kaunitz, Zhong, and Kreiner (2017).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClvMetrics {
    /// Fractional expected value per unit staked: $p_{\text{close}} \cdot O_{\text{placed}} - 1$.
    pub ev_fraction: f64,
    /// Expected-value percentage: $100 \times (p_{\text{close}} \cdot O_{\text{placed}} - 1)$.
    pub ev_pct: f64,
    /// Implied-probability edge: $p_{\text{close}} - 1 / O_{\text{placed}}$.
    pub implied_prob_edge: f64,
    /// Log-odds (logit) shift: $\operatorname{logit}(p_{\text{close}}) - \operatorname{logit}(p_{\text{placed}})$.
    pub log_odds_shift: f64,
}

/// Validated settled wager record for zero-allocation CLV, calibration, and yield audits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SettledBet {
    /// Decimal odds locked at bet placement.
    pub placed_odds: Odds,
    /// Devigged fair closing win probability in $(0, 1)$.
    pub closing_prob: UnitInterval,
    /// Capital staked on the wager ($> 0$).
    pub stake: f64,
    /// Whether the wager settled as a win (`true`) or loss (`false`).
    pub won: bool,
}

impl SettledBet {
    /// Validates and constructs a [`SettledBet`] record.
    ///
    /// # Errors
    ///
    /// Returns [`KellyError::InvalidProbability`] if `closing_prob` is at the degenerate
    /// boundary $\{0, 1\}$, or [`KellyError::InvalidStake`] if `stake` is non-finite or $\le 0$.
    #[verified_engine::verified]
    pub fn new(
        placed_odds: Odds,
        closing_prob: UnitInterval,
        stake: f64,
        won: bool,
    ) -> Result<Self, KellyError> {
        let p = closing_prob.value();
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err(KellyError::InvalidProbability { value: p });
        }
        if !stake.is_finite() || stake <= 0.0 {
            return Err(KellyError::InvalidStake { value: stake });
        }
        Ok(Self {
            placed_odds,
            closing_prob,
            stake,
            won,
        })
    }
}

/// Zero-allocation one-pass summary of Closing Line Value across a settled bet ledger.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClvSummary {
    /// Number of bets summarized.
    pub num_bets: usize,
    /// Total capital staked across all bets.
    pub total_staked: f64,
    /// Unweighted mean fractional CLV ($\bar{c}$).
    pub unweighted_mean_clv: f64,
    /// Unweighted mean CLV percentage ($100 \times \bar{c}$).
    pub unweighted_mean_clv_pct: f64,
    /// Stake-weighted mean fractional CLV ($\sum s_i c_i / \sum s_i$).
    pub stake_weighted_mean_clv: f64,
    /// Stake-weighted mean CLV percentage.
    pub stake_weighted_mean_clv_pct: f64,
    /// Unweighted mean implied-probability edge.
    pub mean_prob_edge: f64,
    /// Unweighted mean log-odds shift.
    pub mean_log_odds_shift: f64,
    /// Sample standard error of unweighted fractional CLV ($s / \sqrt{n}$).
    pub std_error: f64,
    /// Student/normal z-score $\bar{c} / \text{SE}$ testing $H_0: \mu_{\text{CLV}} = 0$.
    pub z_score: f64,
    /// One-sided upper-tail p-value $P(Z \ge z)$ testing positive pricing skill.
    pub p_value_one_sided: f64,
    /// Two-sided p-value $P(|Z| \ge |z|)$.
    pub p_value_two_sided: f64,
}

pub(crate) fn normal_tail_p_values(z: f64) -> (f64, f64) {
    if !z.is_finite() {
        let p_one = if z.is_sign_positive() { 0.0 } else { 1.0 };
        return (p_one, 0.0);
    }
    let scaled = z * std::f64::consts::FRAC_1_SQRT_2;
    let p_one = (0.5 * erfc(scaled)).clamp(0.0, 1.0);
    let p_two = erfc(scaled.abs()).clamp(0.0, 1.0);
    (p_one, p_two)
}

/// Removes bookmaker vigorish from a two-way closing market via proportional normalization.
///
/// # Example
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::kelly::Odds;
/// use math_explorer::pure_math::statistics::validation::devig_two_way;
///
/// let side_a = Odds::from_american(-110.0).unwrap();
/// let side_b = Odds::from_american(-110.0).unwrap();
/// let (p_a, p_b) = devig_two_way(&side_a, &side_b).unwrap();
/// assert!((p_a.value() - 0.5).abs() < 1e-12);
/// assert!((p_b.value() - 0.5).abs() < 1e-12);
/// ```
#[verified_engine::verified]
pub fn devig_two_way(
    odds_a: &Odds,
    odds_b: &Odds,
) -> Result<(UnitInterval, UnitInterval), KellyError> {
    let q_a = odds_a.implied_probability();
    let q_b = odds_b.implied_probability();
    let sum = q_a + q_b;
    if !sum.is_finite() || sum <= 0.0 {
        return Err(KellyError::InvalidOdds { value: sum });
    }
    let p_a = UnitInterval::new(q_a / sum)
        .map_err(|_| KellyError::InvalidProbability { value: q_a / sum })?;
    let p_b = UnitInterval::new(q_b / sum)
        .map_err(|_| KellyError::InvalidProbability { value: q_b / sum })?;
    Ok((p_a, p_b))
}

/// Removes bookmaker vigorish from an $N$-way closing market into a preallocated slice.
///
/// Returns the raw market booksum (overround $\sum_{i=1}^N 1 / O_i$) with zero heap allocations.
#[verified_engine::verified]
pub fn devig_multi_way_into(
    market_odds: &[Odds],
    out_probs: &mut [UnitInterval],
) -> Result<f64, KellyError> {
    if market_odds.len() < 2 {
        return Err(KellyError::InsufficientBets {
            required: 2,
            actual: market_odds.len(),
        });
    }
    if out_probs.len() != market_odds.len() {
        return Err(KellyError::DimensionMismatch {
            expected: market_odds.len(),
            actual: out_probs.len(),
        });
    }
    let mut booksum = 0.0;
    for odds in market_odds {
        booksum += odds.implied_probability();
    }
    if !booksum.is_finite() || booksum <= 0.0 {
        return Err(KellyError::InvalidOdds { value: booksum });
    }
    for (odds, slot) in market_odds.iter().zip(out_probs.iter_mut()) {
        let p = odds.implied_probability() / booksum;
        *slot = UnitInterval::new(p).map_err(|_| KellyError::InvalidProbability { value: p })?;
    }
    Ok(booksum)
}

/// Removes bookmaker vigorish from an $N$-way closing market, returning normalized probabilities.
#[verified_engine::verified]
pub fn devig_multi_way(market_odds: &[Odds]) -> Result<Vec<UnitInterval>, KellyError> {
    if market_odds.len() < 2 {
        return Err(KellyError::InsufficientBets {
            required: 2,
            actual: market_odds.len(),
        });
    }
    let zero = UnitInterval::new(0.5).map_err(|_| KellyError::InvalidProbability { value: 0.5 })?;
    let mut probs = vec![zero; market_odds.len()];
    devig_multi_way_into(market_odds, &mut probs)?;
    Ok(probs)
}

/// Computes single-bet Closing Line Value from placed [`Odds`] and devigged closing [`UnitInterval`].
///
/// # Example
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::kelly::{Odds, UnitInterval};
/// use math_explorer::pure_math::statistics::validation::single_bet_clv;
///
/// let placed = Odds::new(2.10).unwrap();
/// let closing = UnitInterval::new(0.50).unwrap();
/// let clv = single_bet_clv(&placed, &closing).unwrap();
/// assert!((clv.ev_pct - 5.0).abs() < 1e-9);
/// ```
#[verified_engine::verified]
pub fn single_bet_clv(
    placed_odds: &Odds,
    closing_prob: &UnitInterval,
) -> Result<ClvMetrics, KellyError> {
    let p_close = closing_prob.value();
    if !p_close.is_finite() || p_close <= 0.0 || p_close >= 1.0 {
        return Err(KellyError::InvalidProbability { value: p_close });
    }
    let o_placed = placed_odds.value();
    let p_placed = placed_odds.implied_probability();
    let ev_fraction = p_close * o_placed - 1.0;
    let implied_prob_edge = p_close - p_placed;
    let log_odds_shift =
        (p_close / (1.0 - p_close)).ln() + placed_odds.net_profit_multiplier().ln();
    Ok(ClvMetrics {
        ev_fraction,
        ev_pct: ev_fraction * 100.0,
        implied_prob_edge,
        log_odds_shift,
    })
}

/// Computes single-bet CLV directly from placed [`Odds`] and two-way closing [`Odds`].
#[verified_engine::verified]
pub fn single_bet_clv_two_way(
    placed_odds: &Odds,
    closing_for: &Odds,
    closing_against: &Odds,
) -> Result<ClvMetrics, KellyError> {
    let (p_close, _) = devig_two_way(closing_for, closing_against)?;
    single_bet_clv(placed_odds, &p_close)
}

/// Summarizes a settled bet ledger in one zero-allocation Welford pass.
///
/// # Errors
///
/// Returns [`KellyError::InsufficientBets`] if fewer than 2 bets are supplied, or typed domain
/// errors if any bet carries non-positive stake or degenerate closing probability.
#[verified_engine::verified]
pub fn summarize_clv(bets: &[SettledBet]) -> Result<ClvSummary, KellyError> {
    if bets.len() < 2 {
        return Err(KellyError::InsufficientBets {
            required: 2,
            actual: bets.len(),
        });
    }
    let mut n = 0usize;
    let mut mean_clv = 0.0;
    let mut m2_clv = 0.0;
    let mut total_staked = 0.0;
    let mut weighted_clv_sum = 0.0;
    let mut sum_prob_edge = 0.0;
    let mut sum_logit_shift = 0.0;

    for bet in bets {
        if !bet.stake.is_finite() || bet.stake <= 0.0 {
            return Err(KellyError::InvalidStake { value: bet.stake });
        }
        let metrics = single_bet_clv(&bet.placed_odds, &bet.closing_prob)?;
        n += 1;
        let nf = n as f64;
        let delta = metrics.ev_fraction - mean_clv;
        mean_clv += delta / nf;
        let delta2 = metrics.ev_fraction - mean_clv;
        m2_clv += delta * delta2;
        total_staked += bet.stake;
        weighted_clv_sum += bet.stake * metrics.ev_fraction;
        sum_prob_edge += metrics.implied_prob_edge;
        sum_logit_shift += metrics.log_odds_shift;
    }

    let nf = n as f64;
    let sample_var = m2_clv / ((n - 1) as f64);
    let std_error = (sample_var / nf).max(0.0).sqrt();
    let z_score = if std_error > 0.0 {
        mean_clv / std_error
    } else if mean_clv == 0.0 {
        0.0
    } else {
        mean_clv.signum() * f64::INFINITY
    };
    let (p_value_one_sided, p_value_two_sided) = normal_tail_p_values(z_score);
    let stake_weighted_mean_clv = weighted_clv_sum / total_staked;

    Ok(ClvSummary {
        num_bets: n,
        total_staked,
        unweighted_mean_clv: mean_clv,
        unweighted_mean_clv_pct: mean_clv * 100.0,
        stake_weighted_mean_clv,
        stake_weighted_mean_clv_pct: stake_weighted_mean_clv * 100.0,
        mean_prob_edge: sum_prob_edge / nf,
        mean_log_odds_shift: sum_logit_shift / nf,
        std_error,
        z_score,
        p_value_one_sided,
        p_value_two_sided,
    })
}

// [cite:clinical_trials_statistics]
