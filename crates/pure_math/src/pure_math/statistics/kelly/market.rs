//! Exact Smoczynski-Tomkins (2010) active-set solver for mutually exclusive `N`-way markets.
//!
//! In a mutually exclusive market (e.g., 3-way soccer `1X2` or multi-runner racing), at most one
//! outcome wins while stakes placed on all other runners are lost (`W_i = 1 - F + f_i * O_i`).
//! Ordering runners by gross expected return `p_i * O_i` descending and pruning outcomes below the
//! reserve rate `r(S) = (1 - sum_{i in S} p_i) / (1 - sum_{i in S} 1 / O_i)` yields the exact
//! Karush-Kuhn-Tucker global optimum in `O(N^2)` stack operations with zero heap allocations.

use super::core::{MAX_MARKET_OUTCOMES, Odds, PROBABILITY_SUM_TOLERANCE, UnitInterval};
use crate::error::KellyError;

/// Single runner / outcome in a mutually exclusive `N`-way market.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarketOutcome {
    probability: UnitInterval,
    odds: Odds,
}

impl MarketOutcome {
    /// Creates a mutually exclusive market outcome with true probability and offered decimal odds.
    #[verified_engine::verified]
    pub fn new(probability: UnitInterval, odds: Odds) -> Self {
        Self { probability, odds }
    }

    /// Returns the outcome's true win probability.
    #[verified_engine::verified]
    pub fn probability(&self) -> UnitInterval {
        self.probability
    }

    /// Returns the offered decimal odds `O_i > 1.0`.
    #[verified_engine::verified]
    pub fn odds(&self) -> Odds {
        self.odds
    }

    /// Returns the gross expected return per unit staked (`p_i * O_i`).
    #[verified_engine::verified]
    pub fn gross_return(&self) -> f64 {
        self.probability.value() * self.odds.value()
    }
}

/// Caller-owned stack scratch space for zero-allocation mutually exclusive market sizing.
#[derive(Debug, Clone)]
pub struct MarketScratch {
    order: [usize; MAX_MARKET_OUTCOMES],
    gross_returns: [f64; MAX_MARKET_OUTCOMES],
}

impl Default for MarketScratch {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

impl MarketScratch {
    /// Creates a zero-initialized scratch workspace for up to [`MAX_MARKET_OUTCOMES`] runners.
    #[verified_engine::verified]
    pub const fn new() -> Self {
        Self {
            order: [0; MAX_MARKET_OUTCOMES],
            gross_returns: [0.0; MAX_MARKET_OUTCOMES],
        }
    }
}

/// Optimal bankroll allocation across a mutually exclusive `N`-way market.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarketAllocation {
    len: usize,
    stakes: [f64; MAX_MARKET_OUTCOMES],
    reserve_rate: f64,
    total_fraction: f64,
    expected_value: f64,
    expected_growth_rate: f64,
    kkt_residual: f64,
}

impl MarketAllocation {
    /// Returns the optimal stake slice `&[f64]` of length `outcomes.len()`.
    #[verified_engine::verified]
    pub fn stakes(&self) -> &[f64] {
        &self.stakes[..self.len]
    }

    /// Returns the total fraction of bankroll wagered `F = sum_i f_i`.
    #[verified_engine::verified]
    pub fn total_fraction(&self) -> f64 {
        self.total_fraction
    }

    /// Returns the Smoczynski-Tomkins reserve rate `r(S) = 1 - F`.
    #[verified_engine::verified]
    pub fn reserve_rate(&self) -> f64 {
        self.reserve_rate
    }

    /// Returns expected net profit `E[W - 1]` as a fraction of bankroll.
    #[verified_engine::verified]
    pub fn expected_value(&self) -> f64 {
        self.expected_value
    }

    /// Returns expected logarithmic bankroll growth `E[ln W]`.
    #[verified_engine::verified]
    pub fn expected_growth_rate(&self) -> f64 {
        self.expected_growth_rate
    }

    /// Returns the maximum KKT complementarity / stationarity residual across all outcomes.
    #[verified_engine::verified]
    pub fn kkt_residual(&self) -> f64 {
        self.kkt_residual
    }
}

fn sort_outcomes_descending(outcomes: &[MarketOutcome], scratch: &mut MarketScratch) {
    let n = outcomes.len();
    for (i, outcome) in outcomes.iter().enumerate() {
        scratch.order[i] = i;
        scratch.gross_returns[i] = outcome.gross_return();
    }
    for i in 1..n {
        let idx = scratch.order[i];
        let key = scratch.gross_returns[idx];
        let mut j = i;
        while j > 0 && scratch.gross_returns[scratch.order[j - 1]] < key {
            scratch.order[j] = scratch.order[j - 1];
            j -= 1;
        }
        scratch.order[j] = idx;
    }
}

fn evaluate_market_metrics(
    outcomes: &[MarketOutcome],
    stakes: &[f64; MAX_MARKET_OUTCOMES],
    total_f: f64,
    prob_sum_all: f64,
) -> (f64, f64, f64) {
    let n = outcomes.len();
    let rem_prob = (1.0 - prob_sum_all).max(0.0);
    let mut ev = 0.0;
    let mut growth = 0.0;
    let mut dual_sum = 0.0;

    for i in 0..n {
        let p_i = outcomes[i].probability().value();
        let o_i = outcomes[i].odds().value();
        let f_i = stakes[i];
        ev += f_i * (p_i * o_i - 1.0);
        if p_i > 0.0 {
            let w_i = 1.0 - total_f + f_i * o_i;
            growth += p_i * w_i.ln();
            dual_sum += p_i / w_i;
        }
    }
    if rem_prob > 0.0 {
        let w_0 = 1.0 - total_f;
        growth += rem_prob * w_0.ln();
        dual_sum += rem_prob / w_0;
    }

    let mut max_res = 0.0_f64;
    for i in 0..n {
        let p_i = outcomes[i].probability().value();
        let o_i = outcomes[i].odds().value();
        let f_i = stakes[i];
        let w_i = 1.0 - total_f + f_i * o_i;
        let grad_i = if p_i > 0.0 {
            (p_i * o_i) / w_i - dual_sum
        } else {
            -dual_sum
        };
        let res = if f_i > 0.0 {
            grad_i.abs()
        } else {
            grad_i.max(0.0)
        };
        max_res = max_res.max(res);
    }
    (ev, growth, max_res)
}

/// Solves a mutually exclusive `N`-way market exactly via Smoczynski & Tomkins (2010).
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::kelly::{
///     MarketOutcome, MarketScratch, Odds, UnitInterval, solve_mutually_exclusive_market,
/// };
///
/// let outcomes = [
///     MarketOutcome::new(UnitInterval::new(0.50).unwrap(), Odds::new(2.20).unwrap()),
///     MarketOutcome::new(UnitInterval::new(0.28).unwrap(), Odds::new(4.00).unwrap()),
///     MarketOutcome::new(UnitInterval::new(0.22).unwrap(), Odds::new(3.20).unwrap()),
/// ];
/// let mut scratch = MarketScratch::new();
/// let alloc = solve_mutually_exclusive_market(&outcomes, &mut scratch).unwrap();
/// assert!((alloc.stakes()[0] - 21.0 / 130.0).abs() < 1e-12);
/// assert!((alloc.stakes()[1] - 61.0 / 650.0).abs() < 1e-12);
/// assert_eq!(alloc.stakes()[2], 0.0);
/// ```
#[verified_engine::verified]
pub fn solve_mutually_exclusive_market(
    outcomes: &[MarketOutcome],
    scratch: &mut MarketScratch,
) -> Result<MarketAllocation, KellyError> {
    let n = outcomes.len();
    if n == 0 {
        return Err(KellyError::EmptyPortfolio);
    }
    if n > MAX_MARKET_OUTCOMES {
        return Err(KellyError::PortfolioTooLarge {
            count: n,
            max: MAX_MARKET_OUTCOMES,
        });
    }

    let mut prob_sum_all = 0.0;
    let mut inv_odds_all = 0.0;
    let mut all_positive_prob = true;
    for outcome in outcomes {
        let p = outcome.probability().value();
        prob_sum_all += p;
        inv_odds_all += 1.0 / outcome.odds().value();
        all_positive_prob &= p > 0.0;
    }
    if prob_sum_all > 1.0 + PROBABILITY_SUM_TOLERANCE {
        return Err(KellyError::InvalidMarketProbabilitySum { sum: prob_sum_all });
    }

    let mut stakes = [0.0; MAX_MARKET_OUTCOMES];
    if (prob_sum_all - 1.0).abs() <= PROBABILITY_SUM_TOLERANCE
        && all_positive_prob
        && inv_odds_all < 1.0 - 1e-12
    {
        // Complete-market Dutch-book arbitrage: W_i = p_i * O_i > 0 in every possible state.
        let norm = 1.0 / prob_sum_all;
        for (i, outcome) in outcomes.iter().enumerate() {
            stakes[i] = outcome.probability().value() * norm;
        }
        let (ev, growth, kkt_res) = evaluate_market_metrics(outcomes, &stakes, 1.0, 1.0);
        return Ok(MarketAllocation {
            len: n,
            stakes,
            reserve_rate: 0.0,
            total_fraction: 1.0,
            expected_value: ev,
            expected_growth_rate: growth,
            kkt_residual: kkt_res,
        });
    }

    sort_outcomes_descending(outcomes, scratch);
    let mut p_active = 0.0;
    let mut sigma_active = 0.0;
    let mut reserve = 1.0;
    let mut active_count = 0;

    for pos in 0..n {
        let idx = scratch.order[pos];
        let p_i = outcomes[idx].probability().value();
        let inv_o_i = 1.0 / outcomes[idx].odds().value();
        let gross_i = scratch.gross_returns[idx];
        if gross_i <= reserve || sigma_active + inv_o_i >= 1.0 - 1e-15 {
            break;
        }
        let next_p = (p_active + p_i).min(1.0);
        let next_sigma = sigma_active + inv_o_i;
        let next_reserve = (1.0 - next_p) / (1.0 - next_sigma);
        if gross_i <= next_reserve {
            break;
        }
        p_active = next_p;
        sigma_active = next_sigma;
        reserve = next_reserve;
        active_count = pos + 1;
    }

    let mut total_f = 0.0;
    for &idx in scratch.order.iter().take(active_count) {
        let f_i =
            (outcomes[idx].probability().value() - reserve / outcomes[idx].odds().value()).max(0.0);
        stakes[idx] = f_i;
        total_f += f_i;
    }
    let (ev, growth, kkt_res) =
        evaluate_market_metrics(outcomes, &stakes, total_f, prob_sum_all.min(1.0));
    Ok(MarketAllocation {
        len: n,
        stakes,
        reserve_rate: reserve,
        total_fraction: total_f,
        expected_value: ev,
        expected_growth_rate: growth,
        kkt_residual: kkt_res,
    })
}
