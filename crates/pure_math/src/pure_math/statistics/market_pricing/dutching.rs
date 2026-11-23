//! Cross-bookmaker surebet arbitrage and equal-payout subset Dutching calculators.

use super::MAX_MARKET_OUTCOMES;
use super::overround::{booksum, validate_odds_slice};
use crate::error::MarketPricingError;
use crate::pure_math::statistics::kelly::Odds;

/// Risk-free cross-bookmaker arbitrage sizing and guaranteed return summary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArbitrageReport {
    /// Number of mutually exclusive outcomes covered across books.
    pub outcome_count: usize,
    /// Composite implied probability sum `R = sum(1 / O_i) < 1.0`.
    pub reciprocal_sum: f64,
    /// Total capital deployed across all legs (`sum(stakes)`).
    pub total_budget: f64,
    /// Identical gross payout returned regardless of which outcome wins (`total_budget / R`).
    pub guaranteed_payout: f64,
    /// Risk-free net profit locked in (`guaranteed_payout - total_budget`).
    pub guaranteed_profit: f64,
    /// Risk-free return on investment (`(1.0 - R) / R`).
    pub roi: f64,
}

/// Sizing constraint for multi-outcome Dutching.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DutchingMode {
    /// Allocate a fixed total stake `budget` across selected outcomes for equal payout.
    FixedBudget(f64),
    /// Size stakes so that hitting any selected outcome yields net profit `target_profit`.
    TargetProfit(f64),
}

/// Equal-payout Dutching allocation summary over a subset of contenders.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DutchingReport {
    /// Number of contenders included in the Dutching ticket.
    pub selected_count: usize,
    /// Sum of implied probabilities across selected contenders (`R = sum_{j in S} 1 / O_j`).
    pub combined_implied_probability: f64,
    /// Total stake wagered across all selected contenders.
    pub total_stake: f64,
    /// Gross payout returned if any selected contender wins (`total_stake / R`).
    pub winning_payout: f64,
    /// Net profit if any selected contender wins (`winning_payout - total_stake`).
    pub profit_if_hit: f64,
    /// Return on investment if any selected contender wins (`profit_if_hit / total_stake`).
    pub roi_if_hit: f64,
    /// `true` when `combined_implied_probability < 1.0` (hitting a selected outcome yields profit).
    pub is_profitable_on_hit: bool,
}

/// Returns `true` when best cross-book odds form a risk-free surebet (`sum(1 / O_i) < 1.0`).
#[verified_engine::verified]
pub fn is_arbitrage(best_odds: &[Odds]) -> Result<bool, MarketPricingError> {
    Ok(booksum(best_odds)? < 1.0)
}

/// Allocates `total_budget` across `best_odds` into `stakes_out` to lock in equal arbitrage profit.
#[verified_engine::verified]
pub fn evaluate_arbitrage_into(
    best_odds: &[Odds],
    total_budget: f64,
    stakes_out: &mut [f64],
) -> Result<ArbitrageReport, MarketPricingError> {
    if !total_budget.is_finite() || total_budget <= 0.0 {
        return Err(MarketPricingError::InvalidStakeOrBudget {
            value: total_budget,
        });
    }
    if stakes_out.len() != best_odds.len() {
        return Err(MarketPricingError::SliceLengthMismatch {
            expected: best_odds.len(),
            actual: stakes_out.len(),
        });
    }
    let r = booksum(best_odds)?;
    if r >= 1.0 {
        return Err(MarketPricingError::NoArbitrage { reciprocal_sum: r });
    }
    let guaranteed_payout = total_budget / r;
    let mut i = 0;
    while i < best_odds.len() {
        stakes_out[i] = guaranteed_payout / best_odds[i].value();
        i += 1;
    }
    let guaranteed_profit = guaranteed_payout - total_budget;
    Ok(ArbitrageReport {
        outcome_count: best_odds.len(),
        reciprocal_sum: r,
        total_budget,
        guaranteed_payout,
        guaranteed_profit,
        roi: guaranteed_profit / total_budget,
    })
}

/// Convenience helper for 2-way cross-book arbitrage returning `(ArbitrageReport, [stake_a, stake_b])`.
#[verified_engine::verified]
pub fn evaluate_two_way_arbitrage(
    odds_a: Odds,
    odds_b: Odds,
    total_budget: f64,
) -> Result<(ArbitrageReport, [f64; 2]), MarketPricingError> {
    let pair = [odds_a, odds_b];
    let mut stakes = [0.0_f64; 2];
    let report = evaluate_arbitrage_into(&pair, total_budget, &mut stakes)?;
    Ok((report, stakes))
}

/// Computes equal-payout Dutching stakes over `selection_odds` for [`DutchingMode`].
#[verified_engine::verified]
pub fn dutch_outcomes_into(
    selection_odds: &[Odds],
    mode: DutchingMode,
    stakes_out: &mut [f64],
) -> Result<DutchingReport, MarketPricingError> {
    validate_odds_slice(selection_odds, 1)?;
    if stakes_out.len() != selection_odds.len() {
        return Err(MarketPricingError::SliceLengthMismatch {
            expected: selection_odds.len(),
            actual: stakes_out.len(),
        });
    }
    let mut r = 0.0;
    let mut i = 0;
    while i < selection_odds.len() {
        r += selection_odds[i].implied_probability();
        i += 1;
    }
    let total_stake = match mode {
        DutchingMode::FixedBudget(budget) => {
            if !budget.is_finite() || budget <= 0.0 {
                return Err(MarketPricingError::InvalidStakeOrBudget { value: budget });
            }
            budget
        }
        DutchingMode::TargetProfit(target) => {
            if !target.is_finite() || target <= 0.0 {
                return Err(MarketPricingError::InvalidStakeOrBudget { value: target });
            }
            if r >= 1.0 {
                return Err(MarketPricingError::InfeasibleTargetProfit { reciprocal_sum: r });
            }
            target * r / (1.0 - r)
        }
    };
    let winning_payout = total_stake / r;
    let mut j = 0;
    while j < selection_odds.len() {
        stakes_out[j] = winning_payout / selection_odds[j].value();
        j += 1;
    }
    let profit_if_hit = winning_payout - total_stake;
    Ok(DutchingReport {
        selected_count: selection_odds.len(),
        combined_implied_probability: r,
        total_stake,
        winning_payout,
        profit_if_hit,
        roi_if_hit: profit_if_hit / total_stake,
        is_profitable_on_hit: r < 1.0,
    })
}

/// Computes equal-payout Dutching stakes for a subset `selected_indices` of a full `market_odds`
/// slate, zeroing unselected entries in `market_stakes_out`.
#[verified_engine::verified]
pub fn dutch_subset_into(
    market_odds: &[Odds],
    selected_indices: &[usize],
    mode: DutchingMode,
    market_stakes_out: &mut [f64],
) -> Result<DutchingReport, MarketPricingError> {
    validate_odds_slice(market_odds, 2)?;
    if market_stakes_out.len() != market_odds.len() {
        return Err(MarketPricingError::SliceLengthMismatch {
            expected: market_odds.len(),
            actual: market_stakes_out.len(),
        });
    }
    if selected_indices.is_empty() {
        return Err(MarketPricingError::InsufficientOutcomes {
            count: 0,
            min_required: 1,
        });
    }
    let mut seen = [false; MAX_MARKET_OUTCOMES];
    let mut subset_odds = [market_odds[0]; MAX_MARKET_OUTCOMES];
    let mut subset_stakes = [0.0_f64; MAX_MARKET_OUTCOMES];
    let mut k = 0;
    while k < selected_indices.len() {
        let idx = selected_indices[k];
        if idx >= market_odds.len() {
            return Err(MarketPricingError::OutcomeIndexOutOfBounds {
                index: idx,
                outcome_count: market_odds.len(),
            });
        }
        if seen[idx] {
            return Err(MarketPricingError::DuplicateOutcomeIndex { index: idx });
        }
        seen[idx] = true;
        subset_odds[k] = market_odds[idx];
        k += 1;
    }
    let m = selected_indices.len();
    let report = dutch_outcomes_into(&subset_odds[..m], mode, &mut subset_stakes[..m])?;
    let mut i = 0;
    while i < market_stakes_out.len() {
        market_stakes_out[i] = 0.0;
        i += 1;
    }
    let mut j = 0;
    while j < m {
        market_stakes_out[selected_indices[j]] = subset_stakes[j];
        j += 1;
    }
    Ok(report)
}

// [cite:clinical_trials_statistics]
