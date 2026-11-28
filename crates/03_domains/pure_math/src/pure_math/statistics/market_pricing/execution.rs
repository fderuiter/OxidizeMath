//! Spread/total middle break-even & EV calculators and live/futures hedge stake optimizers.

use crate::error::MarketPricingError;
use crate::pure_math::statistics::kelly::{Odds, UnitInterval};

/// Balanced or custom-stake spread/total middle analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MiddleReport {
    /// Stake placed on Side A (e.g., Favorite `-2.5`).
    pub stake_a: f64,
    /// Stake placed on Side B (e.g., Underdog `+3.5`).
    pub stake_b: f64,
    /// Total capital risked across both legs (`stake_a + stake_b`).
    pub total_stake: f64,
    /// Sum of implied probabilities `1 / O_a + 1 / O_b`.
    pub reciprocal_sum: f64,
    /// Net P&L when Side A wins and Side B loses.
    pub side_a_only_pnl: f64,
    /// Net P&L when Side B wins and Side A loses.
    pub side_b_only_pnl: f64,
    /// Non-negative qualifying hold loss on a single-win outcome (`max(0, -min_single_pnl)`).
    pub qualifying_loss: f64,
    /// Net profit when the final score lands inside the middle window and BOTH bets cash.
    pub middle_profit: f64,
    /// Minimum middle hit probability required for zero expected value (`0.0` for arb-middles).
    pub break_even_probability: f64,
    /// Expected net profit in dollars when `middle_probability` is supplied.
    pub expected_profit: Option<f64>,
    /// Expected return per unit staked (`expected_profit / total_stake`) when `middle_probability` is supplied.
    pub ev_per_unit: Option<f64>,
}

/// Objective for hedging an open pre-game or futures ticket (`original_stake` at `original_odds`).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum HedgeMode {
    /// Lock in identical guaranteed profit regardless of whether the original or hedge wins.
    EqualProfit,
    /// Minimize hedge stake subject to worst-case net P&L `>= floor` (`0.0` = money-back freeroll).
    TargetFloor(f64),
}

/// Live or futures hedge execution summary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HedgeReport {
    /// Original sunk/committed stake `S_0`.
    pub original_stake: f64,
    /// Original ticket decimal [`Odds`] `O_0`.
    pub original_odds: Odds,
    /// Live or futures hedge [`Odds`] `O_h` on the mutually exclusive complement.
    pub hedge_odds: Odds,
    /// Recommended stake `H` to place on the hedge side.
    pub hedge_stake: f64,
    /// Total capital committed across original and hedge legs (`S_0 + H`).
    pub total_exposure: f64,
    /// Net profit if the original ticket wins and the hedge loses (`S_0 * (O_0 - 1) - H`).
    pub net_profit_if_original_wins: f64,
    /// Net profit if the hedge wins and the original ticket loses (`H * (O_h - 1) - S_0`).
    pub net_profit_if_hedge_wins: f64,
    /// Guaranteed worst-case net P&L (`min(net_profit_if_original_wins, net_profit_if_hedge_wins)`).
    pub guaranteed_floor: f64,
    /// Maximum equalized profit attainable (`S_0 * O_0 * (1 - 1 / O_h) - S_0`).
    pub equal_profit_ceiling: f64,
}

/// Sizes a balanced equal-payout spread/total middle (`s_a * O_a = s_b * O_b`) for `total_budget`,
/// computing qualifying hold loss, middle jackpot profit, break-even hit rate, and optional EV.
#[verified_engine::verified]
pub fn evaluate_middle(
    odds_a: Odds,
    odds_b: Odds,
    total_budget: f64,
    middle_probability: Option<UnitInterval>,
) -> Result<MiddleReport, MarketPricingError> {
    if !total_budget.is_finite() || total_budget <= 0.0 {
        return Err(MarketPricingError::InvalidStakeOrBudget {
            value: total_budget,
        });
    }
    let oa = odds_a.value();
    let ob = odds_b.value();
    if !oa.is_finite() || oa <= 1.0 {
        return Err(MarketPricingError::InvalidOdds { value: oa });
    }
    if !ob.is_finite() || ob <= 1.0 {
        return Err(MarketPricingError::InvalidOdds { value: ob });
    }
    let r = odds_a.implied_probability() + odds_b.implied_probability();
    let stake_a = total_budget / (r * oa);
    let stake_b = total_budget / (r * ob);
    evaluate_middle_custom_stakes(odds_a, stake_a, odds_b, stake_b, middle_probability)
}

/// Evaluates middle P&L, break-even probability, and optional EV for explicit `stake_a`, `stake_b`.
#[verified_engine::verified]
pub fn evaluate_middle_custom_stakes(
    odds_a: Odds,
    stake_a: f64,
    odds_b: Odds,
    stake_b: f64,
    middle_probability: Option<UnitInterval>,
) -> Result<MiddleReport, MarketPricingError> {
    if !stake_a.is_finite() || stake_a <= 0.0 {
        return Err(MarketPricingError::InvalidStakeOrBudget { value: stake_a });
    }
    if !stake_b.is_finite() || stake_b <= 0.0 {
        return Err(MarketPricingError::InvalidStakeOrBudget { value: stake_b });
    }
    let oa = odds_a.value();
    let ob = odds_b.value();
    if !oa.is_finite() || oa <= 1.0 {
        return Err(MarketPricingError::InvalidOdds { value: oa });
    }
    if !ob.is_finite() || ob <= 1.0 {
        return Err(MarketPricingError::InvalidOdds { value: ob });
    }
    let total_stake = stake_a + stake_b;
    let r = odds_a.implied_probability() + odds_b.implied_probability();
    let side_a_only_pnl = stake_a * oa - total_stake;
    let side_b_only_pnl = stake_b * ob - total_stake;
    let worst_single = side_a_only_pnl.min(side_b_only_pnl);
    let qualifying_loss = (-worst_single).max(0.0);
    let middle_profit = stake_a * oa + stake_b * ob - total_stake;
    // Under worst-case non-middle outcome `worst_single`: (1 - p) * worst_single + p * middle_profit = 0.
    let break_even_probability = if worst_single >= 0.0 {
        0.0
    } else {
        let denom = middle_profit - worst_single;
        if denom <= 0.0 {
            1.0
        } else {
            ((-worst_single) / denom).clamp(0.0, 1.0)
        }
    };
    let (expected_profit, ev_per_unit) = match middle_probability {
        Some(p_unit) => {
            let p = p_unit.value();
            let avg_single = 0.5 * (side_a_only_pnl + side_b_only_pnl);
            // For balanced stakes `side_a_only_pnl == side_b_only_pnl`, exact for any split of non-middles!
            let exp_pnl = (1.0 - p) * avg_single + p * middle_profit;
            (Some(exp_pnl), Some(exp_pnl / total_stake))
        }
        None => (None, None),
    };
    Ok(MiddleReport {
        stake_a,
        stake_b,
        total_stake,
        reciprocal_sum: r,
        side_a_only_pnl,
        side_b_only_pnl,
        qualifying_loss,
        middle_profit,
        break_even_probability,
        expected_profit,
        ev_per_unit,
    })
}

/// Computes optimal live/futures hedge stake and net P&L across both outcomes for [`HedgeMode`].
#[verified_engine::verified]
pub fn evaluate_hedge(
    original_stake: f64,
    original_odds: Odds,
    hedge_odds: Odds,
    mode: HedgeMode,
) -> Result<HedgeReport, MarketPricingError> {
    if !original_stake.is_finite() || original_stake <= 0.0 {
        return Err(MarketPricingError::InvalidStakeOrBudget {
            value: original_stake,
        });
    }
    let o0 = original_odds.value();
    let oh = hedge_odds.value();
    if !o0.is_finite() || o0 <= 1.0 {
        return Err(MarketPricingError::InvalidOdds { value: o0 });
    }
    if !oh.is_finite() || oh <= 1.0 {
        return Err(MarketPricingError::InvalidOdds { value: oh });
    }
    let equal_stake = (original_stake * o0) / oh;
    let equal_profit_ceiling = original_stake * (o0 - 1.0) - equal_stake;
    let hedge_stake = match mode {
        HedgeMode::EqualProfit => equal_stake,
        HedgeMode::TargetFloor(floor) => {
            if !floor.is_finite() {
                return Err(MarketPricingError::InvalidStakeOrBudget { value: floor });
            }
            if floor > equal_profit_ceiling + 1e-9 {
                return Err(MarketPricingError::UnattainableFloor {
                    floor,
                    max_guaranteed_profit: equal_profit_ceiling,
                });
            }
            ((original_stake + floor) / (oh - 1.0))
                .max(0.0)
                .min(equal_stake)
        }
    };
    let net_orig = original_stake * (o0 - 1.0) - hedge_stake;
    let net_hedge = hedge_stake * (oh - 1.0) - original_stake;
    Ok(HedgeReport {
        original_stake,
        original_odds,
        hedge_odds,
        hedge_stake,
        total_exposure: original_stake + hedge_stake,
        net_profit_if_original_wins: net_orig,
        net_profit_if_hedge_wins: net_hedge,
        guaranteed_floor: net_orig.min(net_hedge),
        equal_profit_ceiling,
    })
}

// [cite:clinical_trials_statistics]
