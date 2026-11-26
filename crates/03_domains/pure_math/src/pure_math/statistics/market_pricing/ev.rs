//! `+EV` valuation pairing devigged fair probabilities with Kelly Criterion bet sizing.

use super::MAX_MARKET_OUTCOMES;
use super::devig::{DevigMetadata, DevigMethod, devig_into};
use crate::error::{KellyError, MarketPricingError};
use crate::pure_math::statistics::kelly::{
    BankrollFraction, Odds, UnitInterval, expected_growth_rate, expected_value, kelly_fraction,
    variants,
};

/// Expected-value and Kelly sizing diagnostics for a soft offered price against a fair probability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvEvaluation {
    /// Calibrated vig-free win probability in `[0.0, 1.0]`.
    pub fair_probability: UnitInterval,
    /// Vig-free fair decimal odds (`1.0 / fair_probability`, or `INFINITY` if `p == 0.0`).
    pub fair_decimal_odds: f64,
    /// Offered soft-book [`Odds`] being evaluated.
    pub offered_odds: Odds,
    /// Expected profit per `$1` staked (`p * (O - 1) - (1 - p) = p * O - 1`).
    pub ev_per_unit: f64,
    /// Expected return on investment in percentage points (`ev_per_unit * 100.0`).
    pub ev_percent: f64,
    /// Probability edge over the offered implied break-even rate (`p - 1 / O`).
    pub probability_edge: f64,
    /// `true` when `ev_per_unit > 0.0` and `kelly_fraction` yields a positive stake.
    pub has_edge: bool,
    /// Full Kelly bankroll fraction (`0.0` when `has_edge == false`).
    pub full_kelly: BankrollFraction,
    /// Half-Kelly bankroll fraction (`0.5 * full_kelly`).
    pub half_kelly: BankrollFraction,
    /// Quarter-Kelly bankroll fraction (`0.25 * full_kelly`).
    pub quarter_kelly: BankrollFraction,
    /// Expected logarithmic wealth growth per bet at `full_kelly` (`0.0` when `has_edge == false`).
    pub expected_growth: f64,
    /// Optional devigging telemetry when `fair_probability` was derived from a sharp market.
    pub devig_metadata: Option<DevigMetadata>,
}

impl EvEvaluation {
    /// Computes recommended dollar stakes `(full, half, quarter)` for a positive `bankroll`.
    #[verified_engine::verified]
    pub fn recommended_stakes(&self, bankroll: f64) -> Result<(f64, f64, f64), KellyError> {
        let full = self.full_kelly.bet_amount(bankroll)?;
        let half = self.half_kelly.bet_amount(bankroll)?;
        let quarter = self.quarter_kelly.bet_amount(bankroll)?;
        Ok((full, half, quarter))
    }
}

/// Side-by-side `+EV` evaluation of a single outcome across all four devigging models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AllMethodsEvReport {
    /// Target outcome index within the sharp market slice.
    pub outcome_index: usize,
    /// Evaluation under proportional (`Multiplicative`) vig removal.
    pub multiplicative: EvEvaluation,
    /// Evaluation under equal-margin (`Additive`) vig removal (simplex-clamped if violated).
    pub additive: EvEvaluation,
    /// Evaluation under constant-exponent (`Power`) vig removal.
    pub power: EvEvaluation,
    /// Evaluation under `Shin` insider-trading equilibrium vig removal.
    pub shin: EvEvaluation,
}

impl AllMethodsEvReport {
    /// Selects the [`EvEvaluation`] corresponding to `method`.
    #[verified_engine::verified]
    pub fn for_method(&self, method: DevigMethod) -> &EvEvaluation {
        match method {
            DevigMethod::Multiplicative => &self.multiplicative,
            DevigMethod::Additive => &self.additive,
            DevigMethod::Power => &self.power,
            DevigMethod::Shin => &self.shin,
        }
    }
}

/// Evaluates `+EV`, edge, and Full / Half / Quarter Kelly fractions from a known `fair_probability`.
#[verified_engine::verified]
pub fn evaluate_ev_from_probability(
    fair_probability: UnitInterval,
    offered_odds: Odds,
) -> Result<EvEvaluation, MarketPricingError> {
    let p = fair_probability.value();
    if !p.is_finite() {
        return Err(MarketPricingError::InvalidProbability { value: p });
    }
    let fair_decimal_odds = if p <= 0.0 { f64::INFINITY } else { 1.0 / p };
    let ev_per_unit = expected_value(&fair_probability, &offered_odds);
    let probability_edge = p - offered_odds.implied_probability();
    let zero_frac = BankrollFraction::new(0.0).unwrap_or_else(|_| zero_bankroll_fraction());

    let (has_edge, full_kelly, half_kelly, quarter_kelly, expected_growth) =
        match kelly_fraction(&fair_probability, &offered_odds) {
            Ok(full) => {
                let half =
                    variants::half_kelly(&fair_probability, &offered_odds).unwrap_or(zero_frac);
                let quarter =
                    variants::quarter_kelly(&fair_probability, &offered_odds).unwrap_or(zero_frac);
                let growth = expected_growth_rate(&fair_probability, &offered_odds, &full);
                (true, full, half, quarter, growth)
            }
            Err(_) => (false, zero_frac, zero_frac, zero_frac, 0.0),
        };

    Ok(EvEvaluation {
        fair_probability,
        fair_decimal_odds,
        offered_odds,
        ev_per_unit,
        ev_percent: ev_per_unit * 100.0,
        probability_edge,
        has_edge,
        full_kelly,
        half_kelly,
        quarter_kelly,
        expected_growth,
        devig_metadata: None,
    })
}

#[verified_engine::verified]
fn zero_bankroll_fraction() -> BankrollFraction {
    match BankrollFraction::new(0.0) {
        Ok(z) => z,
        Err(_) => unreachable!(),
    }
}

/// Devigs `sharp_market_odds` using `method` and evaluates `offered_odds` on `outcome_index`.
#[verified_engine::verified]
pub fn evaluate_outcome_ev(
    sharp_market_odds: &[Odds],
    outcome_index: usize,
    offered_odds: Odds,
    method: DevigMethod,
) -> Result<EvEvaluation, MarketPricingError> {
    let n = sharp_market_odds.len();
    if n > MAX_MARKET_OUTCOMES {
        return Err(MarketPricingError::TooManyOutcomes {
            count: n,
            max_supported: MAX_MARKET_OUTCOMES,
        });
    }
    if outcome_index >= n {
        return Err(MarketPricingError::OutcomeIndexOutOfBounds {
            index: outcome_index,
            outcome_count: n,
        });
    }
    let mut probs = [0.0_f64; MAX_MARKET_OUTCOMES];
    let meta = devig_into(sharp_market_odds, method, &mut probs[..n])?;
    let p_clamped = probs[outcome_index].clamp(0.0, 1.0);
    let unit_p =
        UnitInterval::new(p_clamped).map_err(|_| MarketPricingError::InvalidProbability {
            value: probs[outcome_index],
        })?;
    let mut eval = evaluate_ev_from_probability(unit_p, offered_odds)?;
    eval.devig_metadata = Some(meta);
    Ok(eval)
}

/// Evaluates `offered_odds` on `outcome_index` across all four devigging models simultaneously.
#[verified_engine::verified]
pub fn compare_outcome_ev_all_methods(
    sharp_market_odds: &[Odds],
    outcome_index: usize,
    offered_odds: Odds,
) -> Result<AllMethodsEvReport, MarketPricingError> {
    let multiplicative = evaluate_outcome_ev(
        sharp_market_odds,
        outcome_index,
        offered_odds,
        DevigMethod::Multiplicative,
    )?;
    let additive = evaluate_outcome_ev(
        sharp_market_odds,
        outcome_index,
        offered_odds,
        DevigMethod::Additive,
    )?;
    let power = evaluate_outcome_ev(
        sharp_market_odds,
        outcome_index,
        offered_odds,
        DevigMethod::Power,
    )?;
    let shin = evaluate_outcome_ev(
        sharp_market_odds,
        outcome_index,
        offered_odds,
        DevigMethod::Shin,
    )?;
    Ok(AllMethodsEvReport {
        outcome_index,
        multiplicative,
        additive,
        power,
        shin,
    })
}

// [cite:clinical_trials_statistics]
