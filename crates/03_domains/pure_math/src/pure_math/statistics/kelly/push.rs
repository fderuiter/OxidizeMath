//! Closed-form Kelly Criterion for three-state pushable wagers (win / push / loss).
//!
//! Whole-number sports spreads (e.g., `-3` in American football) and draw-no-bet contracts
//! return the original stake unchanged (`0` net profit) when the outcome lands on the line.
//! Ignoring the push state by setting `q = 1 - p_win` misprices positive-EV wagers; the exact
//! three-state log-optimal stake scales the binary formula by `1 / (1 - p_push)`.

use super::core::{BankrollFraction, Odds, PROBABILITY_SUM_TOLERANCE};
use crate::error::KellyError;

/// Realized settlement state of a three-state pushable wager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushOutcome {
    /// Wager wins net profit `+b * stake`.
    Win,
    /// Wager pushes (voids) and returns the original stake unchanged (`0` net profit).
    Push,
    /// Wager loses the staked amount (`-stake`).
    Loss,
}

/// Validated three-state pushable wager with win, push, and loss probabilities plus decimal odds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PushWager {
    p_win: f64,
    p_push: f64,
    p_loss: f64,
    odds: Odds,
}

impl PushWager {
    /// Creates a validated three-state wager from explicit `(p_win, p_push, p_loss)` probabilities.
    ///
    /// Probabilities must each lie in `[0, 1]` and sum to `1.0` within `1e-9`.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::{
    ///     Odds, PushWager, push_expected_value, push_kelly_fraction,
    /// };
    ///
    /// let odds = Odds::from_american(-110.0).unwrap();
    /// let wager = PushWager::new(0.50, 0.10, 0.40, odds).unwrap();
    /// assert!((push_expected_value(&wager) - 3.0 / 55.0).abs() < 1e-12);
    /// assert!((push_kelly_fraction(&wager).unwrap().value() - 1.0 / 15.0).abs() < 1e-12);
    /// ```
    #[verified_engine::verified]
    pub fn new(p_win: f64, p_push: f64, p_loss: f64, odds: Odds) -> Result<Self, KellyError> {
        for val in [p_win, p_push, p_loss] {
            if !val.is_finite() || !(0.0..=1.0).contains(&val) {
                return Err(KellyError::InvalidProbability { value: val });
            }
        }
        let sum = p_win + p_push + p_loss;
        if !sum.is_finite() || (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
            return Err(KellyError::InvalidProbabilityDistribution { sum });
        }
        Ok(Self {
            p_win,
            p_push,
            p_loss,
            odds,
        })
    }

    /// Creates a three-state wager from `p_win` and `p_push`, setting `p_loss = 1 - p_win - p_push`.
    #[verified_engine::verified]
    pub fn from_win_and_push(p_win: f64, p_push: f64, odds: Odds) -> Result<Self, KellyError> {
        for val in [p_win, p_push] {
            if !val.is_finite() || !(0.0..=1.0).contains(&val) {
                return Err(KellyError::InvalidProbability { value: val });
            }
        }
        let raw_loss = 1.0 - p_win - p_push;
        if raw_loss < -PROBABILITY_SUM_TOLERANCE {
            return Err(KellyError::InvalidProbabilityDistribution {
                sum: p_win + p_push,
            });
        }
        Self::new(p_win, p_push, raw_loss.max(0.0), odds)
    }

    /// Returns the win probability `p_win`.
    #[verified_engine::verified]
    pub fn p_win(&self) -> f64 {
        self.p_win
    }

    /// Returns the push (refund) probability `p_push`.
    #[verified_engine::verified]
    pub fn p_push(&self) -> f64 {
        self.p_push
    }

    /// Returns the loss probability `p_loss`.
    #[verified_engine::verified]
    pub fn p_loss(&self) -> f64 {
        self.p_loss
    }

    /// Returns the decimal odds offered on the wager.
    #[verified_engine::verified]
    pub fn odds(&self) -> Odds {
        self.odds
    }

    /// Maps a uniform `[0, 1)` variate `u` deterministically onto a [`PushOutcome`].
    #[verified_engine::verified]
    pub fn sample_outcome(&self, u: f64) -> PushOutcome {
        if u < self.p_win {
            PushOutcome::Win
        } else if u < self.p_win + self.p_push {
            PushOutcome::Push
        } else {
            PushOutcome::Loss
        }
    }

    /// Settles `bankroll` after wagering `fraction` and observing `outcome`.
    #[verified_engine::verified]
    pub fn settle(&self, bankroll: f64, fraction: &BankrollFraction, outcome: PushOutcome) -> f64 {
        let stake = bankroll * fraction.value();
        match outcome {
            PushOutcome::Win => (bankroll + stake * self.odds.net_profit_multiplier()).max(0.0),
            PushOutcome::Push => bankroll.max(0.0),
            PushOutcome::Loss => (bankroll - stake).max(0.0),
        }
    }
}

/// Computes the exact unit expected value `EV = b * p_win - p_loss` for a three-state push wager.
#[verified_engine::verified]
pub fn push_expected_value(wager: &PushWager) -> f64 {
    wager.odds.net_profit_multiplier() * wager.p_win - wager.p_loss
}

/// Computes the expected logarithmic growth rate `g(f) = p_win * ln(1 + b * f) + p_loss * ln(1 - f)`.
#[verified_engine::verified]
pub fn push_expected_growth_rate(wager: &PushWager, fraction: &BankrollFraction) -> f64 {
    let f = fraction.value();
    let b = wager.odds.net_profit_multiplier();
    if f < 0.0 || 1.0 + b * f <= 0.0 || (wager.p_loss > 0.0 && f >= 1.0) {
        return f64::NEG_INFINITY;
    }
    let win_term = if wager.p_win > 0.0 {
        wager.p_win * (1.0 + b * f).ln()
    } else {
        0.0
    };
    let loss_term = if wager.p_loss > 0.0 {
        wager.p_loss * (1.0 - f).ln()
    } else {
        0.0
    };
    win_term + loss_term
}

/// Computes the closed-form optimal Kelly stake `f* = (b * p_win - p_loss) / (b * (1 - p_push))`.
#[verified_engine::verified]
pub fn push_kelly_fraction(wager: &PushWager) -> Result<BankrollFraction, KellyError> {
    let ev = push_expected_value(wager);
    let action_prob = 1.0 - wager.p_push;
    if ev <= 0.0 || action_prob <= 0.0 {
        return Err(KellyError::NoEdge {
            probability: wager.p_win,
            odds: wager.odds.value(),
        });
    }
    let b = wager.odds.net_profit_multiplier();
    let f_star = (ev / (b * action_prob)).min(1.0);
    BankrollFraction::new(f_star)
}
