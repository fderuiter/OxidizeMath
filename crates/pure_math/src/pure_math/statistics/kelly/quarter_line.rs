//! Five-state Asian quarter-line Kelly Criterion via bracketed root-finding.
//!
//! Asian quarter-handicaps (such as `-0.25`, `-2.75`, or `+1.25`) split a unit stake equally across
//! two adjacent half/whole lines, inducing five distinct settlement states with net payoffs
//! `{+b, +b/2, 0, -1/2, -1}`. Because `g'(f)` is strictly decreasing on `[0, 1)`, the exact
//! log-optimal stake is solved deterministically with zero heap allocations via [`Bisection`].

use super::core::{BankrollFraction, Odds, PROBABILITY_SUM_TOLERANCE};
use crate::error::KellyError;
use crate::pure_math::analysis::roots::{AnalysisError, Bisection, RootFinder};

const QUARTER_LINE_ROOT_ITERS: usize = 128;
const QUARTER_LINE_ROOT_TOL: f64 = 1e-13;

/// Realized settlement state of a five-state Asian quarter-line wager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuarterLineOutcome {
    /// Both split half-stakes win (`+b` net profit per unit staked).
    FullWin,
    /// One split half-stake wins and one pushes (`+b / 2` net profit per unit staked).
    HalfWin,
    /// Net settlement is zero (`0` net profit; stake returned).
    Push,
    /// One split half-stake pushes and one loses (`-1 / 2` net return per unit staked).
    HalfLoss,
    /// Both split half-stakes lose (`-1` net return per unit staked).
    FullLoss,
}

/// Validated five-state Asian quarter-line wager.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuarterLineWager {
    p_full_win: f64,
    p_half_win: f64,
    p_push: f64,
    p_half_loss: f64,
    p_full_loss: f64,
    odds: Odds,
}

impl QuarterLineWager {
    /// Constructs a validated five-state quarter-line wager whose probabilities sum to `1.0`.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::{
    ///     Odds, QuarterLineWager, quarter_line_expected_value, quarter_line_kelly_fraction,
    /// };
    ///
    /// let odds = Odds::from_american(-110.0).unwrap();
    /// let ql = QuarterLineWager::new(0.44, 0.12, 0.0, 0.0, 0.44, odds).unwrap();
    /// assert!((quarter_line_expected_value(&ql) - 0.16 / 11.0).abs() < 1e-12);
    /// assert!((quarter_line_kelly_fraction(&ql).unwrap().value() - 0.01751665).abs() < 1e-7);
    /// ```
    #[verified_engine::verified]
    pub fn new(
        p_full_win: f64,
        p_half_win: f64,
        p_push: f64,
        p_half_loss: f64,
        p_full_loss: f64,
        odds: Odds,
    ) -> Result<Self, KellyError> {
        for val in [p_full_win, p_half_win, p_push, p_half_loss, p_full_loss] {
            if !val.is_finite() || !(0.0..=1.0).contains(&val) {
                return Err(KellyError::InvalidProbability { value: val });
            }
        }
        let sum = p_full_win + p_half_win + p_push + p_half_loss + p_full_loss;
        if !sum.is_finite() || (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
            return Err(KellyError::InvalidProbabilityDistribution { sum });
        }
        Ok(Self {
            p_full_win,
            p_half_win,
            p_push,
            p_half_loss,
            p_full_loss,
            odds,
        })
    }

    /// Returns the full-win probability `p_full_win` (`+b`).
    #[verified_engine::verified]
    pub fn p_full_win(&self) -> f64 {
        self.p_full_win
    }

    /// Returns the half-win probability `p_half_win` (`+b / 2`).
    #[verified_engine::verified]
    pub fn p_half_win(&self) -> f64 {
        self.p_half_win
    }

    /// Returns the full-push probability `p_push` (`0`).
    #[verified_engine::verified]
    pub fn p_push(&self) -> f64 {
        self.p_push
    }

    /// Returns the half-loss probability `p_half_loss` (`-1 / 2`).
    #[verified_engine::verified]
    pub fn p_half_loss(&self) -> f64 {
        self.p_half_loss
    }

    /// Returns the full-loss probability `p_full_loss` (`-1`).
    #[verified_engine::verified]
    pub fn p_full_loss(&self) -> f64 {
        self.p_full_loss
    }

    /// Returns the decimal odds offered on the quarter-line wager.
    #[verified_engine::verified]
    pub fn odds(&self) -> Odds {
        self.odds
    }

    /// Maps a uniform `[0, 1)` variate `u` deterministically onto a [`QuarterLineOutcome`].
    #[verified_engine::verified]
    pub fn sample_outcome(&self, u: f64) -> QuarterLineOutcome {
        let c1 = self.p_full_win;
        let c2 = c1 + self.p_half_win;
        let c3 = c2 + self.p_push;
        let c4 = c3 + self.p_half_loss;
        if u < c1 {
            QuarterLineOutcome::FullWin
        } else if u < c2 {
            QuarterLineOutcome::HalfWin
        } else if u < c3 {
            QuarterLineOutcome::Push
        } else if u < c4 {
            QuarterLineOutcome::HalfLoss
        } else {
            QuarterLineOutcome::FullLoss
        }
    }

    /// Settles `bankroll` after wagering `fraction` and observing `outcome`.
    #[verified_engine::verified]
    pub fn settle(
        &self,
        bankroll: f64,
        fraction: &BankrollFraction,
        outcome: QuarterLineOutcome,
    ) -> f64 {
        let stake = bankroll * fraction.value();
        let b = self.odds.net_profit_multiplier();
        match outcome {
            QuarterLineOutcome::FullWin => (bankroll + stake * b).max(0.0),
            QuarterLineOutcome::HalfWin => (bankroll + 0.5 * stake * b).max(0.0),
            QuarterLineOutcome::Push => bankroll.max(0.0),
            QuarterLineOutcome::HalfLoss => (bankroll - 0.5 * stake).max(0.0),
            QuarterLineOutcome::FullLoss => (bankroll - stake).max(0.0),
        }
    }
}

/// Computes unit expected value `EV = b * p_fw + (b/2) * p_hw - (1/2) * p_hl - p_fl`.
#[verified_engine::verified]
pub fn quarter_line_expected_value(wager: &QuarterLineWager) -> f64 {
    let b = wager.odds.net_profit_multiplier();
    b * wager.p_full_win + 0.5 * b * wager.p_half_win - 0.5 * wager.p_half_loss - wager.p_full_loss
}

/// Computes logarithmic bankroll growth rate `g(f)` across all five quarter-line states.
#[verified_engine::verified]
pub fn quarter_line_expected_growth_rate(
    wager: &QuarterLineWager,
    fraction: &BankrollFraction,
) -> f64 {
    let f = fraction.value();
    let b = wager.odds.net_profit_multiplier();
    if f < 0.0
        || 1.0 + b * f <= 0.0
        || (wager.p_full_loss > 0.0 && f >= 1.0)
        || (wager.p_half_loss > 0.0 && f >= 2.0)
    {
        return f64::NEG_INFINITY;
    }
    let mut g = 0.0;
    if wager.p_full_win > 0.0 {
        g += wager.p_full_win * (1.0 + b * f).ln();
    }
    if wager.p_half_win > 0.0 {
        g += wager.p_half_win * (1.0 + 0.5 * b * f).ln();
    }
    if wager.p_half_loss > 0.0 {
        g += wager.p_half_loss * (1.0 - 0.5 * f).ln();
    }
    if wager.p_full_loss > 0.0 {
        g += wager.p_full_loss * (1.0 - f).ln();
    }
    g
}

fn growth_derivative_inner(wager: &QuarterLineWager, f: f64) -> f64 {
    let b = wager.odds.net_profit_multiplier();
    let half_b = 0.5 * b;
    let fw = (b * wager.p_full_win) / (1.0 + b * f);
    let hw = (half_b * wager.p_half_win) / (1.0 + half_b * f);
    let hl = (0.5 * wager.p_half_loss) / (1.0 - 0.5 * f);
    let fl = if wager.p_full_loss > 0.0 {
        wager.p_full_loss / (1.0 - f)
    } else {
        0.0
    };
    fw + hw - hl - fl
}

/// Evaluates the strictly decreasing first derivative `g'(f)` of expected log-growth.
#[verified_engine::verified]
pub fn quarter_line_growth_derivative(wager: &QuarterLineWager, f: f64) -> f64 {
    growth_derivative_inner(wager, f)
}

/// Solves `g'(f*) = 0` on `[0, 1]` via bracketed [`Bisection`] with zero heap allocations.
#[verified_engine::verified]
pub fn quarter_line_kelly_fraction(
    wager: &QuarterLineWager,
) -> Result<BankrollFraction, KellyError> {
    let g_zero = quarter_line_expected_value(wager);
    if g_zero <= 0.0 {
        return Err(KellyError::NoEdge {
            probability: wager.p_full_win + 0.5 * wager.p_half_win,
            odds: wager.odds.value(),
        });
    }
    if wager.p_full_loss == 0.0 && growth_derivative_inner(wager, 1.0) >= 0.0 {
        return BankrollFraction::new(1.0);
    }

    let mut gap = 1e-12;
    let mut upper = 1.0 - gap;
    while growth_derivative_inner(wager, upper) >= 0.0 && gap > 1e-16 {
        gap *= 0.25;
        upper = 1.0 - gap;
    }
    if growth_derivative_inner(wager, upper) >= 0.0 {
        return BankrollFraction::new(1.0);
    }

    let solver = Bisection::new(QUARTER_LINE_ROOT_ITERS, QUARTER_LINE_ROOT_TOL);
    let root = match solver.find_root(|f| growth_derivative_inner(wager, f), 0.0, upper) {
        Ok(r) => r,
        Err(AnalysisError::ConvergenceError(best)) => best,
        Err(AnalysisError::InvalidParameters(_)) => {
            return Err(KellyError::ConvergenceFailed {
                iterations: QUARTER_LINE_ROOT_ITERS,
                residual: growth_derivative_inner(wager, upper).abs(),
            });
        }
    };
    BankrollFraction::new(root.clamp(0.0, 1.0))
}
