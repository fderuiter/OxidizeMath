//! Exact match-market probability and fair-odds pricers over [`ScoreGrid`]:
//! `1X2`, Draw-No-Bet, BTTS, Asian Handicap, and Over/Under totals (whole, half, and quarter lines).

use super::grid::ScoreGrid;
use crate::error::MatchModelError;
use crate::pure_math::statistics::kelly::{Odds, UnitInterval};
use serde::{Deserialize, Serialize};

#[verified_engine::verified]
fn prob_to_fair_odds(prob: f64, total_active: f64) -> Result<Odds, MatchModelError> {
    if prob <= 0.0 || total_active <= prob || !prob.is_finite() || !total_active.is_finite() {
        return Err(MatchModelError::DegenerateMarket { probability: prob });
    }
    Odds::new(total_active / prob)
        .map_err(|_| MatchModelError::DegenerateMarket { probability: prob })
}

#[verified_engine::verified]
fn parse_quarter_line(line: f64) -> Result<i64, MatchModelError> {
    if !line.is_finite() || line.abs() > 1_000.0 {
        return Err(MatchModelError::InvalidLine { line });
    }
    let scaled = line * 4.0;
    let rounded = scaled.round();
    if (scaled - rounded).abs() > 1e-9 {
        return Err(MatchModelError::InvalidLine { line });
    }
    Ok(rounded as i64)
}

/// Three-way `1X2` match outcome probabilities derived from a [`ScoreGrid`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OneXTwoQuote {
    /// Home win probability $\sum_{i > j} P(i, j)$.
    pub home_win: f64,
    /// Draw probability $\sum_{i = j} P(i, j)$.
    pub draw: f64,
    /// Away win probability $\sum_{i < j} P(i, j)$.
    pub away_win: f64,
}

impl OneXTwoQuote {
    /// Converts `(home_win, draw, away_win)` into fair zero-vig decimal [`Odds`].
    #[verified_engine::verified]
    pub fn fair_odds(&self) -> Result<(Odds, Odds, Odds), MatchModelError> {
        let total = self.home_win + self.draw + self.away_win;
        Ok((
            prob_to_fair_odds(self.home_win, total)?,
            prob_to_fair_odds(self.draw, total)?,
            prob_to_fair_odds(self.away_win, total)?,
        ))
    }
}

/// Draw-No-Bet (DNB / `AH 0.0`) probabilities with draw stake refund and conditional win shares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawNoBetQuote {
    /// Unconditional home win probability.
    pub home_win: f64,
    /// Draw (stake refund / push) probability.
    pub push: f64,
    /// Unconditional away win probability.
    pub away_win: f64,
    /// Conditional home win probability given no draw (`home_win / (home_win + away_win)`).
    pub conditional_home: UnitInterval,
    /// Conditional away win probability given no draw (`away_win / (home_win + away_win)`).
    pub conditional_away: UnitInterval,
}

impl DrawNoBetQuote {
    /// Fair zero-vig Draw-No-Bet decimal [`Odds`] `(home_odds, away_odds)`.
    #[verified_engine::verified]
    pub fn fair_odds(&self) -> Result<(Odds, Odds), MatchModelError> {
        let decisive = self.home_win + self.away_win;
        Ok((
            prob_to_fair_odds(self.home_win, decisive)?,
            prob_to_fair_odds(self.away_win, decisive)?,
        ))
    }
}

/// Both-Teams-To-Score (`BTTS`) probabilities over a [`ScoreGrid`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BttsQuote {
    /// Probability both teams score at least one goal ($i \ge 1 \land j \ge 1$).
    pub yes: f64,
    /// Probability at least one team fails to score ($i = 0 \lor j = 0$).
    pub no: f64,
}

impl BttsQuote {
    /// Fair zero-vig decimal [`Odds`] `(yes_odds, no_odds)`.
    #[verified_engine::verified]
    pub fn fair_odds(&self) -> Result<(Odds, Odds), MatchModelError> {
        let total = self.yes + self.no;
        Ok((
            prob_to_fair_odds(self.yes, total)?,
            prob_to_fair_odds(self.no, total)?,
        ))
    }
}

/// Exact Asian Handicap settlement probabilities for the home selection at handicap `line`
/// across whole (`.0`), half (`.5`), and split quarter (`.25`, `.75`) lines.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsianHandicapQuote {
    /// Home handicap line (e.g., `-1.0`, `-0.75`, `-0.5`, `-0.25`, `0.0`, `+0.25`).
    pub line: f64,
    /// Effective stake-weighted win probability (`full_win + 0.5 * half_win`).
    pub win: f64,
    /// Effective stake-weighted push probability (`full_push + 0.5 * (half_win + half_loss)`).
    pub push: f64,
    /// Effective stake-weighted loss probability (`full_loss + 0.5 * half_loss`).
    pub loss: f64,
    /// Full-stake win probability.
    pub full_win: f64,
    /// Half-win / half-push probability (quarter lines only).
    pub half_win: f64,
    /// Full-stake push probability (whole lines only).
    pub full_push: f64,
    /// Half-loss / half-push probability (quarter lines only).
    pub half_loss: f64,
    /// Full-stake loss probability.
    pub full_loss: f64,
}

impl AsianHandicapQuote {
    /// Fair zero-vig decimal [`Odds`] `(home_odds, away_odds)` accounting for push and half-stake splits.
    #[verified_engine::verified]
    pub fn fair_odds(&self) -> Result<(Odds, Odds), MatchModelError> {
        let active = self.win + self.loss;
        Ok((
            prob_to_fair_odds(self.win, active)?,
            prob_to_fair_odds(self.loss, active)?,
        ))
    }
}

/// Exact Over/Under total-goals settlement probabilities at `line` across whole, half, and quarter lines.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverUnderQuote {
    /// Goal total line (e.g., `2.0`, `2.25`, `2.5`, `2.75`, `3.0`).
    pub line: f64,
    /// Effective stake-weighted Over probability (`full_over + 0.5 * half_over`).
    pub over: f64,
    /// Effective stake-weighted push probability (`full_push + 0.5 * (half_over + half_under)`).
    pub push: f64,
    /// Effective stake-weighted Under probability (`full_under + 0.5 * half_under`).
    pub under: f64,
    /// Full-stake Over probability.
    pub full_over: f64,
    /// Half-Over / half-push probability (quarter lines only).
    pub half_over: f64,
    /// Full-stake push probability (whole lines only).
    pub full_push: f64,
    /// Half-Under / half-push probability (quarter lines only).
    pub half_under: f64,
    /// Full-stake Under probability.
    pub full_under: f64,
}

impl OverUnderQuote {
    /// Fair zero-vig decimal [`Odds`] `(over_odds, under_odds)` accounting for push and half-stake splits.
    #[verified_engine::verified]
    pub fn fair_odds(&self) -> Result<(Odds, Odds), MatchModelError> {
        let active = self.over + self.under;
        Ok((
            prob_to_fair_odds(self.over, active)?,
            prob_to_fair_odds(self.under, active)?,
        ))
    }
}

/// Computes exact `1X2` (`home_win`, `draw`, `away_win`) probabilities over `grid`.
#[verified_engine::verified]
pub fn one_x_two(grid: &ScoreGrid) -> OneXTwoQuote {
    let max_g = grid.max_goals();
    let mut home_win = 0.0;
    let mut draw = 0.0;
    let mut away_win = 0.0;
    for h in 0..=max_g {
        for a in 0..=max_g {
            let p = grid.prob_unchecked(h, a);
            if h > a {
                home_win += p;
            } else if h == a {
                draw += p;
            } else {
                away_win += p;
            }
        }
    }
    OneXTwoQuote {
        home_win,
        draw,
        away_win,
    }
}

/// Computes exact Draw-No-Bet probabilities and conditional non-draw shares over `grid`.
#[verified_engine::verified]
pub fn draw_no_bet(grid: &ScoreGrid) -> Result<DrawNoBetQuote, MatchModelError> {
    let q = one_x_two(grid);
    let decisive = q.home_win + q.away_win;
    if decisive <= 0.0 || !decisive.is_finite() {
        return Err(MatchModelError::DegenerateMarket {
            probability: decisive,
        });
    }
    let ch = (q.home_win / decisive).clamp(0.0, 1.0);
    let ca = (q.away_win / decisive).clamp(0.0, 1.0);
    let conditional_home =
        UnitInterval::new(ch).map_err(|_| MatchModelError::DegenerateMarket { probability: ch })?;
    let conditional_away =
        UnitInterval::new(ca).map_err(|_| MatchModelError::DegenerateMarket { probability: ca })?;
    Ok(DrawNoBetQuote {
        home_win: q.home_win,
        push: q.draw,
        away_win: q.away_win,
        conditional_home,
        conditional_away,
    })
}

/// Computes exact Both-Teams-To-Score (`BTTS`) probabilities over `grid`.
#[verified_engine::verified]
pub fn btts(grid: &ScoreGrid) -> BttsQuote {
    let max_g = grid.max_goals();
    let mut yes = 0.0;
    let mut no = 0.0;
    for h in 0..=max_g {
        for a in 0..=max_g {
            let p = grid.prob_unchecked(h, a);
            if h >= 1 && a >= 1 {
                yes += p;
            } else {
                no += p;
            }
        }
    }
    BttsQuote { yes, no }
}

#[inline]
#[verified_engine::verified]
fn accumulate_quarter_diff(d4: i64, is_quarter: bool, p: f64, buckets: &mut [f64; 5]) {
    if !is_quarter {
        if d4 > 0 {
            buckets[0] += p;
        } else if d4 == 0 {
            buckets[2] += p;
        } else {
            buckets[4] += p;
        }
    } else if d4 >= 2 {
        buckets[0] += p;
    } else if d4 == 1 {
        buckets[1] += p;
    } else if d4 == -1 {
        buckets[3] += p;
    } else {
        buckets[4] += p;
    }
}

/// Computes exact home Asian Handicap probabilities at `line` (`0.25` increments).
///
/// Quarter lines (e.g., `-0.25`, `-0.75`, `+0.25`) equal the exact 50/50 stake average
/// of adjacent whole and half lines (`line - 0.25` and `line + 0.25`).
#[verified_engine::verified]
pub fn asian_handicap(grid: &ScoreGrid, line: f64) -> Result<AsianHandicapQuote, MatchModelError> {
    let q = parse_quarter_line(line)?;
    let is_quarter = q % 2 != 0;
    let max_g = grid.max_goals();
    let mut b = [0.0_f64; 5];
    for h in 0..=max_g {
        for a in 0..=max_g {
            let d4 = 4 * (h as i64 - a as i64) + q;
            accumulate_quarter_diff(d4, is_quarter, grid.prob_unchecked(h, a), &mut b);
        }
    }
    Ok(AsianHandicapQuote {
        line,
        win: b[0] + 0.5 * b[1],
        push: b[2] + 0.5 * (b[1] + b[3]),
        loss: b[4] + 0.5 * b[3],
        full_win: b[0],
        half_win: b[1],
        full_push: b[2],
        half_loss: b[3],
        full_loss: b[4],
    })
}

/// Computes exact Over/Under total-goals probabilities at `line` (`>= 0.0`, `0.25` increments).
///
/// Quarter lines (e.g., `2.25`, `2.75`) equal the exact 50/50 stake average of adjacent
/// whole and half total lines (`line - 0.25` and `line + 0.25`).
#[verified_engine::verified]
pub fn over_under(grid: &ScoreGrid, line: f64) -> Result<OverUnderQuote, MatchModelError> {
    if line < 0.0 {
        return Err(MatchModelError::InvalidLine { line });
    }
    let q = parse_quarter_line(line)?;
    let is_quarter = q % 2 != 0;
    let max_g = grid.max_goals();
    let mut b = [0.0_f64; 5];
    for h in 0..=max_g {
        for a in 0..=max_g {
            let t4 = 4 * ((h + a) as i64) - q;
            accumulate_quarter_diff(t4, is_quarter, grid.prob_unchecked(h, a), &mut b);
        }
    }
    Ok(OverUnderQuote {
        line,
        over: b[0] + 0.5 * b[1],
        push: b[2] + 0.5 * (b[1] + b[3]),
        under: b[4] + 0.5 * b[3],
        full_over: b[0],
        half_over: b[1],
        full_push: b[2],
        half_under: b[3],
        full_under: b[4],
    })
}
