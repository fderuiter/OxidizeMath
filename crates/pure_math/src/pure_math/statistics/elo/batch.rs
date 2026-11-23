//! Frozen-period and sequential multi-team Elo batch updates with zero heap allocations.

use super::core::{EloConfig, EloRating, Venue};
use super::update::elo_rating_delta;
use crate::error::EloError;
use serde::{Deserialize, Deserializer, Serialize};

/// Head-to-head match outcome within a multi-club Elo schedule or rating period.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct EloBatchMatch {
    /// Index of the first (home) club in the roster slice.
    pub home_team: usize,
    /// Index of the second (away) club in the roster slice.
    pub away_team: usize,
    /// Match score from `home_team`'s perspective in `[0.0, 1.0]`.
    pub score_home: f64,
    /// Absolute goal/point margin (`|home_goals - away_goals|`).
    pub goal_diff: u32,
    /// Venue indicator (`Venue::Home` applies `config.home_advantage()`).
    pub venue: Venue,
}

impl EloBatchMatch {
    /// Creates a validated batch match from an explicit score and goal margin.
    #[verified_engine::verified]
    pub fn new(
        home_team: usize,
        away_team: usize,
        score_home: f64,
        goal_diff: u32,
        venue: Venue,
    ) -> Result<Self, EloError> {
        if home_team == away_team {
            return Err(EloError::SelfMatch {
                team_index: home_team,
            });
        }
        if !(0.0..=1.0).contains(&score_home) || !score_home.is_finite() {
            return Err(EloError::InvalidScore { value: score_home });
        }
        Ok(Self {
            home_team,
            away_team,
            score_home,
            goal_diff,
            venue,
        })
    }

    /// Creates a validated batch match from integer team goals.
    #[verified_engine::verified]
    pub fn from_goals(
        home_team: usize,
        away_team: usize,
        home_goals: u32,
        away_goals: u32,
        venue: Venue,
    ) -> Result<Self, EloError> {
        let score_home = if home_goals > away_goals {
            1.0
        } else if home_goals < away_goals {
            0.0
        } else {
            0.5
        };
        Self::new(
            home_team,
            away_team,
            score_home,
            home_goals.abs_diff(away_goals),
            venue,
        )
    }

    /// Complementary match score from `away_team`'s perspective (`1.0 - score_home`).
    #[verified_engine::verified]
    pub fn score_away(&self) -> f64 {
        1.0 - self.score_home
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEloBatchMatch {
    home_team: usize,
    away_team: usize,
    score_home: f64,
    goal_diff: u32,
    venue: Venue,
}

impl<'de> Deserialize<'de> for EloBatchMatch {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawEloBatchMatch::deserialize(deserializer)?;
        Self::new(
            raw.home_team,
            raw.away_team,
            raw.score_home,
            raw.goal_diff,
            raw.venue,
        )
        .map_err(serde::de::Error::custom)
    }
}

#[verified_engine::verified]
fn validate_batch_matches(roster_len: usize, matches: &[EloBatchMatch]) -> Result<(), EloError> {
    for m in matches {
        if m.home_team >= roster_len || m.away_team >= roster_len {
            let index = if m.home_team >= roster_len {
                m.home_team
            } else {
                m.away_team
            };
            return Err(EloError::PlayerIndexOutOfBounds { index, roster_len });
        }
        if m.home_team == m.away_team {
            return Err(EloError::SelfMatch {
                team_index: m.home_team,
            });
        }
        if !(0.0..=1.0).contains(&m.score_home) || !m.score_home.is_finite() {
            return Err(EloError::InvalidScore {
                value: m.score_home,
            });
        }
    }
    Ok(())
}

#[verified_engine::verified]
fn collect_team_deltas(
    team_idx: usize,
    frozen_roster: &[EloRating],
    matches: &[EloBatchMatch],
    config: &EloConfig,
    scratch_deltas: &mut [f64],
) -> Result<f64, EloError> {
    let mut count = 0usize;
    for m in matches {
        if m.home_team == team_idx || m.away_team == team_idx {
            let home_delta = elo_rating_delta(
                frozen_roster[m.home_team],
                frozen_roster[m.away_team],
                m.score_home,
                m.goal_diff,
                m.venue,
                config,
            )?;
            scratch_deltas[count] = if m.home_team == team_idx {
                home_delta
            } else {
                -home_delta
            };
            count += 1;
        }
    }
    scratch_deltas[..count].sort_unstable_by(f64::total_cmp);
    let mut total = 0.0;
    for &d in &scratch_deltas[..count] {
        total += d;
    }
    Ok(total)
}

/// Evaluates a frozen Elo rating period into pre-allocated slices with zero heap allocations.
///
/// Sorting each team's match deltas via `f64::total_cmp` guarantees bit-exact invariance to
/// match ordering within `matches`.
#[verified_engine::verified]
pub fn evaluate_elo_batch_into(
    roster: &[EloRating],
    matches: &[EloBatchMatch],
    config: &EloConfig,
    scratch_deltas: &mut [f64],
    out_roster: &mut [EloRating],
) -> Result<(), EloError> {
    if out_roster.len() != roster.len() {
        return Err(EloError::DimensionMismatch {
            expected: roster.len(),
            actual: out_roster.len(),
        });
    }
    if scratch_deltas.len() < matches.len() {
        return Err(EloError::DimensionMismatch {
            expected: matches.len(),
            actual: scratch_deltas.len(),
        });
    }
    validate_batch_matches(roster.len(), matches)?;
    for (idx, &rating) in roster.iter().enumerate() {
        let delta = collect_team_deltas(idx, roster, matches, config, scratch_deltas)?;
        out_roster[idx] = EloRating::new(rating.value() + delta)?;
    }
    Ok(())
}

/// Evaluates a frozen Elo rating period simultaneously across `roster`, returning an updated `Vec<EloRating>`.
#[verified_engine::verified]
pub fn evaluate_elo_batch(
    roster: &[EloRating],
    matches: &[EloBatchMatch],
    config: &EloConfig,
) -> Result<Vec<EloRating>, EloError> {
    let mut scratch = vec![0.0; matches.len()];
    let mut out = vec![EloRating::default(); roster.len()];
    evaluate_elo_batch_into(roster, matches, config, &mut scratch, &mut out)?;
    Ok(out)
}

/// Applies Elo matches sequentially in slice order into `out_roster` with zero heap allocations.
#[verified_engine::verified]
pub fn evaluate_elo_sequential_into(
    roster: &[EloRating],
    matches: &[EloBatchMatch],
    config: &EloConfig,
    out_roster: &mut [EloRating],
) -> Result<(), EloError> {
    if out_roster.len() != roster.len() {
        return Err(EloError::DimensionMismatch {
            expected: roster.len(),
            actual: out_roster.len(),
        });
    }
    validate_batch_matches(roster.len(), matches)?;
    out_roster.copy_from_slice(roster);
    for m in matches {
        let home = out_roster[m.home_team];
        let away = out_roster[m.away_team];
        let delta = elo_rating_delta(home, away, m.score_home, m.goal_diff, m.venue, config)?;
        out_roster[m.home_team] = EloRating::new(home.value() + delta)?;
        out_roster[m.away_team] = EloRating::new(away.value() - delta)?;
    }
    Ok(())
}

/// Applies Elo matches sequentially one fixture at a time, mutating intermediate ratings across the schedule.
#[verified_engine::verified]
pub fn evaluate_elo_sequential(
    roster: &[EloRating],
    matches: &[EloBatchMatch],
    config: &EloConfig,
) -> Result<Vec<EloRating>, EloError> {
    let mut out = vec![EloRating::default(); roster.len()];
    evaluate_elo_sequential_into(roster, matches, config, &mut out)?;
    Ok(out)
}
