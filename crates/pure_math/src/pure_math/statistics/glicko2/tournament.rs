//! Multi-player frozen-period evaluation, head-to-head win matrices, and tournament pairings
//! for the Glicko-2 rating system.

use super::core::{GlickoPlayer, MatchResult, SystemConstant, expected_outcome};
use super::rating::{increase_rd_for_inactivity, update_rating};
use crate::error::Glicko2Error;
use nalgebra::DMatrix;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;
use serde::{Deserialize, Deserializer, Serialize};

pub use super::swiss::swiss_pairings;

/// Head-to-head match outcome recorded within a tournament rating period.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PeriodMatch {
    /// Index of the first player in the tournament roster.
    pub player_a: usize,
    /// Index of the second player in the tournament roster.
    pub player_b: usize,
    /// Score achieved by `player_a` in `[0.0, 1.0]` (`1.0` = win, `0.5` = draw, `0.0` = loss).
    pub score_a: f64,
}

impl PeriodMatch {
    /// Creates a validated head-to-head period match.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::glicko2::PeriodMatch;
    ///
    /// let game = PeriodMatch::new(0, 1, 1.0).unwrap();
    /// assert_eq!(game.score_b(), 0.0);
    /// ```
    #[verified_engine::verified]
    pub fn new(player_a: usize, player_b: usize, score_a: f64) -> Result<Self, Glicko2Error> {
        if player_a == player_b {
            return Err(Glicko2Error::SelfMatch {
                player_index: player_a,
            });
        }
        if !(0.0..=1.0).contains(&score_a) || !score_a.is_finite() {
            return Err(Glicko2Error::InvalidScore { value: score_a });
        }
        Ok(Self {
            player_a,
            player_b,
            score_a,
        })
    }

    /// Complementary match score from `player_b`'s perspective (`1.0 - score_a`).
    #[verified_engine::verified]
    pub fn score_b(&self) -> f64 {
        1.0 - self.score_a
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPeriodMatch {
    player_a: usize,
    player_b: usize,
    score_a: f64,
}

impl<'de> Deserialize<'de> for PeriodMatch {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawPeriodMatch::deserialize(deserializer)?;
        Self::new(raw.player_a, raw.player_b, raw.score_a).map_err(serde::de::Error::custom)
    }
}

/// Scheduled head-to-head pairing between two roster indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pairing {
    /// First competitor index in the roster.
    pub player_a: usize,
    /// Second competitor index in the roster.
    pub player_b: usize,
}

impl Pairing {
    /// Creates a validated pairing between two distinct competitors.
    #[verified_engine::verified]
    pub fn new(player_a: usize, player_b: usize) -> Result<Self, Glicko2Error> {
        if player_a == player_b {
            return Err(Glicko2Error::SelfMatch {
                player_index: player_a,
            });
        }
        Ok(Self { player_a, player_b })
    }
}

/// Deterministic Swiss-system round pairing plan, including an optional odd-roster bye.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SwissRoundPlan {
    /// Head-to-head board pairings ordered from top table downward.
    pub pairings: Vec<Pairing>,
    /// Player index assigned a deterministic bye when the roster count is odd.
    pub bye_player: Option<usize>,
}

#[verified_engine::verified]
fn validate_player_pair(a: usize, b: usize, roster_len: usize) -> Result<(), Glicko2Error> {
    if a >= roster_len || b >= roster_len {
        let index = if a >= roster_len { a } else { b };
        return Err(Glicko2Error::PlayerIndexOutOfBounds { index, roster_len });
    }
    if a == b {
        return Err(Glicko2Error::SelfMatch { player_index: a });
    }
    Ok(())
}

#[verified_engine::verified]
pub(super) fn validate_period_matches(
    roster_len: usize,
    matches: &[PeriodMatch],
) -> Result<(), Glicko2Error> {
    for m in matches {
        validate_player_pair(m.player_a, m.player_b, roster_len)?;
        if !(0.0..=1.0).contains(&m.score_a) || !m.score_a.is_finite() {
            return Err(Glicko2Error::InvalidScore { value: m.score_a });
        }
    }
    Ok(())
}

fn match_result_sort_key(res: &MatchResult) -> (u64, u64, u64, u64) {
    (
        res.opponent.rating.value().to_bits(),
        res.opponent.rating_deviation.value().to_bits(),
        res.opponent.volatility.value().to_bits(),
        res.score.to_bits(),
    )
}

#[verified_engine::verified]
fn collect_player_results(
    player_idx: usize,
    frozen_roster: &[GlickoPlayer],
    matches: &[PeriodMatch],
    scratch_results: &mut [MatchResult],
) -> Result<usize, Glicko2Error> {
    let mut count = 0usize;
    for m in matches {
        if m.player_a == player_idx {
            scratch_results[count] = MatchResult::new(frozen_roster[m.player_b], m.score_a)?;
            count += 1;
        } else if m.player_b == player_idx {
            scratch_results[count] = MatchResult::new(frozen_roster[m.player_a], m.score_b())?;
            count += 1;
        }
    }
    scratch_results[..count].sort_unstable_by_key(match_result_sort_key);
    Ok(count)
}

/// Evaluates a frozen Glicko-2 rating period into pre-allocated buffers without heap allocation.
#[verified_engine::verified]
pub fn evaluate_rating_period_into(
    roster: &[GlickoPlayer],
    matches: &[PeriodMatch],
    tau: &SystemConstant,
    scratch_results: &mut [MatchResult],
    out_roster: &mut [GlickoPlayer],
) -> Result<(), Glicko2Error> {
    if out_roster.len() != roster.len() {
        return Err(Glicko2Error::DimensionMismatch {
            expected: roster.len(),
            actual: out_roster.len(),
        });
    }
    if scratch_results.len() < matches.len() {
        return Err(Glicko2Error::DimensionMismatch {
            expected: matches.len(),
            actual: scratch_results.len(),
        });
    }
    validate_period_matches(roster.len(), matches)?;
    for (idx, player) in roster.iter().enumerate() {
        let count = collect_player_results(idx, roster, matches, scratch_results)?;
        out_roster[idx] = if count == 0 {
            increase_rd_for_inactivity(player)
        } else {
            update_rating(player, &scratch_results[..count], tau)?
        };
    }
    Ok(())
}

/// Evaluates a Glicko-2 rating period simultaneously across a frozen pre-period roster.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::glicko2::*;
///
/// let roster = [GlickoPlayer::default(), GlickoPlayer::default()];
/// let updated = evaluate_rating_period(&roster, &[PeriodMatch::new(0, 1, 1.0).unwrap()], &SystemConstant::default()).unwrap();
/// assert!(updated[0].rating.value() > updated[1].rating.value());
/// ```
#[verified_engine::verified]
pub fn evaluate_rating_period(
    roster: &[GlickoPlayer],
    matches: &[PeriodMatch],
    tau: &SystemConstant,
) -> Result<Vec<GlickoPlayer>, Glicko2Error> {
    let placeholder = MatchResult::new(GlickoPlayer::default(), 0.5)?;
    let mut scratch = vec![placeholder; matches.len()];
    let mut out = vec![GlickoPlayer::default(); roster.len()];
    evaluate_rating_period_into(roster, matches, tau, &mut scratch, &mut out)?;
    Ok(out)
}

/// Applies matches sequentially one game at a time (mutating ratings mid-period) to measure
/// order-dependent sequential drift relative to [`evaluate_rating_period`].
#[verified_engine::verified]
pub fn evaluate_sequential_period(
    roster: &[GlickoPlayer],
    matches: &[PeriodMatch],
    tau: &SystemConstant,
) -> Result<Vec<GlickoPlayer>, Glicko2Error> {
    validate_period_matches(roster.len(), matches)?;
    let mut state = roster.to_vec();
    let mut played = vec![false; roster.len()];
    for m in matches {
        let (pre_a, pre_b) = (state[m.player_a], state[m.player_b]);
        state[m.player_a] = update_rating(&pre_a, &[MatchResult::new(pre_b, m.score_a)?], tau)?;
        state[m.player_b] = update_rating(&pre_b, &[MatchResult::new(pre_a, m.score_b())?], tau)?;
        played[m.player_a] = true;
        played[m.player_b] = true;
    }
    for (idx, was_active) in played.into_iter().enumerate() {
        if !was_active {
            state[idx] = increase_rd_for_inactivity(&state[idx]);
        }
    }
    Ok(state)
}

/// Computes head-to-head expected win probability between two players using combined Glicko-2
/// uncertainty $\phi_{ij} = \sqrt{\phi_i^2 + \phi_j^2}$ so $P_{ij} + P_{ji} = 1$.
#[verified_engine::verified]
pub fn head_to_head_win_probability(player_i: &GlickoPlayer, player_j: &GlickoPlayer) -> f64 {
    let (mu_i, phi_i, _) = player_i.to_glicko2_scale();
    let (mu_j, phi_j, _) = player_j.to_glicko2_scale();
    expected_outcome(mu_i, mu_j, (phi_i * phi_i + phi_j * phi_j).sqrt())
}

/// Computes head-to-head expected win probability bounded in [`math_commons::primitives::UnitInterval`].
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::glicko2::{
///     GlickoPlayer, head_to_head_win_probability_unit,
/// };
///
/// let p1 = GlickoPlayer::default();
/// let p2 = GlickoPlayer::default();
/// let win_prob = head_to_head_win_probability_unit(&p1, &p2);
/// assert!((win_prob.value() - 0.5).abs() < 1e-12);
/// ```
#[verified_engine::verified]
pub fn head_to_head_win_probability_unit(
    player_i: &GlickoPlayer,
    player_j: &GlickoPlayer,
) -> math_commons::primitives::UnitInterval {
    math_commons::primitives::UnitInterval::new_clamped(head_to_head_win_probability(
        player_i, player_j,
    ))
}

/// Computes the $N \times N$ pairwise head-to-head win-probability matrix for `roster`.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::glicko2::{GlickoPlayer, pairwise_win_matrix};
///
/// let roster = vec![GlickoPlayer::default(), GlickoPlayer::default()];
/// let matrix = pairwise_win_matrix(&roster);
/// assert!((matrix[(0, 1)] - 0.5).abs() < 1e-12);
/// ```
#[verified_engine::verified]
pub fn pairwise_win_matrix(roster: &[GlickoPlayer]) -> DMatrix<f64> {
    let n = roster.len();
    let mut matrix = DMatrix::from_element(n, n, 0.5);
    for i in 0..n {
        for j in (i + 1)..n {
            let p_ij = head_to_head_win_probability(&roster[i], &roster[j]);
            matrix[(i, j)] = p_ij;
            matrix[(j, i)] = 1.0 - p_ij;
        }
    }
    matrix
}

/// Generates a deterministic single round-robin tournament schedule using the circle method.
#[verified_engine::verified]
pub fn round_robin_schedule(num_players: usize) -> Result<Vec<Vec<Pairing>>, Glicko2Error> {
    if num_players < 2 {
        return Err(Glicko2Error::InsufficientPlayers { count: num_players });
    }
    let slots = if num_players.is_multiple_of(2) {
        num_players
    } else {
        num_players + 1
    };
    let (bye_sentinel, num_rounds, half) = (num_players, slots - 1, slots / 2);
    let mut ring: Vec<usize> = (0..slots).collect();
    let mut schedule = Vec::with_capacity(num_rounds);

    for _ in 0..num_rounds {
        let mut round_pairings = Vec::with_capacity(half);
        for board in 0..half {
            let (left, right) = (ring[board], ring[slots - 1 - board]);
            if left != bye_sentinel && right != bye_sentinel {
                round_pairings.push(Pairing {
                    player_a: left.min(right),
                    player_b: left.max(right),
                });
            }
        }
        schedule.push(round_pairings);
        let last = ring.pop().unwrap_or(0);
        ring.insert(1, last);
    }
    Ok(schedule)
}

/// Simulates stochastic match outcomes for `pairings` using [`OxidizeRng`] while preserving exact
/// Glicko-2 expected scores across `draw_probability` in `[0.0, 1.0]`.
#[verified_engine::verified]
pub fn simulate_period_matches(
    skill_roster: &[GlickoPlayer],
    pairings: &[Pairing],
    draw_probability: f64,
    rng: &mut OxidizeRng,
) -> Result<Vec<PeriodMatch>, Glicko2Error> {
    if !(0.0..=1.0).contains(&draw_probability) || !draw_probability.is_finite() {
        return Err(Glicko2Error::InvalidScore {
            value: draw_probability,
        });
    }
    let roster_len = skill_roster.len();
    let mut outcomes = Vec::with_capacity(pairings.len());
    for pairing in pairings {
        validate_player_pair(pairing.player_a, pairing.player_b, roster_len)?;
        let p_exp = head_to_head_win_probability(
            &skill_roster[pairing.player_a],
            &skill_roster[pairing.player_b],
        );
        let closeness = p_exp.min(1.0 - p_exp);
        let p_draw = 2.0 * draw_probability * closeness;
        let p_win_a = p_exp - draw_probability * closeness;
        let u: f64 = rng.r#gen();
        let score_a = if u < p_win_a {
            1.0
        } else if u < p_win_a + p_draw {
            0.5
        } else {
            0.0
        };
        outcomes.push(PeriodMatch::new(
            pairing.player_a,
            pairing.player_b,
            score_a,
        )?);
    }
    Ok(outcomes)
}

// [cite:sports_quantitative_modeling]
