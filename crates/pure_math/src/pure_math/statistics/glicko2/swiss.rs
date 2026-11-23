//! Deterministic Swiss-system pairing helpers for Glicko-2 tournament schedules.

use super::core::GlickoPlayer;
use super::tournament::{Pairing, PeriodMatch, SwissRoundPlan, validate_period_matches};
use crate::error::Glicko2Error;
use nalgebra::DMatrix;
use std::cmp::Ordering;

fn compare_standing_order(
    lhs: usize,
    rhs: usize,
    scores: &[f64],
    roster: &[GlickoPlayer],
) -> Ordering {
    scores[rhs]
        .total_cmp(&scores[lhs])
        .then_with(|| {
            roster[rhs]
                .rating
                .value()
                .total_cmp(&roster[lhs].rating.value())
        })
        .then_with(|| {
            roster[lhs]
                .rating_deviation
                .value()
                .total_cmp(&roster[rhs].rating_deviation.value())
        })
        .then_with(|| lhs.cmp(&rhs))
}

fn select_odd_bye(order: &mut Vec<usize>, games_played: &[usize]) -> Option<usize> {
    if order.len().is_multiple_of(2) {
        return None;
    }
    let mut chosen_pos = order.len() - 1;
    let mut best_games = games_played[order[chosen_pos]];
    for pos in (0..order.len()).rev() {
        if games_played[order[pos]] > best_games {
            best_games = games_played[order[pos]];
            chosen_pos = pos;
        }
    }
    Some(order.remove(chosen_pos))
}

fn greedy_swiss_pair(order: &[usize], played: &DMatrix<bool>) -> Vec<Pairing> {
    let active_len = order.len();
    let mut used = vec![false; active_len];
    let mut pairings = Vec::with_capacity(active_len / 2);
    for i in 0..active_len {
        if used[i] {
            continue;
        }
        used[i] = true;
        let p_a = order[i];
        let partner_j = ((i + 1)..active_len)
            .find(|&j| !used[j] && !played[(p_a, order[j])])
            .or_else(|| ((i + 1)..active_len).find(|&j| !used[j]))
            .unwrap_or(i);
        used[partner_j] = true;
        pairings.push(Pairing {
            player_a: p_a,
            player_b: order[partner_j],
        });
    }
    pairings
}

fn repair_swiss_rematches(pairings: &mut [Pairing], played: &DMatrix<bool>) {
    for idx in (0..pairings.len()).rev() {
        let cur = pairings[idx];
        if !played[(cur.player_a, cur.player_b)] {
            continue;
        }
        for prev_idx in (0..idx).rev() {
            let prev = pairings[prev_idx];
            if !played[(prev.player_a, cur.player_a)] && !played[(prev.player_b, cur.player_b)] {
                pairings[prev_idx] = Pairing {
                    player_a: prev.player_a,
                    player_b: cur.player_a,
                };
                pairings[idx] = Pairing {
                    player_a: prev.player_b,
                    player_b: cur.player_b,
                };
                break;
            }
            if !played[(prev.player_a, cur.player_b)] && !played[(prev.player_b, cur.player_a)] {
                pairings[prev_idx] = Pairing {
                    player_a: prev.player_a,
                    player_b: cur.player_b,
                };
                pairings[idx] = Pairing {
                    player_a: prev.player_b,
                    player_b: cur.player_a,
                };
                break;
            }
        }
    }
}

/// Computes deterministic Swiss-system pairings from current standings and prior match history.
#[verified_engine::verified]
pub fn swiss_pairings(
    roster: &[GlickoPlayer],
    scores: &[f64],
    prior_matches: &[PeriodMatch],
) -> Result<SwissRoundPlan, Glicko2Error> {
    let n = roster.len();
    if n < 2 {
        return Err(Glicko2Error::InsufficientPlayers { count: n });
    }
    if scores.len() != n {
        return Err(Glicko2Error::DimensionMismatch {
            expected: n,
            actual: scores.len(),
        });
    }
    for &s in scores {
        if !s.is_finite() {
            return Err(Glicko2Error::InvalidScore { value: s });
        }
    }
    validate_period_matches(n, prior_matches)?;

    let mut played = DMatrix::from_element(n, n, false);
    let mut games_played = vec![0usize; n];
    for m in prior_matches {
        played[(m.player_a, m.player_b)] = true;
        played[(m.player_b, m.player_a)] = true;
        games_played[m.player_a] += 1;
        games_played[m.player_b] += 1;
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| compare_standing_order(a, b, scores, roster));
    let bye_player = select_odd_bye(&mut order, &games_played);
    let mut pairings = greedy_swiss_pair(&order, &played);
    repair_swiss_rematches(&mut pairings, &played);
    Ok(SwissRoundPlan {
        pairings,
        bye_player,
    })
}
