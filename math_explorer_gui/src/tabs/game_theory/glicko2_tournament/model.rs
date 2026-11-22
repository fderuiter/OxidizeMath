#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::Glicko2TournamentTool;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::glicko2::{
    expected_outcome, increase_rd_for_inactivity, update_rating, GlickoPlayer, MatchResult, Rating,
    RatingDeviation, SystemConstant, Volatility,
};

pub const MAX_TOURNAMENT_PLAYERS: usize = 64;
pub const MAX_TOURNAMENT_MATCHES: usize = 1024;
pub const MAX_CSV_IMPORT_BYTES: usize = 64 * 1024;
pub const MAX_PLAYER_NAME_CHARS: usize = 64;
pub const MAX_RATING_PERIOD: usize = 10_000;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RosterPlayer {
    pub name: String,
    pub rating: f64,
    pub rd: f64,
    pub volatility: f64,
}

impl RosterPlayer {
    pub fn to_glicko_player(&self) -> Result<GlickoPlayer, String> {
        if !(100.0..=4000.0).contains(&self.rating)
            || !(1.0..=500.0).contains(&self.rd)
            || !(0.001..=1.5).contains(&self.volatility)
        {
            return Err(format!("Player '{}' parameters out of bounds", self.name));
        }
        let r = Rating::new(self.rating).map_err(|e| format!("{e}"))?;
        let rd = RatingDeviation::new(self.rd).map_err(|e| format!("{e}"))?;
        let vol = Volatility::new(self.volatility).map_err(|e| format!("{e}"))?;
        Ok(GlickoPlayer::new(r, rd, vol))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduledMatch {
    pub player_a: usize,
    pub player_b: usize,
    pub score_a: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glicko2TournamentState {
    pub tau: f64,
    pub period: usize,
    pub roster: Vec<RosterPlayer>,
    pub matches: Vec<ScheduledMatch>,
}

impl Glicko2TournamentState {
    pub fn validate(&self) -> Result<(), String> {
        let _ = SystemConstant::new(self.tau).map_err(|e| format!("Invalid tau: {e}"))?;
        if !(1..=MAX_RATING_PERIOD).contains(&self.period) {
            return Err("Rating period index out of bounds".into());
        }
        if self.roster.len() < 2 || self.roster.len() > MAX_TOURNAMENT_PLAYERS {
            return Err("Roster must contain between 2 and 64 players".into());
        }
        if self.matches.len() > MAX_TOURNAMENT_MATCHES {
            return Err("Match log exceeds maximum of 1,024 matches".into());
        }
        for p in &self.roster {
            let trimmed = p.name.trim();
            if trimmed.is_empty() || p.name.chars().count() > MAX_PLAYER_NAME_CHARS {
                return Err("Player name must be 1..=64 characters".into());
            }
            let _ = p.to_glicko_player()?;
        }
        let dummy = GlickoPlayer::default();
        for m in &self.matches {
            if m.player_a >= self.roster.len()
                || m.player_b >= self.roster.len()
                || m.player_a == m.player_b
            {
                return Err("Invalid match player indices".into());
            }
            let _ = MatchResult::new(dummy, m.score_a).map_err(|e| format!("{e}"))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct StandingRow {
    pub rank: usize,
    pub player_idx: usize,
    pub name: String,
    pub pre_rating: f64,
    pub pre_rd: f64,
    pub post_rating: f64,
    pub post_rd: f64,
    pub post_volatility: f64,
    pub matches_played: usize,
    pub score_sum: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

pub fn glickman_canonical_roster_and_matches() -> (Vec<RosterPlayer>, Vec<ScheduledMatch>) {
    let roster = vec![
        RosterPlayer {
            name: "Player A".into(),
            rating: 1500.0,
            rd: 200.0,
            volatility: 0.06,
        },
        RosterPlayer {
            name: "Opponent 1".into(),
            rating: 1400.0,
            rd: 30.0,
            volatility: 0.06,
        },
        RosterPlayer {
            name: "Opponent 2".into(),
            rating: 1550.0,
            rd: 100.0,
            volatility: 0.06,
        },
        RosterPlayer {
            name: "Opponent 3".into(),
            rating: 1700.0,
            rd: 300.0,
            volatility: 0.06,
        },
        RosterPlayer {
            name: "Inactive Reserve".into(),
            rating: 1500.0,
            rd: 80.0,
            volatility: 0.06,
        },
    ];
    let matches = vec![
        ScheduledMatch {
            player_a: 0,
            player_b: 1,
            score_a: 1.0,
        },
        ScheduledMatch {
            player_a: 0,
            player_b: 2,
            score_a: 0.0,
        },
        ScheduledMatch {
            player_a: 0,
            player_b: 3,
            score_a: 0.0,
        },
    ];
    (roster, matches)
}

pub type PeriodEvaluation = (Vec<GlickoPlayer>, Vec<(usize, f64)>);

pub fn evaluate_simultaneous_period(
    roster: &[RosterPlayer],
    matches: &[ScheduledMatch],
    tau_val: f64,
) -> Result<PeriodEvaluation, String> {
    let tau = SystemConstant::new(tau_val).map_err(|e| format!("{e}"))?;
    let pre_snapshot: Vec<GlickoPlayer> = roster
        .iter()
        .map(RosterPlayer::to_glicko_player)
        .collect::<Result<Vec<_>, _>>()?;

    let n = pre_snapshot.len();
    let mut staged: Vec<Vec<(usize, u64, MatchResult)>> = vec![Vec::new(); n];
    let mut tallies = vec![(0usize, 0.0_f64); n];

    for m in matches {
        if m.player_a >= n || m.player_b >= n || m.player_a == m.player_b {
            return Err("Match references invalid or identical player index".into());
        }
        let score_b = (1.0 - m.score_a).clamp(0.0, 1.0);
        let res_a = MatchResult::new(pre_snapshot[m.player_b], m.score_a)
            .map_err(|e| format!("Invalid match score: {e}"))?;
        let res_b = MatchResult::new(pre_snapshot[m.player_a], score_b)
            .map_err(|e| format!("Invalid match score: {e}"))?;

        staged[m.player_a].push((m.player_b, m.score_a.to_bits(), res_a));
        staged[m.player_b].push((m.player_a, score_b.to_bits(), res_b));
        tallies[m.player_a].0 += 1;
        tallies[m.player_a].1 += m.score_a;
        tallies[m.player_b].0 += 1;
        tallies[m.player_b].1 += score_b;
    }

    let mut post_players = Vec::with_capacity(n);
    for (idx, pre_player) in pre_snapshot.iter().enumerate() {
        if staged[idx].is_empty() {
            post_players.push(increase_rd_for_inactivity(pre_player));
        } else {
            staged[idx].sort_by_key(|(opp_idx, score_bits, _)| (*opp_idx, *score_bits));
            let ordered_results: Vec<MatchResult> =
                staged[idx].iter().map(|(_, _, r)| *r).collect();
            let updated =
                update_rating(pre_player, &ordered_results, &tau).map_err(|e| format!("{e}"))?;
            post_players.push(updated);
        }
    }
    Ok((post_players, tallies))
}

pub fn compute_pairwise_win_matrix(players: &[GlickoPlayer]) -> Vec<Vec<f64>> {
    let scaled: Vec<(f64, f64, f64)> = players.iter().map(GlickoPlayer::to_glicko2_scale).collect();
    let n = scaled.len();
    let mut matrix = vec![vec![0.5; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                matrix[i][j] = expected_outcome(scaled[i].0, scaled[j].0, scaled[j].1);
            }
        }
    }
    matrix
}

fn build_ibeam_error_bar(x: f64, ci_low: f64, ci_high: f64) -> [PlotPoint; 6] {
    const CAP: f64 = 0.18;
    [
        PlotPoint::new(x - CAP, ci_low),
        PlotPoint::new(x + CAP, ci_low),
        PlotPoint::new(x, ci_low),
        PlotPoint::new(x, ci_high),
        PlotPoint::new(x - CAP, ci_high),
        PlotPoint::new(x + CAP, ci_high),
    ]
}

impl Glicko2TournamentTool {
    pub fn recalculate(&mut self) {
        self.standings.clear();
        self.win_prob_matrix.clear();
        self.error_bar_lines.clear();
        self.post_rating_points.clear();
        self.pre_rating_points.clear();
        self.eval_error = None;

        let (post_players, tallies) =
            match evaluate_simultaneous_period(&self.roster, &self.matches, self.tau) {
                Ok(res) => res,
                Err(err) => {
                    self.eval_error = Some(err);
                    return;
                }
            };

        self.win_prob_matrix = compute_pairwise_win_matrix(&post_players);
        let mut rows: Vec<StandingRow> = self
            .roster
            .iter()
            .enumerate()
            .map(|(idx, pre)| {
                let post_r = post_players[idx].rating.value();
                let post_rd = post_players[idx].rating_deviation.value();
                StandingRow {
                    rank: 0,
                    player_idx: idx,
                    name: pre.name.clone(),
                    pre_rating: pre.rating,
                    pre_rd: pre.rd,
                    post_rating: post_r,
                    post_rd,
                    post_volatility: post_players[idx].volatility.value(),
                    matches_played: tallies[idx].0,
                    score_sum: tallies[idx].1,
                    ci_low: post_r - 2.0 * post_rd,
                    ci_high: post_r + 2.0 * post_rd,
                }
            })
            .collect();

        rows.sort_by(|a, b| {
            b.post_rating
                .total_cmp(&a.post_rating)
                .then_with(|| a.player_idx.cmp(&b.player_idx))
        });

        for (rank_zero, row) in rows.iter_mut().enumerate() {
            row.rank = rank_zero + 1;
            let x = row.rank as f64;
            self.post_rating_points
                .push(PlotPoint::new(x, row.post_rating));
            self.pre_rating_points
                .push(PlotPoint::new(x, row.pre_rating));
            self.error_bar_lines.push((
                format!("#{} {} (95% CI)", row.rank, row.name),
                build_ibeam_error_bar(x, row.ci_low, row.ci_high),
            ));
        }
        self.standings = rows;
    }

    pub fn advance_rating_period(&mut self) {
        if self.eval_error.is_some() || self.standings.is_empty() {
            return;
        }
        for row in &self.standings {
            if let Some(slot) = self.roster.get_mut(row.player_idx) {
                slot.rating = row.post_rating;
                slot.rd = row.post_rd.min(500.0);
                slot.volatility = row.post_volatility.clamp(0.001, 1.5);
            }
        }
        self.period = (self.period + 1).min(MAX_RATING_PERIOD);
        self.recalculate();
    }

    pub fn load_glickman_preset(&mut self) {
        let (roster, matches) = glickman_canonical_roster_and_matches();
        self.tau = 0.5;
        self.period = 1;
        self.roster = roster;
        self.matches = matches;
        self.draft_match_a = 0;
        self.draft_match_b = 1;
        self.recalculate();
    }
}
