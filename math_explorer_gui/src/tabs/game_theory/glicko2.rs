//! Interactive client-side Glicko-2 tournament simulator (`Glicko2TournamentTool`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[cfg(test)]
use crate::framework::InteractiveTool;
use crate::presets::{
    Glicko2MatchPreset, Glicko2PairingMode, Glicko2PeriodPreset, Glicko2TournamentPresetConfig,
    MAX_TOURNAMENT_PERIODS, MAX_TOURNAMENT_PLAYERS,
};
#[cfg(test)]
use eframe::egui;
use math_explorer::pure_math::statistics::glicko2::{
    evaluate_rating_period, evaluate_sequential_period, round_robin_schedule,
    simulate_period_matches, swiss_pairings, GlickoPlayer, Pairing, PeriodMatch, Rating,
    RatingDeviation, SystemConstant, Volatility,
};
use oxidize_core::rng::OxidizeRng;
use std::collections::HashSet;

#[path = "glicko2/io.rs"]
pub(crate) mod io;
#[path = "glicko2/io_dialog.rs"]
mod io_dialog;
#[path = "glicko2/plots.rs"]
mod plots;
#[path = "glicko2/registration.rs"]
mod registration;
#[path = "glicko2/ui.rs"]
mod ui;

#[cfg(test)]
#[path = "glicko2/tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glicko2PlotMode {
    ConfidenceBands,
    Volatility,
    WinMatrix,
    SequentialDrift,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BottomViewTab {
    Standings,
    NextPairings,
    WinMatrixTable,
}

#[derive(Clone, Debug)]
pub struct CompetitorEntry {
    pub name: String,
    pub initial: GlickoPlayer,
    pub current: GlickoPlayer,
    pub sequential: GlickoPlayer,
    pub inactive: bool,
    pub points: f64,
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
    pub byes: usize,
}

impl CompetitorEntry {
    fn new(name: String, initial: GlickoPlayer, inactive: bool) -> Self {
        Self {
            name,
            initial,
            current: initial,
            sequential: initial,
            inactive,
            points: 0.0,
            wins: 0,
            draws: 0,
            losses: 0,
            byes: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PeriodSnapshot {
    pub period: usize,
    pub frozen: Vec<GlickoPlayer>,
    pub sequential: Vec<GlickoPlayer>,
}

#[derive(Clone, Debug)]
pub struct ScheduledPairingRow {
    pub pairing: Pairing,
    pub manual_score_a: f64,
}

pub struct Glicko2TournamentTool {
    pub(crate) tau: f64,
    pub(crate) draw_probability: f64,
    pub(crate) pairing_mode: Glicko2PairingMode,
    pub(crate) target_rounds: usize,
    pub(crate) rng_seed: u64,
    pub(crate) competitors: Vec<CompetitorEntry>,
    pub(crate) completed_periods: Vec<Glicko2PeriodPreset>,
    pub(crate) snapshots: Vec<PeriodSnapshot>,
    pub(crate) next_pairings: Vec<ScheduledPairingRow>,
    pub(crate) next_bye_player: Option<usize>,
    pub(crate) plot_mode: Glicko2PlotMode,
    pub(crate) bottom_tab: BottomViewTab,
    pub(crate) focus_player: Option<usize>,
    pub(crate) new_player_name: String,
    pub(crate) state_rev: u64,
    pub(crate) io_state: io_dialog::TournamentIoState,
}

pub(crate) fn match_preset(player_a: usize, player_b: usize, score_a: f64) -> Glicko2MatchPreset {
    Glicko2MatchPreset {
        player_a,
        player_b,
        score_a,
    }
}

const DEFAULT_CONFIG_JSON: &str = r#"{"tau":0.5,"draw_probability":0.15,"pairing_mode":"Swiss","target_rounds":4,"rng_seed":2026,"players":[{"name":"Carlsen","rating":1750.0,"rd":75.0,"volatility":0.06},{"name":"Nakamura","rating":1690.0,"rd":85.0,"volatility":0.06},{"name":"Gukesh","rating":1580.0,"rd":150.0,"volatility":0.07},{"name":"Praggnanandhaa","rating":1540.0,"rd":175.0,"volatility":0.08}],"periods":[{"matches":[{"player_a":0,"player_b":1,"score_a":0.5},{"player_a":2,"player_b":3,"score_a":1.0}]},{"matches":[{"player_a":2,"player_b":0,"score_a":1.0},{"player_a":1,"player_b":3,"score_a":1.0}]}]}"#;

pub(crate) fn default_preset_config() -> Glicko2TournamentPresetConfig {
    serde_json::from_str(DEFAULT_CONFIG_JSON).expect("valid embedded default Glicko-2 config")
}

impl Default for Glicko2TournamentTool {
    fn default() -> Self {
        let mut tool = Self {
            tau: 0.5,
            draw_probability: 0.15,
            pairing_mode: Glicko2PairingMode::Swiss,
            target_rounds: 4,
            rng_seed: 2026,
            competitors: Vec::new(),
            completed_periods: Vec::new(),
            snapshots: Vec::new(),
            next_pairings: Vec::new(),
            next_bye_player: None,
            plot_mode: Glicko2PlotMode::ConfidenceBands,
            bottom_tab: BottomViewTab::Standings,
            focus_player: None,
            new_player_name: "Challenger".to_string(),
            state_rev: 1,
            io_state: io_dialog::TournamentIoState::default(),
        };
        let _ = tool.apply_config(default_preset_config());
        tool
    }
}

fn record_outcomes(comps: &mut [CompetitorEntry], matches: &[PeriodMatch], bye: Option<usize>) {
    if let Some(comp) = bye.and_then(|b| comps.get_mut(b)) {
        comp.points += 1.0;
        comp.byes += 1;
    }
    for m in matches {
        comps[m.player_a].points += m.score_a;
        comps[m.player_b].points += m.score_b();
        if (m.score_a - 0.5).abs() <= 0.05 {
            comps[m.player_a].draws += 1;
            comps[m.player_b].draws += 1;
        } else if m.score_a > 0.5 {
            comps[m.player_a].wins += 1;
            comps[m.player_b].losses += 1;
        } else {
            comps[m.player_a].losses += 1;
            comps[m.player_b].wins += 1;
        }
    }
}

impl Glicko2TournamentTool {
    pub(crate) fn bump_revision(&mut self) {
        self.state_rev = self.state_rev.wrapping_add(1).max(1);
    }

    pub fn rebuild_trajectory(&mut self) -> Result<(), String> {
        let tau = SystemConstant::new(self.tau).map_err(|e| e.to_string())?;
        let mut frozen: Vec<GlickoPlayer> = self.competitors.iter().map(|c| c.initial).collect();
        let mut sequential = frozen.clone();
        for c in &mut self.competitors {
            (c.points, c.wins, c.draws, c.losses, c.byes) = (0.0, 0, 0, 0, 0);
        }
        let mut snapshots = vec![PeriodSnapshot {
            period: 0,
            frozen: frozen.clone(),
            sequential: sequential.clone(),
        }];
        for (idx, period) in self.completed_periods.iter().enumerate() {
            let matches: Vec<PeriodMatch> = period
                .matches
                .iter()
                .map(|m| {
                    PeriodMatch::new(m.player_a, m.player_b, m.score_a).map_err(|e| e.to_string())
                })
                .collect::<Result<_, _>>()?;
            frozen = evaluate_rating_period(&frozen, &matches, &tau).map_err(|e| e.to_string())?;
            sequential = evaluate_sequential_period(&sequential, &matches, &tau)
                .map_err(|e| e.to_string())?;
            record_outcomes(&mut self.competitors, &matches, period.bye_player);
            snapshots.push(PeriodSnapshot {
                period: idx + 1,
                frozen: frozen.clone(),
                sequential: sequential.clone(),
            });
        }
        for (i, comp) in self.competitors.iter_mut().enumerate() {
            (comp.current, comp.sequential) = (frozen[i], sequential[i]);
        }
        self.snapshots = snapshots;
        self.refresh_next_pairings();
        self.bump_revision();
        Ok(())
    }

    pub(crate) fn refresh_next_pairings(&mut self) {
        let active: Vec<usize> = self
            .competitors
            .iter()
            .enumerate()
            .filter_map(|(i, c)| (!c.inactive).then_some(i))
            .collect();
        if active.len() < 2 {
            self.next_pairings.clear();
            self.next_bye_player = active.first().copied();
            return;
        }
        match self.pairing_mode {
            Glicko2PairingMode::Swiss => self.refresh_swiss_pairings(&active),
            Glicko2PairingMode::RoundRobin => self.refresh_round_robin_pairings(&active),
        }
    }

    fn refresh_swiss_pairings(&mut self, active: &[usize]) {
        let mut inv = vec![None; self.competitors.len()];
        for (li, &gi) in active.iter().enumerate() {
            inv[gi] = Some(li);
        }
        let players: Vec<GlickoPlayer> = active
            .iter()
            .map(|&i| self.competitors[i].current)
            .collect();
        let points: Vec<f64> = active.iter().map(|&i| self.competitors[i].points).collect();
        let mut prior = Vec::new();
        for m in self.completed_periods.iter().flat_map(|p| &p.matches) {
            if let (Some(la), Some(lb)) = (inv[m.player_a], inv[m.player_b]) {
                if let Ok(pm) = PeriodMatch::new(la, lb, m.score_a) {
                    prior.push(pm);
                }
            }
        }
        if let Ok(plan) = swiss_pairings(&players, &points, &prior) {
            self.next_pairings = plan
                .pairings
                .into_iter()
                .map(|p| ScheduledPairingRow {
                    pairing: Pairing {
                        player_a: active[p.player_a],
                        player_b: active[p.player_b],
                    },
                    manual_score_a: 0.5,
                })
                .collect();
            self.next_bye_player = plan.bye_player.map(|b| active[b]);
        }
    }

    fn refresh_round_robin_pairings(&mut self, active: &[usize]) {
        if let Ok(schedule) = round_robin_schedule(active.len()) {
            let round = &schedule[self.completed_periods.len() % schedule.len()];
            let mut matched = HashSet::with_capacity(round.len() * 2);
            self.next_pairings = round
                .iter()
                .map(|p| {
                    matched.insert(p.player_a);
                    matched.insert(p.player_b);
                    ScheduledPairingRow {
                        pairing: Pairing {
                            player_a: active[p.player_a],
                            player_b: active[p.player_b],
                        },
                        manual_score_a: 0.5,
                    }
                })
                .collect();
            self.next_bye_player = (0..active.len())
                .find(|li| !matched.contains(li))
                .map(|li| active[li]);
        }
    }

    pub fn simulate_next_period(&mut self) -> Result<(), String> {
        if self.completed_periods.len() >= MAX_TOURNAMENT_PERIODS {
            return Err(format!(
                "Maximum period cap ({MAX_TOURNAMENT_PERIODS}) reached"
            ));
        }
        let pairings: Vec<Pairing> = self.next_pairings.iter().map(|r| r.pairing).collect();
        let roster: Vec<GlickoPlayer> = self.competitors.iter().map(|c| c.current).collect();
        let seed = self.rng_seed.wrapping_add(
            (self.completed_periods.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        let mut rng = OxidizeRng::new(seed);
        let matches = simulate_period_matches(&roster, &pairings, self.draw_probability, &mut rng)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|m| match_preset(m.player_a, m.player_b, m.score_a))
            .collect();
        self.completed_periods.push(Glicko2PeriodPreset {
            matches,
            bye_player: self.next_bye_player,
        });
        self.rebuild_trajectory()
    }

    pub fn commit_manual_period(&mut self) -> Result<(), String> {
        if self.completed_periods.len() >= MAX_TOURNAMENT_PERIODS {
            return Err(format!(
                "Maximum period cap ({MAX_TOURNAMENT_PERIODS}) reached"
            ));
        }
        let matches = self
            .next_pairings
            .iter()
            .map(|r| {
                match_preset(
                    r.pairing.player_a,
                    r.pairing.player_b,
                    r.manual_score_a.clamp(0.0, 1.0),
                )
            })
            .collect();
        self.completed_periods.push(Glicko2PeriodPreset {
            matches,
            bye_player: self.next_bye_player,
        });
        self.rebuild_trajectory()
    }

    pub fn simulate_to_target_rounds(&mut self) -> Result<(), String> {
        while self.completed_periods.len() < self.target_rounds {
            self.simulate_next_period()?;
        }
        Ok(())
    }

    pub fn reset_periods(&mut self) {
        self.completed_periods.clear();
        let _ = self.rebuild_trajectory();
    }

    #[must_use]
    pub fn standings_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.competitors.len()).collect();
        order.sort_by(|&a, &b| {
            let (ca, cb) = (&self.competitors[a], &self.competitors[b]);
            cb.points
                .total_cmp(&ca.points)
                .then_with(|| {
                    cb.current
                        .rating
                        .value()
                        .total_cmp(&ca.current.rating.value())
                })
                .then_with(|| {
                    ca.current
                        .rating_deviation
                        .value()
                        .total_cmp(&cb.current.rating_deviation.value())
                })
                .then_with(|| a.cmp(&b))
        });
        order
    }

    pub fn add_competitor(&mut self) -> Result<(), String> {
        if self.competitors.len() >= MAX_TOURNAMENT_PLAYERS {
            return Err(format!("Roster capped at {MAX_TOURNAMENT_PLAYERS} players"));
        }
        let base = if self.new_player_name.trim().is_empty() {
            "Player"
        } else {
            self.new_player_name.trim()
        };
        let mut name = base.to_string();
        let mut suffix = self.competitors.len() + 1;
        while self
            .competitors
            .iter()
            .any(|c| c.name.eq_ignore_ascii_case(&name))
        {
            name = format!("{base} {suffix}");
            suffix += 1;
        }
        self.competitors
            .push(CompetitorEntry::new(name, GlickoPlayer::default(), false));
        self.rebuild_trajectory()
    }

    pub fn remove_competitor(&mut self, idx: usize) -> Result<(), String> {
        if self.competitors.len() <= 2 || idx >= self.competitors.len() {
            return Err("Tournament roster requires at least 2 competitors".to_string());
        }
        self.competitors.remove(idx);
        self.focus_player = match self.focus_player {
            Some(fp) if fp == idx => None,
            Some(fp) if fp > idx => Some(fp - 1),
            other => other,
        };
        self.completed_periods.clear();
        self.rebuild_trajectory()
    }

    pub(crate) fn apply_config(
        &mut self,
        config: Glicko2TournamentPresetConfig,
    ) -> Result<(), String> {
        config.validate()?;
        let mut competitors = Vec::with_capacity(config.players.len());
        for p in &config.players {
            let initial = GlickoPlayer::new(
                Rating::new(p.rating).map_err(|e| e.to_string())?,
                RatingDeviation::new(p.rd).map_err(|e| e.to_string())?,
                Volatility::new(p.volatility).map_err(|e| e.to_string())?,
            );
            competitors.push(CompetitorEntry::new(
                p.name.trim().to_string(),
                initial,
                p.inactive,
            ));
        }
        (self.tau, self.draw_probability, self.pairing_mode) =
            (config.tau, config.draw_probability, config.pairing_mode);
        (self.target_rounds, self.rng_seed) = (config.target_rounds, config.rng_seed);
        (self.competitors, self.completed_periods) = (competitors, config.periods);
        if self
            .focus_player
            .is_some_and(|i| i >= self.competitors.len())
        {
            self.focus_player = None;
        }
        self.rebuild_trajectory()
    }
}
