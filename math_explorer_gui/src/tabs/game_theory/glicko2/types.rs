//! Shared types and initial preset helpers for [`super::Glicko2TournamentTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::presets::{Glicko2MatchPreset, Glicko2TournamentPresetConfig};
use math_explorer::pure_math::statistics::glicko2::{GlickoPlayer, Pairing, PeriodMatch};

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
    pub(super) fn new(name: String, initial: GlickoPlayer, inactive: bool) -> Self {
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

pub(super) fn record_outcomes(
    comps: &mut [CompetitorEntry],
    matches: &[PeriodMatch],
    bye: Option<usize>,
) {
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
