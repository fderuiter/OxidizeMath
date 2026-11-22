//! Preset configuration models and validation routines for `Glicko2TournamentTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Pairing system used to schedule tournament rounds in `Glicko2TournamentTool`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glicko2PairingMode {
    /// Standings-ordered Swiss pairings with deterministic odd-roster byes.
    Swiss,
    /// Circle-method single round-robin schedule.
    RoundRobin,
}

/// Competitor entry in a serialized Glicko-2 tournament preset.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Glicko2PlayerPreset {
    /// Competitor display name.
    pub name: String,
    /// Initial rating on the Glicko scale.
    pub rating: f64,
    /// Initial rating deviation (`> 0`).
    pub rd: f64,
    /// Initial rating volatility (`> 0`).
    pub volatility: f64,
    /// Whether this competitor sits out newly simulated periods.
    #[serde(default)]
    pub inactive: bool,
}

/// Match record within a serialized Glicko-2 rating period.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Glicko2MatchPreset {
    /// First competitor index in `players`.
    pub player_a: usize,
    /// Second competitor index in `players`.
    pub player_b: usize,
    /// Outcome score for `player_a` in `[0.0, 1.0]`.
    pub score_a: f64,
}

/// Completed rating period in a serialized Glicko-2 tournament preset.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Glicko2PeriodPreset {
    /// Matches evaluated simultaneously against the pre-period roster.
    pub matches: Vec<Glicko2MatchPreset>,
    /// Optional competitor index awarded a bye during this period.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bye_player: Option<usize>,
}

/// Configuration struct for the Glicko-2 Tournament Simulator tool.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Glicko2TournamentPresetConfig {
    /// System constant tau constraining volatility updates (`[0.3, 1.2]`).
    pub tau: f64,
    /// Draw rate between evenly matched competitors (`[0.0, 1.0]`).
    pub draw_probability: f64,
    /// Active tournament pairing mode (`Swiss` or `RoundRobin`).
    pub pairing_mode: Glicko2PairingMode,
    /// Target number of rounds when simulating a full tournament (`1..=32`).
    pub target_rounds: usize,
    /// Deterministic seed for `OxidizeRng`.
    pub rng_seed: u64,
    /// Initial competitor roster.
    pub players: Vec<Glicko2PlayerPreset>,
    /// Chronological sequence of completed rating periods.
    pub periods: Vec<Glicko2PeriodPreset>,
}

impl Glicko2PairingMode {
    /// Human-readable pairing label for UI selectors and CSV/JSON exports.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Swiss => "Swiss",
            Self::RoundRobin => "Round-Robin",
        }
    }
}

/// Maximum roster size permitted in interactive tournament presets.
pub const MAX_TOURNAMENT_PLAYERS: usize = 64;

/// Maximum number of completed or target rating periods permitted in presets.
pub const MAX_TOURNAMENT_PERIODS: usize = 64;

fn validate_players(players: &[Glicko2PlayerPreset]) -> Result<(), String> {
    if players.len() < 2 || players.len() > MAX_TOURNAMENT_PLAYERS {
        return Err(format!(
            "Tournament roster must contain between 2 and {MAX_TOURNAMENT_PLAYERS} players (got {})",
            players.len()
        ));
    }
    let mut seen_names = HashSet::with_capacity(players.len());
    for (idx, p) in players.iter().enumerate() {
        let trimmed = p.name.trim();
        if trimmed.is_empty() {
            return Err(format!("Player at index {idx} has an empty name"));
        }
        if !seen_names.insert(trimmed.to_ascii_lowercase()) {
            return Err(format!("Duplicate player name '{}' in roster", trimmed));
        }
        if !p.rating.is_finite() || !(100.0..=4000.0).contains(&p.rating) {
            return Err(format!(
                "Player '{}' rating must be finite in [100, 4000] (got {})",
                trimmed, p.rating
            ));
        }
        if !p.rd.is_finite() || !(1.0..=600.0).contains(&p.rd) {
            return Err(format!(
                "Player '{}' RD must be finite in [1, 600] (got {})",
                trimmed, p.rd
            ));
        }
        if !p.volatility.is_finite() || !(0.001..=1.5).contains(&p.volatility) {
            return Err(format!(
                "Player '{}' volatility must be finite in [0.001, 1.5] (got {})",
                trimmed, p.volatility
            ));
        }
    }
    Ok(())
}

fn validate_periods(periods: &[Glicko2PeriodPreset], num_players: usize) -> Result<(), String> {
    if periods.len() > MAX_TOURNAMENT_PERIODS {
        return Err(format!(
            "Completed periods ({}) exceed maximum allowed ({MAX_TOURNAMENT_PERIODS})",
            periods.len()
        ));
    }
    for (p_idx, period) in periods.iter().enumerate() {
        if let Some(bye) = period.bye_player {
            if bye >= num_players {
                return Err(format!(
                    "Period {} bye_player index {bye} is out of bounds for {num_players} players",
                    p_idx + 1
                ));
            }
        }
        for (m_idx, m) in period.matches.iter().enumerate() {
            if m.player_a >= num_players || m.player_b >= num_players {
                return Err(format!(
                    "Period {} match {} player index out of bounds ({} vs {} for {num_players} players)",
                    p_idx + 1,
                    m_idx + 1,
                    m.player_a,
                    m.player_b
                ));
            }
            if m.player_a == m.player_b {
                return Err(format!(
                    "Period {} match {} contains invalid self-match for player index {}",
                    p_idx + 1,
                    m_idx + 1,
                    m.player_a
                ));
            }
            if period
                .bye_player
                .is_some_and(|b| b == m.player_a || b == m.player_b)
            {
                return Err(format!(
                    "Period {} bye_player cannot also participate in match {}",
                    p_idx + 1,
                    m_idx + 1
                ));
            }
            if !m.score_a.is_finite() || !(0.0..=1.0).contains(&m.score_a) {
                return Err(format!(
                    "Period {} match {} score_a must be finite in [0.0, 1.0] (got {})",
                    p_idx + 1,
                    m_idx + 1,
                    m.score_a
                ));
            }
        }
    }
    Ok(())
}

impl Glicko2TournamentPresetConfig {
    /// Validates tournament constants, competitor roster bounds, and period match invariants.
    pub fn validate(&self) -> Result<(), String> {
        if !self.tau.is_finite() || !(0.3..=1.2).contains(&self.tau) {
            return Err(format!(
                "System constant tau must be finite in [0.3, 1.2] (got {})",
                self.tau
            ));
        }
        if !self.draw_probability.is_finite() || !(0.0..=1.0).contains(&self.draw_probability) {
            return Err(format!(
                "Draw probability must be finite in [0.0, 1.0] (got {})",
                self.draw_probability
            ));
        }
        if !(1..=MAX_TOURNAMENT_PERIODS).contains(&self.target_rounds) {
            return Err(format!(
                "Target rounds must be in 1..={MAX_TOURNAMENT_PERIODS} (got {})",
                self.target_rounds
            ));
        }
        validate_players(&self.players)?;
        validate_periods(&self.periods, self.players.len())
    }
}
