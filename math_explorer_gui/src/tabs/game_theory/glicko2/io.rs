//! Non-blocking local CSV/JSON match-log import and standings/pairings export for [`super::Glicko2TournamentTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::parse::{csv_escape_field, parse_csv_periods, parse_json_match_array};
use super::Glicko2TournamentTool;
use crate::framework::InteractiveTool;
use crate::presets::{Glicko2PlayerPreset, Glicko2TournamentPresetConfig, MAX_TOURNAMENT_PERIODS};
use crate::web_io::{
    spawn_open_file_dialog, spawn_save_as_file_dialog, strip_utf8_bom, OpenDialogConfig,
    SaveAsDialogConfig,
};
use eframe::egui;
use math_explorer::pure_math::statistics::glicko2::head_to_head_win_probability;
use std::sync::mpsc::{channel, Receiver};

const MATCH_LOG_OPEN_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "Match Log or Preset (.csv, .json)",
    extensions: &["csv", "json"],
    accept: ".csv,.json,text/csv,application/json",
    read_error_prefix: "Failed to read tournament file",
};

const CSV_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "CSV Spreadsheet (.csv)",
    extensions: &["csv"],
    fallback_filename: "glicko2_export.csv",
    write_error_prefix: "Failed to write CSV export",
};

const JSON_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "JSON Document (.json)",
    extensions: &["json"],
    fallback_filename: "glicko2_export.json",
    write_error_prefix: "Failed to write JSON export",
};

#[derive(Default)]
pub(crate) struct TournamentIoState {
    open_rx: Option<Receiver<Result<(String, String), String>>>,
    save_rx: Option<Receiver<Result<String, String>>>,
    pub status_message: Option<(bool, String)>,
}

impl Glicko2TournamentTool {
    pub(crate) fn export_config(&self) -> Glicko2TournamentPresetConfig {
        let players = self
            .competitors
            .iter()
            .map(|c| Glicko2PlayerPreset {
                name: c.name.clone(),
                rating: c.initial.rating.value(),
                rd: c.initial.rating_deviation.value(),
                volatility: c.initial.volatility.value(),
                inactive: c.inactive,
            })
            .collect();
        Glicko2TournamentPresetConfig {
            tau: self.tau,
            draw_probability: self.draw_probability,
            pairing_mode: self.pairing_mode,
            target_rounds: self.target_rounds,
            rng_seed: self.rng_seed,
            players,
            periods: self.completed_periods.clone(),
        }
    }

    #[must_use]
    pub fn estimated_url_state_bytes(&self) -> usize {
        serde_json::to_value(self.export_config())
            .ok()
            .and_then(|v| {
                crate::presets::serialize_compact_preset_envelope(self.name(), self.domain(), v)
                    .ok()
            })
            .map_or(0, |s| s.len())
    }

    #[must_use]
    pub fn url_share_warning(&self) -> Option<String> {
        let (bytes, cap) = (
            self.estimated_url_state_bytes(),
            crate::url_state::MAX_DECODED_URL_STATE_BYTES,
        );
        (bytes > cap).then(|| {
            format!(
                "URL share payload ({bytes} B) exceeds the {cap} B URL cap; Copy Share Link will be rejected. Local CSV/JSON and Preset JSON exports remain available."
            )
        })
    }

    /// Atomically imports a UTF-8 (or BOM-prefixed) CSV/JSON match log or JSON tournament preset.
    pub fn import_match_log_str(&mut self, raw: &str) -> Result<usize, String> {
        let cleaned = strip_utf8_bom(raw).trim();
        if cleaned.is_empty() {
            return Err("Imported tournament file is empty".to_string());
        }
        if cleaned.starts_with('{') {
            if cleaned.contains("\"metadata\"") && cleaned.contains("\"config\"") {
                self.import_preset_json(cleaned)?;
            } else {
                let cfg: Glicko2TournamentPresetConfig = serde_json::from_str(cleaned)
                    .map_err(|e| format!("Invalid tournament JSON config: {e}"))?;
                self.apply_config(cfg)?;
            }
            return Ok(self.completed_periods.iter().map(|p| p.matches.len()).sum());
        }

        let names: Vec<String> = self.competitors.iter().map(|c| c.name.clone()).collect();
        let base_period = self.completed_periods.len();
        let (new_periods, count) = if cleaned.starts_with('[') {
            parse_json_match_array(cleaned, &names, base_period)?
        } else {
            parse_csv_periods(cleaned, &names, base_period)?
        };

        let mut candidate = self.export_config();
        if candidate.periods.len() + new_periods.len() > MAX_TOURNAMENT_PERIODS {
            return Err(format!(
                "Import exceeds period cap ({MAX_TOURNAMENT_PERIODS})"
            ));
        }
        candidate.periods.extend(new_periods);
        candidate.validate()?;
        self.apply_config(candidate)?;
        Ok(count)
    }

    /// Formats ranked standings, 95% confidence bounds, volatility, and sequential drift as CSV.
    #[must_use]
    pub fn format_standings_csv(&self) -> String {
        let mut out = String::from(
            "rank,name,rating,rd,ci95_low,ci95_high,volatility,points,wins,draws,losses,byes,inactive,sequential_rating,sequential_drift\n",
        );
        for (rank_zero, idx) in self.standings_order().into_iter().enumerate() {
            let c = &self.competitors[idx];
            let (r, rd, seq_r) = (
                c.current.rating.value(),
                c.current.rating_deviation.value(),
                c.sequential.rating.value(),
            );
            out.push_str(&format!(
                "{},{},{r:.2},{rd:.2},{:.2},{:.2},{:.5},{:.1},{},{},{},{},{},{seq_r:.2},{:+.2}\n",
                rank_zero + 1,
                csv_escape_field(&c.name),
                r - 2.0 * rd,
                r + 2.0 * rd,
                c.current.volatility.value(),
                c.points,
                c.wins,
                c.draws,
                c.losses,
                c.byes,
                c.inactive,
                seq_r - r
            ));
        }
        out
    }

    /// Formats ranked standings as pretty-printed JSON.
    pub fn format_standings_json(&self) -> Result<String, String> {
        let rows: Vec<serde_json::Value> = self
            .standings_order()
            .into_iter()
            .enumerate()
            .map(|(rank_zero, idx)| {
                let c = &self.competitors[idx];
                let (r, rd, seq_r) = (
                    c.current.rating.value(),
                    c.current.rating_deviation.value(),
                    c.sequential.rating.value(),
                );
                serde_json::json!({
                    "rank": rank_zero + 1,
                    "name": c.name,
                    "rating": r,
                    "rd": rd,
                    "ci95_low": r - 2.0 * rd,
                    "ci95_high": r + 2.0 * rd,
                    "volatility": c.current.volatility.value(),
                    "points": c.points,
                    "wins": c.wins,
                    "draws": c.draws,
                    "losses": c.losses,
                    "byes": c.byes,
                    "inactive": c.inactive,
                    "sequential_rating": seq_r,
                    "sequential_drift": seq_r - r,
                })
            })
            .collect();
        serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())
    }

    /// Formats upcoming round pairings, win probabilities, and bye assignment as CSV.
    #[must_use]
    pub fn format_pairings_csv(&self) -> String {
        let (next_period, mode) = (self.completed_periods.len() + 1, self.pairing_mode.label());
        let mut out = String::from(
            "next_period,pairing_mode,board,player_a,rating_a,player_b,rating_b,expected_score_a,manual_score_a\n",
        );
        for (b_idx, row) in self.next_pairings.iter().enumerate() {
            let (ca, cb) = (
                &self.competitors[row.pairing.player_a],
                &self.competitors[row.pairing.player_b],
            );
            let win_prob = head_to_head_win_probability(&ca.current, &cb.current);
            out.push_str(&format!(
                "{next_period},{mode},{},{},{:.2},{},{:.2},{win_prob:.4},{:.1}\n",
                b_idx + 1,
                csv_escape_field(&ca.name),
                ca.current.rating.value(),
                csv_escape_field(&cb.name),
                cb.current.rating.value(),
                row.manual_score_a
            ));
        }
        if let Some(bye_idx) = self.next_bye_player {
            let cb = &self.competitors[bye_idx];
            out.push_str(&format!(
                "{next_period},{mode},0,{},{:.2},BYE,0.00,1.0000,1.0\n",
                csv_escape_field(&cb.name),
                cb.current.rating.value()
            ));
        }
        out
    }

    /// Formats upcoming round pairings as pretty-printed JSON.
    pub fn format_pairings_json(&self) -> Result<String, String> {
        let (next_period, mode) = (self.completed_periods.len() + 1, self.pairing_mode.label());
        let rows: Vec<serde_json::Value> = self
            .next_pairings
            .iter()
            .enumerate()
            .map(|(b_idx, row)| {
                let (ca, cb) = (
                    &self.competitors[row.pairing.player_a],
                    &self.competitors[row.pairing.player_b],
                );
                serde_json::json!({
                    "next_period": next_period,
                    "pairing_mode": mode,
                    "board": b_idx + 1,
                    "player_a": ca.name,
                    "rating_a": ca.current.rating.value(),
                    "player_b": cb.name,
                    "rating_b": cb.current.rating.value(),
                    "expected_score_a": head_to_head_win_probability(&ca.current, &cb.current),
                    "manual_score_a": row.manual_score_a,
                })
            })
            .collect();
        serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())
    }
    pub(crate) fn poll_io(&mut self, ctx: &egui::Context) {
        if let Some(Ok(outcome)) = self.io_state.open_rx.as_ref().map(Receiver::try_recv) {
            self.io_state.open_rx = None;
            let status = match outcome {
                Ok((name, raw)) => match self.import_match_log_str(&raw) {
                    Ok(count) => (false, format!("Imported {count} match(es) from {name}")),
                    Err(err) => (true, format!("Import rejected ({name}): {err}")),
                },
                Err(err) => (true, err),
            };
            self.io_state.status_message = Some(status);
            ctx.request_repaint();
        }
        if let Some(Ok(outcome)) = self.io_state.save_rx.as_ref().map(Receiver::try_recv) {
            self.io_state.save_rx = None;
            self.io_state.status_message = Some(match outcome {
                Ok(saved) => (false, format!("Exported tournament file: {saved}")),
                Err(err) => (true, err),
            });
            ctx.request_repaint();
        }
    }
}

impl TournamentIoState {
    pub(crate) fn trigger_import_dialog(&mut self) {
        let (tx, rx) = channel();
        self.open_rx = Some(rx);
        spawn_open_file_dialog(tx, MATCH_LOG_OPEN_CONFIG, |p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("match_log")
                .to_string()
        });
    }

    pub(crate) fn trigger_save_csv(&mut self, default_filename: &str, content: String) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        spawn_save_as_file_dialog(tx, CSV_SAVE_CONFIG, default_filename, content, |p, fb| {
            p.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(fb)
                .to_string()
        });
    }

    pub(crate) fn trigger_save_json(
        &mut self,
        default_filename: &str,
        content: Result<String, String>,
    ) {
        match content {
            Ok(payload) => {
                let (tx, rx) = channel();
                self.save_rx = Some(rx);
                spawn_save_as_file_dialog(
                    tx,
                    JSON_SAVE_CONFIG,
                    default_filename,
                    payload,
                    |p, fb| {
                        p.file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or(fb)
                            .to_string()
                    },
                );
            }
            Err(err) => self.status_message = Some((true, err)),
        }
    }
}
