#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::model::{
    RosterPlayer, ScheduledMatch, StandingRow, MAX_CSV_IMPORT_BYTES, MAX_PLAYER_NAME_CHARS,
    MAX_TOURNAMENT_MATCHES, MAX_TOURNAMENT_PLAYERS,
};
use super::Glicko2TournamentTool;
use crate::accessibility::{announce_live_assertive_status, announce_live_status};
use crate::web_io::{
    spawn_open_file_dialog, spawn_save_as_file_dialog, strip_utf8_bom, OpenDialogConfig,
    SaveAsDialogConfig,
};
use eframe::egui;
use egui_plot::export::{copy_text_to_clipboard, escape_csv_field, split_line};
use math_explorer::pure_math::statistics::glicko2::{GlickoPlayer, MatchResult};
use std::collections::HashMap;
use std::sync::mpsc::channel;

const MATCH_CSV_OPEN_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "Match Log CSV (*.csv)",
    extensions: &["csv", "txt"],
    accept: ".csv,text/csv,text/plain",
    read_error_prefix: "Failed to read match CSV",
};

const CSV_EXPORT_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "CSV File (*.csv)",
    extensions: &["csv"],
    fallback_filename: "glicko2_standings.csv",
    write_error_prefix: "Failed to write CSV export",
};

const MD_EXPORT_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "Markdown File (*.md)",
    extensions: &["md"],
    fallback_filename: "glicko2_standings.md",
    write_error_prefix: "Failed to write Markdown export",
};

fn is_header_row(tokens: &[String]) -> bool {
    tokens.len() >= 3
        && tokens[2].trim().parse::<f64>().is_err()
        && (tokens[0].to_ascii_lowercase().contains("player")
            || tokens[2].to_ascii_lowercase().contains("score")
            || tokens[2].to_ascii_lowercase().contains("result"))
}

fn find_or_insert_player(
    name: &str,
    existing_by_name: &HashMap<String, RosterPlayer>,
    roster: &mut Vec<RosterPlayer>,
    index_by_name: &mut HashMap<String, usize>,
) -> Result<usize, String> {
    if let Some(&idx) = index_by_name.get(name) {
        return Ok(idx);
    }
    if roster.len() >= MAX_TOURNAMENT_PLAYERS {
        return Err(format!(
            "CSV exceeds maximum of {MAX_TOURNAMENT_PLAYERS} players"
        ));
    }
    let player = existing_by_name.get(name).cloned().unwrap_or_else(|| {
        let def = GlickoPlayer::default();
        RosterPlayer {
            name: name.to_string(),
            rating: def.rating.value(),
            rd: def.rating_deviation.value(),
            volatility: def.volatility.value(),
        }
    });
    let idx = roster.len();
    roster.push(player);
    index_by_name.insert(name.to_string(), idx);
    Ok(idx)
}

fn parse_csv_match_row(
    line_num: usize,
    tokens: &[String],
    dummy: &GlickoPlayer,
) -> Result<(String, String, f64), String> {
    if tokens.len() != 3 {
        return Err(format!(
            "Row {line_num}: expected 3 columns (player_a,player_b,score)"
        ));
    }
    let name_a = tokens[0].trim();
    let name_b = tokens[1].trim();
    if name_a.is_empty()
        || name_b.is_empty()
        || name_a.chars().count() > MAX_PLAYER_NAME_CHARS
        || name_b.chars().count() > MAX_PLAYER_NAME_CHARS
    {
        return Err(format!("Row {line_num}: player names must be 1..=64 chars"));
    }
    if name_a == name_b {
        return Err(format!(
            "Row {line_num}: self-match '{name_a}' vs '{name_b}' not allowed"
        ));
    }
    let score: f64 = tokens[2]
        .trim()
        .parse()
        .map_err(|_| format!("Row {line_num}: invalid numeric score '{}'", tokens[2]))?;
    let _ = MatchResult::new(*dummy, score).map_err(|e| format!("Row {line_num}: {e}"))?;
    Ok((name_a.to_string(), name_b.to_string(), score))
}

pub fn parse_match_csv(
    raw: &str,
    existing_roster: &[RosterPlayer],
) -> Result<(Vec<RosterPlayer>, Vec<ScheduledMatch>), String> {
    let cleaned = strip_utf8_bom(raw);
    if cleaned.len() > MAX_CSV_IMPORT_BYTES {
        return Err(format!(
            "CSV file ({} bytes) exceeds 64 KiB limit",
            cleaned.len()
        ));
    }

    let mut existing_by_name = HashMap::new();
    let mut roster: Vec<RosterPlayer> = Vec::new();
    let mut index_by_name: HashMap<String, usize> = HashMap::new();
    for p in existing_roster {
        let key = p.name.trim().to_string();
        existing_by_name.insert(key.clone(), p.clone());
        if roster.len() < MAX_TOURNAMENT_PLAYERS && !index_by_name.contains_key(&key) {
            index_by_name.insert(key, roster.len());
            roster.push(p.clone());
        }
    }

    let mut matches = Vec::new();
    let dummy = GlickoPlayer::default();

    for (line_idx, raw_line) in cleaned.lines().enumerate() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let tokens = split_line(trimmed, ',');
        if line_idx == 0 && is_header_row(&tokens) {
            continue;
        }
        let (name_a, name_b, score) = parse_csv_match_row(line_idx + 1, &tokens, &dummy)?;
        if matches.len() >= MAX_TOURNAMENT_MATCHES {
            return Err(format!(
                "CSV exceeds maximum of {MAX_TOURNAMENT_MATCHES} matches"
            ));
        }
        let idx_a =
            find_or_insert_player(&name_a, &existing_by_name, &mut roster, &mut index_by_name)?;
        let idx_b =
            find_or_insert_player(&name_b, &existing_by_name, &mut roster, &mut index_by_name)?;
        matches.push(ScheduledMatch {
            player_a: idx_a,
            player_b: idx_b,
            score_a: score,
        });
    }

    if matches.is_empty() || roster.len() < 2 {
        return Err("CSV match log must contain at least 1 valid match between 2 players".into());
    }
    Ok((roster, matches))
}

pub fn format_leaderboard_csv(standings: &[StandingRow]) -> String {
    let mut out = String::from(
        "Rank,Player,Pre_Rating,Pre_RD,Post_Rating,Post_RD,Volatility,CI_95_Low,CI_95_High,Matches,Points\n",
    );
    for s in standings {
        let name = escape_csv_field(&s.name);
        out.push_str(&format!(
            "{},{name},{:.2},{:.2},{:.2},{:.2},{:.5},{:.2},{:.2},{},{:.1}\n",
            s.rank,
            s.pre_rating,
            s.pre_rd,
            s.post_rating,
            s.post_rd,
            s.post_volatility,
            s.ci_low,
            s.ci_high,
            s.matches_played,
            s.score_sum
        ));
    }
    out
}

pub fn format_leaderboard_markdown(standings: &[StandingRow], tau: f64, period: usize) -> String {
    let mut out = format!(
        "### Glicko-2 Standings (Period {period}, τ = {tau:.2})\n\n| Rank | Player | Rating (r) | RD | σ | 95% Interval [r - 2RD, r + 2RD] | Matches | Score |\n| ---: | :--- | ---: | ---: | ---: | :---: | ---: | ---: |\n"
    );
    for s in standings {
        let safe_name = s.name.replace('|', "\\|").replace(['\r', '\n'], " ");
        out.push_str(&format!(
            "| {} | {safe_name} | {:.2} | {:.2} | {:.5} | [{:.2}, {:.2}] | {} | {:.1} |\n",
            s.rank,
            s.post_rating,
            s.post_rd,
            s.post_volatility,
            s.ci_low,
            s.ci_high,
            s.matches_played,
            s.score_sum
        ));
    }
    out
}

pub fn format_matchup_csv(roster: &[RosterPlayer], matrix: &[Vec<f64>]) -> String {
    let mut out = String::from("Player");
    for p in roster {
        out.push(',');
        out.push_str(&escape_csv_field(&p.name));
    }
    out.push('\n');
    for (i, p) in roster.iter().enumerate() {
        out.push_str(&escape_csv_field(&p.name));
        for j in 0..roster.len() {
            let val = matrix.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.5);
            out.push_str(&format!(",{val:.4}"));
        }
        out.push('\n');
    }
    out
}

pub fn format_matchup_markdown(roster: &[RosterPlayer], matrix: &[Vec<f64>]) -> String {
    let mut out = String::from("| Player \\ Opponent |");
    let mut sep = String::from("| :--- |");
    for p in roster {
        let clean = p.name.replace('|', "\\|").replace(['\r', '\n'], " ");
        out.push_str(&format!(" {clean} |"));
        sep.push_str(" ---: |");
    }
    out.push('\n');
    out.push_str(&sep);
    out.push('\n');
    for (i, p) in roster.iter().enumerate() {
        let clean = p.name.replace('|', "\\|").replace(['\r', '\n'], " ");
        out.push_str(&format!("| **{clean}** |"));
        for j in 0..roster.len() {
            if i == j {
                out.push_str(" — |");
            } else {
                let val = matrix.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.5);
                out.push_str(&format!(" {:.1}% |", val * 100.0));
            }
        }
        out.push('\n');
    }
    out
}

impl Glicko2TournamentTool {
    pub(super) fn trigger_csv_import_dialog(&mut self) {
        let (tx, rx) = channel();
        self.open_csv_rx = Some(rx);
        spawn_open_file_dialog(tx, MATCH_CSV_OPEN_CONFIG, |p| p.display().to_string());
    }

    pub(super) fn copy_text_artifact(&mut self, ctx: &egui::Context, text: &str, label: &str) {
        match copy_text_to_clipboard(text) {
            Ok(()) => {
                let msg = format!("Copied {label} to clipboard");
                announce_live_status(ctx, &msg);
                self.io_status = Some((msg, false));
            }
            Err(e) => {
                let err = format!("Clipboard copy failed: {e}");
                announce_live_assertive_status(ctx, &err);
                self.io_status = Some((err, true));
            }
        }
    }

    pub(super) fn save_text_artifact(&mut self, filename: &str, content: String, markdown: bool) {
        let (tx, rx) = channel();
        self.save_file_rx = Some(rx);
        let cfg = if markdown {
            MD_EXPORT_CONFIG
        } else {
            CSV_EXPORT_CONFIG
        };
        spawn_save_as_file_dialog(tx, cfg, filename, content, |p, _| {
            format!("Saved tournament artifact to {}", p.display())
        });
    }

    pub(super) fn poll_io_channels(&mut self, ctx: &egui::Context) {
        if let Some(res) = self.open_csv_rx.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.open_csv_rx = None;
            match res.and_then(|(path, content)| {
                parse_match_csv(&content, &self.roster).map(|(r, m)| (path, r, m))
            }) {
                Ok((path, roster, matches)) => {
                    let count = matches.len();
                    self.roster = roster;
                    self.matches = matches;
                    self.recalculate();
                    let msg = format!("Imported {count} matches from {path}");
                    announce_live_status(ctx, &msg);
                    self.io_status = Some((msg, false));
                }
                Err(err) => {
                    announce_live_assertive_status(ctx, &err);
                    self.io_status = Some((err, true));
                }
            }
        }
        if let Some(res) = self.save_file_rx.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.save_file_rx = None;
            match res {
                Ok(msg) => {
                    announce_live_status(ctx, &msg);
                    self.io_status = Some((msg, false));
                }
                Err(err) => {
                    announce_live_assertive_status(ctx, &err);
                    self.io_status = Some((err, true));
                }
            }
        }
    }
}
