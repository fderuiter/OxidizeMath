//! Non-blocking local CSV/JSON match-log import and standings/pairings export for [`super::Glicko2TournamentTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub(crate) use super::state::TournamentIoState;
use super::Glicko2TournamentTool;
use crate::framework::InteractiveTool;
use crate::presets::{
    Glicko2MatchPreset, Glicko2PeriodPreset, Glicko2PlayerPreset, Glicko2TournamentPresetConfig,
    MAX_TOURNAMENT_PERIODS,
};
use crate::web_io::strip_utf8_bom;
use math_explorer::pure_math::statistics::glicko2::head_to_head_win_probability;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(untagged)]
enum PlayerSelector {
    Index(usize),
    Name(String),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonMatchRecord {
    #[serde(default)]
    period: Option<usize>,
    player_a: PlayerSelector,
    player_b: PlayerSelector,
    score_a: f64,
}

fn resolve_player_token(token: &str, names: &[String]) -> Result<usize, String> {
    let trimmed = token.trim().trim_matches('"');
    if trimmed.is_empty() {
        return Err("Empty competitor identifier in match log".to_string());
    }
    if let Some(idx) = names.iter().position(|n| n.eq_ignore_ascii_case(trimmed)) {
        return Ok(idx);
    }
    if let Ok(idx) = trimmed.parse::<usize>() {
        return (idx < names.len()).then_some(idx).ok_or_else(|| {
            format!(
                "Player index {idx} out of bounds for {} players",
                names.len()
            )
        });
    }
    Err(format!("Unknown competitor '{trimmed}' in match log"))
}

fn resolve_selector(sel: &PlayerSelector, names: &[String]) -> Result<usize, String> {
    match sel {
        PlayerSelector::Index(idx) => (*idx < names.len()).then_some(*idx).ok_or_else(|| {
            format!(
                "Player index {idx} out of bounds for {} players",
                names.len()
            )
        }),
        PlayerSelector::Name(name) => resolve_player_token(name, names),
    }
}

fn validate_match_entry(
    player_a: usize,
    player_b: usize,
    score_a: f64,
    row_num: usize,
) -> Result<Glicko2MatchPreset, String> {
    if player_a == player_b {
        return Err(format!(
            "Row {row_num}: invalid self-match for player index {player_a}"
        ));
    }
    if !score_a.is_finite() || !(0.0..=1.0).contains(&score_a) {
        return Err(format!(
            "Row {row_num}: score_a must be finite in [0.0, 1.0] (got {score_a})"
        ));
    }
    Ok(Glicko2MatchPreset {
        player_a,
        player_b,
        score_a,
    })
}

fn is_csv_header_line(cols: &[&str]) -> bool {
    let first = cols
        .first()
        .map(|s| s.trim().trim_matches('"'))
        .unwrap_or("");
    matches!(
        first.to_ascii_lowercase().as_str(),
        "period" | "round" | "player_a" | "white"
    )
}

fn parse_csv_row<'a>(
    cols: &[&'a str],
    line_num: usize,
    base_period: usize,
) -> Result<(usize, &'a str, &'a str, &'a str), String> {
    match cols {
        [pa, pb, sc] => Ok((base_period + 1, *pa, *pb, *sc)),
        [per, pa, pb, sc, ..] => {
            let p = per
                .trim()
                .parse::<usize>()
                .map_err(|_| format!("Line {line_num}: invalid period '{}'", per.trim()))?;
            if p == 0 {
                return Err(format!("Line {line_num}: period number must be >= 1"));
            }
            Ok((p, *pa, *pb, *sc))
        }
        _ => Err(format!(
            "Line {line_num}: expected 3 or 4 CSV columns ([period,]player_a,player_b,score_a)"
        )),
    }
}

fn parse_csv_periods(
    cleaned: &str,
    names: &[String],
    base_period: usize,
) -> Result<(Vec<Glicko2PeriodPreset>, usize), String> {
    let mut grouped: BTreeMap<usize, Vec<Glicko2MatchPreset>> = BTreeMap::new();
    let mut match_count = 0usize;
    for (line_idx, raw_line) in cleaned.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if line_idx == 0 && is_csv_header_line(&cols) {
            continue;
        }
        let line_num = line_idx + 1;
        let (period_key, pa_raw, pb_raw, score_raw) = parse_csv_row(&cols, line_num, base_period)?;
        let pa =
            resolve_player_token(pa_raw, names).map_err(|e| format!("Line {line_num}: {e}"))?;
        let pb =
            resolve_player_token(pb_raw, names).map_err(|e| format!("Line {line_num}: {e}"))?;
        let score_a = score_raw
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("Line {line_num}: malformed score '{}'", score_raw.trim()))?;
        grouped
            .entry(period_key)
            .or_default()
            .push(validate_match_entry(pa, pb, score_a, line_num)?);
        match_count += 1;
    }
    if match_count == 0 {
        return Err("CSV match log contained no match rows".to_string());
    }
    let periods = grouped
        .into_values()
        .map(|matches| Glicko2PeriodPreset {
            matches,
            bye_player: None,
        })
        .collect();
    Ok((periods, match_count))
}

fn parse_json_match_array(
    cleaned: &str,
    names: &[String],
    base_period: usize,
) -> Result<(Vec<Glicko2PeriodPreset>, usize), String> {
    let records: Vec<JsonMatchRecord> =
        serde_json::from_str(cleaned).map_err(|e| format!("Invalid JSON match log array: {e}"))?;
    if records.is_empty() {
        return Err("JSON match log array is empty".to_string());
    }
    let mut grouped: BTreeMap<usize, Vec<Glicko2MatchPreset>> = BTreeMap::new();
    for (idx, rec) in records.iter().enumerate() {
        let period_key = rec.period.unwrap_or(base_period + 1);
        if period_key == 0 {
            return Err(format!("Entry {}: period must be >= 1", idx + 1));
        }
        let pa = resolve_selector(&rec.player_a, names)
            .map_err(|e| format!("Entry {}: {e}", idx + 1))?;
        let pb = resolve_selector(&rec.player_b, names)
            .map_err(|e| format!("Entry {}: {e}", idx + 1))?;
        grouped
            .entry(period_key)
            .or_default()
            .push(validate_match_entry(pa, pb, rec.score_a, idx + 1)?);
    }
    let periods = grouped
        .into_values()
        .map(|matches| Glicko2PeriodPreset {
            matches,
            bye_player: None,
        })
        .collect();
    Ok((periods, records.len()))
}

fn csv_escape_field(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
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
}
