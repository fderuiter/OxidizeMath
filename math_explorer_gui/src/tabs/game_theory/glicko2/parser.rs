//! CSV and JSON match-log parsers for [`super::Glicko2TournamentTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::presets::{Glicko2MatchPreset, Glicko2PeriodPreset};
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

pub(super) fn parse_csv_periods(
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

pub(super) fn parse_json_match_array(
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

pub(super) fn csv_escape_field(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}
