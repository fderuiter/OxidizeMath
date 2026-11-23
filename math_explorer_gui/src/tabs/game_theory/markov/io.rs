//! Atomic UTF-8/BOM CSV & JSON import, URL share budget guards, and takeaway artifact exports
//! for [`super::MarkovSequenceWorkbenchTool`].
// [cite:clinical_trials_statistics]

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::dmatrix_to_rows;
use super::MarkovSequenceWorkbenchTool;
use crate::framework::{InteractiveTool, ToolStateBounds};
use crate::presets::{
    deserialize_preset_envelope_bounded, DtmcStateKind, MarkovWorkbenchMode,
    MarkovWorkbenchPresetConfig, MAX_OBSERVATION_SEQUENCE_LEN,
};
use crate::web_io::strip_utf8_bom;
use math_explorer::pure_math::statistics::markov::{
    ContinuousMarkovChain, HiddenMarkovModel, MarkovChain, StateType,
};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(untagged)]
enum JsonObsToken {
    Index(usize),
    Label(String),
}

fn resolve_symbol_token(token: &str, symbol_names: &[String]) -> Result<usize, String> {
    let trimmed = token.trim().trim_matches('"');
    if trimmed.is_empty() {
        return Err("Empty observation token".to_string());
    }
    if let Some(idx) = symbol_names
        .iter()
        .position(|s| s.eq_ignore_ascii_case(trimmed))
    {
        return Ok(idx);
    }
    let idx = trimmed
        .parse::<usize>()
        .map_err(|_| format!("Unknown observation symbol '{trimmed}'"))?;
    if idx >= symbol_names.len() {
        return Err(format!(
            "Observation index {idx} out of bounds for {} symbols",
            symbol_names.len()
        ));
    }
    Ok(idx)
}

fn is_obs_csv_header(cols: &[&str], symbol_names: &[String]) -> bool {
    let first = cols.first().map(|s| s.trim().trim_matches('"')).unwrap_or("");
    if resolve_symbol_token(first, symbol_names).is_ok() {
        return false;
    }
    matches!(
        first.to_ascii_lowercase().as_str(),
        "step" | "t" | "time" | "index" | "observation" | "obs" | "symbol"
    )
}

fn parse_hmm_observations_csv(raw: &str, symbol_names: &[String]) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for (line_idx, raw_line) in raw.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if line_idx == 0 && is_obs_csv_header(&cols, symbol_names) {
            continue;
        }
        let line_num = line_idx + 1;
        if cols.len() == 2
            && cols[0].trim().parse::<usize>().is_ok()
            && line.contains(',')
            && raw.lines().count() > 1
        {
            let sym = resolve_symbol_token(cols[1], symbol_names)
                .map_err(|e| format!("Line {line_num}: {e}"))?;
            out.push(sym);
        } else {
            for col in cols {
                let sym = resolve_symbol_token(col, symbol_names)
                    .map_err(|e| format!("Line {line_num}: {e}"))?;
                out.push(sym);
            }
        }
        if out.len() > MAX_OBSERVATION_SEQUENCE_LEN {
            return Err(format!(
                "Observation count exceeds cap ({MAX_OBSERVATION_SEQUENCE_LEN})"
            ));
        }
    }
    if out.is_empty() {
        return Err("CSV contained no observations".to_string());
    }
    Ok(out)
}

fn parse_numeric_matrix_csv(raw: &str) -> Result<Vec<Vec<f64>>, String> {
    let mut rows = Vec::new();
    for (line_idx, raw_line) in raw.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if line_idx == 0 && cols[0].trim().parse::<f64>().is_err() {
            continue;
        }
        let mut row = Vec::with_capacity(cols.len());
        for col in cols {
            let val = col.trim().parse::<f64>().map_err(|_| {
                format!(
                    "Line {}: invalid numeric matrix entry '{}'",
                    line_idx + 1,
                    col.trim()
                )
            })?;
            row.push(val);
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("CSV matrix is empty".to_string());
    }
    Ok(rows)
}

fn default_labels(prefix: &str, count: usize) -> Vec<String> {
    (0..count).map(|i| format!("{prefix} {i}")).collect()
}

impl MarkovSequenceWorkbenchTool {
    pub(crate) fn export_config(&self) -> MarkovWorkbenchPresetConfig {
        self.config.clone()
    }

    /// Estimates the compact JSON byte size used when encoding a URL share link.
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

    /// Returns a proactive warning banner when the active state exceeds the `4,096`-byte URL cap.
    #[must_use]
    pub fn url_share_warning(&self) -> Option<String> {
        let bytes = self.estimated_url_state_bytes();
        let cap = crate::url_state::MAX_DECODED_URL_STATE_BYTES;
        (bytes > cap).then(|| {
            format!(
                "URL share payload ({bytes} B) exceeds the {cap} B URL cap; Copy Share Link will be rejected. Local CSV/JSON takeaway exports remain available."
            )
        })
    }

    fn apply_imported_hmm_model(&mut self, hmm: HiddenMarkovModel<f64>) -> Result<String, String> {
        let mut candidate = self.export_config();
        let (n, m) = (hmm.num_states(), hmm.num_observations());
        candidate.mode = MarkovWorkbenchMode::Hmm;
        if candidate.hmm.state_names.len() != n {
            candidate.hmm.state_names = default_labels("State", n);
        }
        if candidate.hmm.symbol_names.len() != m {
            candidate.hmm.symbol_names = default_labels("Symbol", m);
        }
        candidate.hmm.initial = hmm.initial().iter().copied().collect();
        candidate.hmm.transitions = dmatrix_to_rows(hmm.transitions());
        candidate.hmm.emissions = dmatrix_to_rows(hmm.emissions());
        if candidate.hmm.observations.iter().any(|&o| o >= m) {
            candidate.hmm.observations = (0..m.clamp(1, 8)).collect();
        }
        self.apply_config(candidate)?;
        Ok(format!("{n}-state, {m}-symbol HMM model"))
    }

    fn apply_imported_ctmc_model(
        &mut self,
        ctmc: ContinuousMarkovChain<f64>,
    ) -> Result<String, String> {
        let mut candidate = self.export_config();
        let n = ctmc.num_states();
        candidate.mode = MarkovWorkbenchMode::Ctmc;
        if candidate.ctmc.state_names.len() != n {
            candidate.ctmc.state_names = default_labels("State", n);
        }
        candidate.ctmc.generator = dmatrix_to_rows(ctmc.generator());
        candidate.ctmc.source_state = candidate.ctmc.source_state.min(n - 1);
        candidate.ctmc.transient_states = (0..n)
            .filter(|&i| candidate.ctmc.generator[i].iter().any(|&v| v.abs() > 1e-9))
            .collect();
        if candidate.ctmc.transient_states.len() == n {
            candidate.ctmc.transient_states.clear();
        }
        self.apply_config(candidate)?;
        Ok(format!("{n}-state CTMC generator"))
    }

    fn apply_imported_dtmc_model(&mut self, dtmc: MarkovChain<f64>) -> Result<String, String> {
        let mut candidate = self.export_config();
        let n = dtmc.transition_matrix().nrows();
        candidate.mode = MarkovWorkbenchMode::AbsorbingDtmc;
        if candidate.dtmc.state_names.len() != n {
            candidate.dtmc.state_names = default_labels("State", n);
        }
        candidate.dtmc.transition_matrix = dmatrix_to_rows(dtmc.transition_matrix());
        candidate.dtmc.state_types = dtmc
            .state_types()
            .iter()
            .map(|t| match t {
                StateType::Transient => DtmcStateKind::Transient,
                StateType::Absorbing => DtmcStateKind::Absorbing,
            })
            .collect();
        let num_abs = dtmc.num_absorbing();
        if candidate.dtmc.absorbing_rewards.len() != num_abs {
            candidate.dtmc.absorbing_rewards =
                (0..num_abs).map(|k| if k == 0 { 1.0 } else { 0.0 }).collect();
        }
        self.apply_config(candidate)?;
        Ok(format!("{n}-state Absorbing DTMC"))
    }

    fn import_json_object(&mut self, cleaned: &str) -> Result<String, String> {
        ToolStateBounds::DEFAULT
            .for_preset_envelope()
            .validate_json(cleaned)?;
        if cleaned.contains("\"metadata\"") && cleaned.contains("\"config\"") {
            let cfg: MarkovWorkbenchPresetConfig = deserialize_preset_envelope_bounded(
                cleaned,
                self.domain(),
                self.name(),
                ToolStateBounds::DEFAULT,
            )?;
            self.apply_config(cfg)?;
            return Ok("preset envelope".to_string());
        }
        if cleaned.contains("\"hmm\"") && cleaned.contains("\"ctmc\"") {
            let cfg: MarkovWorkbenchPresetConfig =
                serde_json::from_str(cleaned).map_err(|e| format!("Invalid workbench JSON: {e}"))?;
            self.apply_config(cfg)?;
            return Ok("workbench configuration".to_string());
        }
        if cleaned.contains("\"emissions\"") {
            let hmm: HiddenMarkovModel<f64> =
                serde_json::from_str(cleaned).map_err(|e| format!("Invalid HMM JSON: {e}"))?;
            return self.apply_imported_hmm_model(hmm);
        }
        if cleaned.contains("\"generator\"") {
            let ctmc: ContinuousMarkovChain<f64> =
                serde_json::from_str(cleaned).map_err(|e| format!("Invalid CTMC JSON: {e}"))?;
            return self.apply_imported_ctmc_model(ctmc);
        }
        if cleaned.contains("\"transition_matrix\"") {
            let dtmc: MarkovChain<f64> =
                serde_json::from_str(cleaned).map_err(|e| format!("Invalid DTMC JSON: {e}"))?;
            return self.apply_imported_dtmc_model(dtmc);
        }
        Err("Unrecognized JSON object schema for Markov workbench".to_string())
    }

    fn import_json_observations(&mut self, cleaned: &str) -> Result<String, String> {
        ToolStateBounds::DEFAULT.validate_json(cleaned)?;
        let tokens: Vec<JsonObsToken> = serde_json::from_str(cleaned)
            .map_err(|e| format!("Invalid JSON observation array: {e}"))?;
        if tokens.is_empty() || tokens.len() > MAX_OBSERVATION_SEQUENCE_LEN {
            return Err(format!(
                "JSON observation count ({}) must be in [1, {MAX_OBSERVATION_SEQUENCE_LEN}]",
                tokens.len()
            ));
        }
        let mut obs = Vec::with_capacity(tokens.len());
        for (idx, tok) in tokens.iter().enumerate() {
            let sym = match tok {
                JsonObsToken::Index(v) => (*v < self.config.hmm.symbol_names.len())
                    .then_some(*v)
                    .ok_or_else(|| format!("Entry {}: symbol {v} out of bounds", idx + 1))?,
                JsonObsToken::Label(s) => resolve_symbol_token(s, &self.config.hmm.symbol_names)
                    .map_err(|e| format!("Entry {}: {e}", idx + 1))?,
            };
            obs.push(sym);
        }
        let mut candidate = self.export_config();
        candidate.mode = MarkovWorkbenchMode::Hmm;
        candidate.hmm.observations = obs;
        let count = candidate.hmm.observations.len();
        self.apply_config(candidate)?;
        Ok(format!("{count} JSON observation(s)"))
    }

    fn import_csv_by_mode(&mut self, cleaned: &str) -> Result<String, String> {
        let mut candidate = self.export_config();
        match candidate.mode {
            MarkovWorkbenchMode::Hmm => {
                let obs = parse_hmm_observations_csv(cleaned, &candidate.hmm.symbol_names)?;
                let count = obs.len();
                candidate.hmm.observations = obs;
                self.apply_config(candidate)?;
                Ok(format!("{count} CSV observation(s)"))
            }
            MarkovWorkbenchMode::Ctmc => {
                let rows = parse_numeric_matrix_csv(cleaned)?;
                let n = rows.len();
                if candidate.ctmc.state_names.len() != n {
                    candidate.ctmc.state_names = default_labels("State", n);
                }
                candidate.ctmc.generator = rows;
                candidate.ctmc.source_state = candidate.ctmc.source_state.min(n.saturating_sub(1));
                candidate.ctmc.transient_states = (0..n)
                    .filter(|&i| candidate.ctmc.generator[i].iter().any(|&v| v.abs() > 1e-9))
                    .collect();
                if candidate.ctmc.transient_states.len() == n {
                    candidate.ctmc.transient_states.clear();
                }
                self.apply_config(candidate)?;
                Ok(format!("{n}x{n} CSV CTMC generator"))
            }
            MarkovWorkbenchMode::AbsorbingDtmc => self.import_dtmc_csv_matrix(cleaned, candidate),
        }
    }

    fn import_dtmc_csv_matrix(
        &mut self,
        cleaned: &str,
        mut candidate: MarkovWorkbenchPresetConfig,
    ) -> Result<String, String> {
        let rows = parse_numeric_matrix_csv(cleaned)?;
        let n = rows.len();
        if candidate.dtmc.state_names.len() != n {
            candidate.dtmc.state_names = default_labels("State", n);
        }
        candidate.dtmc.state_types = rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                if r.get(i).is_some_and(|&diag| (diag - 1.0).abs() <= 1e-6) {
                    DtmcStateKind::Absorbing
                } else {
                    DtmcStateKind::Transient
                }
            })
            .collect();
        let num_abs = candidate
            .dtmc
            .state_types
            .iter()
            .filter(|&&k| k == DtmcStateKind::Absorbing)
            .count();
        if candidate.dtmc.absorbing_rewards.len() != num_abs {
            candidate.dtmc.absorbing_rewards =
                (0..num_abs).map(|k| if k == 0 { 1.0 } else { 0.0 }).collect();
        }
        candidate.dtmc.transition_matrix = rows;
        self.apply_config(candidate)?;
        Ok(format!("{n}x{n} CSV DTMC transition matrix"))
    }

    /// Atomically imports a UTF-8 or UTF-8 BOM-prefixed `.csv` or `.json` file without mutating
    /// active state when validation or parsing fails.
    pub fn import_file_str(&mut self, raw: &str) -> Result<String, String> {
        let cleaned = strip_utf8_bom(raw).trim();
        if cleaned.is_empty() {
            return Err("Imported Markov file is empty".to_string());
        }
        if cleaned.starts_with('{') {
            return self.import_json_object(cleaned);
        }
        if cleaned.starts_with('[') {
            return self.import_json_observations(cleaned);
        }
        self.import_csv_by_mode(cleaned)
    }
}
