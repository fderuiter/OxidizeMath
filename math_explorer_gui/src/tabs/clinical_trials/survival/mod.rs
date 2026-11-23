//! Multi-arm Kaplan-Meier survival analysis tool with `try_estimate_hazard_ratio`,
//! `TimePoint` risk-set tables, local CSV/TSV cohort import, and `.csv`/`.md`/`.svg` exports.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod export;
pub mod ui;

#[cfg(test)]
mod tests;

use super::clinical_io::ClinicalIoState;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::applied::clinical_trials::survival_analysis::{
    kaplan_meier, try_estimate_hazard_ratio, Observation, TimePoint,
};
use math_explorer::applied::clinical_trials::types::SurvivalTime;

/// Canonical two-arm survival cohort example (`Arm, Time, Event`) for comparative KM & HR analysis.
pub const TWO_ARM_EXAMPLE_CSV: &str = "\
Arm, Time, Event
Treatment, 6.0, 1
Treatment, 9.0, 0
Treatment, 11.0, 1
Treatment, 15.0, 0
Treatment, 19.0, 1
Treatment, 24.0, 0
Treatment, 28.0, 1
Treatment, 34.0, 0
Control, 3.0, 1
Control, 5.0, 1
Control, 7.0, 1
Control, 8.0, 0
Control, 10.0, 1
Control, 12.0, 1
Control, 15.0, 1
Control, 18.0, 0";

/// Per-arm Kaplan-Meier cohort and evaluated `TimePoint` step trajectory.
#[derive(Clone, Debug)]
pub struct ArmSurvival {
    /// Display name of the study arm (e.g., `"Treatment"`, `"Control"`, or `"Cohort"`).
    pub name: String,
    /// Validated right-censored survival observations in this arm.
    pub observations: Vec<Observation>,
    /// Evaluated Kaplan-Meier `TimePoint` risk-set steps for this arm.
    pub curve: Vec<TimePoint>,
}

impl ArmSurvival {
    /// First event time `t` at which estimated survival `S(t) <= 0.50`, if reached.
    #[must_use]
    pub fn median_survival_time(&self) -> Option<f64> {
        self.curve
            .iter()
            .find(|pt| pt.survival_probability <= 0.5 + 1e-12)
            .map(|pt| pt.time)
    }

    /// Builds step-function `[time, S(t)]` plot coordinates starting from `(0, 1.0)`.
    #[must_use]
    pub fn build_step_points(&self) -> Vec<[f64; 2]> {
        build_curve_step_points(&self.curve)
    }
}

pub(crate) fn build_curve_step_points(curve: &[TimePoint]) -> Vec<[f64; 2]> {
    let mut plot_points: Vec<[f64; 2]> = vec![[0.0, 1.0]];
    for point in curve {
        if let Some(last) = plot_points.last() {
            plot_points.push([point.time, last[1]]);
        }
        plot_points.push([point.time, point.survival_probability]);
    }
    if let Some(last) = plot_points.last() {
        plot_points.push([last[0] * 1.1, last[1]]);
    }
    plot_points
}

/// Interactive single-arm and multi-arm Kaplan-Meier survival and hazard ratio workbench.
pub struct SurvivalAnalysisTool {
    /// Raw single-arm (`time, event`) or multi-arm (`arm, time, event`) CSV/TSV text.
    pub input_text: String,
    /// Flattened validated observations across all arms (capped at `MAX_ITERATIONS`).
    pub(crate) observations: Vec<Observation>,
    /// Primary arm Kaplan-Meier curve (single-arm curve or first study arm).
    pub(crate) curve: Vec<TimePoint>,
    /// Per-arm Kaplan-Meier curves in display order.
    pub arms: Vec<ArmSurvival>,
    /// Incidence-density hazard ratio `HR` between `arms[0]` and `arms[1]` when `arms.len() >= 2`.
    pub hazard_ratio: Option<Result<f64, String>>,
    /// Parse or validation error message when `input_text` is malformed.
    pub(crate) error_message: Option<String>,
    /// Non-blocking file picker and `.csv`/`.md`/`.svg` download state.
    pub(crate) io_state: ClinicalIoState,
}

impl Default for SurvivalAnalysisTool {
    fn default() -> Self {
        let default_text = "6.0, 1\n6.0, 1\n6.0, 1\n7.0, 1\n10.0, 1\n13.0, 1\n16.0, 1\n22.0, 1\n23.0, 1\n6.0, 0\n9.0, 0\n10.0, 0\n11.0, 0\n17.0, 0\n19.0, 0\n20.0, 0\n25.0, 0\n32.0, 0\n32.0, 0\n34.0, 0\n35.0, 0";
        let mut tool = Self {
            input_text: default_text.to_string(),
            observations: Vec::new(),
            curve: Vec::new(),
            arms: Vec::new(),
            hazard_ratio: None,
            error_message: None,
            io_state: ClinicalIoState::default(),
        };
        tool.recalculate();
        tool
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SurvivalState {
    input_text: String,
}

fn parse_event_flag(line_idx: usize, event_str: &str) -> Result<bool, String> {
    match event_str.parse::<u8>() {
        Ok(1) => Ok(true),
        Ok(0) => Ok(false),
        _ if event_str.eq_ignore_ascii_case("true") || event_str.eq_ignore_ascii_case("event") => {
            Ok(true)
        }
        _ if event_str.eq_ignore_ascii_case("false")
            || event_str.eq_ignore_ascii_case("censored") =>
        {
            Ok(false)
        }
        _ => Err(format!(
            "Line {}: Event must be 1 (event) or 0 (censored)",
            line_idx + 1
        )),
    }
}

fn is_header_line(parts: &[&str]) -> bool {
    parts.iter().any(|raw| {
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "time" | "survival_time" | "arm" | "group" | "cohort"
        )
    })
}

fn split_fields(line: &str) -> Vec<&str> {
    if line.contains('\t') {
        line.split('\t').collect()
    } else {
        line.split(',').collect()
    }
}

fn parse_survival_row(line_idx: usize, parts: &[&str]) -> Result<(String, Observation), String> {
    let (arm_name, time_str, event_str) = match parts.len() {
        2 => ("Cohort".to_string(), parts[0].trim(), parts[1].trim()),
        3 => {
            let p0 = parts[0].trim();
            let p1 = parts[1].trim();
            let p2 = parts[2].trim();
            if p1.parse::<f64>().is_ok() && parse_event_flag(line_idx, p2).is_ok() {
                if p0.is_empty() {
                    return Err(format!("Line {}: Arm label cannot be empty", line_idx + 1));
                }
                (p0.to_string(), p1, p2)
            } else if p0.parse::<f64>().is_ok() && parse_event_flag(line_idx, p1).is_ok() {
                if p2.is_empty() {
                    return Err(format!("Line {}: Arm label cannot be empty", line_idx + 1));
                }
                (p2.to_string(), p0, p1)
            } else {
                return Err(format!(
                    "Line {}: Invalid 3-column survival row (expected 'arm, time, event')",
                    line_idx + 1
                ));
            }
        }
        _ => {
            return Err(format!(
                "Line {}: Expected 'time, event' or 'arm, time, event'",
                line_idx + 1
            ));
        }
    };

    let time_val = time_str
        .parse::<f64>()
        .map_err(|_| format!("Line {}: Invalid time '{}'", line_idx + 1, time_str))?;
    let event_occurred = parse_event_flag(line_idx, event_str)?;
    let time = SurvivalTime::new(time_val).map_err(|e| format!("Line {}: {}", line_idx + 1, e))?;

    Ok((
        arm_name,
        Observation {
            time,
            event_occurred,
        },
    ))
}

impl SurvivalAnalysisTool {
    /// Recomputes single-arm or multi-arm Kaplan-Meier curves and two-arm hazard ratio.
    pub fn recalculate(&mut self) {
        self.clear_computed();
        let cleaned = crate::web_io::strip_utf8_bom(&self.input_text);
        let mut arm_buckets: Vec<(String, Vec<Observation>)> = Vec::new();
        let mut saw_data_row = false;

        for (line_idx, raw_line) in cleaned.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts = split_fields(line);
            if !saw_data_row && is_header_line(&parts) {
                saw_data_row = true;
                continue;
            }
            saw_data_row = true;

            if self.observations.len() >= math_commons::registry::MAX_ITERATIONS {
                self.clear_computed();
                self.error_message = Some(format!(
                    "Line {}: Maximum of {} observations exceeded",
                    line_idx + 1,
                    math_commons::registry::MAX_ITERATIONS
                ));
                return;
            }

            match parse_survival_row(line_idx, &parts) {
                Ok((arm, obs)) => {
                    self.observations.push(obs);
                    if let Some((_, bucket)) = arm_buckets
                        .iter_mut()
                        .find(|(existing, _)| existing.eq_ignore_ascii_case(&arm))
                    {
                        bucket.push(obs);
                    } else {
                        arm_buckets.push((arm, vec![obs]));
                    }
                }
                Err(err) => {
                    self.clear_computed();
                    self.error_message = Some(err);
                    return;
                }
            }
        }

        self.finalize_arms(arm_buckets);
    }

    fn clear_computed(&mut self) {
        self.observations.clear();
        self.curve.clear();
        self.arms.clear();
        self.hazard_ratio = None;
        self.error_message = None;
    }

    fn finalize_arms(&mut self, mut arm_buckets: Vec<(String, Vec<Observation>)>) {
        if arm_buckets.len() == 2
            && arm_buckets[0].0.eq_ignore_ascii_case("control")
            && arm_buckets[1].0.eq_ignore_ascii_case("treatment")
        {
            arm_buckets.swap(0, 1);
        }

        for (name, obs) in arm_buckets {
            let curve = kaplan_meier(&obs);
            self.arms.push(ArmSurvival {
                name,
                observations: obs,
                curve,
            });
        }

        if let Some(first) = self.arms.first() {
            self.curve = first.curve.clone();
        }
        if self.arms.len() >= 2 {
            self.hazard_ratio = Some(
                try_estimate_hazard_ratio(&self.arms[0].observations, &self.arms[1].observations)
                    .map_err(|e| e.to_string()),
            );
        }
    }

    /// Atomically validates and imports a CSV or TSV survival cohort string.
    pub fn import_cohort_str(&mut self, raw: &str) -> Result<(usize, usize), String> {
        let mut candidate = Self {
            input_text: crate::web_io::strip_utf8_bom(raw).to_string(),
            ..Self::default()
        };
        candidate.recalculate();
        if let Some(err) = candidate.error_message {
            return Err(err);
        }
        if candidate.observations.is_empty() {
            return Err("Imported survival file contains no observations".into());
        }
        let summary = (candidate.arms.len(), candidate.observations.len());
        self.input_text = candidate.input_text;
        self.observations = candidate.observations;
        self.curve = candidate.curve;
        self.arms = candidate.arms;
        self.hazard_ratio = candidate.hazard_ratio;
        self.error_message = None;
        Ok(summary)
    }

    /// Polls non-blocking file import and export channels.
    pub fn poll_io(&mut self, ctx: &egui::Context) {
        self.io_state.poll_save("Kaplan-Meier export");
        if let Some(outcome) = self.io_state.poll_open() {
            match outcome {
                Ok((filename, content)) => match self.import_cohort_str(&content) {
                    Ok((n_arms, n_obs)) => {
                        self.io_state.record_info(format!(
                            "Imported {filename} ({n_obs} observations across {n_arms} arm(s))"
                        ));
                    }
                    Err(err) => {
                        self.io_state
                            .record_error(format!("Cohort import rejected ({filename}): {err}"));
                    }
                },
                Err(err) => self.io_state.record_error(err),
            }
            ctx.request_repaint();
        }
    }

    pub(crate) fn median_survival_time(&self) -> Option<f64> {
        self.curve
            .iter()
            .find(|pt| pt.survival_probability <= 0.5 + 1e-12)
            .map(|pt| pt.time)
    }

    pub(crate) fn build_step_points(&self) -> Vec<[f64; 2]> {
        build_curve_step_points(&self.curve)
    }
}

impl InteractiveTool for SurvivalAnalysisTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Survival Curves (Kaplan-Meier)"
    }

    fn domain(&self) -> &'static str {
        "clinical_trials"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        export::collect_export_datasets(self)
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&SurvivalState {
            input_text: self.input_text.clone(),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: SurvivalState =
            serde_json::from_str(state).map_err(|e| format!("Invalid survival state: {e}"))?;
        let mut candidate = Self {
            input_text: s.input_text,
            ..Self::default()
        };
        candidate.recalculate();
        if let Some(err) = candidate.error_message {
            return Err(err);
        }
        if candidate.observations.is_empty() {
            return Err("Survival observations cannot be empty".into());
        }
        *self = candidate;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        ui::show_survival_inside(self, ui);
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SurvivalAnalysisTool",
        display_name: "Survival Curves (Kaplan-Meier)",
        domain: "clinical_trials",
        tags: &["survival", "kaplan_meier", "hazard_ratio", "censored"],
        build: || Box::new(SurvivalAnalysisTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for SurvivalAnalysisTool {
    fn theory_description(&self) -> String {
        format!(
            "Kaplan-Meier non-parametric maximum likelihood estimator of survival function $S(t) = \\prod_{{t_i \\le t}} \\left(1 - \\frac{{d_i}}{{n_i}}\\right)$ and incidence-density hazard ratio across {} observations ({} arm(s)).",
            self.observations.len(),
            self.arms.len().max(1)
        )
    }

    fn phonetic_description(&self) -> String {
        "Kaplan-Meier non-parametric survival analysis and hazard ratio theory".into()
    }

    fn theory_citation(&self) -> String {
        "Nonparametric Estimation from Incomplete Observations (Kaplan & Meier, 1958)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Kaplan-Meier Product-Limit".into(),
            "Stepwise right-censored survival probability estimation with risk-set tracking".into(),
        );
        map.insert(
            "Incidence-Density Hazard Ratio".into(),
            "Two-arm event-rate ratio HR = (d_1 / T_1) / (d_2 / T_2)".into(),
        );
        map
    }
}

/// Starter preset JSON payload for Kaplan-Meier survival analysis.
pub const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Kaplan-Meier Survival",
    "domain": "clinical_trials",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "input_text": "3.0, 1\n5.0, 1\n6.0, 0\n8.0, 1\n11.0, 0\n14.0, 1\n18.0, 1\n21.0, 0\n24.0, 1\n30.0, 0"
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "survival_analysis_starter",
        title: "Kaplan-Meier Survival",
        description: "Ŝ(t) = ∏(1 - d_i/n_i): Non-parametric product-limit estimator for time-to-event censorship data.",
        category: "Survival Models",
        icon: "⏱️",
        domain: "clinical_trials",
        tool_name: Some("SurvivalAnalysisTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:clinical_trials_statistics]
