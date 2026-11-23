//! Multi-arm Kaplan-Meier survival analysis tool with `try_estimate_hazard_ratio`,
//! `TimePoint` risk-set tables, local CSV/TSV cohort import, and `.csv`/`.md`/`.svg` exports.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod export;
mod parser;
pub mod ui;

#[cfg(test)]
mod tests;

use super::clinical_io::ClinicalIoState;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::applied::clinical_trials::survival_analysis::{
    kaplan_meier, try_estimate_hazard_ratio, Observation, TimePoint,
};

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
    /// Optional separate Control arm (`time, event`) text.
    pub control_text: String,
    /// Flattened validated observations across all arms (capped at `MAX_ITERATIONS`).
    pub(crate) observations: Vec<Observation>,
    /// Primary arm Kaplan-Meier curve (single-arm curve or first study arm).
    pub(crate) curve: Vec<TimePoint>,
    /// Per-arm Kaplan-Meier curves in display order.
    pub arms: Vec<ArmSurvival>,
    /// Incidence-density hazard ratio `HR` between `arms[0]` and `arms[1]` when `arms.len() >= 2`.
    pub hazard_ratio: Option<Result<f64, String>>,
    /// Parse or validation error message when `input_text` or `control_text` is malformed.
    pub(crate) error_message: Option<String>,
    /// Non-blocking file picker and `.csv`/`.md`/`.svg` download state.
    pub(crate) io_state: ClinicalIoState,
}

impl Default for SurvivalAnalysisTool {
    fn default() -> Self {
        let default_text = "6.0, 1\n6.0, 1\n6.0, 1\n7.0, 1\n10.0, 1\n13.0, 1\n16.0, 1\n22.0, 1\n23.0, 1\n6.0, 0\n9.0, 0\n10.0, 0\n11.0, 0\n17.0, 0\n19.0, 0\n20.0, 0\n25.0, 0\n32.0, 0\n32.0, 0\n34.0, 0\n35.0, 0";
        let mut tool = Self {
            input_text: default_text.to_string(),
            control_text: String::new(),
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
    #[serde(default, skip_serializing_if = "String::is_empty")]
    control_text: String,
}

impl SurvivalAnalysisTool {
    /// Recomputes single-arm or multi-arm Kaplan-Meier curves and two-arm hazard ratio.
    pub fn recalculate(&mut self) {
        self.clear_computed();
        match parser::parse_survival_cohorts(&self.input_text, &self.control_text) {
            Ok((all_obs, arm_buckets)) => {
                self.observations = all_obs;
                self.finalize_arms(arm_buckets);
            }
            Err(err) => {
                self.clear_computed();
                self.error_message = Some(err);
            }
        }
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
        self.control_text.clear();
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
            control_text: self.control_text.clone(),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: SurvivalState =
            serde_json::from_str(state).map_err(|e| format!("Invalid survival state: {e}"))?;
        let mut candidate = Self {
            input_text: s.input_text,
            control_text: s.control_text,
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
        tags: &[
            "kaplan-meier",
            "censoring",
            "hazard",
            "hazard-ratio",
            "hazard ratio",
            "survival",
            "time-to-event",
            "nonparametric",
            "biostatistics",
        ],
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
