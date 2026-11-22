#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::InteractiveTool;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use math_explorer::applied::clinical_trials::survival_analysis::{
    kaplan_meier, try_estimate_hazard_ratio, Observation, TimePoint,
};
use math_explorer::applied::clinical_trials::types::SurvivalTime;

const DEFAULT_TREATMENT_TEXT: &str = "6.0, 1\n6.0, 1\n6.0, 1\n7.0, 1\n10.0, 1\n13.0, 1\n16.0, 1\n22.0, 1\n23.0, 1\n6.0, 0\n9.0, 0\n10.0, 0\n11.0, 0\n17.0, 0\n19.0, 0\n20.0, 0\n25.0, 0\n32.0, 0\n32.0, 0\n34.0, 0\n35.0, 0";
const DEMO_CONTROL_TEXT: &str = "1.0, 1\n1.0, 1\n2.0, 1\n2.0, 1\n3.0, 1\n4.0, 1\n4.0, 1\n5.0, 1\n5.0, 1\n8.0, 1\n8.0, 1\n8.0, 1\n8.0, 1\n11.0, 1\n11.0, 1\n12.0, 1\n12.0, 1\n15.0, 1\n17.0, 1\n22.0, 1\n23.0, 1";

pub struct SurvivalAnalysisTool {
    input_text: String,
    control_text: String,
    observations: Vec<Observation>,
    control_observations: Vec<Observation>,
    curve: Vec<TimePoint>,
    control_curve: Vec<TimePoint>,
    hazard_ratio: Option<Result<f64, String>>,
    error_message: Option<String>,
}

impl Default for SurvivalAnalysisTool {
    fn default() -> Self {
        let mut tool = Self {
            input_text: DEFAULT_TREATMENT_TEXT.to_string(),
            control_text: String::new(),
            observations: Vec::new(),
            control_observations: Vec::new(),
            curve: Vec::new(),
            control_curve: Vec::new(),
            hazard_ratio: None,
            error_message: None,
        };
        tool.recalculate();
        tool
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ParsedArm {
    Treatment,
    Control,
}

fn parse_event_flag(line_idx: usize, event_str: &str) -> Result<bool, String> {
    match event_str.parse::<u8>() {
        Ok(1) => Ok(true),
        Ok(0) => Ok(false),
        _ if event_str.eq_ignore_ascii_case("true") => Ok(true),
        _ if event_str.eq_ignore_ascii_case("false") => Ok(false),
        _ => Err(format!(
            "Line {}: Event must be 1 (event) or 0 (censored)",
            line_idx + 1
        )),
    }
}

fn parse_arm_label(line_idx: usize, raw: &str) -> Result<ParsedArm, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "treatment" | "trt" | "active" | "1" => Ok(ParsedArm::Treatment),
        "control" | "ctrl" | "placebo" | "0" => Ok(ParsedArm::Control),
        other => Err(format!(
            "Line {}: Invalid arm '{}' (expected treatment/control)",
            line_idx + 1,
            other
        )),
    }
}

fn build_step_points(curve: &[TimePoint]) -> Vec<[f64; 2]> {
    let mut plot_points = Vec::with_capacity(curve.len() * 2 + 2);
    plot_points.push([0.0, 1.0]);
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

impl SurvivalAnalysisTool {
    fn parse_line(line_idx: usize, line: &str) -> Result<(ParsedArm, Observation), String> {
        let parts: Vec<&str> = line.split(',').collect();
        let (arm, time_str, event_str) = match parts.as_slice() {
            [t, e] => (ParsedArm::Treatment, t.trim(), e.trim()),
            [arm_raw, t, e] => (parse_arm_label(line_idx, arm_raw)?, t.trim(), e.trim()),
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
        let time =
            SurvivalTime::new(time_val).map_err(|e| format!("Line {}: {}", line_idx + 1, e))?;
        Ok((
            arm,
            Observation {
                time,
                event_occurred,
            },
        ))
    }

    fn parse_arm_block(
        text: &str,
        allow_three_col: bool,
    ) -> Result<(Vec<Observation>, Vec<Observation>), String> {
        let mut trt = Vec::new();
        let mut ctrl = Vec::new();
        for (line_idx, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line_idx == 0 && line.to_ascii_lowercase().contains("time") {
                continue;
            }
            if trt.len() + ctrl.len() >= math_commons::registry::MAX_ITERATIONS {
                return Err(format!(
                    "Line {}: Maximum of {} observations exceeded",
                    line_idx + 1,
                    math_commons::registry::MAX_ITERATIONS
                ));
            }
            let (arm, obs) = Self::parse_line(line_idx, line)?;
            if !allow_three_col && arm == ParsedArm::Control {
                return Err(format!("Line {}: Expected 'time, event'", line_idx + 1));
            }
            match arm {
                ParsedArm::Treatment => trt.push(obs),
                ParsedArm::Control => ctrl.push(obs),
            }
        }
        Ok((trt, ctrl))
    }

    fn clear_computed(&mut self) {
        self.observations.clear();
        self.control_observations.clear();
        self.curve.clear();
        self.control_curve.clear();
        self.hazard_ratio = None;
    }

    fn recalculate(&mut self) {
        self.clear_computed();
        self.error_message = None;

        let (trt, mut inline_ctrl) = match Self::parse_arm_block(&self.input_text, true) {
            Ok(pair) => pair,
            Err(err) => {
                self.error_message = Some(err);
                return;
            }
        };
        if !self.control_text.trim().is_empty() {
            match Self::parse_arm_block(&self.control_text, false) {
                Ok((extra_ctrl, _)) => inline_ctrl.extend(extra_ctrl),
                Err(err) => {
                    self.error_message = Some(format!("Control arm {err}"));
                    return;
                }
            }
        }
        self.observations = trt;
        self.control_observations = inline_ctrl;
        self.curve = kaplan_meier(&self.observations);
        if !self.control_observations.is_empty() {
            self.control_curve = kaplan_meier(&self.control_observations);
            self.hazard_ratio = Some(
                try_estimate_hazard_ratio(&self.observations, &self.control_observations)
                    .map_err(|e| e.to_string()),
            );
        }
    }

    fn render_side_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Survival Cohort Input");
        ui.label("Treatment ('time, event' or 'arm, time, event'):");
        let mut changed = ui
            .add(
                egui::TextEdit::multiline(&mut self.input_text)
                    .desired_width(f32::INFINITY)
                    .desired_rows(10),
            )
            .changed();
        ui.add_space(4.0);
        ui.label("Optional Control Arm ('time, event'):");
        changed |= ui
            .add(
                egui::TextEdit::multiline(&mut self.control_text)
                    .desired_width(f32::INFINITY)
                    .desired_rows(6),
            )
            .changed();
        ui.horizontal(|ui| {
            if ui.button("Load 2-Arm Demo").clicked() {
                self.input_text = DEFAULT_TREATMENT_TEXT.to_string();
                self.control_text = DEMO_CONTROL_TEXT.to_string();
                changed = true;
            }
            if !self.control_text.is_empty() && ui.button("Single Arm").clicked() {
                self.control_text.clear();
                changed = true;
            }
        });
        if changed {
            self.recalculate();
        }
        self.render_metrics(ui);
    }

    fn render_metrics(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if let Some(err) = &self.error_message {
            ui.colored_label(palette.error, err);
        }
        ui.separator();
        render_copyable_metric(ui, "Treatment N", &self.observations.len().to_string());
        if !self.control_observations.is_empty() {
            render_copyable_metric(
                ui,
                "Control N",
                &self.control_observations.len().to_string(),
            );
            match &self.hazard_ratio {
                Some(Ok(hr)) => {
                    render_copyable_metric(ui, "Hazard Ratio (Trt/Ctrl)", &format!("{hr:.3}"));
                }
                Some(Err(err)) => {
                    ui.colored_label(palette.warning, format!("HR unavailable: {err}"));
                }
                None => {}
            }
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SurvivalState {
    input_text: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    control_text: String,
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

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&SurvivalState {
            input_text: self.input_text.clone(),
            control_text: self.control_text.clone(),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: SurvivalState =
            serde_json::from_str(state).map_err(|e| format!("Invalid survival state: {e}"))?;
        let mut candidate = Self {
            input_text: s.input_text,
            control_text: s.control_text,
            observations: Vec::new(),
            control_observations: Vec::new(),
            curve: Vec::new(),
            control_curve: Vec::new(),
            hazard_ratio: None,
            error_message: None,
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
        let panel_id = crate::framework::scoped_panel_id(ui, "survival_input_panel");
        crate::framework::PaneSidePanel::left(panel_id)
            .show_inside(ui, |ui| self.render_side_panel(ui));

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Kaplan-Meier Survival Curves");
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let trt_pts = build_step_points(&self.curve);
            let ctrl_pts = if self.control_curve.is_empty() {
                Vec::new()
            } else {
                build_step_points(&self.control_curve)
            };

            Plot::new("kaplan_meier_plot")
                .legend(Legend::default())
                .view_aspect(2.0)
                .x_axis_label("Follow-up Time")
                .y_axis_label("Survival Probability")
                .include_y(0.0)
                .include_y(1.05)
                .show(ui, |plot_ui| {
                    let trt_label = if ctrl_pts.is_empty() {
                        "Survival Probability"
                    } else {
                        "Treatment Survival"
                    };
                    plot_ui.line(
                        Line::new(trt_label, PlotPoints::new(trt_pts))
                            .color(palette.accent)
                            .width(2.5_f32),
                    );
                    if !ctrl_pts.is_empty() {
                        plot_ui.line(
                            Line::new("Control Survival", PlotPoints::new(ctrl_pts))
                                .color(palette.warning)
                                .width(2.2_f32),
                        );
                    }
                });
        });
    }
}

// [cite:clinical_trials_statistics]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SurvivalAnalysisTool",
        display_name: "Survival Curves (Kaplan-Meier)",
        domain: "clinical_trials",
        tags: &["survival", "kaplan-meier", "hazard-ratio"],
        build: || Box::new(SurvivalAnalysisTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for SurvivalAnalysisTool {
    fn theory_description(&self) -> String {
        format!(
            "Kaplan-Meier non-parametric maximum likelihood estimator of survival function $S(t) = \\prod_{{t_i \\le t}} \\left(1 - \\frac{{d_i}}{{n_i}}\\right)$ across {} treatment and {} control observations, paired with incidence-rate hazard ratio estimation.",
            self.observations.len(),
            self.control_observations.len()
        )
    }
    fn phonetic_description(&self) -> String {
        "Kaplan-Meier non-parametric survival analysis and hazard ratio theory".into()
    }
    fn theory_citation(&self) -> String {
        "Nonparametric Estimation from Incomplete Observations (Kaplan & Meier, 1958)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reject_infinite_and_non_finite_presets() {
        let inf_preset = r#"{"metadata":{"schema_version":"1.0","title":"KM","domain":"clinical_trials","timestamp":"2026-11-19T00:00:00Z"},"config":{"input_text":"inf, 1"}}"#;
        let mut tool = SurvivalAnalysisTool::default();
        let baseline_len = tool.observations.len();
        assert!(tool.import_preset_json(inf_preset).is_err());
        assert_eq!(tool.observations.len(), baseline_len);

        for invalid_input in ["-inf, 1", "Infinity, 0", "NaN, 1", "-1.0, 1"] {
            let payload = format!(
                r#"{{"metadata":{{"schema_version":"1.0","title":"KM","domain":"clinical_trials","timestamp":"2026-11-19T00:00:00Z"}},"config":{{"input_text":"{invalid_input}"}}}}"#
            );
            assert!(
                tool.import_preset_json(&payload).is_err(),
                "Expected error for preset input '{invalid_input}'"
            );
            assert_eq!(tool.observations.len(), baseline_len);
        }
    }

    #[test]
    fn test_observation_cap_enforced_in_recalculate_and_load_state() {
        let max_obs = math_commons::registry::MAX_ITERATIONS;
        let at_cap_input: String = (0..max_obs)
            .map(|idx| format!("{}.0, 1", idx + 1))
            .collect::<Vec<_>>()
            .join("\n");

        let mut tool = SurvivalAnalysisTool::default();
        let state_ok = serde_json::to_string(&SurvivalState {
            input_text: at_cap_input,
            control_text: String::new(),
        })
        .unwrap();
        assert!(tool.try_load_state(&state_ok).is_ok());
        assert_eq!(tool.observations.len(), max_obs);
        assert!(tool.error_message.is_none());

        let over_cap_input: String = (0..=max_obs)
            .map(|idx| format!("{}.0, 1", idx + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let state_over = serde_json::to_string(&SurvivalState {
            input_text: over_cap_input.clone(),
            control_text: String::new(),
        })
        .unwrap();
        let err = tool.try_load_state(&state_over).unwrap_err();
        assert!(err.contains(&max_obs.to_string()));
        assert_eq!(tool.observations.len(), max_obs);

        tool.input_text = over_cap_input;
        tool.recalculate();
        assert!(tool.error_message.is_some());
        assert!(tool.observations.is_empty());
        assert!(tool.curve.is_empty());
    }

    #[test]
    fn test_starter_preset_preserves_survival_probabilities() {
        let mut tool = SurvivalAnalysisTool::default();
        tool.import_preset_json(STARTER_PRESET_JSON).unwrap();
        assert_eq!(tool.observations.len(), 10);
        assert_eq!(tool.curve.len(), 10);
        assert!((tool.curve[0].survival_probability - 0.9).abs() < 1e-12);
        assert!((tool.curve[1].survival_probability - 0.8).abs() < 1e-12);
        assert!((tool.curve[2].survival_probability - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_two_arm_kaplan_meier_estimates_hazard_ratio() {
        let mut tool = SurvivalAnalysisTool::default();
        let state = serde_json::to_string(&SurvivalState {
            input_text: "10.0, 1\n12.0, 0\n15.0, 1".to_string(),
            control_text: "4.0, 1\n5.0, 1\n6.0, 1".to_string(),
        })
        .unwrap();
        tool.try_load_state(&state).unwrap();
        assert_eq!(tool.observations.len(), 3);
        assert_eq!(tool.control_observations.len(), 3);
        assert!(!tool.control_curve.is_empty());
        let hr = tool.hazard_ratio.as_ref().unwrap().as_ref().unwrap();
        assert!(*hr < 1.0 && hr.is_finite());
    }
}
