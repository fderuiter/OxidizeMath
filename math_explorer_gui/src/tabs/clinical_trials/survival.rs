#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::framework::{ExportCapabilities, InteractiveTool};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Legend, Line, MarkerShape, Plot, PlotPoints, Points};
use math_explorer::applied::clinical_trials::survival_analysis::{
    LogRankResult, Observation, TimePoint,
};

use super::survival_io::{
    build_censor_points, build_km_step_points, format_risk_table_csv, parse_survival_cohort,
    ArmSurvivalSummary, SurvivalDialogIo, TWO_ARM_SAMPLE_TEXT,
};

pub struct SurvivalAnalysisTool {
    input_text: String,
    observations: Vec<Observation>,
    curve: Vec<TimePoint>,
    arms: Vec<ArmSurvivalSummary>,
    hazard_ratio: Option<Result<f64, String>>,
    log_rank: Option<Result<LogRankResult, String>>,
    error_message: Option<String>,
    dialog_io: SurvivalDialogIo,
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
            log_rank: None,
            error_message: None,
            dialog_io: SurvivalDialogIo::default(),
        };
        tool.recalculate();
        tool
    }
}

impl SurvivalAnalysisTool {
    fn recalculate(&mut self) {
        self.observations.clear();
        self.curve.clear();
        self.arms.clear();
        self.hazard_ratio = None;
        self.log_rank = None;
        self.error_message = None;

        match parse_survival_cohort(&self.input_text) {
            Ok(parsed) => {
                self.observations = parsed.all_observations;
                self.curve = parsed
                    .arms
                    .first()
                    .map_or_else(Vec::new, |arm| arm.curve.clone());
                self.arms = parsed.arms;
                self.hazard_ratio = parsed.hazard_ratio;
                self.log_rank = parsed.log_rank;
            }
            Err(err) => {
                self.error_message = Some(err);
            }
        }
    }

    fn render_input_panel(&mut self, ui: &mut egui::Ui) {
        if let Some(loaded) = self.dialog_io.poll() {
            self.input_text = loaded;
            self.recalculate();
        }
        ui.heading("Input Data");
        ui.label("Format: 'time, event' or 'time, event, arm' (CSV/TSV)");
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📂 Load CSV/TSV")
                .accessible_hover_text("Load local survival cohort CSV or TSV file")
                .clicked()
            {
                self.dialog_io.trigger_open();
            }
            if ui
                .button("🧪 Two-Arm Sample")
                .accessible_hover_text("Load sample two-arm censored oncology cohort")
                .clicked()
            {
                self.input_text = TWO_ARM_SAMPLE_TEXT.to_string();
                self.recalculate();
            }
            let can_export = !self.arms.is_empty() && !self.curve.is_empty();
            if ui
                .add_enabled(can_export, egui::Button::new("💾 Export Risk Table"))
                .accessible_hover_text("Download TimePoint risk counts as CSV")
                .clicked()
            {
                self.dialog_io
                    .trigger_save_risk_table(format_risk_table_csv(&self.arms));
            }
        });
        ui.add_space(4.0);
        let text_edit = egui::TextEdit::multiline(&mut self.input_text)
            .desired_width(f32::INFINITY)
            .desired_rows(12);
        if ui.add(text_edit).changed() {
            self.recalculate();
        }
        self.render_metrics_summary(ui);
    }

    fn render_metrics_summary(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if let Some((is_err, msg)) = &self.dialog_io.status {
            let color = if *is_err { palette.error } else { palette.info };
            ui.colored_label(color, msg);
        }
        if let Some(err) = &self.error_message {
            ui.colored_label(palette.error, err);
        }
        ui.separator();
        render_copyable_metric(ui, "Observations", &self.observations.len().to_string());
        render_copyable_metric(ui, "Study Arms", &self.arms.len().to_string());
        if let Some(hr_res) = &self.hazard_ratio {
            match hr_res {
                Ok(hr) => {
                    let _ = render_copyable_metric(ui, "Hazard Ratio (HR)", &format!("{hr:.3}"));
                }
                Err(e) => {
                    ui.colored_label(palette.warning, e);
                }
            }
        }
        if let Some(lr_res) = &self.log_rank {
            match lr_res {
                Ok(lr) => {
                    render_copyable_metric(ui, "Log-Rank χ²", &format!("{:.3}", lr.statistic));
                    render_copyable_metric(ui, "Log-Rank p-value", &format!("{:.4}", lr.p_value));
                }
                Err(e) => {
                    ui.colored_label(palette.warning, e);
                }
            }
        }
    }

    fn render_curve_and_risk_table(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Kaplan-Meier Survival Curve");
        Plot::new("kaplan_meier_plot")
            .legend(Legend::default())
            .view_aspect(2.1)
            .x_axis_label("Time")
            .y_axis_label("Survival Probability")
            .show(ui, |plot_ui| {
                if self.arms.is_empty() {
                    plot_ui.line(Line::new("Empty", vec![[0.0, 1.0]]).color(palette.muted_text));
                }
                for (idx, arm) in self.arms.iter().enumerate() {
                    let color = palette.series_color(idx);
                    let line_name = if self.arms.len() == 1 {
                        "Survival Probability".to_string()
                    } else {
                        format!("{} S(t)", arm.label)
                    };
                    let step_pts = build_km_step_points(&arm.curve);
                    plot_ui.line(
                        Line::new(line_name.clone(), PlotPoints::new(step_pts))
                            .name(line_name)
                            .color(color),
                    );
                    let censor_pts = build_censor_points(&arm.curve);
                    if !censor_pts.is_empty() {
                        plot_ui.points(
                            Points::new(format!("{} Censored", arm.label), censor_pts)
                                .shape(MarkerShape::Plus)
                                .radius(4.5_f32)
                                .color(color),
                        );
                    }
                }
            });

        ui.separator();
        self.render_risk_table(ui);
    }

    fn render_risk_table(&self, ui: &mut egui::Ui) {
        ui.strong("Kaplan-Meier Risk Table (TimePoint Counts)");
        egui::ScrollArea::vertical()
            .max_height(150.0)
            .show(ui, |ui| {
                egui::Grid::new("km_risk_table_grid")
                    .striped(true)
                    .num_columns(6)
                    .show(ui, |ui| {
                        for h in ["Arm", "Time", "S(t)", "At Risk", "Events", "Censored"] {
                            ui.strong(h);
                        }
                        ui.end_row();
                        for arm in &self.arms {
                            for pt in &arm.curve {
                                ui.label(&arm.label);
                                ui.label(format!("{:.2}", pt.time));
                                ui.label(format!("{:.3}", pt.survival_probability));
                                ui.label(pt.n_at_risk.to_string());
                                ui.label(pt.n_events.to_string());
                                ui.label(pt.n_censored.to_string());
                                ui.end_row();
                            }
                        }
                    });
            });
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SurvivalState {
    input_text: String,
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
        let mut datasets = Vec::new();
        for arm in &self.arms {
            if arm.curve.is_empty() {
                continue;
            }
            datasets.push((
                format!("{} Survival Probability", arm.label),
                build_km_step_points(&arm.curve),
            ));
            datasets.push((
                format!("{} n_at_risk", arm.label),
                arm.curve
                    .iter()
                    .map(|p| [p.time, p.n_at_risk as f64])
                    .collect(),
            ));
            datasets.push((
                format!("{} n_events", arm.label),
                arm.curve
                    .iter()
                    .map(|p| [p.time, p.n_events as f64])
                    .collect(),
            ));
            datasets.push((
                format!("{} n_censored", arm.label),
                arm.curve
                    .iter()
                    .map(|p| [p.time, p.n_censored as f64])
                    .collect(),
            ));
        }
        datasets
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
            observations: Vec::new(),
            curve: Vec::new(),
            arms: Vec::new(),
            hazard_ratio: None,
            log_rank: None,
            error_message: None,
            dialog_io: SurvivalDialogIo::default(),
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
            .min_width(290.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.render_input_panel(ui);
                });
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.render_curve_and_risk_table(ui);
        });
    }
}

// [cite:clinical_trials_statistics]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SurvivalAnalysisTool",
        display_name: "Survival Curves (Kaplan-Meier)",
        domain: "clinical_trials",
        tags: &[],
        build: || Box::new(SurvivalAnalysisTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for SurvivalAnalysisTool {
    fn theory_description(&self) -> String {
        format!(
            "Kaplan-Meier non-parametric maximum likelihood estimator of survival function $S(t) = \\prod_{{t_i \\le t}} \\left(1 - \\frac{{d_i}}{{n_i}}\\right)$ with Mantel-Haenszel two-arm log-rank test across {} observations.",
            self.observations.len()
        )
    }
    fn phonetic_description(&self) -> String {
        "Kaplan-Meier non-parametric survival analysis theory".into()
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
    fn test_two_arm_tsv_bom_log_rank_hazard_ratio_and_risk_table_exports() {
        let bom_tsv = format!("\u{feff}{}", TWO_ARM_SAMPLE_TEXT.replace(',', "\t"));
        let state = serde_json::to_string(&SurvivalState {
            input_text: bom_tsv,
        })
        .unwrap();
        let mut tool = SurvivalAnalysisTool::default();
        tool.try_load_state(&state).expect("two-arm TSV state valid");
        assert_eq!(tool.arms.len(), 2);
        let hr = tool.hazard_ratio.as_ref().unwrap().as_ref().unwrap();
        assert!(*hr < 1.0 && hr.is_finite());
        let lr = tool.log_rank.as_ref().unwrap().as_ref().unwrap();
        assert!(lr.statistic > 0.0 && (0.0..=1.0).contains(&lr.p_value));

        let csv = format_risk_table_csv(&tool.arms);
        assert!(csv.starts_with("arm,time,survival_probability,n_at_risk,n_events,n_censored\n"));
        assert!(csv.contains("Treatment,"));
        assert!(csv.contains("Control,"));
        let datasets = tool.export_datasets();
        assert_eq!(datasets.len(), 8);
    }
}
