#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::framework::{ExportCapabilities, InteractiveTool};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{HLine, Legend, Line, LineStyle, MarkerShape, Plot, PlotPoints, Points};
use math_explorer::applied::clinical_trials::survival_analysis::{
    LogRankResult, Observation, TimePoint,
};

use super::survival_io::{
    build_censor_points, build_km_step_points, format_risk_table_csv, parse_survival_with_control,
    ArmSurvivalSummary, SurvivalDialogIo, TWO_ARM_SAMPLE_TEXT,
};

const DEFAULT_TREATMENT_TEXT: &str = "6.0, 1\n6.0, 1\n6.0, 1\n7.0, 1\n10.0, 1\n13.0, 1\n16.0, 1\n22.0, 1\n23.0, 1\n6.0, 0\n9.0, 0\n10.0, 0\n11.0, 0\n17.0, 0\n19.0, 0\n20.0, 0\n25.0, 0\n32.0, 0\n32.0, 0\n34.0, 0\n35.0, 0";
const DEMO_CONTROL_TEXT: &str = "1.0, 1\n1.0, 1\n2.0, 1\n2.0, 1\n3.0, 1\n4.0, 1\n4.0, 1\n5.0, 1\n5.0, 1\n8.0, 1\n8.0, 1\n8.0, 1\n8.0, 1\n11.0, 1\n11.0, 1\n12.0, 1\n12.0, 1\n15.0, 1\n17.0, 1\n22.0, 1\n23.0, 1";

pub struct SurvivalAnalysisTool {
    input_text: String,
    control_text: String,
    observations: Vec<Observation>,
    control_observations: Vec<Observation>,
    curve: Vec<TimePoint>,
    control_curve: Vec<TimePoint>,
    arms: Vec<ArmSurvivalSummary>,
    hazard_ratio: Option<Result<f64, String>>,
    log_rank: Option<Result<LogRankResult, String>>,
    error_message: Option<String>,
    dialog_io: SurvivalDialogIo,
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
        self.control_observations.clear();
        self.curve.clear();
        self.control_curve.clear();
        self.arms.clear();
        self.hazard_ratio = None;
        self.log_rank = None;
        self.error_message = None;

        match parse_survival_with_control(&self.input_text, &self.control_text) {
            Ok(parsed) => {
                if let Some(first) = parsed.arms.first() {
                    self.observations = first.observations.clone();
                    self.curve = first.curve.clone();
                }
                if let Some(second) = parsed.arms.get(1) {
                    self.control_observations = second.observations.clone();
                    self.control_curve = second.curve.clone();
                }
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
            self.control_text.clear();
            self.recalculate();
        }
        ui.heading("Survival Cohort Input");
        ui.label("Format: 'time, event' or 'time, event, arm' (CSV/TSV)");
        self.render_io_toolbar(ui);
        ui.add_space(4.0);
        let mut changed = ui
            .add(
                egui::TextEdit::multiline(&mut self.input_text)
                    .desired_width(f32::INFINITY)
                    .desired_rows(10),
            )
            .changed();
        ui.collapsing("Optional Separate Control Arm ('time, event')", |ui| {
            changed |= ui
                .add(
                    egui::TextEdit::multiline(&mut self.control_text)
                        .desired_width(f32::INFINITY)
                        .desired_rows(5),
                )
                .changed();
            if ui.button("Load 2-Arm Demo Split").clicked() {
                self.input_text = DEFAULT_TREATMENT_TEXT.to_string();
                self.control_text = DEMO_CONTROL_TEXT.to_string();
                changed = true;
            }
        });
        if changed {
            self.recalculate();
        }
        self.render_metrics_summary(ui);
    }

    fn render_io_toolbar(&mut self, ui: &mut egui::Ui) {
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
                self.control_text.clear();
                self.recalculate();
            }
            if !self.control_text.is_empty() && ui.button("Single Arm").clicked() {
                self.control_text.clear();
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
        let primary_label = if self.control_observations.is_empty() {
            "Observations"
        } else {
            "Treatment N"
        };
        render_copyable_metric(ui, primary_label, &self.observations.len().to_string());
        if !self.control_observations.is_empty() {
            render_copyable_metric(
                ui,
                "Control N",
                &self.control_observations.len().to_string(),
            );
        }
        render_copyable_metric(ui, "Study Arms", &self.arms.len().to_string());
        let median_text = match self.median_survival_time() {
            Some(t) => format!("{t:.2}"),
            None => "Not reached".to_owned(),
        };
        render_copyable_metric(ui, "Median Survival", &median_text);
        if let Some(hr_res) = &self.hazard_ratio {
            match hr_res {
                Ok(hr) => {
                    render_copyable_metric(ui, "Hazard Ratio (HR)", &format!("{hr:.3}"));
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
            .x_axis_label("Follow-up Time")
            .y_axis_label("Survival Probability")
            .include_y(0.0)
            .include_y(1.05)
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
                            .color(color)
                            .width(2.2_f32),
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
                plot_ui.hline(
                    HLine::new("S(t) = 0.50", 0.50)
                        .color(palette.warning)
                        .style(LineStyle::Dashed { length: 8.0 }),
                );
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
            observations: Vec::new(),
            control_observations: Vec::new(),
            curve: Vec::new(),
            control_curve: Vec::new(),
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

impl SurvivalAnalysisTool {
    pub(crate) fn median_survival_time(&self) -> Option<f64> {
        self.curve
            .iter()
            .find(|pt| pt.survival_probability <= 0.5 + 1e-12)
            .map(|pt| pt.time)
    }

    #[cfg(test)]
    pub(crate) fn build_step_points(&self) -> Vec<[f64; 2]> {
        build_km_step_points(&self.curve)
    }
}

// [cite:clinical_trials_statistics]

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
            "Kaplan-Meier non-parametric maximum likelihood estimator of survival function $S(t) = \\prod_{{t_i \\le t}} \\left(1 - \\frac{{d_i}}{{n_i}}\\right)$ across {} primary and {} control observations, paired with Mantel-Haenszel log-rank testing and incidence-rate hazard ratio estimation.",
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
mod tests;
