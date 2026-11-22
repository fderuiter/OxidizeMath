#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::framework::{ExportCapabilities, InteractiveTool};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{HLine, Legend, Line, Plot, PlotPoints, Points};
use math_explorer::applied::clinical_trials::sample_size::{
    calculate_power_means, calculate_power_proportions, calculate_sample_size_means,
    calculate_sample_size_proportions,
};

#[derive(Debug, PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum CalculationMode {
    Means,
    Proportions,
}

pub struct SampleSizeCalculatorTool {
    mode: CalculationMode,
    alpha: f64,
    power: f64,
    delta: f64,
    sigma: f64,
    p1: f64,
    p2: f64,
    result: Result<usize, String>,
}

impl Default for SampleSizeCalculatorTool {
    fn default() -> Self {
        let mut tool = Self {
            mode: CalculationMode::Means,
            alpha: 0.05,
            power: 0.80,
            delta: 0.5,
            sigma: 1.0,
            p1: 0.5,
            p2: 0.6,
            result: Ok(0),
        };
        tool.recalculate();
        tool
    }
}

impl SampleSizeCalculatorTool {
    fn recalculate(&mut self) {
        self.result = match self.mode {
            CalculationMode::Means => {
                calculate_sample_size_means(self.alpha, self.power, self.delta, self.sigma)
                    .map_err(|e| e.to_string())
            }
            CalculationMode::Proportions => {
                calculate_sample_size_proportions(self.alpha, self.power, self.p1, self.p2)
                    .map_err(|e| e.to_string())
            }
        };
    }

    fn evaluate_power_at(&self, n_per_group: usize) -> Option<f64> {
        match self.mode {
            CalculationMode::Means => {
                calculate_power_means(self.alpha, n_per_group, self.delta, self.sigma).ok()
            }
            CalculationMode::Proportions => {
                calculate_power_proportions(self.alpha, n_per_group, self.p1, self.p2).ok()
            }
        }
    }

    fn build_power_curve(&self) -> Vec<[f64; 2]> {
        let req_n = self.result.as_ref().copied().unwrap_or(100);
        let max_n = req_n.saturating_mul(2).clamp(24, 2_000);
        let steps = 64usize;
        let mut pts = Vec::with_capacity(steps);
        for idx in 0..steps {
            let n = 2 + ((max_n - 2) * idx) / (steps - 1);
            if let Some(pwr) = self.evaluate_power_at(n) {
                pts.push([n as f64, pwr]);
            }
        }
        pts
    }

    fn render_common_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Mode:");
            changed |= ui
                .radio_value(&mut self.mode, CalculationMode::Means, "Means (t-test)")
                .changed();
            changed |= ui
                .radio_value(
                    &mut self.mode,
                    CalculationMode::Proportions,
                    "Proportions (Z-test)",
                )
                .changed();
        });
        ui.separator();
        ui.group(|ui| {
            ui.heading("Statistical Targets");
            changed |= ui
                .add(egui::Slider::new(&mut self.alpha, 0.001..=0.20).text("Alpha (α)"))
                .accessible_hover_text("Two-sided Type I error rate (e.g., 0.05)")
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.power, 0.5..=0.999).text("Target Power (1-β)"))
                .accessible_hover_text("Target probability of detecting an effect (e.g., 0.80)")
                .changed();
        });
        changed
    }

    fn render_mode_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.group(|ui| match self.mode {
            CalculationMode::Means => {
                ui.heading("Continuous Means");
                changed |= ui
                    .add(egui::Slider::new(&mut self.delta, 0.01..=5.0).text("Effect Size (δ)"))
                    .accessible_hover_text("Minimum clinically important difference between arms")
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.sigma, 0.1..=10.0).text("Std Dev (σ)"))
                    .accessible_hover_text("Assumed within-group standard deviation")
                    .changed();
            }
            CalculationMode::Proportions => {
                ui.heading("Binary Proportions");
                changed |= ui
                    .add(egui::Slider::new(&mut self.p1, 0.01..=0.99).text("Control (p₁)"))
                    .accessible_hover_text("Expected event proportion in Control arm")
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.p2, 0.01..=0.99).text("Treatment (p₂)"))
                    .accessible_hover_text("Expected event proportion in Treatment arm")
                    .changed();
            }
        });
        changed
    }

    fn render_summary_metrics(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        match &self.result {
            Ok(n) => {
                render_copyable_metric(ui, "Required N / Arm", &n.to_string());
                render_copyable_metric(ui, "Total Study N", &(n * 2).to_string());
                if let Some(achieved) = self.evaluate_power_at(*n) {
                    render_copyable_metric(
                        ui,
                        "Achieved Power",
                        &format!("{:.1}%", achieved * 100.0),
                    );
                }
            }
            Err(e) => {
                ui.colored_label(palette.error, format!("Error: {e}"));
            }
        }
    }

    fn render_power_curve_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let curve = self.build_power_curve();
        let req_marker = self
            .result
            .as_ref()
            .ok()
            .and_then(|&n| self.evaluate_power_at(n).map(|p| vec![[n as f64, p]]))
            .unwrap_or_default();
        let target_power = self.power;

        Plot::new("sample_size_analytical_power_plot")
            .legend(Legend::default())
            .x_axis_label("Sample Size per Arm (N)")
            .y_axis_label("Statistical Power (1 - β)")
            .include_y(0.0)
            .include_y(1.0)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Analytical Power Curve", PlotPoints::new(curve))
                        .color(palette.accent)
                        .width(2.5_f32),
                );
                plot_ui.hline(
                    HLine::new("Target Power (1 - β)", target_power)
                        .color(palette.warning)
                        .style(egui_plot::LineStyle::dashed_dense()),
                );
                if !req_marker.is_empty() {
                    plot_ui.points(
                        Points::new("Required N per Arm", PlotPoints::new(req_marker))
                            .color(palette.success)
                            .radius(6.0_f32),
                    );
                }
            });
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SampleSizeState {
    mode: CalculationMode,
    alpha: f64,
    power: f64,
    delta: f64,
    sigma: f64,
    p1: f64,
    p2: f64,
}

impl InteractiveTool for SampleSizeCalculatorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Sample Size Calculator"
    }
    fn domain(&self) -> &'static str {
        "clinical_trials"
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut out = Vec::new();
        let curve = self.build_power_curve();
        if !curve.is_empty() {
            out.push(("Analytical Power Curve".to_string(), curve));
        }
        if let Ok(req_n) = self.result {
            if let Some(pwr) = self.evaluate_power_at(req_n) {
                out.push(("Required N per Arm".to_string(), vec![[req_n as f64, pwr]]));
            }
        }
        out
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&SampleSizeState {
            mode: self.mode,
            alpha: self.alpha,
            power: self.power,
            delta: self.delta,
            sigma: self.sigma,
            p1: self.p1,
            p2: self.p2,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: SampleSizeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid sample size state: {e}"))?;
        let valid = [s.alpha, s.power, s.delta, s.sigma, s.p1, s.p2]
            .iter()
            .all(|v| v.is_finite())
            && (0.001..=0.20).contains(&s.alpha)
            && (0.5..=0.999).contains(&s.power)
            && (0.01..=5.0).contains(&s.delta)
            && (0.1..=10.0).contains(&s.sigma)
            && (0.01..=0.99).contains(&s.p1)
            && (0.01..=0.99).contains(&s.p2);
        if !valid {
            return Err("Sample size parameters out of bounds".into());
        }
        let result = match s.mode {
            CalculationMode::Means => {
                calculate_sample_size_means(s.alpha, s.power, s.delta, s.sigma)
                    .map_err(|e| e.to_string())?
            }
            CalculationMode::Proportions => {
                calculate_sample_size_proportions(s.alpha, s.power, s.p1, s.p2)
                    .map_err(|e| e.to_string())?
            }
        };
        self.mode = s.mode;
        self.alpha = s.alpha;
        self.power = s.power;
        self.delta = s.delta;
        self.sigma = s.sigma;
        self.p1 = s.p1;
        self.p2 = s.p2;
        self.result = Ok(result);
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "sample_size_controls_panel");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("Sample Size Sizing");
            let mut changed = self.render_common_controls(ui);
            ui.add_space(6.0);
            changed |= self.render_mode_controls(ui);
            if changed {
                self.recalculate();
            }
            ui.separator();
            self.render_summary_metrics(ui);
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Analytical Statistical Power by Sample Size");
            self.render_power_curve_plot(ui);
        });
    }
}

// [cite:clinical_trials_statistics]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SampleSizeCalculatorTool",
        display_name: "Sample Size Calculator",
        domain: "clinical_trials",
        tags: &["power", "sample-size", "clinical-trials"],
        build: || Box::new(SampleSizeCalculatorTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for SampleSizeCalculatorTool {
    fn theory_description(&self) -> String {
        match self.mode {
            CalculationMode::Means => format!(
                "Two-sample hypothesis test sample size for means: $N = \\frac{{2 (Z_{{\\alpha/2}} + Z_\\beta)^2 \\sigma^2}}{{\\delta^2}}$ with significance $\\alpha = {:.3}$, power $1-\\beta = {:.2}$, effect size $\\delta = {:.2}$, $\\sigma = {:.2}$.",
                self.alpha, self.power, self.delta, self.sigma
            ),
            CalculationMode::Proportions => format!(
                "Two-sample hypothesis test sample size for proportions: $N = \\frac{{(Z_{{\\alpha/2}} \\sqrt{{2\\bar{{p}}(1-\\bar{{p}})}} + Z_\\beta \\sqrt{{p_1(1-p_1) + p_2(1-p_2)}})^2}}{{(p_1 - p_2)^2}}$ with $\\alpha = {:.3}$, power $1-\\beta = {:.2}$, $p_1 = {:.2}$, $p_2 = {:.2}$.",
                self.alpha, self.power, self.p1, self.p2
            ),
        }
    }
    fn phonetic_description(&self) -> String {
        "Clinical trial sample size and statistical power theory".into()
    }
    fn theory_citation(&self) -> String {
        "Sample Size Calculations in Clinical Research (Chow et al., 2017)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Means".into(),
            "Normal approximation for difference between two independent continuous means".into(),
        );
        map.insert(
            "Proportions".into(),
            "Normal approximation with pooled variance for binary outcome proportions".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Statistical Power & Sample Size",
    "domain": "clinical_trials",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "mode": "Means",
    "alpha": 0.05,
    "power": 0.85,
    "delta": 0.5,
    "sigma": 1.0,
    "p1": 0.5,
    "p2": 0.65
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "sample_size_starter",
        title: "Statistical Power & Sample Size",
        description: "N = 2(z_α + z_β)²σ² / δ²: Neyman-Pearson hypothesis test sizing for minimum detectable effect.",
        category: "Trial Design",
        icon: "👥",
        domain: "clinical_trials",
        tool_name: Some("SampleSizeCalculatorTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
