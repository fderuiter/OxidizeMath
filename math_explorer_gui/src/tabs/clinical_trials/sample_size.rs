#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::framework::{ExportCapabilities, InteractiveTool};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, Points};
use math_explorer::applied::clinical_trials::sample_size::{
    calculate_sample_size_means, calculate_sample_size_proportions,
};

use super::sample_size_cards::{
    evaluate_hypothesis_cards, render_contingency_and_risk_card, render_t_test_card,
    HypothesisCardsEval, HypothesisCardsState,
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
    cards: HypothesisCardsState,
    result: Result<usize, String>,
    per_arm_curve: Vec<[f64; 2]>,
    total_curve: Vec<[f64; 2]>,
    cards_eval: HypothesisCardsEval,
}

impl Default for SampleSizeCalculatorTool {
    fn default() -> Self {
        let cards = HypothesisCardsState::default();
        let cards_eval = evaluate_hypothesis_cards(&cards, 0.05);
        let mut tool = Self {
            mode: CalculationMode::Means,
            alpha: 0.05,
            power: 0.80,
            delta: 0.5,
            sigma: 1.0,
            p1: 0.5,
            p2: 0.6,
            cards,
            result: Ok(0),
            per_arm_curve: Vec::new(),
            total_curve: Vec::new(),
            cards_eval,
        };
        tool.recalculate();
        tool
    }
}

impl SampleSizeCalculatorTool {
    fn sample_size_at_power(&self, power_val: f64) -> Result<usize, String> {
        match self.mode {
            CalculationMode::Means => {
                calculate_sample_size_means(self.alpha, power_val, self.delta, self.sigma)
                    .map_err(|e| e.to_string())
            }
            CalculationMode::Proportions => {
                calculate_sample_size_proportions(self.alpha, power_val, self.p1, self.p2)
                    .map_err(|e| e.to_string())
            }
        }
    }

    fn recalculate(&mut self) {
        self.result = self.sample_size_at_power(self.power);
        self.per_arm_curve.clear();
        self.total_curve.clear();
        let steps = 24usize;
        for idx in 0..=steps {
            let pwr = 0.50 + 0.48 * (idx as f64 / steps as f64);
            if let Ok(n) = self.sample_size_at_power(pwr) {
                self.per_arm_curve.push([pwr, n as f64]);
                self.total_curve.push([pwr, (n * 2) as f64]);
            }
        }
        self.cards_eval = evaluate_hypothesis_cards(&self.cards, self.alpha);
    }

    fn render_sizing_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("Sample Size & Hypothesis Suite");
        ui.horizontal_wrapped(|ui| {
            changed |= ui
                .radio_value(&mut self.mode, CalculationMode::Means, "Means (t-test)")
                .changed();
            changed |= ui
                .radio_value(
                    &mut self.mode,
                    CalculationMode::Proportions,
                    "Proportions (χ²)",
                )
                .changed();
        });
        ui.group(|ui| {
            ui.strong("Power & Significance");
            changed |= ui
                .add(egui::Slider::new(&mut self.alpha, 0.001..=0.20).text("α (Type I Error)"))
                .accessible_hover_text("Two-sided significance level α")
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.power, 0.50..=0.999).text("1 - β (Power)"))
                .accessible_hover_text("Target statistical power 1 - β")
                .changed();
        });
        ui.group(|ui| match self.mode {
            CalculationMode::Means => {
                ui.strong("Continuous Endpoint Parameters");
                changed |= ui
                    .add(egui::Slider::new(&mut self.delta, 0.05..=5.0).text("Effect Size δ"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.sigma, 0.1..=10.0).text("Std Dev σ"))
                    .changed();
            }
            CalculationMode::Proportions => {
                ui.strong("Binary Proportion Parameters");
                changed |= ui
                    .add(egui::Slider::new(&mut self.p1, 0.01..=0.99).text("Control p₁"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.p2, 0.01..=0.99).text("Treatment p₂"))
                    .changed();
            }
        });
        changed
    }

    fn render_side_panel(&mut self, ui: &mut egui::Ui) {
        let mut changed = self.render_sizing_controls(ui);
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        match &self.result {
            Ok(n) => {
                render_copyable_metric(ui, "Required N per Group", &n.to_string());
                render_copyable_metric(ui, "Total Participants (2N)", &(n * 2).to_string());
            }
            Err(e) => {
                ui.colored_label(palette.error, format!("Error: {e}"));
            }
        }
        ui.separator();
        changed |= render_t_test_card(ui, &mut self.cards, &self.cards_eval);
        ui.add_space(4.0);
        changed |= render_contingency_and_risk_card(ui, &mut self.cards, &self.cards_eval);
        if changed {
            self.recalculate();
        }
    }

    fn render_power_curve_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Statistical Power vs Required Sample Size Curve");
        Plot::new("sample_size_power_curve_plot")
            .legend(Legend::default())
            .view_aspect(1.9)
            .x_axis_label("Statistical Power (1 - β)")
            .y_axis_label("Required Sample Size")
            .show(ui, |plot_ui| {
                if self.per_arm_curve.is_empty() {
                    plot_ui
                        .line(Line::new("Invalid Parameters", vec![[0.5, 0.0]]).color(palette.error));
                    return;
                }
                plot_ui.line(
                    Line::new(
                        "Sample Size per Group (n)",
                        PlotPoints::new(self.per_arm_curve.clone()),
                    )
                    .color(palette.series_color(0)),
                );
                plot_ui.line(
                    Line::new(
                        "Total Study Size (2n)",
                        PlotPoints::new(self.total_curve.clone()),
                    )
                    .color(palette.series_color(1)),
                );
                if let Ok(active_n) = self.result {
                    plot_ui.points(
                        Points::new("Target Operating Point", vec![[self.power, active_n as f64]])
                            .radius(5.0_f32)
                            .color(palette.success),
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
    #[serde(default, skip_serializing_if = "HypothesisCardsState::is_default")]
    cards: HypothesisCardsState,
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
        if self.per_arm_curve.is_empty() {
            return Vec::new();
        }
        vec![
            ("Sample Size per Group (n)".into(), self.per_arm_curve.clone()),
            ("Total Study Size (2n)".into(), self.total_curve.clone()),
        ]
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
            cards: self.cards.clone(),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: SampleSizeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid sample size state: {e}"))?;
        if ![s.alpha, s.power, s.delta, s.sigma, s.p1, s.p2]
            .iter()
            .all(|v| v.is_finite())
            || !(0.001..=0.20).contains(&s.alpha)
            || !(0.5..=0.999).contains(&s.power)
            || !(0.01..=5.0).contains(&s.delta)
            || !(0.1..=10.0).contains(&s.sigma)
            || !(0.01..=0.99).contains(&s.p1)
            || !(0.01..=0.99).contains(&s.p2)
        {
            return Err("Sample size parameters out of bounds".into());
        }
        s.cards.validate()?;
        let result = match s.mode {
            CalculationMode::Means => calculate_sample_size_means(s.alpha, s.power, s.delta, s.sigma)
                .map_err(|e| e.to_string())?,
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
        self.cards = s.cards;
        self.recalculate();
        self.result = Ok(result);
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "sample_size_side_panel");
        crate::framework::PaneSidePanel::left(panel_id)
            .min_width(320.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.render_side_panel(ui);
                });
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
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
        tags: &[],
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
        description: "N = 2(z_α + z_β)²σ² / δ²: Neyman-Pearson hypothesis test sizing, power curves, and 2×2 risk metrics.",
        category: "Trial Design",
        icon: "👥",
        domain: "clinical_trials",
        tool_name: Some("SampleSizeCalculatorTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legacy_roundtrip_power_curves_and_hypothesis_cards() {
        let mut tool = SampleSizeCalculatorTool::default();
        let legacy =
            r#"{"mode":"Proportions","alpha":0.01,"power":0.9,"delta":0.4,"sigma":1.2,"p1":0.3,"p2":0.5}"#;
        tool.try_load_state(legacy)
            .expect("legacy proportions state loads");
        assert_eq!(tool.save_state().as_deref(), Some(legacy));

        let datasets = tool.export_datasets();
        assert_eq!(datasets.len(), 2);
        assert!(datasets.iter().all(|(_, pts)| pts.len() >= 20));

        let t_res = tool.cards_eval.t_test.as_ref().expect("t-test evaluated");
        assert!(t_res.statistic > 0.0 && t_res.is_significant);
        let chi_res = tool
            .cards_eval
            .chi_square
            .as_ref()
            .expect("chi-square evaluated");
        assert!(chi_res.statistic > 0.0);
        let rm = tool
            .cards_eval
            .risk_metrics
            .as_ref()
            .expect("risk metrics evaluated");
        assert!(rm.relative_risk < 1.0 && rm.odds_ratio < 1.0);
    }
}
