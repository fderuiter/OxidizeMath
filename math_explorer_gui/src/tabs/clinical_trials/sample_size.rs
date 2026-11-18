use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::applied::clinical_trials::sample_size::{
    calculate_sample_size_means, calculate_sample_size_proportions,
};

#[derive(Debug, PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum CalculationMode {
    Means,
    Proportions,
}

pub struct SampleSizeCalculatorTool {
    mode: CalculationMode,
    // Common Parameters
    alpha: f64,
    power: f64,
    // Means Parameters
    delta: f64,
    sigma: f64,
    // Proportions Parameters
    p1: f64,
    p2: f64,
    // Output
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
}

#[derive(serde::Serialize, serde::Deserialize)]
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
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Sample Size Calculator"
    }
    fn domain(&self) -> &'static str {
        "clinical_trials"
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

    fn load_state(&mut self, state: &str) {
        if let Ok(s) = serde_json::from_str::<SampleSizeState>(state) {
            self.mode = s.mode;
            self.alpha = s.alpha;
            self.power = s.power;
            self.delta = s.delta;
            self.sigma = s.sigma;
            self.p1 = s.p1;
            self.p2 = s.p2;
            self.recalculate();
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Sample Size Calculator");
            ui.label("Calculate the required sample size per group for a two-arm study.");
            ui.add_space(10.0);

            // Mode Selection
            ui.horizontal(|ui| {
                ui.label("Calculation Mode:");
                if ui
                    .radio_value(
                        &mut self.mode,
                        CalculationMode::Means,
                        "Compare Means (t-test)",
                    )
                    .changed()
                {
                    self.recalculate();
                }
                if ui
                    .radio_value(
                        &mut self.mode,
                        CalculationMode::Proportions,
                        "Compare Proportions (Chi-square)",
                    )
                    .changed()
                {
                    self.recalculate();
                }
            });

            ui.separator();

            let mut changed = false;

            // Common Inputs
            ui.group(|ui| {
                ui.heading("Statistical Parameters");
                egui::Grid::new("common_params_grid").show(ui, |ui| {
                    changed |= ui.add(egui::Slider::new(&mut self.alpha, 0.001..=0.20).text("Type I Error Rate (α): - Alpha"))
                        .accessible_hover_text("Significance level (e.g., 0.05 for 5%)")
                        .changed();
                    ui.end_row();

                    changed |= ui.add(egui::Slider::new(&mut self.power, 0.5..=0.999).text("Statistical Power (1-β): - Power"))
                        .accessible_hover_text(
                            "Probability of detecting an effect if it exists (e.g., 0.80)",
                        )
                        .changed();
                    ui.end_row();
                });
            });

            ui.add_space(5.0);

            // Mode-specific Inputs
            ui.group(|ui| match self.mode {
                CalculationMode::Means => {
                    ui.heading("Means Parameters");
                    egui::Grid::new("means_params_grid").show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut self.delta, 0.01..=5.0).text("Effect Size (δ): - Delta"))
                            .accessible_hover_text("Minimum difference to detect between groups")
                            .changed();
                        ui.end_row();

                        changed |= ui.add(egui::Slider::new(&mut self.sigma, 0.1..=10.0).text("Standard Deviation (σ): - Sigma"))
                            .accessible_hover_text("Assumed standard deviation of the population")
                            .changed();
                        ui.end_row();
                    });
                }
                CalculationMode::Proportions => {
                    ui.heading("Proportions Parameters");
                    egui::Grid::new("props_params_grid").show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut self.p1, 0.01..=0.99).text("Group 1 Proportion (p1): - p1"))
                            .accessible_hover_text("Expected proportion in Control group")
                            .changed();
                        ui.end_row();

                        changed |= ui.add(egui::Slider::new(&mut self.p2, 0.01..=0.99).text("Group 2 Proportion (p2): - p2"))
                            .accessible_hover_text("Expected proportion in Treatment group")
                            .changed();
                        ui.end_row();
                    });
                }
            });

            if changed {
                self.recalculate();
            }

            ui.add_space(20.0);
            ui.separator();
            ui.add_space(10.0);

            // Results Display
            match &self.result {
                Ok(n) => {
                    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Required Sample Size per Group:")
                                .size(18.0)
                                .strong(),
                        );
                        ui.label(
                            egui::RichText::new(n.to_string())
                                .size(24.0)
                                .color(palette.success)
                                .strong(),
                        );
                    });
                    ui.label(format!("Total participants needed: {}", n * 2));
                }
                Err(e) => {
                    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                    ui.colored_label(palette.error, format!("Error: {}", e));
                }
            }
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
    fn phonetic_description(&self) -> String { "Clinical trial sample size and statistical power theory".into() }
    fn theory_citation(&self) -> String { "Sample Size Calculations in Clinical Research (Chow et al., 2017)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("Means".into(), "Normal approximation for difference between two independent continuous means".into());
        map.insert("Proportions".into(), "Normal approximation with pooled variance for binary outcome proportions".into());
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
    "alpha": 0.05,
    "power": 0.8,
    "effect_size": 0.5
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
