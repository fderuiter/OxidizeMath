use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::applied::battery_degradation::{Capacity, DepthOfDischarge, PowerLawModel};

pub struct LifetimeEstimatorTool {
    target_capacity: f64,
    dod: f64,
    cycles_per_day: f64,
}

impl Default for LifetimeEstimatorTool {
    fn default() -> Self {
        Self {
            target_capacity: 80.0,
            dod: 50.0,
            cycles_per_day: 1.0,
        }
    }
}

impl LifetimeEstimatorTool {
    fn compute_lifetime_table(&self) -> Option<crate::framework::ExportTable> {
        use crate::framework::export_table::{ExportCell, ExportTable};
        if !self.cycles_per_day.is_finite() || self.cycles_per_day <= 0.0 {
            return None;
        }
        let dod = DepthOfDischarge::new(self.dod).ok()?;
        let cap = Capacity::new(self.target_capacity / 100.0).ok()?;
        let total_cycles = PowerLawModel::standard()
            .cycles_to_capacity(cap, dod)
            .as_f64();
        let days = total_cycles / self.cycles_per_day;
        let years = days / 365.25;
        Some(ExportTable::new(
            "Battery Lifetime Estimate",
            vec![
                "dod_pct".into(),
                "target_capacity_pct".into(),
                "cycles_per_day".into(),
                "estimated_total_cycles".into(),
                "estimated_days".into(),
                "estimated_years".into(),
            ],
            vec![vec![
                ExportCell::Float(self.dod),
                ExportCell::Float(self.target_capacity),
                ExportCell::Float(self.cycles_per_day),
                ExportCell::Float(total_cycles),
                ExportCell::Float(days),
                ExportCell::Float(years),
            ]],
        ))
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LifetimeEstimatorState {
    target_capacity: f64,
    dod: f64,
    cycles_per_day: f64,
}

impl InteractiveTool for LifetimeEstimatorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Lifetime Estimator"
    }
    fn domain(&self) -> &'static str {
        "battery_degradation"
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        if self.compute_lifetime_table().is_some() {
            crate::framework::ExportCapabilities::dataset_only()
        } else {
            crate::framework::ExportCapabilities::unpopulated(
                "Configure valid depth-of-discharge, target capacity, and cycles per day before exporting.",
            )
        }
    }
    fn export_tables(&self) -> Vec<crate::framework::ExportTable> {
        self.compute_lifetime_table().into_iter().collect()
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&LifetimeEstimatorState {
            target_capacity: self.target_capacity,
            dod: self.dod,
            cycles_per_day: self.cycles_per_day,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: LifetimeEstimatorState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid lifetime estimator state: {e}"))?;
        if !s.target_capacity.is_finite()
            || !(10.0..=99.0).contains(&s.target_capacity)
            || !s.dod.is_finite()
            || !(0.1..=100.0).contains(&s.dod)
            || !s.cycles_per_day.is_finite()
            || !(0.1..=10.0).contains(&s.cycles_per_day)
        {
            return Err("Lifetime estimator parameters out of bounds".into());
        }
        self.target_capacity = s.target_capacity;
        self.dod = s.dod;
        self.cycles_per_day = s.cycles_per_day;
        Ok(())
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show_inside(&mut self, ui: &mut egui::Ui) {
        // Enforce valid ranges
        self.target_capacity = self.target_capacity.clamp(0.1, 99.9);
        self.dod = self.dod.clamp(0.1, 100.0);
        self.cycles_per_day = self.cycles_per_day.max(0.1);

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Battery Lifetime Estimator");
            ui.label("Calculate the expected battery life based on specific usage profiles.");
            ui.add_space(10.0);

            ui.group(|ui| {
                ui.heading("Usage Profile");
                egui::Grid::new("lifetime_params_grid").show(ui, |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.dod, 0.1..=100.0)
                            .text("Depth of Discharge (DoD): - %"),
                    )
                    .accessible_hover_text("Percentage of battery capacity used per cycle.");
                    ui.end_row();

                    ui.add(
                        egui::Slider::new(&mut self.target_capacity, 10.0..=99.0)
                            .text("Target Capacity (End of Life): - %"),
                    )
                    .accessible_hover_text(
                        "The threshold capacity to be considered 'end of life'.",
                    );
                    ui.end_row();

                    ui.add(
                        egui::Slider::new(&mut self.cycles_per_day, 0.1..=10.0)
                            .text("Cycles per Day: - cycles"),
                    )
                    .accessible_hover_text("How many charge/discharge cycles happen per day.");
                    ui.end_row();
                });
            });

            ui.add_space(20.0);
            ui.separator();
            ui.add_space(10.0);

            // Calculation
            let model = PowerLawModel::standard();
            let dod_val = DepthOfDischarge::new(self.dod);
            let capacity_val = Capacity::new(self.target_capacity / 100.0);

            let total_cycles = if let (Ok(c), Ok(d)) = (capacity_val, dod_val) {
                model.cycles_to_capacity(c, d).as_f64()
            } else {
                0.0
            };
            let lifetime_days = total_cycles / self.cycles_per_day;
            let lifetime_years = lifetime_days / 365.25;

            // Results Display
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Estimated Total Cycles:")
                        .size(18.0)
                        .strong(),
                );
                ui.label(
                    egui::RichText::new(format!("{:.0}", total_cycles))
                        .size(24.0)
                        .color(palette.success)
                        .strong(),
                );
            });

            ui.add_space(5.0);

            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Estimated Lifetime:")
                        .size(18.0)
                        .strong(),
                );
                ui.label(
                    egui::RichText::new(format!("{:.1} years", lifetime_years))
                        .size(24.0)
                        .color(palette.success)
                        .strong(),
                );
            });

            ui.label(format!("({:.0} days)", lifetime_days));
        });
    }
}

// [cite:algorithmic_information_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "LifetimeEstimatorTool",
        display_name: "Lifetime Estimator",
        domain: "battery_degradation",
        tags: &[],
        build: || Box::new(LifetimeEstimatorTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for LifetimeEstimatorTool {
    fn theory_description(&self) -> String {
        format!(
            "Battery cycle lifetime estimator modeling cycles-to-failure $N_{{70}}(D) = \\alpha D^\\beta$ for target capacity {:.1}% at {:.1}% depth of discharge.",
            self.target_capacity, self.dod
        )
    }
    fn phonetic_description(&self) -> String {
        "Battery lifetime estimator and power-law cycle life theory".into()
    }
    fn theory_citation(&self) -> String {
        "Capacity Fade Model for Li-ion Batteries (Smith et al., 2011)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Battery Lifetime Estimator",
    "domain": "battery_degradation",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "target_capacity": 80.0,
    "dod": 60.0,
    "cycles_per_day": 1.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lifetime_estimator_starter",
        title: "Battery Lifetime Estimator",
        description: "Power-law battery cycle life and calendar lifetime estimator from depth-of-discharge usage profiles.",
        category: "Battery Systems",
        icon: "⏳",
        domain: "battery_degradation",
        tool_name: Some("LifetimeEstimatorTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
