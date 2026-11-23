use crate::framework::InteractiveTool;
use crate::framework::empirical_csv::{
    EmpiricalCsvState, EmpiricalCsvTable, EmpiricalPanelOptions,
};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{HLine, Line, MarkerShape, Plot, PlotPoints, PlotUi, Points};
use math_explorer::applied::battery_degradation::{
    Capacity, Cycles, DepthOfDischarge, PowerLawModel,
};
use std::hash::{Hash, Hasher};

#[path = "capacity_fade_calibration.rs"]
mod calibration;

const DEFAULT_ALPHA: f64 = 1.019e5;
const DEFAULT_BETA: f64 = -1.2639;
pub(super) const MIN_ALPHA: f64 = 1_000.0;
pub(super) const MAX_ALPHA: f64 = 2_000_000.0;
pub(super) const MIN_BETA: f64 = -4.0;
pub(super) const MAX_BETA: f64 = -0.1;

fn default_alpha() -> f64 {
    DEFAULT_ALPHA
}

fn default_beta() -> f64 {
    DEFAULT_BETA
}

fn is_default_alpha(val: &f64) -> bool {
    (*val - DEFAULT_ALPHA).abs() <= 1e-9
}

fn is_default_beta(val: &f64) -> bool {
    (*val - DEFAULT_BETA).abs() <= 1e-9
}

pub struct CapacityFadeTool {
    pub(crate) dod: f64,
    pub(crate) temperature: f64,
    pub(crate) cycles_to_simulate: f64,
    pub(crate) alpha: f64,
    pub(crate) beta: f64,
    pub(crate) empirical_csv: EmpiricalCsvState,
    revision: u64,
}

impl Default for CapacityFadeTool {
    fn default() -> Self {
        Self {
            dod: 80.0,
            temperature: 25.0,
            cycles_to_simulate: 2000.0,
            alpha: DEFAULT_ALPHA,
            beta: DEFAULT_BETA,
            empirical_csv: EmpiricalCsvState::default(),
            revision: 1,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CapacityFadeState {
    dod: f64,
    temperature: f64,
    cycles_to_simulate: f64,
    #[serde(default = "default_alpha", skip_serializing_if = "is_default_alpha")]
    alpha: f64,
    #[serde(default = "default_beta", skip_serializing_if = "is_default_beta")]
    beta: f64,
}

impl CapacityFadeTool {
    fn render_sidebar(&mut self, ui: &mut egui::Ui) {
        ui.heading("Parameters");
        ui.add_space(5.0);
        ui.add(
            egui::Slider::new(&mut self.dod, 0.0..=100.0).text("Depth of Discharge (DoD) - %"),
        );
        ui.small("Percentage of battery capacity used per cycle.");
        ui.add_space(8.0);
        ui.add(
            egui::Slider::new(&mut self.alpha, MIN_ALPHA..=MAX_ALPHA)
                .logarithmic(true)
                .text("Power-Law Scale (alpha)"),
        );
        ui.add(egui::Slider::new(&mut self.beta, MIN_BETA..=MAX_BETA).text("DoD Exponent (beta)"));
        if ui.small_button("Reset Standard Model").clicked() {
            self.alpha = DEFAULT_ALPHA;
            self.beta = DEFAULT_BETA;
        }
        ui.add_space(8.0);
        ui.add_enabled_ui(false, |ui| {
            ui.add(egui::Slider::new(&mut self.temperature, -20.0..=60.0).text("Temperature (°C)"));
        })
        .response
        .on_disabled_hover_text("Temperature dependency is not yet implemented in the core model.");
        ui.add_space(8.0);
        ui.add(
            egui::Slider::new(&mut self.cycles_to_simulate, 100.0..=10_000.0)
                .text("Simulation Range - Cycles"),
        );
        ui.separator();
        let opts = EmpiricalPanelOptions {
            id_salt: "capacity_fade_csv",
            x_label: "Cycles Column (X)",
            y_label: "Capacity / SOH Column (Y)",
            aux_label: Some("Optional DoD (%) Column"),
            residual_filename: "capacity_fade_residuals.csv",
            fit_button_label: "Fit Parameters",
        };
        if self.empirical_csv.show_panel(ui, &opts) {
            let ctx = ui.ctx().clone();
            let _ = self.calibrate_from_loaded_csv(Some(&ctx));
        }
    }

    fn render_empirical_points(&self, plot_ui: &mut PlotUi<'_>, color: egui::Color32) {
        let Some(ref table) = self.empirical_csv.table else {
            return;
        };
        let mut pts = table.scatter_points(self.empirical_csv.x_col, self.empirical_csv.y_col);
        if pts.is_empty() {
            return;
        }
        let max_y = pts.iter().fold(0.0_f64, |acc, p| acc.max(p[1]));
        if max_y > 1.5 {
            for pt in &mut pts {
                pt[1] /= 100.0;
            }
        }
        plot_ui.points(
            Points::new(format!("Measured Fade ({})", table.name), pts)
                .radius(4.0_f32)
                .shape(MarkerShape::Circle)
                .color(color),
        );
    }
}

impl InteractiveTool for CapacityFadeTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Capacity Fade"
    }
    fn domain(&self) -> &'static str {
        "battery_degradation"
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut h);
        self.dod.to_bits().hash(&mut h);
        self.temperature.to_bits().hash(&mut h);
        self.cycles_to_simulate.to_bits().hash(&mut h);
        self.alpha.to_bits().hash(&mut h);
        self.beta.to_bits().hash(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        [
            self.dod,
            self.temperature,
            self.cycles_to_simulate,
            self.alpha,
            self.beta,
        ]
        .into_iter()
        .all(f64::is_finite)
    }

    fn supports_empirical_csv(&self) -> bool {
        true
    }

    fn ingest_empirical_dataset(&mut self, dataset: EmpiricalCsvTable) -> Result<String, String> {
        let auto_dod = dataset.headers.iter().enumerate().skip(2).find_map(|(i, h)| {
            let low = h.to_ascii_lowercase();
            (low.contains("dod") || low.contains("depth")).then_some(i)
        });
        self.revision = self.revision.wrapping_add(1).max(1);
        let summary = self.empirical_csv.set_table(dataset);
        self.empirical_csv.aux_col = auto_dod;
        Ok(summary)
    }

    fn notify_empirical_csv_error(&mut self, error: String) {
        self.empirical_csv.record_error(None, error);
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&CapacityFadeState {
            dod: self.dod,
            temperature: self.temperature,
            cycles_to_simulate: self.cycles_to_simulate,
            alpha: self.alpha,
            beta: self.beta,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: CapacityFadeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid capacity fade state: {e}"))?;
        if !s.dod.is_finite()
            || !(0.0..=100.0).contains(&s.dod)
            || !s.temperature.is_finite()
            || !(-20.0..=60.0).contains(&s.temperature)
            || !s.cycles_to_simulate.is_finite()
            || !(100.0..=10_000.0).contains(&s.cycles_to_simulate)
            || !s.alpha.is_finite()
            || !(MIN_ALPHA..=MAX_ALPHA).contains(&s.alpha)
            || !s.beta.is_finite()
            || !(MIN_BETA..=MAX_BETA).contains(&s.beta)
        {
            return Err("Capacity fade parameters out of bounds".into());
        }
        self.dod = s.dod;
        self.temperature = s.temperature;
        self.cycles_to_simulate = s.cycles_to_simulate;
        self.alpha = s.alpha;
        self.beta = s.beta;
        self.revision = self.revision.wrapping_add(1).max(1);
        Ok(())
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![("Capacity".to_string(), self.capacity_curve_points())]
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.dod = self.dod.clamp(0.0, 100.0);
        self.cycles_to_simulate = self.cycles_to_simulate.clamp(100.0, 10_000.0);
        self.alpha = self.alpha.clamp(MIN_ALPHA, MAX_ALPHA);
        self.beta = self.beta.clamp(MIN_BETA, MAX_BETA);
        self.render_controls_panel(ui);
        self.render_projection_panel(ui);
    }
}

impl CapacityFadeTool {
    pub(crate) fn capacity_curve_points(&self) -> Vec<[f64; 2]> {
        let model = PowerLawModel::new(self.alpha, self.beta);
        let dod_val = DepthOfDischarge::new(self.dod).ok();
        (0..=100)
            .map(|i| {
                let cycle = (self.cycles_to_simulate / 100.0) * (i as f64);
                if let (Ok(c), Some(d)) = (Cycles::new(cycle), dod_val) {
                    [cycle, model.capacity(c, d).as_f64()]
                } else {
                    [cycle, 0.0]
                }
            })
            .collect()
    }

    pub(crate) fn projected_life_cycles_for_model(
        model: &PowerLawModel,
        dod_percent: f64,
    ) -> (f64, f64) {
        if let Ok(d) = DepthOfDischarge::new(dod_percent) {
            let n80 = model
                .cycles_to_capacity(Capacity::new_clamped(0.8), d)
                .as_f64();
            let n70 = model.n70(d).as_f64();
            (n80, n70)
        } else {
            (0.0, 0.0)
        }
    }

    #[cfg(test)]
    pub(crate) fn projected_life_cycles(dod_percent: f64) -> (f64, f64) {
        let model = PowerLawModel::standard();
        Self::projected_life_cycles_for_model(&model, dod_percent)
    }

    fn render_controls_panel(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "battery_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.render_sidebar(ui);
            });
        });
    }

    fn render_projection_panel(&self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.render_fade_plot_and_metrics(ui);
        });
    }

    fn render_fade_plot_and_metrics(&self, ui: &mut egui::Ui) {
        ui.heading("Capacity Fade Projection");
        ui.label("Estimation based on calibrated Power Law Model for Li-ion batteries.");

        let model = PowerLawModel::new(self.alpha, self.beta);
        let points = self.capacity_curve_points();
        let line = Line::new("Capacity", PlotPoints::new(points));
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        Plot::new("capacity_fade_plot")
            .view_aspect(2.0)
            .x_axis_label("Cycles")
            .y_axis_label("State of Health (Capacity)")
            .include_y(0.0)
            .include_y(1.0)
            .show(ui, |plot_ui| {
                plot_ui.line(line);
                plot_ui.hline(
                    HLine::new("End of Life (70%)", 0.7)
                        .color(palette.error)
                        .style(egui_plot::LineStyle::Dashed { length: 10.0 }),
                );
                plot_ui.hline(
                    HLine::new("First Life (80%)", 0.8)
                        .color(palette.warning)
                        .style(egui_plot::LineStyle::Dashed { length: 10.0 }),
                );
                self.render_empirical_points(plot_ui, palette.series_color(2));
            });

        ui.add_space(10.0);
        let (n80, n70) = Self::projected_life_cycles_for_model(&model, self.dod);
        render_copyable_metric(ui, "First Life (80%)", &format!("{n80:.0} cycles"));
        render_copyable_metric(
            ui,
            "Projected Cycle Life (to 70%)",
            &format!("{n70:.0} cycles"),
        );
    }
}

// [cite:mmwave_radiotherapy_setup]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "CapacityFadeTool",
        display_name: "Capacity Fade",
        domain: "battery_degradation",
        tags: &["battery", "calibration", "power-law"],
        build: || Box::new(CapacityFadeTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for CapacityFadeTool {
    fn theory_description(&self) -> String {
        format!(
            "Lithium-ion capacity fade power-law degradation: $N_{{70}} = \\alpha \\cdot \\text{{DoD}}^{{\\beta}}$ ($\\alpha = {:.0}$, $\\beta = {:.4}$) with remaining capacity $C(N) = 0.7^{{N / N_{{70}}}}$ at {:.1}% depth of discharge.",
            self.alpha, self.beta, self.dod
        )
    }
    fn phonetic_description(&self) -> String {
        "Battery capacity fade power-law degradation theory".into()
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
    "title": "Capacity Fade",
    "domain": "battery_degradation",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "dod": 80.0,
    "temperature": 25.0,
    "cycles_to_simulate": 2500.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "capacity_fade_starter",
        title: "Capacity Fade",
        description: "Power-law lithium-ion state-of-health capacity retention curve across charge-discharge cycles.",
        category: "Electrochemistry",
        icon: "🔋",
        domain: "battery_degradation",
        tool_name: Some("CapacityFadeTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::InteractiveTool;

    #[test]
    fn test_first_life_80_threshold_error_and_power_law_fit() {
        let mut tool = CapacityFadeTool::default();
        let (n80, n70) = CapacityFadeTool::projected_life_cycles(tool.dod);
        assert!(n80 > 0.0 && n80 < n70);
        tool.cycles_to_simulate = (n70 * 1.2).max(2000.0);

        let datasets = tool.export_datasets();
        let cap_pts: Vec<egui_plot::PlotPoint> = datasets[0]
            .1
            .iter()
            .map(|p| egui_plot::PlotPoint::new(p[0], p[1]))
            .collect();
        let crossings = egui_plot::poi::interpolate_threshold_crossings(&cap_pts, 0.8);
        assert_eq!(crossings.len(), 1);
        assert!(
            (crossings[0] - n80).abs() < 0.5,
            "80% threshold cycle error = {} cycles",
            (crossings[0] - n80).abs()
        );

        let n70_curve: Vec<[f64; 2]> = (10..=100)
            .step_by(5)
            .map(|dod| {
                let (_, life70) = CapacityFadeTool::projected_life_cycles(dod as f64);
                [dod as f64, life70]
            })
            .collect();
        let fit = egui_plot::fit_model_with_axes(
            &n70_curve,
            egui_plot::FitType::PowerLaw,
            "PowerLawModel::n70",
            false,
            false,
            false,
        )
        .expect("power law fit");
        assert!(fit.r_squared >= 0.99);
    }
}

