#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{InteractiveTool, MathExpressionInputBar, TraceSeries, TraceSnapshotManager};
use crate::presets::{InteractivePreset, OdePreset, OdePresetConfig};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

mod annotations;
mod models;

use models::*;

// ----------------------------------------------------------------------------
// Tool Implementation
// ----------------------------------------------------------------------------

pub struct OdeSolverTool {
    preset: OdePreset,
    dt: f64,
    total_time: f64,
    param_k: f64,
    param_r: f64,
    param_cap_k: f64,
    ic_y0: f64,
    ic_v0: f64,
    time_series: Vec<f64>,
    y_series: Vec<f64>,
    v_series: Vec<f64>,
    diverged: bool,
    math_input: MathExpressionInputBar,
    snapshot_manager: TraceSnapshotManager,
    sensitivity_engine: crate::sensitivity::SensitivityEngine,
    last_param_summary: String,
    pub show_annotations: bool,
}

impl Default for OdeSolverTool {
    fn default() -> Self {
        let mut tool = Self {
            preset: OdePreset::Exponential,
            dt: 0.05,
            total_time: 10.0,
            param_k: 1.0,
            param_r: 1.0,
            param_cap_k: 10.0,
            ic_y0: 1.0,
            ic_v0: 0.0,
            time_series: Vec::new(),
            y_series: Vec::new(),
            v_series: Vec::new(),
            diverged: false,
            math_input: MathExpressionInputBar::new(
                "y'",
                "-k * y * sin(t)",
                vec![
                    ("y' = -k*y*sin(t) (Non-linear Sine)", "-k * y * sin(t)"),
                    ("y' = k*y (Exponential)", "k * y"),
                    ("y' = r*y*(1 - y/cap_k) (Logistic)", "r * y * (1 - y / cap_k)"),
                    ("y' = -k * y^2 (Stiff Decay)", "-k * y^2"),
                    ("y' = cos(t) - y (Forced Damped)", "cos(t) - y"),
                ],
            ),
            snapshot_manager: TraceSnapshotManager::new(),
            sensitivity_engine: crate::sensitivity::SensitivityEngine::new(),
            last_param_summary: String::new(),
            show_annotations: true,
        };
        tool.recalculate();
        tool
    }
}

impl OdeSolverTool {
    pub fn param_summary(&self) -> String {
        match self.preset {
            OdePreset::Exponential | OdePreset::HarmonicOscillator => format!(r"k={:.2}, \Delta t={:.2}", self.param_k, self.dt),
            OdePreset::LogisticGrowth => format!(r"r={:.2}, K={:.2}, \Delta t={:.2}", self.param_r, self.param_cap_k, self.dt),
            OdePreset::Custom => format!(r"k={:.2}, \Delta t={:.2}", self.param_k, self.dt),
        }
    }
    #[must_use]
    #[allow(dead_code)]
    pub fn diverged(&self) -> bool {
        self.diverged
    }

    pub fn reset_parameters(&mut self) {
        match self.preset {
            OdePreset::Exponential => { self.param_k = 1.0; self.ic_y0 = 1.0; }
            OdePreset::HarmonicOscillator => { self.param_k = 1.0; self.ic_y0 = 1.0; self.ic_v0 = 0.0; }
            OdePreset::LogisticGrowth => { self.param_r = 1.0; self.param_cap_k = 10.0; self.ic_y0 = 1.0; }
            OdePreset::Custom => { self.param_k = 1.0; self.ic_y0 = 1.0; self.math_input.set_expression("-k * y * sin(t)"); }
        }
        self.dt = 0.05;
        self.total_time = 10.0;
        self.recalculate();
    }

    fn recalculate(&mut self) {
        let res = run_simulation(OdeSimConfig {
            preset: self.preset,
            dt: self.dt,
            total_time: self.total_time,
            k: self.param_k,
            r: self.param_r,
            cap_k: self.param_cap_k,
            y0: self.ic_y0,
            v0: self.ic_v0,
            custom_expr: self.math_input.expr(),
        });
        self.time_series = res.time_series;
        self.y_series = res.y_series;
        self.v_series = res.v_series;
        self.diverged = res.diverged;
        self.last_param_summary = self.param_summary();
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct OdeState {
    #[serde(default)]
    preset: Option<OdePreset>,
    dt: f64,
    total_time: f64,
    param_k: f64,
    param_r: f64,
    param_cap_k: f64,
    ic_y0: f64,
    ic_v0: f64,
    #[serde(default)]
    custom_expr: String,
    #[serde(default)]
    snapshot_manager: TraceSnapshotManager,
    #[serde(default = "default_annotations_enabled")]
    show_annotations: bool,
}

fn default_annotations_enabled() -> bool {
    true
}

impl InteractiveTool for OdeSolverTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str { "ODE Solvers" }
    fn domain(&self) -> &'static str { "analysis" }

    fn export_preset_json(&self) -> Result<String, String> { self.to_json_string() }
    fn import_preset_json(&mut self, json_str: &str) -> Result<(), String> {
        let res = self.from_json_str(json_str);
        if res.is_ok() { self.recalculate(); }
        res
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = egui_plot::get_last_plot_datasets();
        if datasets.is_empty() {
            let mut plot_points_y = Vec::new();
            let mut plot_points_v = Vec::new();
            for (i, &t) in self.time_series.iter().enumerate() {
                if let Some(&y) = self.y_series.get(i) { plot_points_y.push([t, y]); }
                if let Some(&v) = self.v_series.get(i) { plot_points_v.push([t, v]); }
            }
            datasets.push(("y(t)".to_string(), plot_points_y));
            if self.preset == OdePreset::HarmonicOscillator && !plot_points_v.is_empty() {
                datasets.push(("y'(t)".to_string(), plot_points_v));
            }
            datasets.extend(self.snapshot_manager.export_datasets());
        }
        datasets
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&OdeState {
            preset: Some(self.preset),
            dt: self.dt, total_time: self.total_time, param_k: self.param_k,
            param_r: self.param_r, param_cap_k: self.param_cap_k, ic_y0: self.ic_y0,
            ic_v0: self.ic_v0, custom_expr: self.math_input.raw_text.clone(),
            snapshot_manager: self.snapshot_manager.clone(),
            show_annotations: self.show_annotations,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(s) = serde_json::from_str::<OdeState>(state) {
            if let Some(p) = s.preset { self.preset = p; }
            self.dt = s.dt; self.total_time = s.total_time; self.param_k = s.param_k;
            self.param_r = s.param_r; self.param_cap_k = s.param_cap_k;
            self.ic_y0 = s.ic_y0; self.ic_v0 = s.ic_v0;
            if !s.custom_expr.is_empty() { self.math_input.set_expression(&s.custom_expr); }
            self.snapshot_manager = s.snapshot_manager;
            self.show_annotations = s.show_annotations;
            self.recalculate();
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        let mut changed = false;
        let mut plot_points_y = Vec::new();
        let mut plot_points_v = Vec::new();

        for (i, &t) in self.time_series.iter().enumerate() {
            if let Some(&y) = self.y_series.get(i) { plot_points_y.push([t, y]); }
            if let Some(&v) = self.v_series.get(i) { plot_points_v.push([t, v]); }
        }

        egui::SidePanel::left("ode_controls").show(ctx, |ui| {
            ui.heading("ODE System");
            ui.separator();

            egui::ComboBox::from_id_salt("ode_preset")
                .selected_text(self.preset.name())
                .show_ui(ui, |ui| {
                    if ui.selectable_value(&mut self.preset, OdePreset::Exponential, OdePreset::Exponential.name()).changed()
                        || ui.selectable_value(&mut self.preset, OdePreset::HarmonicOscillator, OdePreset::HarmonicOscillator.name()).changed()
                        || ui.selectable_value(&mut self.preset, OdePreset::LogisticGrowth, OdePreset::LogisticGrowth.name()).changed()
                        || ui.selectable_value(&mut self.preset, OdePreset::Custom, OdePreset::Custom.name()).changed()
                    {
                        changed = true;
                    }
                });

            if self.preset == OdePreset::Custom {
                ui.add_space(6.0);
                if self.math_input.ui(ui) {
                    changed = true;
                }
            }

            ui.add_space(10.0);
            ui.heading("Parameters");
            match self.preset {
                OdePreset::Exponential | OdePreset::Custom => {
                    ui.horizontal(|ui| {
                        ui.label("k (Rate / Parameter):");
                        if ui.add(egui::DragValue::new(&mut self.param_k).speed(0.1)).changed() { changed = true; }
                    });
                }
                OdePreset::HarmonicOscillator => {
                    ui.horizontal(|ui| {
                        ui.label("k (Spring Constant):");
                        if ui.add(egui::DragValue::new(&mut self.param_k).speed(0.1)).changed() { changed = true; }
                    });
                }
                OdePreset::LogisticGrowth => {
                    ui.horizontal(|ui| {
                        ui.label("r (Growth Rate):");
                        if ui.add(egui::DragValue::new(&mut self.param_r).speed(0.1)).changed() { changed = true; }
                    });
                    ui.horizontal(|ui| {
                        ui.label("K (Carrying Capacity):");
                        if ui.add(egui::DragValue::new(&mut self.param_cap_k).speed(0.1)).changed() { changed = true; }
                    });
                }
            }

            ui.add_space(10.0);
            ui.heading("Initial Conditions");
            ui.horizontal(|ui| {
                ui.label("y(0):");
                if ui.add(egui::DragValue::new(&mut self.ic_y0).speed(0.1)).changed() { changed = true; }
            });
            if self.preset == OdePreset::HarmonicOscillator {
                ui.horizontal(|ui| {
                    ui.label("y'(0):");
                    if ui.add(egui::DragValue::new(&mut self.ic_v0).speed(0.1)).changed() { changed = true; }
                });
            }

            ui.add_space(10.0);
            ui.heading("Solver Settings");
            ui.horizontal(|ui| {
                ui.label("dt (Step Size):");
                if ui.add(egui::DragValue::new(&mut self.dt).speed(0.01).range(0.001..=1.0)).changed() { changed = true; }
            });
            ui.horizontal(|ui| {
                ui.label("Total Time:");
                if ui.add(egui::DragValue::new(&mut self.total_time).speed(1.0).range(1.0..=100.0)).changed() { changed = true; }
            });

            ui.add_space(5.0);
            ui.checkbox(&mut self.show_annotations, "Show Annotations");

            ui.add_space(10.0);
            let param_summary = self.param_summary();
            let is_harmonic = self.preset == OdePreset::HarmonicOscillator;
            let py = plot_points_y.clone();
            let pv = plot_points_v.clone();
            let current_len = self.snapshot_manager.len();

            self.snapshot_manager.render_ui(ui, || {
                let label = format!("Baseline {}", current_len + 1);
                let mut series = vec![TraceSeries::new("y(t)", py)];
                if is_harmonic && !pv.is_empty() {
                    series.push(TraceSeries::new("y'(t)", pv));
                }
                Some((label, param_summary, series))
            });

            let params = match self.preset {
                OdePreset::Exponential | OdePreset::Custom => vec![("k".to_string(), self.param_k), ("y0".to_string(), self.ic_y0)],
                OdePreset::HarmonicOscillator => vec![
                    ("k".to_string(), self.param_k),
                    ("y0".to_string(), self.ic_y0),
                    ("v0".to_string(), self.ic_v0),
                ],
                OdePreset::LogisticGrowth => vec![
                    ("r".to_string(), self.param_r),
                    ("cap_k".to_string(), self.param_cap_k),
                    ("y0".to_string(), self.ic_y0),
                ],
            };

            let current_preset = self.preset;
            let current_dt = self.dt;
            let current_total_time = self.total_time;
            let custom_expr = self.math_input.expr().cloned();

            self.sensitivity_engine.compute(&params, move |p_map| {
                eval_ode_final_state(current_preset, current_dt, current_total_time, p_map, custom_expr.as_ref())
            });

            crate::sensitivity::render_sensitivity_panel_ui(ui, &self.sensitivity_engine, "ode_tool");
        });

        if changed {
            let mut series = vec![TraceSeries::new("y(t)", plot_points_y.clone())];
            if self.preset == OdePreset::HarmonicOscillator && !plot_points_v.is_empty() {
                series.push(TraceSeries::new("y'(t)", plot_points_v.clone()));
            }
            self.snapshot_manager.push_history_run(&self.last_param_summary, series);
            self.recalculate();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Solution Trajectory");

            if self.diverged {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("⚠️ Solver Divergence Warning: Non-finite state detected (NaN or Inf). Integration terminated.")
                                .color(egui::Color32::RED)
                                .strong(),
                        );
                        if ui.button("↻ Reset Parameters").clicked() {
                            self.reset_parameters();
                            changed = true;
                        }
                    });
                });
            }

            let line_y = Line::new("y(t)", PlotPoints::new(plot_points_y)).color(egui::Color32::LIGHT_BLUE);

            Plot::new("ode_plot")
                .view_aspect(2.0)
                .x_axis_label("t \\text{ (Time)}")
                .y_axis_label("y(t)")
                .legend(egui_plot::Legend::default())
                .show(ui, |plot_ui| {
                    plot_ui.line(line_y);
                    if self.preset == OdePreset::HarmonicOscillator && !plot_points_v.is_empty() {
                        let line_v = Line::new("y'(t)", PlotPoints::new(plot_points_v))
                            .color(egui::Color32::LIGHT_RED);
                        plot_ui.line(line_v);
                    }
                    if self.show_annotations && !self.diverged {
                        annotations::render_ode_callouts(plot_ui, self);
                    }
                    self.snapshot_manager.render_on_plot(plot_ui);
                });
        });
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        name: "OdeSolverTool",
        domain: "analysis",
        tags: &[],
        build: || Box::new(OdeSolverTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for OdeSolverTool {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

impl InteractivePreset for OdeSolverTool {
    type Config = OdePresetConfig;

    fn domain(&self) -> &'static str { "analysis" }
    fn title(&self) -> String { format!("ODE Solver - {}", self.preset.name()) }

    fn export_config(&self) -> Self::Config {
        OdePresetConfig {
            preset: self.preset, dt: self.dt, total_time: self.total_time,
            param_k: self.param_k, param_r: self.param_r, param_cap_k: self.param_cap_k,
            ic_y0: self.ic_y0, ic_v0: self.ic_v0,
            custom_expr: self.math_input.raw_text.clone(),
        }
    }

    fn apply_config(&mut self, config: Self::Config) -> Result<(), String> {
        self.preset = config.preset; self.dt = config.dt; self.total_time = config.total_time;
        self.param_k = config.param_k; self.param_r = config.param_r; self.param_cap_k = config.param_cap_k;
        self.ic_y0 = config.ic_y0; self.ic_v0 = config.ic_v0;
        if !config.custom_expr.is_empty() { self.math_input.set_expression(&config.custom_expr); }
        self.recalculate();
        Ok(())
    }
}

#[cfg(test)]
mod ode_tests;
