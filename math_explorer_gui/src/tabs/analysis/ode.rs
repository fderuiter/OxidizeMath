#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{InteractiveTool, TraceSeries, TraceSnapshotManager};
use crate::presets::{InteractivePreset, OdePreset, OdePresetConfig};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::pure_math::analysis::ode::{
    OdeModel, RungeKutta4, TimeStepper, VecState,
};

mod annotations;
pub mod models;
pub mod recovery;

use models::*;
#[allow(unused_imports)]
pub use recovery::{
    clamp_ode_parameters, render_recovery_banner, render_recovery_ui, OdeDiagnosticReport,
    RecoveryAction,
};


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
    diagnostic_report: Option<OdeDiagnosticReport>,
    snapshot_manager: TraceSnapshotManager,
    sensitivity_engine: crate::sensitivity::SensitivityEngine,
    last_param_summary: String,
    pub show_annotations: bool,
}

impl Default for OdeSolverTool {
    fn default() -> Self {
        let mut tool = Self {
            preset: OdePreset::Exponential,
            dt: 0.05, total_time: 10.0,
            param_k: 1.0, param_r: 1.0, param_cap_k: 10.0,
            ic_y0: 1.0, ic_v0: 0.0,
            time_series: Vec::new(), y_series: Vec::new(), v_series: Vec::new(),
            diverged: false,
            diagnostic_report: None,
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
        }
    }
    #[must_use]
    #[allow(dead_code)]
    pub fn diverged(&self) -> bool { self.diverged }

    #[must_use]
    pub fn diagnostic_report(&self) -> Option<&OdeDiagnosticReport> { self.diagnostic_report.as_ref() }

    pub fn auto_scale_step_size(&mut self) {
        let min_dt = 1e-6;
        self.dt = if !self.dt.is_finite() || self.dt <= 0.0 { 0.05 } else { (self.dt * 0.5).max(min_dt) };
        self.recalculate();
    }

    pub fn clamp_instability_parameters(&mut self) {
        clamp_ode_parameters(
            self.preset, &mut self.param_k, &mut self.param_r,
            &mut self.param_cap_k, &mut self.ic_y0, &mut self.ic_v0,
        );
        self.recalculate();
    }

    pub fn reset_parameters(&mut self) {
        match self.preset {
            OdePreset::Exponential => { self.param_k = 1.0; self.ic_y0 = 1.0; }
            OdePreset::HarmonicOscillator => { self.param_k = 1.0; self.ic_y0 = 1.0; self.ic_v0 = 0.0; }
            OdePreset::LogisticGrowth => { self.param_r = 1.0; self.param_cap_k = 10.0; self.ic_y0 = 1.0; }
        }
        self.dt = 0.05;
        self.total_time = 10.0;
        self.recalculate();
    }

    #[allow(clippy::too_many_lines)]
    fn recalculate(&mut self) {
        self.time_series.clear();
        self.y_series.clear();
        self.v_series.clear();
        self.diverged = false;
        self.diagnostic_report = None;

        if self.dt <= 0.0 || self.total_time <= 0.0 || !self.dt.is_finite() || !self.total_time.is_finite() {
            return;
        }

        let num_steps = (self.total_time / self.dt).ceil() as usize;
        let mut divergence_info = None;

        match self.preset {
            OdePreset::Exponential => {
                let init_state = VecState(vec![self.ic_y0]);
                let dynamics = ExponentialOde { k: self.param_k };
                let solver = RungeKutta4::new(&init_state);
                let mut model = OdeModel::new(init_state, dynamics, solver);
                for i in 0..=num_steps {
                    let y = model.get_state().0[0];
                    if !y.is_finite() {
                        divergence_info = Some((i as f64 * self.dt, i, "y(t)".to_string()));
                        break;
                    }
                    self.time_series.push(i as f64 * self.dt);
                    self.y_series.push(y);
                    model.step(self.dt);
                }
            }
            OdePreset::HarmonicOscillator => {
                let init_state = VecState(vec![self.ic_y0, self.ic_v0]);
                let dynamics = HarmonicOscillatorOde { k: self.param_k };
                let solver = RungeKutta4::new(&init_state);
                let mut model = OdeModel::new(init_state, dynamics, solver);
                for i in 0..=num_steps {
                    let y = model.get_state().0[0];
                    let v = model.get_state().0[1];
                    if !y.is_finite() {
                        divergence_info = Some((i as f64 * self.dt, i, "y(t)".to_string()));
                        break;
                    }
                    if !v.is_finite() {
                        divergence_info = Some((i as f64 * self.dt, i, "y'(t)".to_string()));
                        break;
                    }
                    self.time_series.push(i as f64 * self.dt);
                    self.y_series.push(y);
                    self.v_series.push(v);
                    model.step(self.dt);
                }
            }
            OdePreset::LogisticGrowth => {
                let init_state = VecState(vec![self.ic_y0]);
                let dynamics = LogisticGrowthOde { r: self.param_r, cap_k: self.param_cap_k };
                let solver = RungeKutta4::new(&init_state);
                let mut model = OdeModel::new(init_state, dynamics, solver);
                for i in 0..=num_steps {
                    let y = model.get_state().0[0];
                    if !y.is_finite() {
                        divergence_info = Some((i as f64 * self.dt, i, "y(t)".to_string()));
                        break;
                    }
                    self.time_series.push(i as f64 * self.dt);
                    self.y_series.push(y);
                    model.step(self.dt);
                }
            }
        }

        if let Some((t_div, step_idx, var_name)) = divergence_info {
            self.diverged = true;
            let report = OdeDiagnosticReport::new(
                t_div,
                step_idx,
                var_name.clone(),
                format!(
                    "Simulation diverged at t={:.4}s (step {}) in variable {}",
                    t_div, step_idx, var_name
                ),
            );
            self.diagnostic_report = Some(report);
            recovery::emit_divergence_telemetry(t_div, step_idx, &var_name, self.preset.name(), self.dt);
        }

        self.last_param_summary = self.param_summary();
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct OdeState {
    dt: f64,
    total_time: f64,
    param_k: f64,
    param_r: f64,
    param_cap_k: f64,
    ic_y0: f64,
    ic_v0: f64,
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
                if let Some(&y) = self.y_series.get(i) {
                    plot_points_y.push([t, y]);
                }
                if let Some(&v) = self.v_series.get(i) {
                    plot_points_v.push([t, v]);
                }
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
            dt: self.dt,
            total_time: self.total_time,
            param_k: self.param_k,
            param_r: self.param_r,
            param_cap_k: self.param_cap_k,
            ic_y0: self.ic_y0,
            ic_v0: self.ic_v0,
            snapshot_manager: self.snapshot_manager.clone(),
            show_annotations: self.show_annotations,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(s) = serde_json::from_str::<OdeState>(state) {
            self.dt = s.dt; self.total_time = s.total_time;
            self.param_k = s.param_k; self.param_r = s.param_r; self.param_cap_k = s.param_cap_k;
            self.ic_y0 = s.ic_y0; self.ic_v0 = s.ic_v0;
            self.snapshot_manager = s.snapshot_manager;
            self.show_annotations = s.show_annotations;
            self.recalculate();
        }
    }

    fn apply_preset_json(&mut self, json_str: &str) -> Result<(), String> {
        self.import_preset_json(json_str)
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        let mut changed = false;

        let mut plot_points_y = Vec::new();
        let mut plot_points_v = Vec::new();

        for (i, &t) in self.time_series.iter().enumerate() {
            if let Some(&y) = self.y_series.get(i) {
                plot_points_y.push([t, y]);
            }
            if let Some(&v) = self.v_series.get(i) {
                plot_points_v.push([t, v]);
            }
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
                    {
                        changed = true;
                    }
                });

            let (latex_eq, plain_eq) = match self.preset {
                OdePreset::Exponential => (r"\frac{dy}{dt} = k y", "dy/dt = k*y"),
                OdePreset::HarmonicOscillator => (r"\frac{d^2 y}{dt^2} = -k y", "d^2y/dt^2 = -k*y"),
                OdePreset::LogisticGrowth => (r"\frac{dy}{dt} = r y (1 - \frac{y}{K})", "dy/dt = r*y*(1 - y/K)"),
            };
            crate::reflective_ui::render_formula_with_export(ui, latex_eq, Some(plain_eq));

            ui.add_space(10.0);
            ui.heading("Parameters");
            match self.preset {
                OdePreset::Exponential => {
                    ui.horizontal(|ui| {
                        ui.label("k (Rate):");
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

            if ui.button("🎨 Starter Gallery").on_hover_text("Open Starter Template Gallery").clicked() {
                let cur = ctx.data(|d| d.get_temp::<bool>(egui::Id::new("SHOW_STARTER_GALLERY")).unwrap_or(false));
                ctx.data_mut(|d| d.insert_temp(egui::Id::new("SHOW_STARTER_GALLERY"), !cur));
            }
            let params = match self.preset {
                OdePreset::Exponential => vec![("k".to_string(), self.param_k), ("y0".to_string(), self.ic_y0)],
                OdePreset::HarmonicOscillator => vec![("k".to_string(), self.param_k), ("y0".to_string(), self.ic_y0), ("v0".to_string(), self.ic_v0)],
                OdePreset::LogisticGrowth => vec![("r".to_string(), self.param_r), ("cap_k".to_string(), self.param_cap_k), ("y0".to_string(), self.ic_y0)],
            };

            let current_preset = self.preset;
            let current_dt = self.dt;
            let current_total_time = self.total_time;

            self.sensitivity_engine.compute(&params, |p_map| {
                eval_ode_final_state(current_preset, current_dt, current_total_time, p_map)
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
                recovery::render_recovery_ui(ui, self, &mut changed, &mut plot_points_y, &mut plot_points_v);
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
    fn theory_description(&self) -> String {
        match self.preset {
            OdePreset::Exponential => r"Exponential model: $\frac{dy}{dt} = k y$".into(),
            OdePreset::HarmonicOscillator => r"Harmonic oscillator: $\frac{d^2 y}{dt^2} = -k y$".into(),
            OdePreset::LogisticGrowth => r"Logistic growth: $\frac{dy}{dt} = r y (1 - \frac{y}{K})$".into(),
        }
    }
    fn phonetic_description(&self) -> String { "Ordinary differential equations theory".into() }
    fn theory_citation(&self) -> String { "Differential Equations and Dynamical Systems (Perko, 2001)".into() }
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
        }
    }

    fn apply_config(&mut self, config: Self::Config) -> Result<(), String> {
        self.preset = config.preset; self.dt = config.dt; self.total_time = config.total_time;
        self.param_k = config.param_k; self.param_r = config.param_r; self.param_cap_k = config.param_cap_k;
        self.ic_y0 = config.ic_y0; self.ic_v0 = config.ic_v0;
        self.recalculate();
        Ok(())
    }
}

#[cfg(test)]
mod ode_tests;
