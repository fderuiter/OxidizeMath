#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod annotations;
pub mod comparison;
pub mod recovery;
pub mod systems;
pub mod ui;

use comparison::{simulate_preset, SolverDiscrepancy};

use crate::framework::{InteractiveTool, TraceSeries, TraceSnapshotManager};
use crate::presets::{InteractivePreset, OdePreset, OdePresetConfig};
use eframe::egui;
use math_explorer::pure_math::analysis::ode::IntegrationMethod;

pub use recovery::OdeDiagnosticReport;
use recovery::clamp_ode_parameters;

// ----------------------------------------------------------------------------
// Tool Implementation
// ----------------------------------------------------------------------------

pub struct OdeSolverTool {
    pub(crate) preset: OdePreset,
    pub(crate) method: IntegrationMethod,
    pub(crate) dt: f64,
    pub(crate) total_time: f64,
    pub(crate) param_k: f64,
    pub(crate) param_r: f64,
    pub(crate) param_cap_k: f64,
    pub(crate) ic_y0: f64,
    pub(crate) ic_v0: f64,
    pub time_series: Vec<f64>,
    pub y_series: Vec<f64>,
    pub v_series: Vec<f64>,
    pub(crate) diverged: bool,
    pub(crate) diagnostic_report: Option<OdeDiagnosticReport>,
    pub(crate) comparison_mode: bool,
    pub(crate) discrepancy_euler_rk4: SolverDiscrepancy,
    pub(crate) discrepancy_heun_rk4: SolverDiscrepancy,
    pub(crate) discrepancy_euler_heun: SolverDiscrepancy,
    pub snapshot_manager: TraceSnapshotManager,
    pub(crate) comparison_manager: TraceSnapshotManager,
    pub(crate) sensitivity_engine: crate::sensitivity::SensitivityEngine,
    pub(crate) last_param_summary: String,
    pub show_annotations: bool,
}

impl Default for OdeSolverTool {
    fn default() -> Self {
        let mut tool = Self {
            preset: OdePreset::Exponential,
            method: IntegrationMethod::RungeKutta4,
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
            diagnostic_report: None,
            comparison_mode: false,
            discrepancy_euler_rk4: SolverDiscrepancy::default(),
            discrepancy_heun_rk4: SolverDiscrepancy::default(),
            discrepancy_euler_heun: SolverDiscrepancy::default(),
            snapshot_manager: TraceSnapshotManager::new(),
            comparison_manager: TraceSnapshotManager::new(),
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
            OdePreset::Exponential | OdePreset::HarmonicOscillator => {
                format!(r"k={:.2}, \Delta t={:.2}", self.param_k, self.dt)
            }
            OdePreset::LogisticGrowth => format!(
                r"r={:.2}, K={:.2}, \Delta t={:.2}",
                self.param_r, self.param_cap_k, self.dt
            ),
        }
    }

    #[must_use]
    #[allow(dead_code)]
    pub fn diverged(&self) -> bool {
        self.diverged
    }

    #[must_use]
    pub fn diagnostic_report(&self) -> Option<&OdeDiagnosticReport> {
        self.diagnostic_report.as_ref()
    }

    pub fn auto_scale_step_size(&mut self) {
        let min_dt = 1e-6;
        self.dt = if !self.dt.is_finite() || self.dt <= 0.0 {
            0.05
        } else {
            (self.dt * 0.5).max(min_dt)
        };
        self.recalculate();
    }

    pub fn clamp_instability_parameters(&mut self) {
        clamp_ode_parameters(
            self.preset,
            &mut self.param_k,
            &mut self.param_r,
            &mut self.param_cap_k,
            &mut self.ic_y0,
            &mut self.ic_v0,
        );
        self.recalculate();
    }

    #[cfg(test)]
    #[must_use]
    pub fn method(&self) -> IntegrationMethod {
        self.method
    }

    #[cfg(test)]
    pub fn set_method(&mut self, method: IntegrationMethod) {
        self.method = method;
        self.recalculate();
    }

    #[cfg(test)]
    #[must_use]
    pub fn comparison_mode(&self) -> bool {
        self.comparison_mode
    }

    #[cfg(test)]
    pub fn set_comparison_mode(&mut self, enabled: bool) {
        self.comparison_mode = enabled;
        self.recalculate();
    }

    #[cfg(test)]
    #[must_use]
    pub fn discrepancy_euler_rk4(&self) -> SolverDiscrepancy {
        self.discrepancy_euler_rk4
    }

    #[cfg(test)]
    #[must_use]
    pub fn discrepancy_heun_rk4(&self) -> SolverDiscrepancy {
        self.discrepancy_heun_rk4
    }

    #[cfg(test)]
    #[must_use]
    pub fn discrepancy_euler_heun(&self) -> SolverDiscrepancy {
        self.discrepancy_euler_heun
    }

    #[cfg(test)]
    #[must_use]
    pub fn comparison_manager(&self) -> &TraceSnapshotManager {
        &self.comparison_manager
    }

    pub fn reset_parameters(&mut self) {
        match self.preset {
            OdePreset::Exponential => {
                self.param_k = 1.0;
                self.ic_y0 = 1.0;
            }
            OdePreset::HarmonicOscillator => {
                self.param_k = 1.0;
                self.ic_y0 = 1.0;
                self.ic_v0 = 0.0;
            }
            OdePreset::LogisticGrowth => {
                self.param_r = 1.0;
                self.param_cap_k = 10.0;
                self.ic_y0 = 1.0;
            }
        }
        self.dt = 0.05;
        self.total_time = 10.0;
        self.recalculate();
    }

    pub fn recalculate(&mut self) {
        self.time_series.clear();
        self.y_series.clear();
        self.v_series.clear();
        self.diverged = false;
        self.diagnostic_report = None;

        self.recalculate_active();
        self.recalculate_comparison();
        self.last_param_summary = self.param_summary();
    }

    fn recalculate_active(&mut self) {
        let active_traj = simulate_preset(
            self.preset,
            self.method,
            self.dt,
            self.total_time,
            self.param_k,
            self.param_r,
            self.param_cap_k,
            self.ic_y0,
            self.ic_v0,
        );
        self.time_series = active_traj.time_series;
        self.y_series = active_traj.y_series;
        self.v_series = active_traj.v_series;
        self.diverged = active_traj.diverged;

        if let Some((t_div, step_idx, var_name)) = active_traj.divergence_info {
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
            recovery::emit_divergence_telemetry(
                t_div,
                step_idx,
                &var_name,
                self.preset.name(),
                self.dt,
            );
        }
    }
}

fn default_annotations_enabled() -> bool {
    true
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
    #[serde(default)]
    method: IntegrationMethod,
    #[serde(default)]
    comparison_mode: bool,
    #[serde(default = "default_annotations_enabled")]
    show_annotations: bool,
}

impl InteractiveTool for OdeSolverTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "ODE Solvers"
    }
    fn domain(&self) -> &'static str {
        "analysis"
    }

    fn export_preset_json(&self) -> Result<String, String> {
        self.to_json_string()
    }

    fn import_preset_json(&mut self, json_str: &str) -> Result<(), String> {
        let res = self.from_json_str(json_str);
        if res.is_ok() {
            self.recalculate();
        }
        res
    }

    fn apply_preset_json(&mut self, json_str: &str) -> Result<(), String> {
        self.import_preset_json(json_str)
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
            if self.comparison_mode {
                datasets.extend(self.comparison_manager.export_datasets());
            }
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
            method: self.method,
            comparison_mode: self.comparison_mode,
            show_annotations: self.show_annotations,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(s) = serde_json::from_str::<OdeState>(state) {
            self.dt = s.dt;
            self.total_time = s.total_time;
            self.param_k = s.param_k;
            self.param_r = s.param_r;
            self.param_cap_k = s.param_cap_k;
            self.ic_y0 = s.ic_y0;
            self.ic_v0 = s.ic_v0;
            self.snapshot_manager = s.snapshot_manager;
            self.method = s.method;
            self.comparison_mode = s.comparison_mode;
            self.show_annotations = s.show_annotations;
            self.recalculate();
        }
    }

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
            changed |= self.render_sidebar_controls(ui, &plot_points_y, &plot_points_v);
        });

        if changed {
            let mut series = vec![TraceSeries::new("y(t)", plot_points_y.clone())];
            if self.preset == OdePreset::HarmonicOscillator && !plot_points_v.is_empty() {
                series.push(TraceSeries::new("y'(t)", plot_points_v.clone()));
            }
            self.snapshot_manager
                .push_history_run(&self.last_param_summary, series);
            self.recalculate();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_plot_canvas(ui, plot_points_y, plot_points_v, &mut changed);
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
            OdePreset::HarmonicOscillator => {
                r"Harmonic oscillator: $\frac{d^2 y}{dt^2} = -k y$".into()
            }
            OdePreset::LogisticGrowth => {
                r"Logistic growth: $\frac{dy}{dt} = r y (1 - \frac{y}{K})$".into()
            }
        }
    }
    fn phonetic_description(&self) -> String {
        "Ordinary differential equations theory".into()
    }
    fn theory_citation(&self) -> String {
        "Differential Equations and Dynamical Systems (Perko, 2001)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

impl InteractivePreset for OdeSolverTool {
    type Config = OdePresetConfig;

    fn domain(&self) -> &'static str {
        "analysis"
    }
    fn title(&self) -> String {
        format!("ODE Solver - {}", self.preset.name())
    }

    fn export_config(&self) -> Self::Config {
        OdePresetConfig {
            preset: self.preset,
            dt: self.dt,
            total_time: self.total_time,
            param_k: self.param_k,
            param_r: self.param_r,
            param_cap_k: self.param_cap_k,
            ic_y0: self.ic_y0,
            ic_v0: self.ic_v0,
        }
    }

    fn apply_config(&mut self, config: Self::Config) -> Result<(), String> {
        self.preset = config.preset;
        self.dt = config.dt;
        self.total_time = config.total_time;
        self.param_k = config.param_k;
        self.param_r = config.param_r;
        self.param_cap_k = config.param_cap_k;
        self.ic_y0 = config.ic_y0;
        self.ic_v0 = config.ic_v0;
        self.recalculate();
        Ok(())
    }
}

#[cfg(test)]
mod ode_tests;

#[cfg(test)]
mod recovery_tests;
