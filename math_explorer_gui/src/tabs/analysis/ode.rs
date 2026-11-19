#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod annotations;
pub mod comparison;
mod persistence;
pub mod recovery;
pub mod systems;
pub mod ui;

use comparison::{simulate_preset, SolverDiscrepancy};

use crate::framework::recovery::emit_divergence_telemetry;
pub use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{
    CalloutManager, InteractiveTool, MathExpressionInputBar, TraceSeries, TraceSnapshotManager,
};
use crate::presets::OdePreset;
use eframe::egui;
use math_explorer::pure_math::analysis::ode::IntegrationMethod;
use recovery::{clamp_ode_parameters, clamp_ode_step_settings};

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
    pub(crate) math_input: MathExpressionInputBar,
    pub(crate) diagnostic_report: Option<DiagnosticReport>,
    pub(crate) comparison_mode: bool,
    pub(crate) discrepancy_euler_rk4: SolverDiscrepancy,
    pub(crate) discrepancy_heun_rk4: SolverDiscrepancy,
    pub(crate) discrepancy_euler_heun: SolverDiscrepancy,
    pub snapshot_manager: TraceSnapshotManager,
    pub(crate) comparison_manager: TraceSnapshotManager,
    pub callout_manager: CalloutManager,
    pub(crate) sensitivity_engine: crate::sensitivity::SensitivityEngine,
    pub(crate) last_param_summary: String,
    pub show_annotations: bool,
    pub(crate) revision: u64,
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
            math_input: MathExpressionInputBar::new(
                "y'",
                "-k * y * sin(t)",
                vec![
                    ("y' = -k*y*sin(t) (Non-linear Sine)", "-k * y * sin(t)"),
                    ("y' = k*y (Exponential)", "k * y"),
                    (
                        "y' = r*y*(1 - y/cap_k) (Logistic)",
                        "r * y * (1 - y / cap_k)",
                    ),
                    ("y' = -k * y^2 (Stiff Decay)", "-k * y^2"),
                    ("y' = cos(t) - y (Forced Damped)", "cos(t) - y"),
                ],
            ),
            diagnostic_report: None,
            comparison_mode: false,
            discrepancy_euler_rk4: SolverDiscrepancy::default(),
            discrepancy_heun_rk4: SolverDiscrepancy::default(),
            discrepancy_euler_heun: SolverDiscrepancy::default(),
            snapshot_manager: TraceSnapshotManager::new(),
            comparison_manager: TraceSnapshotManager::new(),
            callout_manager: CalloutManager::new(),
            sensitivity_engine: crate::sensitivity::SensitivityEngine::new(),
            last_param_summary: String::new(),
            show_annotations: true,
            revision: 1,
        };
        tool.recalculate();
        tool
    }
}

impl OdeSolverTool {
    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    pub fn param_summary(&self) -> String {
        match self.preset {
            OdePreset::Exponential | OdePreset::HarmonicOscillator | OdePreset::Custom => {
                format!(r"k={:.2}, \Delta t={:.2}", self.param_k, self.dt)
            }
            OdePreset::LogisticGrowth => format!(
                r"r={:.2}, K={:.2}, \Delta t={:.2}",
                self.param_r, self.param_cap_k, self.dt
            ),
        }
    }

    #[must_use]
    pub fn diverged(&self) -> bool {
        self.diverged || !FiniteStateGuard::is_finite_state(self)
    }

    #[must_use]
    pub fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic_report.as_ref()
    }

    pub fn auto_scale_step_size(&mut self) {
        let min_dt = 1e-6;
        if !self.total_time.is_finite() || self.total_time <= 0.0 {
            self.total_time = 10.0;
        } else {
            self.total_time = self.total_time.min(100.0);
        }
        self.dt = if !self.dt.is_finite() || self.dt <= 0.0 {
            0.05
        } else {
            (self.dt * 0.5).max(min_dt)
        };
        let max_horizon = self.dt * crate::presets::MAX_ODE_STEPS as f64;
        if self.total_time > max_horizon {
            self.total_time = max_horizon;
            for _ in 0..4 {
                if crate::presets::checked_ode_step_count(self.dt, self.total_time).is_some()
                    || self.total_time <= 0.0
                {
                    break;
                }
                self.total_time = self.total_time.next_down();
            }
        }
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
        clamp_ode_step_settings(&mut self.dt, &mut self.total_time);
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
            OdePreset::Custom => {
                self.param_k = 1.0;
                self.ic_y0 = 1.0;
                self.math_input.set_expression("-k * y * sin(t)");
            }
        }
        self.dt = 0.05;
        self.total_time = 10.0;
        self.recalculate();
    }

    pub fn recalculate(&mut self) {
        self.bump_revision();
        self.time_series.clear();
        self.y_series.clear();
        self.v_series.clear();
        self.diverged = false;
        self.diagnostic_report = None;

        self.recalculate_active();
        self.recalculate_comparison();
        self.last_param_summary = self.param_summary();
        self.update_callouts();
    }

    fn update_callouts(&mut self) {
        if !self.diverged() && !self.y_series.is_empty() {
            self.callout_manager.set_auto_callout(
                "initial_condition",
                [0.0, self.ic_y0],
                format!("Initial (0, {:.2})", self.ic_y0),
                [-25.0, -25.0],
            );
            if let (Some(&last_t), Some(&last_y)) = (self.time_series.last(), self.y_series.last())
            {
                self.callout_manager.set_auto_callout(
                    "final_state",
                    [last_t, last_y],
                    format!("Final ({:.2}, {:.2})", last_t, last_y),
                    [25.0, -25.0],
                );
            }
        } else if self.diverged() {
            self.callout_manager.clear_auto();
        }
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
            self.math_input.expr(),
        );
        self.time_series = active_traj.time_series;
        self.y_series = active_traj.y_series;
        self.v_series = active_traj.v_series;
        self.diverged = active_traj.diverged;

        if let Some((t_div, step_idx, var_name)) = active_traj.divergence_info {
            self.diverged = true;
            let message = format!(
                "Simulation diverged at t={:.4}s (step {}) in variable {}",
                t_div, step_idx, var_name
            );
            emit_divergence_telemetry("OdeSolverTool", t_div, step_idx, &var_name, &message);
            self.diagnostic_report =
                Some(DiagnosticReport::new(t_div, step_idx, var_name, message));
        }
    }
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

    fn state_revision(&self) -> u64 {
        self.compute_state_revision()
    }

    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
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
        self.serialize_full_state()
    }

    fn save_recovery_state(&self) -> Option<String> {
        self.serialize_recovery_state()
    }

    fn load_recovery_state(&mut self, state: &str) {
        let _ = self.deserialize_state(state, true);
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.deserialize_state(state, false)
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
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

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "OdeSolverTool",
        display_name: "ODE Solvers",
        domain: "analysis",
        tags: &[],
        build: || Box::new(OdeSolverTool::default()),
    }
}

#[cfg(test)]
mod ode_tests;

#[cfg(test)]
mod recovery_tests;

#[cfg(test)]
mod sensitivity_tests;
