#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod annotations;
mod canvas;
mod code_export;
pub mod comparison;
mod persistence;
pub mod recovery;
pub mod systems;
pub mod ui;

use comparison::{simulate_preset, SolverDiscrepancy};
use systems::{default_coupled_ode_bank, CustomOdeSpec};

use crate::framework::math_input::CoupledExpressionBank;
use crate::framework::recovery::emit_divergence_telemetry;
pub use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
#[cfg(test)]
use crate::framework::InteractiveTool;
use crate::framework::{CalloutManager, MathExpressionInputBar, TraceSeries, TraceSnapshotManager};
use crate::presets::OdePreset;
#[cfg(test)]
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
    pub(crate) display_y: std::sync::Arc<Vec<egui_plot::PlotPoint>>,
    pub(crate) display_v: std::sync::Arc<Vec<egui_plot::PlotPoint>>,
    pub(crate) diverged: bool,
    pub(crate) math_input: MathExpressionInputBar,
    pub(crate) custom_coupled: bool,
    pub(crate) coupled_bank: CoupledExpressionBank,
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
    pub(crate) pending_sweep_run: Option<(String, Vec<TraceSeries>)>,
    pub show_annotations: bool,
    pub(crate) revision: u64,
    pub(crate) synced_legacy_params: [f64; 3],
}

fn default_ode_math_input() -> MathExpressionInputBar {
    MathExpressionInputBar::new(
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
    )
    .with_axes(&["t", "y"])
    .with_disallowed_vars(&[
        ("x", "allowed variables: t, y"),
        ("z", "allowed variables: t, y"),
        (
            "v",
            "allowed variables: t, y; enable Coupled (y', v') mode to use v",
        ),
    ])
    .with_param_spec("k", "k (Rate / Parameter)", -5.0..=5.0, 1.0)
    .with_param_spec("r", "r (Growth Rate)", 0.0..=5.0, 1.0)
    .with_param_spec("cap_k", "K (Carrying Capacity)", 1.0..=100.0, 10.0)
    .with_param_spec("K", "K (Carrying Capacity)", 1.0..=100.0, 10.0)
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
            display_y: std::sync::Arc::new(Vec::new()),
            display_v: std::sync::Arc::new(Vec::new()),
            diverged: false,
            math_input: default_ode_math_input(),
            custom_coupled: false,
            coupled_bank: default_coupled_ode_bank(),
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
            pending_sweep_run: None,
            show_annotations: true,
            revision: 1,
            synced_legacy_params: [1.0, 1.0, 10.0],
        };
        tool.seed_default_custom_params();
        tool.recalculate();
        tool
    }
}

impl OdeSolverTool {
    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    pub(crate) const fn has_second_state(&self) -> bool {
        matches!(self.preset, OdePreset::HarmonicOscillator)
            || (matches!(self.preset, OdePreset::Custom) && self.custom_coupled)
    }

    pub(crate) const fn second_state_label(&self) -> &'static str {
        if matches!(self.preset, OdePreset::HarmonicOscillator) {
            "y'(t)"
        } else {
            "v(t)"
        }
    }

    pub(crate) fn has_custom_input_error(&self) -> bool {
        self.preset == OdePreset::Custom
            && if self.custom_coupled {
                !self.coupled_bank.is_valid()
            } else {
                !self.math_input.is_valid()
            }
    }

    pub(crate) fn build_custom_ode_spec(&self) -> Option<CustomOdeSpec> {
        if self.preset != OdePreset::Custom {
            return None;
        }
        if self.custom_coupled {
            CustomOdeSpec::from_coupled_bank(&self.coupled_bank)
        } else {
            CustomOdeSpec::from_scalar_bar(
                &self.math_input,
                self.param_k,
                self.param_r,
                self.param_cap_k,
            )
        }
    }

    pub fn param_summary(&self) -> String {
        match self.preset {
            OdePreset::Exponential | OdePreset::HarmonicOscillator => {
                format!(r"k={:.2}, \Delta t={:.2}", self.param_k, self.dt)
            }
            OdePreset::LogisticGrowth => format!(
                r"r={:.2}, K={:.2}, \Delta t={:.2}",
                self.param_r, self.param_cap_k, self.dt
            ),
            OdePreset::Custom if self.custom_coupled => self.coupled_param_summary(),
            OdePreset::Custom => self.custom_param_summary(),
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
        self.math_input.clamp_active_params(-10.0..=10.0);
        self.coupled_bank.clamp_active_params(-10.0..=10.0);
        self.push_clamped_legacy_params_to_bar();
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
            OdePreset::Custom if self.custom_coupled => {
                self.ic_y0 = 1.0;
                self.ic_v0 = 0.0;
                self.coupled_bank = default_coupled_ode_bank();
            }
            OdePreset::Custom => {
                self.reset_custom_parameters();
            }
        }
        self.dt = 0.05;
        self.total_time = 10.0;
        self.pending_sweep_run = None;
        self.recalculate();
    }

    pub(super) fn apply_sidebar_changes(&mut self, changed: bool, sweep_active: bool) {
        if sweep_active {
            if changed {
                if self.pending_sweep_run.is_none() {
                    let series = self.build_current_trace_series();
                    self.pending_sweep_run = Some((self.last_param_summary.clone(), series));
                }
                self.recalculate();
            }
            return;
        }

        if let Some((summary, series)) = self.pending_sweep_run.take() {
            self.snapshot_manager.push_history_run(summary, series);
            if changed {
                self.recalculate();
            } else {
                self.bump_revision();
            }
        } else if changed {
            let series = self.build_current_trace_series();
            self.snapshot_manager
                .push_history_run(&self.last_param_summary, series);
            self.recalculate();
        }
    }

    pub fn recalculate(&mut self) {
        self.sync_custom_params();
        self.bump_revision();
        self.time_series.clear();
        self.y_series.clear();
        self.v_series.clear();
        self.diverged = false;
        self.diagnostic_report = None;

        self.recalculate_active();
        let cap = crate::framework::MAX_DISPLAY_POLYLINE_POINTS;
        self.display_y =
            crate::framework::decimate_xy_series_minmax(&self.time_series, &self.y_series, cap);
        self.display_v =
            crate::framework::decimate_xy_series_minmax(&self.time_series, &self.v_series, cap);
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
        let input_has_err = self.has_custom_input_error();
        let custom_spec = self.build_custom_ode_spec();
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
            custom_spec.as_ref(),
        );
        if input_has_err && (active_traj.diverged || active_traj.divergence_info.is_some()) {
            return;
        }
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
mod comparison_tests;

#[cfg(test)]
mod coupled_tests;

#[cfg(test)]
mod custom_tests;

#[cfg(test)]
mod ode_tests;

#[cfg(test)]
mod recovery_tests;

#[cfg(test)]
mod sensitivity_tests;

#[cfg(test)]
mod solver_poi_tests;
