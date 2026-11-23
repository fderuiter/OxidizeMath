#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod annotations;
mod canvas;
pub mod comparison;
mod custom_spec;
mod persistence;
pub mod phase_plane;
pub mod poincare_map;
pub mod recovery;
pub mod systems;
pub mod ui;

#[cfg(test)]
use comparison::simulate_preset;
use comparison::{simulate_ode, SolverDiscrepancy, SolverTrajectory};
use phase_plane::{DEFAULT_PLANAR_X_EXPR, DEFAULT_PLANAR_Y_EXPR, DEFAULT_QUIVER_GRID};
use poincare_map::{trace_det_from_matrix, DEFAULT_MATRIX_A};
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
    pub(crate) planar_x_input: MathExpressionInputBar,
    pub(crate) planar_y_input: MathExpressionInputBar,
    pub(crate) matrix_a: [[f64; 2]; 2],
    pub(crate) phase_plane_mode: bool,
    pub(crate) show_quiver: bool,
    pub(crate) quiver_density: usize,
    pub(crate) show_nullclines: bool,
    pub(crate) seeded_ic: Vec<[f64; 2]>,
    pub(crate) seeded_trajectories: Vec<SolverTrajectory>,
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

fn default_ode_math_inputs() -> (
    MathExpressionInputBar,
    MathExpressionInputBar,
    MathExpressionInputBar,
) {
    let scalar = MathExpressionInputBar::new(
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
    .with_axes(&["t", "y"]);
    let planar_x = MathExpressionInputBar::new(
        "x'",
        DEFAULT_PLANAR_X_EXPR,
        vec![
            ("x' = y (Second-Order Velocity)", "y"),
            ("x' = r*x - k*x*y (Lotka-Volterra)", "r * x - k * x * y"),
            ("x' = y - 0.2*x (Damped Rotation)", "y - 0.2 * x"),
        ],
    );
    let planar_y = MathExpressionInputBar::new(
        "y'",
        DEFAULT_PLANAR_Y_EXPR,
        vec![
            (
                "y' = k*(1 - x^2)*y - x (Van der Pol)",
                "k * (1 - x^2) * y - x",
            ),
            ("y' = k*x*y - y (Lotka-Volterra)", "k * x * y - y"),
            (
                "y' = -k*sin(x) - 0.25*y (Pendulum)",
                "-k * sin(x) - 0.25 * y",
            ),
            ("y' = x - x^3 - 0.3*k*y (Duffing)", "x - x^3 - 0.3 * k * y"),
        ],
    );
    (scalar, planar_x, planar_y)
}

impl Default for OdeSolverTool {
    fn default() -> Self {
        let (math_input, planar_x_input, planar_y_input) = default_ode_math_inputs();
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
            math_input,
            custom_coupled: false,
            coupled_bank: default_coupled_ode_bank(),
            planar_x_input,
            planar_y_input,
            matrix_a: DEFAULT_MATRIX_A,
            phase_plane_mode: false,
            show_quiver: true,
            quiver_density: DEFAULT_QUIVER_GRID,
            show_nullclines: false,
            seeded_ic: Vec::new(),
            seeded_trajectories: Vec::new(),
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

    #[must_use]
    pub(crate) fn has_second_state(&self) -> bool {
        self.preset.is_2d() || (self.preset == OdePreset::Custom && self.custom_coupled)
    }

    #[must_use]
    pub(crate) fn active_state_labels(
        &self,
    ) -> (&'static str, &'static str, &'static str, &'static str) {
        if self.preset == OdePreset::Custom && self.custom_coupled {
            ("y(t)", "v(t)", "y", "v")
        } else {
            self.preset.state_labels()
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
            OdePreset::LinearSystem2D => {
                let (tau, delta) = trace_det_from_matrix(self.matrix_a);
                format!(r"\tau={tau:.2}, \Delta={delta:.2}, \Delta t={:.2}", self.dt)
            }
            OdePreset::PlanarCustom => format!(
                r"k={:.2}, r={:.2}, \Delta t={:.2}",
                self.param_k, self.param_r, self.dt
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
        self.recalculate_seeded_orbits();
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
                    format!("Final ({last_t:.2}, {last_y:.2})"),
                    [25.0, -25.0],
                );
            }
        } else if self.diverged() {
            self.callout_manager.clear_auto();
        }
    }

    fn recalculate_active(&mut self) {
        let custom_spec = self.build_custom_ode_spec();
        let active_traj = simulate_ode(&self.sim_spec(self.method, custom_spec.as_ref()));
        self.time_series = active_traj.time_series;
        self.y_series = active_traj.y_series;
        self.v_series = active_traj.v_series;
        self.diverged = active_traj.diverged;

        if let Some((t_div, step_idx, var_name)) = active_traj.divergence_info {
            self.diverged = true;
            let message = format!(
                "Simulation diverged at t={t_div:.4}s (step {step_idx}) in variable {var_name}"
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
mod planar_poincare_tests;
#[cfg(test)]
mod recovery_tests;
#[cfg(test)]
mod sensitivity_tests;

#[cfg(test)]
mod solver_poi_tests;
