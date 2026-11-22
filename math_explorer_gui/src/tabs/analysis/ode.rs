#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod annotations;
pub mod comparison;
mod persistence;
pub mod phase_plane;
pub mod poincare_map;
pub mod recovery;
pub mod systems;
pub mod ui;

#[cfg(test)]
use comparison::simulate_preset;
use comparison::{SolverDiscrepancy, SolverTrajectory, simulate_ode};
use phase_plane::{DEFAULT_PLANAR_X_EXPR, DEFAULT_PLANAR_Y_EXPR, DEFAULT_QUIVER_GRID};
use poincare_map::{DEFAULT_MATRIX_A, trace_det_from_matrix};

use crate::framework::recovery::emit_divergence_telemetry;
pub use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{
    CalloutManager, InteractiveTool, MathExpressionInputBar, TraceSeries, TraceSnapshotManager,
};
use crate::presets::OdePreset;
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
    pub(crate) diverged: bool,
    pub(crate) math_input: MathExpressionInputBar,
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
    );
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
            ("y' = -k*sin(x) - 0.25*y (Pendulum)", "-k * sin(x) - 0.25 * y"),
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
            diverged: false,
            math_input,
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
            OdePreset::LinearSystem2D => {
                let (tau, delta) = trace_det_from_matrix(self.matrix_a);
                format!(r"\tau={tau:.2}, \Delta={delta:.2}, \Delta t={:.2}", self.dt)
            }
            OdePreset::PlanarCustom => format!(
                r"k={:.2}, r={:.2}, \Delta t={:.2}",
                self.param_k, self.param_r, self.dt
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

    fn build_current_trace_series(
        &self,
        plot_points_y: &[[f64; 2]],
        plot_points_v: &[[f64; 2]],
    ) -> Vec<TraceSeries> {
        let (t1_lbl, t2_lbl, _, _) = self.preset.state_labels();
        let mut series = vec![TraceSeries::new(t1_lbl, plot_points_y.to_vec())];
        if self.preset.is_2d() && !plot_points_v.is_empty() {
            series.push(TraceSeries::new(t2_lbl, plot_points_v.to_vec()));
        }
        series
    }

    fn apply_sidebar_changes(
        &mut self,
        changed: bool,
        sweep_active: bool,
        plot_points_y: &[[f64; 2]],
        plot_points_v: &[[f64; 2]],
    ) {
        if sweep_active {
            if changed {
                if self.pending_sweep_run.is_none() {
                    let series = self.build_current_trace_series(plot_points_y, plot_points_v);
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
            let series = self.build_current_trace_series(plot_points_y, plot_points_v);
            self.snapshot_manager
                .push_history_run(&self.last_param_summary, series);
            self.recalculate();
        }
    }

    pub fn recalculate(&mut self) {
        self.bump_revision();
        self.time_series.clear();
        self.y_series.clear();
        self.v_series.clear();
        self.diverged = false;
        self.diagnostic_report = None;

        self.recalculate_active();
        self.recalculate_seeded_orbits();
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
        let active_traj = simulate_ode(&self.sim_spec(self.method));
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
        self.collect_export_datasets()
    }
    fn save_state(&self) -> Option<String> {
        self.serialize_full_state()
    }
    fn save_recovery_state(&self) -> Option<String> {
        self.serialize_recovery_state()
    }
    fn try_load_recovery_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        self.deserialize_state(state, true)
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.deserialize_state(state, false)
    }
    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
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

        let sweeps_before =
            crate::reflective_ui::slider::live_active_sweep_count(parent_ui.ctx());
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "ode_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            changed |= self.render_sidebar_controls(ui, &plot_points_y, &plot_points_v);
        });
        let sweep_active =
            crate::reflective_ui::slider::live_active_sweep_count(parent_ui.ctx()) > sweeps_before;

        self.apply_sidebar_changes(changed, sweep_active, &plot_points_y, &plot_points_v);

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.render_plot_canvas(ui, plot_points_y, plot_points_v, &mut changed);
        });
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
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
mod comparison_tests;
#[cfg(test)]
mod ode_tests;
#[cfg(test)]
mod planar_poincare_tests;
#[cfg(test)]
mod recovery_tests;
#[cfg(test)]
mod sensitivity_tests;
