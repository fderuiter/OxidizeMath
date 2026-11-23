#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::phase_plane::{
    DEFAULT_PLANAR_X_EXPR, DEFAULT_PLANAR_Y_EXPR, DEFAULT_QUIVER_GRID, MAX_QUIVER_GRID,
    MAX_SEEDED_ORBITS,
};
use super::poincare_map::DEFAULT_MATRIX_A;
use super::systems::default_coupled_ode_bank;
use super::OdeSolverTool;
use crate::framework::recovery::{
    render_recovery_banner, DiagnosticReport, FiniteStateGuard, RecoveryAction,
};
use crate::presets::OdePreset;
use eframe::egui;

impl OdeSolverTool {
    /// Halves `dt` (floored at `1e-6`) and clamps `total_time` within `MAX_ODE_STEPS`.
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

    /// Clamps unstable or non-finite parameters, custom slider maps, planar matrix/orbits, and step settings.
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

        clamp_planar_state(
            &mut self.matrix_a,
            &mut self.quiver_density,
            &mut self.seeded_ic,
        );
        clamp_ode_step_settings(&mut self.dt, &mut self.total_time);
        self.recalculate();
    }

    /// Resets parameters, initial conditions, and seeded orbits for the active preset.
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
            OdePreset::LinearSystem2D => {
                self.matrix_a = DEFAULT_MATRIX_A;
                self.ic_y0 = 1.0;
                self.ic_v0 = 0.0;
            }
            OdePreset::PlanarCustom => {
                self.param_k = 1.0;
                self.param_r = 1.0;
                self.ic_y0 = 1.0;
                self.ic_v0 = 0.0;
                self.planar_x_input.set_expression(DEFAULT_PLANAR_X_EXPR);
                self.planar_y_input.set_expression(DEFAULT_PLANAR_Y_EXPR);
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
        self.seeded_ic.clear();
        self.dt = 0.05;
        self.total_time = 10.0;
        self.pending_sweep_run = None;
        self.recalculate();
    }
}

/// Checks whether an [`OdeSolverTool`] is in a completely finite numerical state.
pub(super) fn is_finite_ode_state(tool: &OdeSolverTool) -> bool {
    let params_finite = match tool.preset {
        OdePreset::Exponential => tool.param_k.is_finite(),
        OdePreset::Custom if tool.custom_coupled => {
            tool.ic_v0.is_finite()
                && tool
                    .coupled_bank
                    .param_slice()
                    .iter()
                    .all(|v| v.is_finite())
        }
        OdePreset::Custom => {
            tool.param_k.is_finite() && tool.math_input.param_slice().iter().all(|v| v.is_finite())
        }
        OdePreset::HarmonicOscillator => tool.param_k.is_finite() && tool.ic_v0.is_finite(),
        OdePreset::LogisticGrowth => tool.param_r.is_finite() && tool.param_cap_k.is_finite(),
        OdePreset::LinearSystem2D => {
            tool.ic_v0.is_finite() && tool.matrix_a.iter().flatten().all(|v| v.is_finite())
        }
        OdePreset::PlanarCustom => {
            tool.param_k.is_finite() && tool.param_r.is_finite() && tool.ic_v0.is_finite()
        }
    };
    let seeds_finite = tool
        .seeded_ic
        .iter()
        .all(|pt| pt[0].is_finite() && pt[1].is_finite())
        && tool.seeded_trajectories.iter().all(|traj| {
            traj.time_series.iter().all(|t| t.is_finite())
                && traj.y_series.iter().all(|x| x.is_finite())
                && traj.v_series.iter().all(|y| y.is_finite())
        });

    !tool.diverged
        && tool.dt.is_finite()
        && tool.total_time.is_finite()
        && tool.ic_y0.is_finite()
        && params_finite
        && seeds_finite
        && tool.time_series.iter().all(|t| t.is_finite())
        && tool.y_series.iter().all(|y| y.is_finite())
        && tool.v_series.iter().all(|v| v.is_finite())
}

impl FiniteStateGuard for OdeSolverTool {
    fn is_finite_state(&self) -> bool {
        is_finite_ode_state(self)
    }

    fn validate_state(&self) -> Result<(), DiagnosticReport> {
        if self.is_finite_state() {
            Ok(())
        } else {
            Err(self.diagnostic_report.clone().unwrap_or_else(|| {
                DiagnosticReport::new(
                    0.0,
                    0,
                    "state",
                    "Non-finite floating-point state encountered.",
                )
            }))
        }
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        OdeSolverTool::diagnostic_report(self)
    }

    fn diverged(&self) -> bool {
        OdeSolverTool::diverged(self)
    }

    fn auto_scale_step_size(&mut self) {
        OdeSolverTool::auto_scale_step_size(self);
    }

    fn clamp_parameters(&mut self) {
        self.clamp_instability_parameters();
    }

    fn reset_parameters(&mut self) {
        OdeSolverTool::reset_parameters(self);
    }
}

/// Clamps solver parameters and initial conditions to stable, finite bounds.
pub fn clamp_ode_parameters(
    preset: OdePreset,
    param_k: &mut f64,
    param_r: &mut f64,
    param_cap_k: &mut f64,
    ic_y0: &mut f64,
    ic_v0: &mut f64,
) {
    *ic_y0 = if ic_y0.is_finite() {
        ic_y0.clamp(-10.0, 10.0)
    } else {
        1.0
    };
    *ic_v0 = if ic_v0.is_finite() {
        ic_v0.clamp(-10.0, 10.0)
    } else {
        0.0
    };

    match preset {
        OdePreset::Exponential => {
            *param_k = if param_k.is_finite() {
                param_k.clamp(-5.0, 5.0)
            } else {
                1.0
            };
        }
        OdePreset::HarmonicOscillator => {
            *param_k = if param_k.is_finite() {
                param_k.clamp(0.01, 10.0)
            } else {
                1.0
            };
        }
        OdePreset::LogisticGrowth => {
            *param_r = if param_r.is_finite() {
                param_r.clamp(0.01, 5.0)
            } else {
                1.0
            };
            *param_cap_k = if param_cap_k.is_finite() {
                param_cap_k.clamp(0.1, 100.0)
            } else {
                10.0
            };
        }
        OdePreset::LinearSystem2D | OdePreset::PlanarCustom | OdePreset::Custom => {
            *param_k = if param_k.is_finite() {
                param_k.clamp(-10.0, 10.0)
            } else {
                1.0
            };
            *param_r = if param_r.is_finite() {
                param_r.clamp(-10.0, 10.0)
            } else {
                1.0
            };
        }
    }
}

/// Clamps 2D linear matrix entries, quiver density, and seeded orbit initial states.
pub fn clamp_planar_state(
    matrix_a: &mut [[f64; 2]; 2],
    quiver_density: &mut usize,
    seeded_ic: &mut Vec<[f64; 2]>,
) {
    for (row_idx, row) in matrix_a.iter_mut().enumerate() {
        for (col_idx, val) in row.iter_mut().enumerate() {
            *val = if val.is_finite() {
                val.clamp(-10.0, 10.0)
            } else {
                DEFAULT_MATRIX_A[row_idx][col_idx]
            };
        }
    }
    if !(2..=MAX_QUIVER_GRID).contains(quiver_density) {
        *quiver_density = DEFAULT_QUIVER_GRID;
    }
    seeded_ic.retain(|pt| pt[0].is_finite() && pt[1].is_finite());
    if seeded_ic.len() > MAX_SEEDED_ORBITS {
        let excess = seeded_ic.len() - MAX_SEEDED_ORBITS;
        seeded_ic.drain(0..excess);
    }
    for pt in seeded_ic.iter_mut() {
        pt[0] = pt[0].clamp(-10.0, 10.0);
        pt[1] = pt[1].clamp(-10.0, 10.0);
    }
}

/// Clamps step size `dt` and `total_time` to valid finite solver bounds.
pub fn clamp_ode_step_settings(dt: &mut f64, total_time: &mut f64) {
    if !total_time.is_finite() || *total_time <= 0.0 {
        *total_time = 10.0;
    } else {
        *total_time = total_time.clamp(0.1, 100.0);
    }
    let min_dt = (*total_time / crate::presets::MAX_ODE_STEPS as f64).max(1e-6);
    if !dt.is_finite() || *dt <= 0.0 {
        *dt = 0.05_f64.max(min_dt);
    } else {
        *dt = dt.clamp(min_dt, 0.1);
    }
    for _ in 0..4 {
        if crate::presets::checked_ode_step_count(*dt, *total_time).is_some() {
            break;
        }
        *dt = dt.next_up();
    }
}

/// Applies a centralized [`RecoveryAction`] to [`OdeSolverTool`] via [`FiniteStateGuard`]
/// and refreshes the active plot point buffers.
pub fn apply_recovery_action(
    tool: &mut OdeSolverTool,
    action: RecoveryAction,
    changed: &mut bool,
    plot_points_y: &mut Vec<[f64; 2]>,
    plot_points_v: &mut Vec<[f64; 2]>,
) {
    match action {
        RecoveryAction::AutoScaleStepSize => FiniteStateGuard::auto_scale_step_size(tool),
        RecoveryAction::ClampParameters => FiniteStateGuard::clamp_parameters(tool),
        RecoveryAction::ResetParameters => FiniteStateGuard::reset_parameters(tool),
    }
    *changed = true;
    plot_points_y.clear();
    plot_points_v.clear();
    for (i, &t) in tool.time_series.iter().enumerate() {
        if let Some(&y) = tool.y_series.get(i) {
            plot_points_y.push([t, y]);
        }
        if let Some(&v) = tool.v_series.get(i) {
            plot_points_v.push([t, v]);
        }
    }
}

/// Renders the centralized recovery banner and immediately applies recovery remedies to the tool.
pub fn render_recovery_ui(
    ui: &mut egui::Ui,
    tool: &mut super::OdeSolverTool,
    changed: &mut bool,
    plot_points_y: &mut Vec<[f64; 2]>,
    plot_points_v: &mut Vec<[f64; 2]>,
) {
    if let Some(action) = render_recovery_banner(ui, FiniteStateGuard::diagnostic_report(tool)) {
        apply_recovery_action(tool, action, changed, plot_points_y, plot_points_v);
    }
}
