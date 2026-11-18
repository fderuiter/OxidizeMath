#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::OdeSolverTool;
use crate::framework::recovery::{
    render_recovery_banner, DiagnosticReport, FiniteStateGuard, RecoveryAction,
};
use crate::presets::OdePreset;
use eframe::egui;

/// Checks whether an [`OdeSolverTool`] is in a completely finite numerical state.
pub(super) fn is_finite_ode_state(tool: &OdeSolverTool) -> bool {
    let params_finite = match tool.preset {
        OdePreset::Exponential | OdePreset::Custom => tool.param_k.is_finite(),
        OdePreset::HarmonicOscillator => tool.param_k.is_finite() && tool.ic_v0.is_finite(),
        OdePreset::LogisticGrowth => tool.param_r.is_finite() && tool.param_cap_k.is_finite(),
    };
    !tool.diverged
        && tool.dt.is_finite()
        && tool.total_time.is_finite()
        && tool.ic_y0.is_finite()
        && params_finite
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
    if !ic_y0.is_finite() {
        *ic_y0 = 1.0;
    } else {
        *ic_y0 = ic_y0.clamp(-10.0, 10.0);
    }

    if !ic_v0.is_finite() {
        *ic_v0 = 0.0;
    } else {
        *ic_v0 = ic_v0.clamp(-10.0, 10.0);
    }

    match preset {
        OdePreset::Exponential => {
            if !param_k.is_finite() {
                *param_k = 1.0;
            } else {
                *param_k = param_k.clamp(-5.0, 5.0);
            }
        }
        OdePreset::HarmonicOscillator => {
            if !param_k.is_finite() {
                *param_k = 1.0;
            } else {
                *param_k = param_k.clamp(0.01, 10.0);
            }
        }
        OdePreset::LogisticGrowth => {
            if !param_r.is_finite() {
                *param_r = 1.0;
            } else {
                *param_r = param_r.clamp(0.01, 5.0);
            }
            if !param_cap_k.is_finite() {
                *param_cap_k = 10.0;
            } else {
                *param_cap_k = param_cap_k.clamp(0.1, 100.0);
            }
        }
        OdePreset::Custom => {
            if !param_k.is_finite() {
                *param_k = 1.0;
            } else {
                *param_k = param_k.clamp(-10.0, 10.0);
            }
        }
    }
}

/// Clamps step size `dt` and `total_time` to valid finite solver bounds.
pub fn clamp_ode_step_settings(dt: &mut f64, total_time: &mut f64) {
    if !dt.is_finite() || *dt <= 0.0 {
        *dt = 0.05;
    } else {
        *dt = dt.clamp(1e-6, 1.0);
    }
    if !total_time.is_finite() || *total_time <= 0.0 {
        *total_time = 10.0;
    } else {
        *total_time = total_time.clamp(1.0, 100.0);
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
