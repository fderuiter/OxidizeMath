#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::recovery::{
    render_recovery_banner, DiagnosticReport, FiniteStateGuard, RecoveryAction,
};
use crate::presets::OdePreset;
use eframe::egui;

impl FiniteStateGuard for super::OdeSolverTool {
    fn is_finite_state(&self) -> bool {
        !self.diverged
            && self.time_series.iter().all(|t| t.is_finite())
            && self.y_series.iter().all(|y| y.is_finite())
            && self.v_series.iter().all(|v| v.is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        super::OdeSolverTool::diagnostic_report(self)
    }

    fn diverged(&self) -> bool {
        super::OdeSolverTool::diverged(self) || !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        super::OdeSolverTool::auto_scale_step_size(self);
    }

    fn clamp_parameters(&mut self) {
        if !self.dt.is_finite() || self.dt <= 0.0 {
            self.dt = 0.05;
        } else {
            self.dt = self.dt.clamp(1e-6, 0.1);
        }
        if !self.total_time.is_finite() || self.total_time <= 0.0 {
            self.total_time = 10.0;
        } else {
            self.total_time = self.total_time.clamp(0.1, 100.0);
        }
        self.clamp_instability_parameters();
    }

    fn reset_parameters(&mut self) {
        super::OdeSolverTool::reset_parameters(self);
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

/// Applies a centralized `RecoveryAction` via `FiniteStateGuard` and refreshes trajectory plot buffers.
pub fn apply_recovery_action(
    tool: &mut super::OdeSolverTool,
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
