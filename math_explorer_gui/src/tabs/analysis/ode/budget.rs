#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Synchronous bytecode/AST work budgets (`instructions * evaluations`) for `OdeSolverTool`
//! primary trajectories, seeded phase-plane orbits, and multi-solver comparisons.

use super::OdeSolverTool;
use super::comparison::OdeSimSpec;
use super::phase_plane::MAX_SEEDED_ORBITS;
use super::systems::CustomOdeSpec;
use crate::presets::{MAX_ODE_STEPS, OdePreset, checked_ode_step_count};
use eframe::egui;
use math_commons::ast::{Expr, MAX_COMPILED_INSTRUCTIONS};
use math_explorer::pure_math::analysis::ode::IntegrationMethod;

/// Maximum synchronous bytecode/AST instruction budget for primary + seeded ODE trajectories.
pub const MAX_ODE_SYNC_INSTRUCTIONS: usize = 2_000_000;

/// Maximum synchronous bytecode/AST instruction budget across Euler + Heun + RK4 comparison runs.
pub const MAX_ODE_COMPARISON_SYNC_INSTRUCTIONS: usize = 1_500_000;

/// Minimum step count preserved when scaling `dt` under the synchronous ODE work budget.
pub const MIN_ODE_BUDGET_STEPS: usize = 64;

/// Returns the derivative-evaluation stage count per integration step for `method`.
#[must_use]
pub const fn stages_for_method(method: IntegrationMethod) -> usize {
    match method {
        IntegrationMethod::Euler => 1,
        IntegrationMethod::Heun => 2,
        IntegrationMethod::RungeKutta4 => 4,
    }
}

/// Counts AST nodes iteratively up to `MAX_COMPILED_INSTRUCTIONS` without recursion.
#[must_use]
pub fn expr_node_cost_bounded(root: &Expr) -> usize {
    let mut stack = Vec::with_capacity(32);
    stack.push(root);
    let mut count = 0_usize;
    while let Some(node) = stack.pop() {
        count += 1;
        if count >= MAX_COMPILED_INSTRUCTIONS {
            return MAX_COMPILED_INSTRUCTIONS;
        }
        match node {
            Expr::Number(_) | Expr::Variable(_) => {}
            Expr::Unary { expr, .. } => stack.push(expr),
            Expr::Binary { left, right, .. } => {
                stack.push(right);
                stack.push(left);
            }
            Expr::Function { args, .. } => {
                for arg in args.iter().rev() {
                    stack.push(arg);
                }
            }
        }
    }
    count.max(1)
}

/// Tracks active step-scaling budget notices and deduplicates polite live-region announcements.
#[derive(Clone, Debug)]
pub struct OdeBudgetState {
    pub effective_dt: f64,
    pub primary_notice: Option<String>,
    pub comparison_notice: Option<String>,
    last_announced_primary: Option<String>,
    last_announced_comparison: Option<String>,
}

impl Default for OdeBudgetState {
    fn default() -> Self {
        Self {
            effective_dt: 0.05,
            primary_notice: None,
            comparison_notice: None,
            last_announced_primary: None,
            last_announced_comparison: None,
        }
    }
}

impl OdeSimSpec<'_> {
    /// Returns the instruction/node cost of one right-hand-side evaluation for `self`.
    #[must_use]
    pub fn rhs_instruction_cost(&self) -> usize {
        match self.preset {
            OdePreset::Exponential
            | OdePreset::HarmonicOscillator
            | OdePreset::LogisticGrowth
            | OdePreset::LinearSystem2D => 1,
            OdePreset::Custom => match self.custom_spec {
                Some(CustomOdeSpec::Scalar { expr, .. }) => expr.instruction_count().max(1),
                Some(CustomOdeSpec::Coupled { dy, dv, .. }) => {
                    (dy.instruction_count() + dv.instruction_count()).max(1)
                }
                None => 1,
            },
            OdePreset::PlanarCustom => {
                let cx = self.planar_x_expr.map_or(1, expr_node_cost_bounded);
                let cy = self.planar_y_expr.map_or(1, expr_node_cost_bounded);
                (cx + cy).max(1)
            }
        }
    }

    /// Returns `(steps, effective_dt)` bounded by both `MAX_ODE_STEPS` and `max_instructions`.
    #[must_use]
    pub fn budgeted_steps_and_dt(&self, max_instructions: usize) -> Option<(usize, f64)> {
        let requested_steps = checked_ode_step_count(self.dt, self.total_time)?;
        let ops_per_step =
            stages_for_method(self.method).saturating_mul(self.rhs_instruction_cost());
        if requested_steps.saturating_mul(ops_per_step) <= max_instructions {
            return Some((requested_steps, self.dt));
        }
        let capped_steps =
            (max_instructions / ops_per_step.max(1)).clamp(MIN_ODE_BUDGET_STEPS, requested_steps);
        let eff_dt = (self.total_time / capped_steps as f64).max(self.dt);
        Some((capped_steps, eff_dt))
    }
}

impl OdeSolverTool {
    /// Returns the per-RHS instruction cost of the active preset or custom expression(s).
    #[must_use]
    pub fn rhs_instruction_cost(&self) -> usize {
        match self.preset {
            OdePreset::Exponential
            | OdePreset::HarmonicOscillator
            | OdePreset::LogisticGrowth
            | OdePreset::LinearSystem2D => 1,
            OdePreset::Custom if self.custom_coupled => {
                self.coupled_bank.total_instruction_count().max(1)
            }
            OdePreset::Custom => self.math_input.instruction_count().max(1),
            OdePreset::PlanarCustom => (self.planar_x_input.instruction_count()
                + self.planar_y_input.instruction_count())
            .max(1),
        }
    }

    /// Computes `(effective_dt, optional_notice)` for primary and seeded phase-plane trajectories.
    #[must_use]
    pub fn effective_primary_dt(&self) -> (f64, Option<String>) {
        let Some(requested_steps) = checked_ode_step_count(self.dt, self.total_time) else {
            return (self.dt, None);
        };
        let orbit_count = if self.has_second_state() {
            (1 + self.seeded_ic.len()).min(1 + MAX_SEEDED_ORBITS)
        } else {
            1
        };
        let step_cap_by_orbits = (MAX_ODE_STEPS / orbit_count).max(1);
        let ops_per_step = orbit_count
            .saturating_mul(stages_for_method(self.method))
            .saturating_mul(self.rhs_instruction_cost());
        let step_cap_by_budget =
            if requested_steps.saturating_mul(ops_per_step) <= MAX_ODE_SYNC_INSTRUCTIONS {
                requested_steps
            } else {
                (MAX_ODE_SYNC_INSTRUCTIONS / ops_per_step.max(1))
                    .clamp(MIN_ODE_BUDGET_STEPS, requested_steps)
            };
        let max_steps = requested_steps
            .min(step_cap_by_orbits)
            .min(step_cap_by_budget);
        if max_steps < requested_steps {
            let eff_dt = (self.total_time / max_steps as f64).max(self.dt).max(1e-6);
            let notice = format!(
                "Solver work budget applied: scaled steps from {requested_steps} to {max_steps} (effective dt = {eff_dt:.4}s, {MAX_ODE_SYNC_INSTRUCTIONS} instruction cap)."
            );
            (eff_dt, Some(notice))
        } else {
            (self.dt, None)
        }
    }

    /// Computes `(comparison_dt, optional_notice)` across Euler + Heun + RK4 (7 total stages/step).
    #[must_use]
    pub fn effective_comparison_dt(&self, primary_dt: f64) -> (f64, Option<String>) {
        let base_dt = primary_dt.max(self.dt);
        let Some(base_steps) = checked_ode_step_count(base_dt, self.total_time) else {
            return (base_dt, None);
        };
        let ops_per_step = 7_usize.saturating_mul(self.rhs_instruction_cost());
        if base_steps.saturating_mul(ops_per_step) <= MAX_ODE_COMPARISON_SYNC_INSTRUCTIONS {
            return (base_dt, None);
        }
        let capped_steps = (MAX_ODE_COMPARISON_SYNC_INSTRUCTIONS / ops_per_step.max(1))
            .clamp(MIN_ODE_BUDGET_STEPS, base_steps);
        if capped_steps < base_steps {
            let comp_dt = (self.total_time / capped_steps as f64)
                .max(base_dt)
                .max(1e-6);
            let notice = format!(
                "Multi-solver comparison budget applied: aligned Euler/Heun/RK4 to {capped_steps} steps (comparison dt = {comp_dt:.4}s, {MAX_ODE_COMPARISON_SYNC_INSTRUCTIONS} instruction cap)."
            );
            (comp_dt, Some(notice))
        } else {
            (base_dt, None)
        }
    }

    /// Returns the active custom formula validation/budget error message when in a custom preset.
    #[must_use]
    pub(crate) fn active_custom_formula_error(&self) -> Option<String> {
        match self.preset {
            OdePreset::Custom if self.custom_coupled => self.coupled_bank.first_error(),
            OdePreset::Custom => self.math_input.error_message(),
            OdePreset::PlanarCustom => self
                .planar_x_input
                .error_message()
                .or_else(|| self.planar_y_input.error_message()),
            _ => None,
        }
    }

    /// Computes `(sens_dt, inst_per_eval)` and updates `sensitivity_engine` with any budget notice.
    pub(super) fn prepare_sensitivity_budget(&mut self, param_count: usize) -> (f64, usize) {
        let primary_dt = self.budget_state.effective_dt.max(self.dt);
        let base_steps = checked_ode_step_count(primary_dt, self.total_time).unwrap_or(1);
        let stages = stages_for_method(self.method);
        let inst_per_rhs = self.rhs_instruction_cost();
        let (sens_steps, sens_notice) =
            crate::sensitivity::SensitivityEngine::scale_steps_for_budget(
                param_count,
                base_steps,
                stages,
                inst_per_rhs,
            );
        let sens_dt = if sens_steps < base_steps {
            (self.total_time / sens_steps as f64).max(primary_dt)
        } else {
            primary_dt
        };
        self.sensitivity_engine.set_budget_notice(sens_notice);
        let inst_per_eval = sens_steps
            .saturating_mul(stages)
            .saturating_mul(inst_per_rhs);
        (sens_dt, inst_per_eval)
    }

    pub(super) fn render_budget_notices(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        announce_and_render_notice(
            ui,
            palette.warning,
            self.budget_state.primary_notice.as_deref(),
            &mut self.budget_state.last_announced_primary,
        );
        announce_and_render_notice(
            ui,
            palette.warning,
            self.budget_state.comparison_notice.as_deref(),
            &mut self.budget_state.last_announced_comparison,
        );
    }
}

fn announce_and_render_notice(
    ui: &mut egui::Ui,
    color: egui::Color32,
    notice: Option<&str>,
    last_announced: &mut Option<String>,
) {
    match notice {
        Some(msg) => {
            ui.add_space(4.0);
            ui.colored_label(color, format!("ℹ️ {msg}"));
            if last_announced.as_deref() != Some(msg) {
                crate::accessibility::announce_status(msg);
                *last_announced = Some(msg.to_string());
            }
        }
        None => {
            *last_announced = None;
        }
    }
}
