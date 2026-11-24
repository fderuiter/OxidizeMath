#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Synchronous bytecode/AST work budgets (`instructions * evaluations`) and atomic state
//! hydration guards for `OdeSolverTool` primary trajectories, seeded phase-plane orbits,
//! and multi-solver comparisons.

use super::comparison::OdeSimSpec;
use super::custom_spec::default_ode_custom_params;
use super::persistence::OdeState;
use super::phase_plane::{MAX_QUIVER_GRID, MAX_SEEDED_ORBITS};
use super::systems::{default_coupled_ode_bank, CustomOdeSpec};
use super::{default_ode_math_inputs, default_scalar_ode_input, OdeSolverTool};
use crate::framework::math_input::MAX_SYNTHESIZED_PARAMS;
use crate::presets::{checked_ode_step_count, OdePreset, MAX_ODE_STEPS};
use eframe::egui;
use math_commons::ast::{Expr, LexError, MAX_COMPILED_INSTRUCTIONS, MAX_EXPR_BYTES};
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

#[cfg(test)]
pub(crate) fn build_test_balanced_expr(var: &str, height: usize) -> String {
    let mut cur = var.to_string();
    for level in 1..=height {
        cur = match level % 3 {
            1 => format!("{cur}*{cur}"),
            2 => format!("{cur}+{cur}"),
            _ => format!("({cur})({cur})"),
        };
    }
    cur
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
            OdePreset::PlanarCustom => {
                let nx = self.planar_x_input.instruction_count().max(
                    self.planar_x_input
                        .cached_expr()
                        .map_or(1, expr_node_cost_bounded),
                );
                let ny = self.planar_y_input.instruction_count().max(
                    self.planar_y_input
                        .cached_expr()
                        .map_or(1, expr_node_cost_bounded),
                );
                (nx + ny).max(1)
            }
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

impl OdeSolverTool {
    pub(super) fn validate_deserialized_ode_state(&self, s: &OdeState) -> Result<(), String> {
        validate_deserialized_ode_bounds(s)?;
        validate_dormant_ode_expr_lengths(s)?;
        match s.preset.unwrap_or(self.preset) {
            OdePreset::Custom if s.custom_coupled => self.validate_loaded_coupled_ode(s),
            OdePreset::Custom => self.validate_loaded_scalar_ode(s),
            OdePreset::PlanarCustom => validate_loaded_planar_ode(s),
            _ => Ok(()),
        }
    }

    fn validate_loaded_scalar_ode(&self, s: &OdeState) -> Result<(), String> {
        let mut bar = default_scalar_ode_input();
        let param_k = s.custom_params.get("k").copied().unwrap_or(s.param_k);
        let param_r = s.custom_params.get("r").copied().unwrap_or(s.param_r);
        let param_cap_k = s
            .custom_params
            .get("cap_k")
            .or_else(|| s.custom_params.get("K"))
            .copied()
            .unwrap_or(s.param_cap_k);
        let mut merged = default_ode_custom_params(param_k, param_r, param_cap_k);
        for (k, &v) in &s.custom_params {
            merged.insert(k.clone(), v);
        }
        let expr = if s.custom_expr.is_empty() {
            self.math_input.raw_text.as_str()
        } else {
            s.custom_expr.as_str()
        };
        bar.set_expression(expr);
        bar.set_params_map(&merged);
        bar.error_message()
            .map_or(Ok(()), |err| reject_ode_formula(&err))
    }

    fn validate_loaded_coupled_ode(&self, s: &OdeState) -> Result<(), String> {
        let mut bank = default_coupled_ode_bank();
        if let Some(ref exprs) = s.coupled_exprs {
            bank.set_expressions(exprs);
        } else {
            bank.set_expressions(&[self.coupled_bank.raw_text(0), self.coupled_bank.raw_text(1)]);
        }
        if !s.coupled_params.is_empty() {
            bank.set_params_map(&s.coupled_params);
        }
        bank.first_error()
            .map_or(Ok(()), |err| reject_ode_formula(&err))
    }
}

fn validate_deserialized_ode_bounds(s: &OdeState) -> Result<(), String> {
    if !s.dt.is_finite()
        || s.dt <= 0.0
        || !s.total_time.is_finite()
        || s.total_time <= 0.0
        || !s.param_k.is_finite()
        || !s.param_r.is_finite()
        || !s.param_cap_k.is_finite()
        || !s.ic_y0.is_finite()
        || !s.ic_v0.is_finite()
        || !s.matrix_a.iter().flatten().all(|v| v.is_finite())
        || !s.custom_params.values().all(|v| v.is_finite())
        || !s.coupled_params.values().all(|v| v.is_finite())
    {
        return Err("ODE parameters must be finite with dt > 0 and total_time > 0".to_string());
    }
    if !(2..=MAX_QUIVER_GRID).contains(&s.quiver_density) {
        return Err(format!(
            "Quiver grid density ({}) must be between 2 and {MAX_QUIVER_GRID}",
            s.quiver_density
        ));
    }
    if s.seeded_ic.len() > MAX_SEEDED_ORBITS
        || !s
            .seeded_ic
            .iter()
            .all(|pt| pt[0].is_finite() && pt[1].is_finite())
    {
        return Err(format!(
            "Seeded orbits must be finite and contain at most {MAX_SEEDED_ORBITS} initial states"
        ));
    }
    if s.custom_params.len() > MAX_SYNTHESIZED_PARAMS
        || s.coupled_params.len() > MAX_SYNTHESIZED_PARAMS
    {
        return Err(format!(
            "ODE custom parameter map exceeds maximum of {MAX_SYNTHESIZED_PARAMS} entries"
        ));
    }
    if checked_ode_step_count(s.dt, s.total_time).is_none() {
        return Err(format!(
            "ODE step count (total_time / dt) exceeds maximum allowed limit ({MAX_ODE_STEPS})"
        ));
    }
    Ok(())
}

fn validate_dormant_ode_expr_lengths(s: &OdeState) -> Result<(), String> {
    for raw in [&s.custom_expr, &s.planar_x_expr, &s.planar_y_expr] {
        if raw.len() > MAX_EXPR_BYTES {
            return reject_ode_formula(&LexError::input_too_long(raw.len()).user_message());
        }
    }
    if let Some(ref exprs) = s.coupled_exprs {
        for raw in exprs {
            if raw.len() > MAX_EXPR_BYTES {
                return reject_ode_formula(&LexError::input_too_long(raw.len()).user_message());
            }
        }
    }
    Ok(())
}

fn validate_loaded_planar_ode(s: &OdeState) -> Result<(), String> {
    let (_, mut px_bar, mut py_bar) = default_ode_math_inputs();
    px_bar.set_expression(&s.planar_x_expr);
    py_bar.set_expression(&s.planar_y_expr);
    px_bar
        .error_message()
        .or_else(|| py_bar.error_message())
        .map_or(Ok(()), |err| reject_ode_formula(&err))
}

fn reject_ode_formula(err: &str) -> Result<(), String> {
    let msg = format!(
        "Invalid OdeSolverTool custom formula: {err}. Simplify or fix the expression and retry."
    );
    crate::accessibility::announce_status_with_priority(&msg, "assertive");
    Err(msg)
}
