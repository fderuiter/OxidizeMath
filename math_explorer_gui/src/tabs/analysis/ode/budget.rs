#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::persistence::OdeState;
use super::systems::default_coupled_ode_bank;
use super::{default_ode_math_input, OdeSolverTool};
use crate::presets::{checked_ode_step_count, OdePreset, MAX_ODE_STEPS};

/// Maximum `steps * ast_nodes` work budget for interactive custom ODE trajectories.
pub(crate) const MAX_CUSTOM_ODE_WORK_BUDGET: usize = 250_000;

/// Maximum `steps * ast_nodes` work budget per parameter perturbation run in sensitivity analysis.
pub(crate) const MAX_SENSITIVITY_ODE_WORK_BUDGET: usize = 100_000;

/// Maximum serialized parameter map entries accepted when hydrating an ODE state payload.
const MAX_ODE_CUSTOM_PARAMS_LEN: usize = 16;

impl OdeSolverTool {
    /// Returns the total compiled AST instruction/node count for the active custom ODE formula(s).
    #[must_use]
    pub(crate) fn custom_ast_node_count(&self) -> usize {
        if self.preset != OdePreset::Custom {
            return 1;
        }
        if self.custom_coupled {
            self.coupled_bank
                .active_compiled()
                .iter()
                .map(|c| c.instructions().len())
                .sum::<usize>()
                .max(1)
        } else {
            self.math_input
                .active_compiled()
                .map_or(1, |c| c.instructions().len().max(1))
        }
    }

    /// Maximum integration step count permitted for the current ODE configuration.
    #[must_use]
    pub(crate) fn max_allowed_ode_steps(&self) -> usize {
        if self.preset != OdePreset::Custom {
            return MAX_ODE_STEPS;
        }
        (MAX_CUSTOM_ODE_WORK_BUDGET / self.custom_ast_node_count()).clamp(1, MAX_ODE_STEPS)
    }

    /// Minimum `dt` for a given `total_time` and step cap, nudged for IEEE-754 rounding safety.
    #[must_use]
    pub(crate) fn min_dt_for_step_cap(total_time: f64, max_steps: usize) -> f64 {
        let cap = max_steps.clamp(1, MAX_ODE_STEPS);
        let horizon = if total_time.is_finite() && total_time > 0.0 {
            total_time
        } else {
            0.1
        };
        let mut min_dt = (horizon / cap as f64).max(1e-6);
        for _ in 0..4 {
            if checked_ode_step_count(min_dt, horizon).is_some_and(|steps| steps <= cap) {
                break;
            }
            min_dt = min_dt.next_up();
        }
        min_dt
    }

    /// Clamps finite positive `dt` in `OdePreset::Custom` mode so `steps * ast_nodes` stays bounded.
    pub(crate) fn clamp_dt_to_work_budget(&mut self) {
        if self.preset != OdePreset::Custom
            || !self.dt.is_finite()
            || self.dt <= 0.0
            || !self.total_time.is_finite()
            || self.total_time <= 0.0
        {
            return;
        }
        let min_dt = Self::min_dt_for_step_cap(self.total_time, self.max_allowed_ode_steps());
        if self.dt < min_dt {
            self.dt = min_dt;
        }
    }

    /// Resolves the UI `dt` DragValue floor and clamps `self.dt` in Custom mode if needed.
    pub(crate) fn resolve_ui_min_dt(&mut self, changed: &mut bool) -> f64 {
        let min_dt =
            Self::min_dt_for_step_cap(self.total_time.max(0.1), self.max_allowed_ode_steps());
        if self.preset == OdePreset::Custom
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.dt < min_dt
        {
            self.dt = min_dt;
            *changed = true;
        }
        min_dt
    }

    /// Effective `dt` for finite-difference sensitivity runs, enforcing `MAX_SENSITIVITY_ODE_WORK_BUDGET`.
    #[must_use]
    pub(crate) fn sensitivity_effective_dt(&self) -> f64 {
        if self.preset != OdePreset::Custom
            || !self.dt.is_finite()
            || self.dt <= 0.0
            || !self.total_time.is_finite()
            || self.total_time <= 0.0
        {
            return self.dt;
        }
        let max_sens_steps = (MAX_SENSITIVITY_ODE_WORK_BUDGET / self.custom_ast_node_count())
            .clamp(1, MAX_ODE_STEPS);
        let min_sens_dt = Self::min_dt_for_step_cap(self.total_time, max_sens_steps);
        self.dt.max(min_sens_dt)
    }
}

pub(super) fn validate_deserialized_ode_state(s: &OdeState) -> Result<(), String> {
    if !s.dt.is_finite()
        || s.dt <= 0.0
        || !s.total_time.is_finite()
        || s.total_time <= 0.0
        || !s.param_k.is_finite()
        || !s.param_r.is_finite()
        || !s.param_cap_k.is_finite()
        || !s.ic_y0.is_finite()
        || !s.ic_v0.is_finite()
        || !s.custom_params.values().all(|v| v.is_finite())
        || !s.coupled_params.values().all(|v| v.is_finite())
    {
        return Err("ODE parameters must be finite with dt > 0 and total_time > 0".to_string());
    }
    if s.custom_params.len() > MAX_ODE_CUSTOM_PARAMS_LEN
        || s.coupled_params.len() > MAX_ODE_CUSTOM_PARAMS_LEN
    {
        return Err(format!(
            "ODE custom parameter map exceeds maximum of {MAX_ODE_CUSTOM_PARAMS_LEN} entries"
        ));
    }
    if checked_ode_step_count(s.dt, s.total_time).is_none() {
        return Err(format!(
            "ODE step count (total_time / dt) exceeds maximum allowed limit ({MAX_ODE_STEPS})"
        ));
    }
    validate_ode_custom_expressions(s)
}

fn validate_ode_custom_expressions(s: &OdeState) -> Result<(), String> {
    if !s.custom_expr.is_empty() || (s.preset == Some(OdePreset::Custom) && !s.custom_coupled) {
        let mut bar = default_ode_math_input();
        bar.set_expression(&s.custom_expr);
        if !bar.is_valid() {
            let reason = bar
                .error_message()
                .unwrap_or_else(|| "invalid custom ODE expression".to_string());
            return Err(format!("Invalid custom ODE expression: {reason}"));
        }
    }
    if let Some(ref exprs) = s.coupled_exprs {
        let mut bank = default_coupled_ode_bank();
        bank.set_expressions(exprs);
        if !bank.is_valid() {
            let reason = bank
                .first_error()
                .unwrap_or_else(|| "invalid coupled ODE expression".to_string());
            return Err(format!("Invalid coupled ODE expression: {reason}"));
        }
    }
    Ok(())
}
