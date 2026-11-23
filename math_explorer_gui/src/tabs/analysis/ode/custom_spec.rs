#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Zero-allocation compiled bytecode ODE specifications for scalar `y' = f(t, y)` and
//! coupled `(y', v') = (f(t, y, v), g(t, y, v))` custom systems.

use super::OdeSolverTool;
use crate::framework::math_input::{
    CoupledExpressionBank, MathExpressionInputBar, MAX_SYNTHESIZED_PARAMS,
};
use math_commons::ast::CompiledExpr;
use math_explorer::pure_math::analysis::ode::{ArrayState, OdeSystem, VecState};
use std::collections::{BTreeMap, HashMap};

/// Default coupled 2-state ODE bank (`y' = f(t, y, v)`, `v' = g(t, y, v)`).
#[must_use]
pub fn default_coupled_ode_bank() -> CoupledExpressionBank {
    CoupledExpressionBank::new(&["t", "y", "v"], &[("y'", "v"), ("v'", "-k * y - c * v")])
        .with_param_range(-5.0..=5.0, 1.0)
        .with_initial_params(&[("k", 1.0), ("c", 0.2)])
        .with_system_presets(vec![
            ("Damped Harmonic Oscillator", vec!["v", "-k * y - c * v"]),
            (
                "Lotka-Volterra Predator-Prey",
                vec!["a * y - b * y * v", "d * y * v - g * v"],
            ),
            (
                "Van der Pol Oscillator",
                vec!["v", "mu * (1 - y^2) * v - y"],
            ),
            (
                "Duffing Forced Oscillator",
                vec!["v", "-c * v - a * y - b * y^3 + f * cos(w * t)"],
            ),
        ])
}

/// Zero-allocation compiled bytecode ODE specification for scalar `y' = f(t, y)` and
/// coupled `(y', v') = (f(t, y, v), g(t, y, v))` custom systems.
#[derive(Clone, Debug)]
pub enum CustomOdeSpec {
    /// Scalar 1st-order ODE `y' = f(t, y)`.
    Scalar {
        expr: CompiledExpr,
        params: [f64; MAX_SYNTHESIZED_PARAMS],
        param_len: usize,
    },
    /// Coupled 2-state ODE `y' = f(t, y, v), v' = g(t, y, v)`.
    Coupled {
        dy: CompiledExpr,
        dv: CompiledExpr,
        params: [f64; MAX_SYNTHESIZED_PARAMS],
        param_len: usize,
    },
}

impl CustomOdeSpec {
    /// Builds a scalar [`CustomOdeSpec`] from a [`MathExpressionInputBar`] with `(t, y)` axes.
    #[must_use]
    pub fn from_scalar_bar(
        bar: &MathExpressionInputBar,
        param_k: f64,
        param_r: f64,
        param_cap_k: f64,
    ) -> Option<Self> {
        let expr = bar.active_compiled()?.clone();
        let names = expr.param_names();
        if names.len() > MAX_SYNTHESIZED_PARAMS {
            return None;
        }
        let mut params = [0.0; MAX_SYNTHESIZED_PARAMS];
        let bar_slice = bar.param_slice();
        for (idx, name) in names.iter().enumerate() {
            params[idx] = match name.as_str() {
                "k" => param_k,
                "r" => param_r,
                "cap_k" | "K" => param_cap_k,
                _ => bar_slice.get(idx).copied().unwrap_or(1.0),
            };
        }
        Some(Self::Scalar {
            param_len: names.len(),
            expr,
            params,
        })
    }

    /// Builds a coupled 2-state [`CustomOdeSpec`] from a [`CoupledExpressionBank`] with `(t, y, v)` axes.
    #[must_use]
    pub fn from_coupled_bank(bank: &CoupledExpressionBank) -> Option<Self> {
        let compiled = bank.active_compiled();
        let dy = compiled.first()?.clone();
        let dv = compiled.get(1)?.clone();
        let slice = bank.param_slice();
        if slice.len() > MAX_SYNTHESIZED_PARAMS {
            return None;
        }
        let mut params = [0.0; MAX_SYNTHESIZED_PARAMS];
        params[..slice.len()].copy_from_slice(slice);
        Some(Self::Coupled {
            dy,
            dv,
            params,
            param_len: slice.len(),
        })
    }

    /// Returns whether this custom ODE integrates a second state component `v(t)`.
    #[must_use]
    pub const fn is_coupled(&self) -> bool {
        matches!(self, Self::Coupled { .. })
    }

    /// Clones this spec while overriding named parameter slots from `p_map` (for sensitivity sweeps).
    #[must_use]
    pub fn with_param_overrides(&self, p_map: &HashMap<String, f64>) -> Self {
        let mut cloned = self.clone();
        match &mut cloned {
            Self::Scalar {
                expr,
                params,
                param_len,
            } => {
                for (idx, name) in expr.param_names().iter().take(*param_len).enumerate() {
                    if let Some(&val) = p_map.get(name) {
                        params[idx] = val;
                    }
                }
            }
            Self::Coupled {
                dy,
                params,
                param_len,
                ..
            } => {
                for (idx, name) in dy.param_names().iter().take(*param_len).enumerate() {
                    if let Some(&val) = p_map.get(name) {
                        params[idx] = val;
                    }
                }
            }
        }
        cloned
    }
}

impl OdeSystem<ArrayState<1>> for CustomOdeSpec {
    fn derivative(&self, t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        let mut out = ArrayState([0.0]);
        self.derivative_in_place(t, state, &mut out);
        out
    }

    fn derivative_in_place(&self, t: f64, state: &ArrayState<1>, out: &mut ArrayState<1>) {
        match self {
            Self::Scalar {
                expr,
                params,
                param_len,
            } => {
                out.0[0] = expr.eval_2d(t, state.0[0], &params[..*param_len]);
            }
            Self::Coupled {
                dy,
                params,
                param_len,
                ..
            } => {
                out.0[0] = dy.eval_3d(t, state.0[0], 0.0, &params[..*param_len]);
            }
        }
    }
}

impl OdeSystem<ArrayState<2>> for CustomOdeSpec {
    fn derivative(&self, t: f64, state: &ArrayState<2>) -> ArrayState<2> {
        let mut out = ArrayState([0.0, 0.0]);
        self.derivative_in_place(t, state, &mut out);
        out
    }

    fn derivative_in_place(&self, t: f64, state: &ArrayState<2>, out: &mut ArrayState<2>) {
        match self {
            Self::Scalar {
                expr,
                params,
                param_len,
            } => {
                out.0[0] = expr.eval_2d(t, state.0[0], &params[..*param_len]);
                out.0[1] = 0.0;
            }
            Self::Coupled {
                dy,
                dv,
                params,
                param_len,
            } => {
                let [y, v] = state.0;
                let p = &params[..*param_len];
                out.0[0] = dy.eval_3d(t, y, v, p);
                out.0[1] = dv.eval_3d(t, y, v, p);
            }
        }
    }
}

impl OdeSystem<VecState> for CustomOdeSpec {
    fn derivative(&self, t: f64, state: &VecState) -> VecState {
        let mut out = VecState(vec![0.0; state.0.len()]);
        self.derivative_in_place(t, state, &mut out);
        out
    }

    fn derivative_in_place(&self, t: f64, state: &VecState, out: &mut VecState) {
        match self {
            Self::Scalar {
                expr,
                params,
                param_len,
            } => {
                out.0[0] = expr.eval_2d(t, state.0[0], &params[..*param_len]);
            }
            Self::Coupled {
                dy,
                dv,
                params,
                param_len,
            } => {
                let y = state.0[0];
                let v = state.0.get(1).copied().unwrap_or(0.0);
                let p = &params[..*param_len];
                out.0[0] = dy.eval_3d(t, y, v, p);
                if out.0.len() > 1 {
                    out.0[1] = dv.eval_3d(t, y, v, p);
                }
            }
        }
    }
}

pub(super) fn default_ode_custom_params(k: f64, r: f64, cap_k: f64) -> BTreeMap<String, f64> {
    BTreeMap::from([
        ("k".to_string(), k),
        ("r".to_string(), r),
        ("cap_k".to_string(), cap_k),
        ("K".to_string(), cap_k),
    ])
}

fn sync_one_legacy_param(
    math_input: &mut MathExpressionInputBar,
    name: &str,
    field: &mut f64,
    last_synced: &mut f64,
) {
    if !field.is_finite() {
        *last_synced = *field;
        return;
    }
    if field.to_bits() != last_synced.to_bits() {
        math_input.set_param(name, *field);
        *last_synced = *field;
    } else if let Some(val) = math_input.active_param_value(name) {
        *field = val;
        *last_synced = val;
    }
}

fn format_named_param_summary(names: &[String], vals: &[f64], dt: f64) -> String {
    if names.is_empty() {
        return format!(r"\Delta t={dt:.2}");
    }
    let mut parts: Vec<String> = names
        .iter()
        .zip(vals)
        .map(|(name, val)| format!("{name}={val:.2}"))
        .collect();
    parts.push(format!(r"\Delta t={dt:.2}"));
    parts.join(", ")
}

impl OdeSolverTool {
    pub(crate) fn seed_default_custom_params(&mut self) {
        self.math_input.set_params_map(&default_ode_custom_params(
            self.param_k,
            self.param_r,
            self.param_cap_k,
        ));
        self.synced_legacy_params = [self.param_k, self.param_r, self.param_cap_k];
    }

    pub(crate) fn reset_custom_parameters(&mut self) {
        self.param_k = 1.0;
        self.param_r = 1.0;
        self.param_cap_k = 10.0;
        self.ic_y0 = 1.0;
        self.seed_default_custom_params();
        self.math_input.set_expression("-k * y * sin(t)");
    }

    pub(crate) fn push_clamped_legacy_params_to_bar(&mut self) {
        self.math_input.set_param("k", self.param_k);
        self.math_input.set_param("r", self.param_r);
        self.math_input.set_param("cap_k", self.param_cap_k);
        self.math_input.set_param("K", self.param_cap_k);
        self.synced_legacy_params = [self.param_k, self.param_r, self.param_cap_k];
    }

    pub(crate) fn sync_legacy_params_from_math_input(&mut self) {
        if let Some(k) = self.math_input.active_param_value("k") {
            self.param_k = k;
        }
        if let Some(r) = self.math_input.active_param_value("r") {
            self.param_r = r;
        }
        if let Some(cap_k) = self
            .math_input
            .active_param_value("cap_k")
            .or_else(|| self.math_input.active_param_value("K"))
        {
            self.param_cap_k = cap_k;
        }
        self.synced_legacy_params = [self.param_k, self.param_r, self.param_cap_k];
    }

    pub(crate) fn sync_custom_params(&mut self) {
        sync_one_legacy_param(
            &mut self.math_input,
            "k",
            &mut self.param_k,
            &mut self.synced_legacy_params[0],
        );
        sync_one_legacy_param(
            &mut self.math_input,
            "r",
            &mut self.param_r,
            &mut self.synced_legacy_params[1],
        );
        let cap_symbol = if self.math_input.active_param_value("K").is_some()
            && self.math_input.active_param_value("cap_k").is_none()
        {
            "K"
        } else {
            "cap_k"
        };
        sync_one_legacy_param(
            &mut self.math_input,
            cap_symbol,
            &mut self.param_cap_k,
            &mut self.synced_legacy_params[2],
        );
    }

    pub(crate) fn custom_param_summary(&self) -> String {
        format_named_param_summary(
            self.math_input.active_param_names(),
            self.math_input.param_slice(),
            self.dt,
        )
    }

    pub(crate) fn coupled_param_summary(&self) -> String {
        format_named_param_summary(
            self.coupled_bank.param_names(),
            self.coupled_bank.param_slice(),
            self.dt,
        )
    }
}
