#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    default_orbit_x0, default_selected_r, default_show_lyapunov, default_x0, BifurcationDiagram,
    BifurcationMapMode, CUSTOM_R_BOUND, DEFAULT_CUSTOM_MAP_EXPR,
};
use std::collections::BTreeMap;

fn is_default_selected_r(v: &f64) -> bool {
    (*v - default_selected_r()).abs() < 1e-12
}

fn is_default_orbit_x0(v: &f64) -> bool {
    (*v - default_orbit_x0()).abs() < 1e-12
}

const fn is_default_show_lyapunov(v: &bool) -> bool {
    *v
}

fn is_default_x0(v: &f64) -> bool {
    (*v - default_x0()).abs() < 1e-12
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BifurcationState {
    r_min: f64,
    r_max: f64,
    steps: usize,
    #[serde(default, skip_serializing_if = "BifurcationMapMode::is_logistic")]
    map_mode: BifurcationMapMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
    #[serde(default = "default_x0", skip_serializing_if = "is_default_x0")]
    x0: f64,
    #[serde(
        default = "default_selected_r",
        skip_serializing_if = "is_default_selected_r"
    )]
    selected_r: f64,
    #[serde(
        default = "default_orbit_x0",
        skip_serializing_if = "is_default_orbit_x0"
    )]
    orbit_x0: f64,
    #[serde(
        default = "default_show_lyapunov",
        skip_serializing_if = "is_default_show_lyapunov"
    )]
    show_lyapunov: bool,
}

impl BifurcationDiagram {
    pub(super) fn serialize_state(&self) -> Option<String> {
        let is_custom = self.map_mode == BifurcationMapMode::Custom;
        let custom_expr = (is_custom || self.math_input.raw_text != DEFAULT_CUSTOM_MAP_EXPR)
            .then(|| self.math_input.raw_text.clone());
        let custom_params = if is_custom {
            self.math_input.params_map()
        } else {
            BTreeMap::new()
        };
        serde_json::to_string(&BifurcationState {
            r_min: self.r_min,
            r_max: self.r_max,
            steps: self.steps,
            map_mode: self.map_mode,
            custom_expr,
            custom_params,
            x0: self.x0,
            selected_r: self.selected_r,
            orbit_x0: self.orbit_x0,
            show_lyapunov: self.show_lyapunov,
        })
        .ok()
    }

    pub(super) fn deserialize_state(&mut self, state: &str) -> Result<(), String> {
        let s: BifurcationState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bifurcation state: {e}"))?;
        let r_bounds = match s.map_mode {
            BifurcationMapMode::Logistic => 0.0..=4.0,
            BifurcationMapMode::Custom => -CUSTOM_R_BOUND..=CUSTOM_R_BOUND,
        };
        if !s.r_min.is_finite()
            || !s.r_max.is_finite()
            || !r_bounds.contains(&s.r_min)
            || !r_bounds.contains(&s.r_max)
            || s.r_min >= s.r_max
            || !(100..=2000).contains(&s.steps)
            || !s.x0.is_finite()
            || !(-10.0..=10.0).contains(&s.x0)
            || !s.selected_r.is_finite()
            || !r_bounds.contains(&s.selected_r)
            || !s.orbit_x0.is_finite()
            || !(0.0..=1.0).contains(&s.orbit_x0)
            || s.custom_params.values().any(|v| !v.is_finite())
        {
            return Err("Bifurcation r bounds, initial seed, or step count invalid".into());
        }
        self.map_mode = s.map_mode;
        self.r_min = s.r_min;
        self.r_max = s.r_max;
        self.steps = s.steps;
        self.x0 = s.x0;
        self.selected_r = s.selected_r.clamp(s.r_min, s.r_max);
        self.orbit_x0 = s.orbit_x0.clamp(0.01, 0.99);
        self.show_lyapunov = s.show_lyapunov;
        self.math_input
            .set_expression(s.custom_expr.as_deref().unwrap_or(DEFAULT_CUSTOM_MAP_EXPR));
        self.math_input.set_params_map(&s.custom_params);
        self.recompute();
        Ok(())
    }
}
