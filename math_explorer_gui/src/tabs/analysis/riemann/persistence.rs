//! State persistence and theory metadata for [`super::RiemannIntegrationTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    FunctionPreset, QuadratureMethod, RiemannIntegrationTool, DEFAULT_CC_TOLERANCE,
    DEFAULT_RIEMANN_EXPR, MAX_CC_TOLERANCE, MIN_CC_TOLERANCE, RIEMANN_X_MAX_BOUND,
    RIEMANN_X_MIN_BOUND,
};
use std::collections::BTreeMap;

// [cite:graph_parameters_rust]

fn default_cc_tolerance() -> f64 {
    DEFAULT_CC_TOLERANCE
}

fn is_default_quadrature(m: &QuadratureMethod) -> bool {
    matches!(m, QuadratureMethod::Trapezoidal)
}

fn is_default_cc_tolerance(v: &f64) -> bool {
    (*v - DEFAULT_CC_TOLERANCE).abs() < 1e-15
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RiemannState {
    preset: FunctionPreset,
    #[serde(default, skip_serializing_if = "is_default_quadrature")]
    quadrature_method: QuadratureMethod,
    #[serde(
        default = "default_cc_tolerance",
        skip_serializing_if = "is_default_cc_tolerance"
    )]
    cc_tolerance: f64,
    min_x: f64,
    max_x: f64,
    steps: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
}

impl RiemannIntegrationTool {
    pub(super) fn save_riemann_state(&self) -> Option<String> {
        let is_custom = self.preset == FunctionPreset::Custom;
        let custom_expr = (is_custom || self.math_input.raw_text != DEFAULT_RIEMANN_EXPR)
            .then(|| self.math_input.raw_text.clone());
        let custom_params = if is_custom {
            self.math_input.params_map()
        } else {
            BTreeMap::new()
        };
        serde_json::to_string(&RiemannState {
            preset: self.preset,
            quadrature_method: self.quadrature_method,
            cc_tolerance: self.cc_tolerance,
            min_x: self.min_x,
            max_x: self.max_x,
            steps: self.steps,
            custom_expr,
            custom_params,
        })
        .ok()
    }

    pub(super) fn try_load_riemann_state(&mut self, state: &str) -> Result<(), String> {
        let s: RiemannState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Riemann state: {e}"))?;
        if !s.min_x.is_finite()
            || !s.max_x.is_finite()
            || !(RIEMANN_X_MIN_BOUND..=RIEMANN_X_MAX_BOUND).contains(&s.min_x)
            || !(RIEMANN_X_MIN_BOUND..=RIEMANN_X_MAX_BOUND).contains(&s.max_x)
            || s.min_x >= s.max_x
            || !(1..=200).contains(&s.steps)
            || !s.cc_tolerance.is_finite()
            || !(MIN_CC_TOLERANCE..=MAX_CC_TOLERANCE).contains(&s.cc_tolerance)
        {
            return Err(
                "Riemann integration interval, partition count, or tolerance invalid".into(),
            );
        }
        self.preset = s.preset;
        self.quadrature_method = s.quadrature_method;
        self.cc_tolerance = s.cc_tolerance;
        self.min_x = s.min_x;
        self.max_x = s.max_x;
        self.steps = s.steps;
        self.math_input
            .set_expression(s.custom_expr.as_deref().unwrap_or(DEFAULT_RIEMANN_EXPR));
        self.math_input.set_params_map(&s.custom_params);
        self.recalculate();
        Ok(())
    }
}

impl scientific_metadata::theory::TheoryDescribable for RiemannIntegrationTool {
    fn theory_description(&self) -> String {
        format!(
            "Numerical approximation of the definite integral $\\int_a^b f(x) dx$ for {} using {} over interval [{:.2}, {:.2}] (estimated error {:.2e}).",
            self.preset.name(),
            self.quadrature_method.name(),
            self.min_x,
            self.max_x,
            self.error_estimate
        )
    }

    fn phonetic_description(&self) -> String {
        "Riemann integration, composite trapezoidal, and Clenshaw-Curtis Chebyshev quadrature theory"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Calculus, Vol. 1 (Apostol, 1967); Clenshaw & Curtis (1960), Numerische Mathematik".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "SinX".into(),
            "Integral of sin(x) showing oscillatory area cancellation".into(),
        );
        map.insert(
            "XSquared".into(),
            "Integral of x^2 demonstrating parabolic quadrature".into(),
        );
        map.insert(
            "EToX".into(),
            "Integral of e^x demonstrating exponential growth integration".into(),
        );
        map.insert(
            "Custom".into(),
            "Composite trapezoidal and Clenshaw-Curtis quadrature over compiled formulas".into(),
        );
        map
    }
}
