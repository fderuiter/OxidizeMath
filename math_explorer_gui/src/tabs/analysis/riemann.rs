#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{InteractiveTool, MathExpressionInputBar};
use eframe::egui;
use math_explorer::pure_math::analysis::integration::{Integrator, Trapezoidal};
use std::collections::BTreeMap;

#[path = "riemann/ui.rs"]
mod ui;

pub(crate) const RIEMANN_X_MIN_BOUND: f64 = -50.0;
pub(crate) const RIEMANN_X_MAX_BOUND: f64 = 50.0;
pub(super) const RIEMANN_MIN_SPAN: f64 = 0.1;
const DEFAULT_RIEMANN_EXPR: &str = "clamp(sin(k * x), -0.5, 0.8)";

#[derive(PartialEq, Eq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) enum FunctionPreset {
    SinX,
    XSquared,
    EToX,
    Custom,
}

impl FunctionPreset {
    fn evaluate(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 0.0;
        }
        match self {
            FunctionPreset::SinX => x.sin(),
            FunctionPreset::XSquared => {
                let clamped = x.clamp(-1e150, 1e150);
                clamped * clamped
            }
            FunctionPreset::EToX => x.clamp(-700.0, 80.0).exp(),
            FunctionPreset::Custom => 0.0,
        }
    }

    pub(super) fn name(&self) -> &'static str {
        match self {
            FunctionPreset::SinX => "f(x) = sin(x)",
            FunctionPreset::XSquared => "f(x) = x^2",
            FunctionPreset::EToX => "f(x) = e^x",
            FunctionPreset::Custom => "f(x) (Custom Formula)",
        }
    }
}

pub struct RiemannIntegrationTool {
    pub(crate) preset: FunctionPreset,
    pub(crate) min_x: f64,
    pub(crate) max_x: f64,
    pub(crate) steps: usize,
    pub(crate) result: f64,
    pub(crate) math_input: MathExpressionInputBar,
    cached_trapezoid_polygons: Vec<Vec<[f64; 2]>>,
    cached_true_curve: Vec<[f64; 2]>,
}

impl Default for RiemannIntegrationTool {
    fn default() -> Self {
        let mut tool = Self {
            preset: FunctionPreset::SinX,
            min_x: 0.0,
            max_x: std::f64::consts::PI,
            steps: 10,
            result: 0.0,
            math_input: MathExpressionInputBar::new(
                "f(x)",
                DEFAULT_RIEMANN_EXPR,
                vec![
                    ("Clipped Sine Wave", "clamp(sin(k * x), -0.5, 0.8)"),
                    ("Boxcar Pulse", "a * rect(x / w)"),
                    ("Step Ramp", "x * step(x - a)"),
                    ("Damped Cosine", "exp(-a * x) * cos(k * x)"),
                ],
            )
            .with_axes(&["x"]),
            cached_trapezoid_polygons: Vec::new(),
            cached_true_curve: Vec::new(),
        };
        tool.recalculate();
        tool
    }
}

impl RiemannIntegrationTool {
    pub(crate) fn evaluate_at(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 0.0;
        }
        match self.preset {
            FunctionPreset::Custom => self.math_input.active_compiled().map_or(0.0, |c| {
                let y = c.eval_1d(x, self.math_input.param_slice());
                if y.is_finite() {
                    y.clamp(-1e6, 1e6)
                } else {
                    0.0
                }
            }),
            other => other.evaluate(x),
        }
    }

    pub(crate) fn recalculate(&mut self) {
        if !self.min_x.is_finite()
            || !self.max_x.is_finite()
            || self.min_x >= self.max_x
            || self.steps == 0
        {
            self.result = 0.0;
            self.cached_trapezoid_polygons.clear();
            self.cached_true_curve.clear();
            return;
        }

        let clamped_min = self.min_x.clamp(-700.0, 80.0);
        let clamped_max = self.max_x.clamp(-700.0, 80.0);
        let integrator = Trapezoidal::new(self.steps);
        let res = {
            let f = |x: f64| self.evaluate_at(x);
            if matches!(self.preset, FunctionPreset::EToX)
                && (clamped_min != self.min_x || clamped_max != self.max_x)
            {
                if clamped_min < clamped_max {
                    integrator
                        .integrate(f, clamped_min, clamped_max, 1e-6)
                        .value
                } else {
                    0.0
                }
            } else {
                integrator.integrate(f, self.min_x, self.max_x, 1e-6).value
            }
        };
        self.result = if res.is_finite() { res } else { 0.0 };

        let span = self.max_x - self.min_x;
        if !span.is_finite() || span <= 0.0 {
            self.cached_trapezoid_polygons.clear();
            self.cached_true_curve.clear();
            return;
        }
        let plot_min = self.min_x - span * 0.1;
        let plot_max = self.max_x + span * 0.1;

        self.cached_true_curve = (0..=500)
            .map(|i| {
                let x = plot_min + (plot_max - plot_min) * (i as f64 / 500.0);
                [x, self.evaluate_at(x)]
            })
            .filter(|pt| pt[0].is_finite() && pt[1].is_finite())
            .collect();

        let dx = span / (self.steps as f64);
        self.cached_trapezoid_polygons = (0..self.steps)
            .filter_map(|i| {
                let x0 = self.min_x + (i as f64) * dx;
                let x1 = x0 + dx;
                let y0 = self.evaluate_at(x0);
                let y1 = self.evaluate_at(x1);
                [x0, x1, y0, y1]
                    .iter()
                    .all(|v| v.is_finite())
                    .then(|| vec![[x0, 0.0], [x1, 0.0], [x1, y1], [x0, y0]])
            })
            .collect();
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RiemannState {
    preset: FunctionPreset,
    min_x: f64,
    max_x: f64,
    steps: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
}

impl InteractiveTool for RiemannIntegrationTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Riemann Integration"
    }

    fn domain(&self) -> &'static str {
        "analysis"
    }

    fn has_finite_parameters(&self) -> bool {
        self.min_x.is_finite()
            && self.max_x.is_finite()
            && (RIEMANN_X_MIN_BOUND..=RIEMANN_X_MAX_BOUND).contains(&self.min_x)
            && (RIEMANN_X_MIN_BOUND..=RIEMANN_X_MAX_BOUND).contains(&self.max_x)
            && self.min_x < self.max_x
            && (1..=200).contains(&self.steps)
            && self.result.is_finite()
    }

    fn save_state(&self) -> Option<String> {
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
            min_x: self.min_x,
            max_x: self.max_x,
            steps: self.steps,
            custom_expr,
            custom_params,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: RiemannState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Riemann state: {e}"))?;
        if !s.min_x.is_finite()
            || !s.max_x.is_finite()
            || !(RIEMANN_X_MIN_BOUND..=RIEMANN_X_MAX_BOUND).contains(&s.min_x)
            || !(RIEMANN_X_MIN_BOUND..=RIEMANN_X_MAX_BOUND).contains(&s.max_x)
            || s.min_x >= s.max_x
            || !(1..=200).contains(&s.steps)
        {
            return Err("Riemann integration interval or partition count invalid".into());
        }
        self.preset = s.preset;
        self.min_x = s.min_x;
        self.max_x = s.max_x;
        self.steps = s.steps;
        self.math_input
            .set_expression(s.custom_expr.as_deref().unwrap_or(DEFAULT_RIEMANN_EXPR));
        self.math_input.set_params_map(&s.custom_params);
        self.recalculate();
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        if self.draw_controls(parent_ui) {
            self.recalculate();
        }
        self.draw_plot(parent_ui);
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "RiemannIntegrationTool",
        display_name: "Riemann Integration",
        domain: "analysis",
        tags: &[],
        build: || Box::new(RiemannIntegrationTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for RiemannIntegrationTool {
    fn theory_description(&self) -> String {
        format!(
            "Numerical approximation of the definite integral $\\int_a^b f(x) dx$ for {} using composite trapezoidal rule: $T_n = \\sum_{{i=1}}^n \\frac{{f(x_{{i-1}}) + f(x_i)}}{{2}} \\Delta x$ with {} steps over interval [{:.2}, {:.2}].",
            self.preset.name(),
            self.steps,
            self.min_x,
            self.max_x
        )
    }

    fn phonetic_description(&self) -> String {
        "Riemann integration and composite trapezoidal approximation theory".into()
    }

    fn theory_citation(&self) -> String {
        "Calculus, Vol. 1 (Apostol, 1967); Principles of Mathematical Analysis (Rudin, 1976)".into()
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
            "Composite quadrature over compiled user-defined formulas and parameter sliders".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Composite Trapezoidal Integration",
    "domain": "analysis",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "preset": "SinX",
    "min_x": 0.0,
    "max_x": 3.141592653589793,
    "steps": 20
  }
}"#;

const CUSTOM_PIECEWISE_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Piecewise Pulse Quadrature",
    "domain": "analysis",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "preset": "Custom",
    "min_x": -3.0,
    "max_x": 3.0,
    "steps": 36,
    "custom_expr": "clamp(sin(k * x), -0.5, 0.8)",
    "custom_params": {
      "k": 3.0
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "riemann_integration_starter",
        title: "Composite Trapezoidal Integration",
        description: "Definite integral numerical quadrature via composite trapezoidal rule partitions.",
        category: "Integral Calculus",
        icon: "📐",
        domain: "analysis",
        tool_name: Some("RiemannIntegrationTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "riemann_custom_piecewise_quadrature",
        title: "Piecewise Pulse Quadrature",
        description: "Trapezoidal quadrature of clamp(sin(k*x), -0.5, 0.8) with live frequency slider k.",
        category: "Integral Calculus",
        icon: "∫",
        domain: "analysis",
        tool_name: Some("RiemannIntegrationTool"),
        preset_generator: || CUSTOM_PIECEWISE_PRESET_JSON,
    }
}

#[cfg(test)]
#[path = "riemann/tests.rs"]
mod tests;
