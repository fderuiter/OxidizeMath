#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{InteractiveTool, MathExpressionInputBar};
use eframe::egui;
use math_explorer::pure_math::analysis::integration::{
    ClenshawCurtis, GaussKronrod15, GkSubinterval, Integrator, Simpson, Trapezoidal,
    GK15_MAX_PANELS,
};
use std::collections::BTreeMap;

#[path = "riemann/presets.rs"]
mod presets;
#[path = "riemann/ui.rs"]
mod ui;

pub(crate) const RIEMANN_X_MIN_BOUND: f64 = -50.0;
pub(crate) const RIEMANN_X_MAX_BOUND: f64 = 50.0;
pub(super) const RIEMANN_MIN_SPAN: f64 = 0.1;
pub(crate) const DEFAULT_CC_TOLERANCE: f64 = 1e-8;
pub(crate) const MIN_CC_TOLERANCE: f64 = 1e-13;
pub(crate) const MAX_CC_TOLERANCE: f64 = 1e-2;
const DEFAULT_RIEMANN_EXPR: &str = "clamp(sin(k * x), -0.5, 0.8)";

#[derive(PartialEq, Eq, Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum QuadratureMethod {
    #[default]
    Trapezoidal,
    Simpson,
    ClenshawCurtis,
    GaussKronrod15,
}

impl QuadratureMethod {
    pub const ALL: [Self; 4] = [
        Self::Trapezoidal,
        Self::Simpson,
        Self::ClenshawCurtis,
        Self::GaussKronrod15,
    ];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Trapezoidal => "Composite Trapezoidal",
            Self::Simpson => "Composite Simpson (1/3)",
            Self::ClenshawCurtis => "Clenshaw-Curtis (Adaptive Chebyshev)",
            Self::GaussKronrod15 => "Gauss-Kronrod 15 (Adaptive G7-K15)",
        }
    }

    #[must_use]
    pub const fn uses_tolerance(self) -> bool {
        matches!(self, Self::ClenshawCurtis | Self::GaussKronrod15)
    }
}

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
    pub(crate) quadrature_method: QuadratureMethod,
    pub(crate) cc_tolerance: f64,
    pub(crate) min_x: f64,
    pub(crate) max_x: f64,
    pub(crate) steps: usize,
    pub(crate) result: f64,
    pub(crate) error_estimate: f64,
    pub(crate) active_panel_count: usize,
    pub(crate) math_input: MathExpressionInputBar,
    cached_trapezoid_polygons: Vec<Vec<[f64; 2]>>,
    cached_true_curve: Vec<[f64; 2]>,
    cached_gk_panels: Vec<GkSubinterval>,
}

impl Default for RiemannIntegrationTool {
    fn default() -> Self {
        let mut tool = Self {
            preset: FunctionPreset::SinX,
            quadrature_method: QuadratureMethod::Trapezoidal,
            cc_tolerance: DEFAULT_CC_TOLERANCE,
            min_x: 0.0,
            max_x: std::f64::consts::PI,
            steps: 10,
            result: 0.0,
            error_estimate: 0.0,
            active_panel_count: 10,
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
            cached_gk_panels: Vec::new(),
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

    fn effective_tolerance(&self) -> f64 {
        if self.cc_tolerance.is_finite() {
            self.cc_tolerance.clamp(MIN_CC_TOLERANCE, MAX_CC_TOLERANCE)
        } else {
            DEFAULT_CC_TOLERANCE
        }
    }

    fn compute_quadrature(&mut self, a: f64, b: f64) -> (f64, f64) {
        self.cached_gk_panels.clear();
        if a >= b {
            return (0.0, 0.0);
        }
        let eps = self.effective_tolerance();
        match self.quadrature_method {
            QuadratureMethod::Trapezoidal => {
                let res =
                    Trapezoidal::new(self.steps).integrate(|x| self.evaluate_at(x), a, b, eps);
                self.active_panel_count = self.steps;
                (res.value, res.error.abs())
            }
            QuadratureMethod::Simpson => {
                let res = Simpson::new(self.steps).integrate(|x| self.evaluate_at(x), a, b, eps);
                let base = self.steps.max(2);
                let even_n = if base % 2 == 1 { base + 1 } else { base };
                self.active_panel_count = even_n / 2;
                (res.value, res.error.abs())
            }
            QuadratureMethod::ClenshawCurtis => {
                let res = ClenshawCurtis.integrate(|x| self.evaluate_at(x), a, b, eps);
                self.active_panel_count = self.steps;
                (res.value, res.error.abs())
            }
            QuadratureMethod::GaussKronrod15 => {
                let mut stack_panels = [GkSubinterval::default(); GK15_MAX_PANELS];
                let (res, count) = GaussKronrod15::new().integrate_panels(
                    |x| self.evaluate_at(x),
                    a,
                    b,
                    eps,
                    &mut stack_panels,
                );
                self.cached_gk_panels
                    .extend_from_slice(&stack_panels[..count]);
                self.active_panel_count = count;
                (res.value, res.error.abs())
            }
        }
    }

    pub(crate) fn recalculate(&mut self) {
        if !self.min_x.is_finite()
            || !self.max_x.is_finite()
            || self.min_x >= self.max_x
            || self.steps == 0
        {
            self.result = 0.0;
            self.error_estimate = 0.0;
            self.active_panel_count = 0;
            self.cached_trapezoid_polygons.clear();
            self.cached_true_curve.clear();
            self.cached_gk_panels.clear();
            return;
        }

        let (eval_min, eval_max) = if matches!(self.preset, FunctionPreset::EToX) {
            (
                self.min_x.clamp(-700.0, 80.0),
                self.max_x.clamp(-700.0, 80.0),
            )
        } else {
            (self.min_x, self.max_x)
        };

        let (val, err) = self.compute_quadrature(eval_min, eval_max);
        self.result = if val.is_finite() { val } else { 0.0 };
        self.error_estimate = if err.is_finite() { err } else { 0.0 };

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

        self.rebuild_partition_polygons(span);
    }

    fn rebuild_partition_polygons(&mut self, span: f64) {
        match self.quadrature_method {
            QuadratureMethod::Simpson => self.rebuild_simpson_polygons(span),
            QuadratureMethod::GaussKronrod15 => self.rebuild_gk_polygons(),
            QuadratureMethod::Trapezoidal | QuadratureMethod::ClenshawCurtis => {
                let n = self.steps;
                let mid = 0.5 * (self.min_x + self.max_x);
                let half = 0.5 * span;
                self.cached_trapezoid_polygons = (0..n)
                    .filter_map(|i| {
                        let (x0, x1) = if self.quadrature_method == QuadratureMethod::Trapezoidal {
                            let dx = span / (n as f64);
                            let left = self.min_x + (i as f64) * dx;
                            (left, left + dx)
                        } else {
                            let t0 = std::f64::consts::PI * (i as f64) / (n as f64);
                            let t1 = std::f64::consts::PI * ((i + 1) as f64) / (n as f64);
                            (mid - half * t0.cos(), mid - half * t1.cos())
                        };
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
    }

    fn rebuild_simpson_polygons(&mut self, span: f64) {
        let base = self.steps.max(2);
        let n_even = if base % 2 == 1 { base + 1 } else { base };
        let h = span / (n_even as f64);
        self.cached_trapezoid_polygons = (0..(n_even / 2))
            .filter_map(|k| {
                let x0 = self.min_x + ((2 * k) as f64) * h;
                let (x1, x2) = (x0 + h, x0 + 2.0 * h);
                let (y0, y1, y2) = (
                    self.evaluate_at(x0),
                    self.evaluate_at(x1),
                    self.evaluate_at(x2),
                );
                if ![x0, x1, x2, y0, y1, y2].iter().all(|v| v.is_finite()) {
                    return None;
                }
                let mut pts = Vec::with_capacity(11);
                pts.extend([[x0, 0.0], [x2, 0.0]]);
                for s in (0..=8).rev() {
                    let u = (s as f64) / 4.0;
                    let (l0, l1, l2) = (
                        0.5 * (u - 1.0) * (u - 2.0),
                        -u * (u - 2.0),
                        0.5 * u * (u - 1.0),
                    );
                    pts.push([x0 + u * h, l0 * y0 + l1 * y1 + l2 * y2]);
                }
                Some(pts)
            })
            .collect();
    }

    fn rebuild_gk_polygons(&mut self) {
        self.cached_trapezoid_polygons = self
            .cached_gk_panels
            .iter()
            .filter_map(|panel| {
                let (x0, x1) = (panel.left, panel.right);
                let xm = 0.5 * (x0 + x1);
                let (y0, ym, y1) = (
                    self.evaluate_at(x0),
                    self.evaluate_at(xm),
                    self.evaluate_at(x1),
                );
                [x0, xm, x1, y0, ym, y1]
                    .iter()
                    .all(|v| v.is_finite())
                    .then(|| vec![[x0, 0.0], [x1, 0.0], [x1, y1], [xm, ym], [x0, y0]])
            })
            .collect();
    }
}

const fn default_cc_tolerance() -> f64 {
    DEFAULT_CC_TOLERANCE
}
const fn is_default_quadrature(m: &QuadratureMethod) -> bool {
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
            && self.cc_tolerance.is_finite()
            && (MIN_CC_TOLERANCE..=MAX_CC_TOLERANCE).contains(&self.cc_tolerance)
            && self.result.is_finite()
            && self.error_estimate.is_finite()
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

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
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
        tags: &["quadrature", "simpson", "trapezoidal", "midpoint", "definite integral", "calculus"],
        build: || Box::new(RiemannIntegrationTool::default()),
    }
}

#[cfg(test)]
use presets::STARTER_PRESET_JSON;

#[cfg(test)]
#[path = "riemann/tests.rs"]
mod tests;
