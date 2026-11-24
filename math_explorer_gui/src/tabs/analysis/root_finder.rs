#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::math_input::CoupledExpressionBank;
use crate::framework::{InteractiveTool, MathExpressionInputBar};
use eframe::egui;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

#[path = "root_finder/compute.rs"]
mod compute;
#[path = "root_finder/presets.rs"]
mod presets;
#[path = "root_finder/ui.rs"]
mod ui;

pub(crate) const ROOT_DOMAIN_MIN: f64 = -50.0;
pub(crate) const ROOT_DOMAIN_MAX: f64 = 50.0;
pub(crate) const ROOT_MIN_SPAN: f64 = 0.05;
pub(crate) const DEFAULT_ROOT_TOL: f64 = 1e-8;
pub(crate) const MIN_ROOT_TOL: f64 = 1e-12;
pub(crate) const MAX_ROOT_TOL: f64 = 1e-2;
pub(crate) const DEFAULT_MAX_ITERS: usize = 60;
const DEFAULT_SCALAR_EXPR: &str = "cos(x) - k * x";

/// Dimensionality mode for [`RootFinderTool`].
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum RootDimension {
    /// Scalar equation `f(x) = 0`.
    #[default]
    Scalar1D,
    /// Coupled 2D nonlinear system `F_1(x, y) = 0, F_2(x, y) = 0`.
    System2D,
}

/// Supported 1D scalar root-finding strategies.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum ScalarRootMethod {
    /// Binary interval halving (`Bisection`).
    Bisection,
    /// Brent-Dekker inverse quadratic / secant / bisection hybrid (`Brent`).
    #[default]
    Brent,
    /// Open tangent iteration (`NewtonRaphson`).
    NewtonRaphson,
    /// Two-point derivative-free chord iteration (`Secant`).
    Secant,
}

impl ScalarRootMethod {
    pub const ALL: [Self; 4] = [
        Self::Bisection,
        Self::Brent,
        Self::NewtonRaphson,
        Self::Secant,
    ];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bisection => "Bisection",
            Self::Brent => "Brent (Hybrid)",
            Self::NewtonRaphson => "Newton-Raphson",
            Self::Secant => "Secant",
        }
    }

    #[must_use]
    pub const fn is_bracketing(self) -> bool {
        matches!(self, Self::Bisection | Self::Brent)
    }
}

/// Recorded geometric state for a single 1D root-finder iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarStepRecord {
    pub step_index: usize,
    pub x: f64,
    pub fx: f64,
    pub bracket: Option<[f64; 2]>,
    pub construction_segment: Option<[[f64; 2]; 2]>,
}

/// Recorded state for a single 2D Multivariate Newton iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct System2DStepRecord {
    pub step_index: usize,
    pub point: [f64; 2],
    pub residual: [f64; 2],
    pub inf_norm: f64,
}

/// Interactive 1D & 2D nonlinear root-finding workbench.
pub struct RootFinderTool {
    pub(crate) dimension: RootDimension,
    pub(crate) scalar_method: ScalarRootMethod,
    pub(crate) bracket_min: f64,
    pub(crate) bracket_max: f64,
    pub(crate) init_2d: [f64; 2],
    pub(crate) domain_span_2d: f64,
    pub(crate) tolerance: f64,
    pub(crate) max_iterations: usize,
    pub(crate) selected_step: usize,
    pub(crate) scalar_input: MathExpressionInputBar,
    pub(crate) coupled_bank: CoupledExpressionBank,
    pub(crate) converged_root_1d: Option<f64>,
    pub(crate) converged_root_2d: Option<[f64; 2]>,
    pub(crate) final_residual: f64,
    pub(crate) status_error: Option<String>,
    pub(crate) scalar_steps: Vec<ScalarStepRecord>,
    pub(crate) system_steps: Vec<System2DStepRecord>,
    pub(crate) cached_1d_curve: Vec<[f64; 2]>,
    pub(crate) contours_f1: Vec<Vec<[f64; 2]>>,
    pub(crate) contours_f2: Vec<Vec<[f64; 2]>>,
    pub(crate) revision: u64,
}

fn default_scalar_bar() -> MathExpressionInputBar {
    MathExpressionInputBar::new(
        "f(x)",
        DEFAULT_SCALAR_EXPR,
        vec![
            ("Dottie Transcendental", "cos(x) - k * x"),
            ("Kepler Anomaly", "x - k * sin(x) - 1.0"),
            ("Cubic Polynomial", "x^3 - 2 * x - 5"),
            ("Exponential Cross", "exp(-x) - x"),
        ],
    )
    .with_axes(&["x"])
    .with_param_spec("k", "k (Coupling)", 0.1..=5.0, 1.0)
}

fn default_coupled_2d_bank() -> CoupledExpressionBank {
    CoupledExpressionBank::new(
        &["x", "y"],
        &[("F1(x, y)", "x^2 + y^2 - r^2"), ("F2(x, y)", "x^2 - y - c")],
    )
    .with_param_range(-5.0..=5.0, 1.0)
    .with_initial_params(&[("r", 2.0), ("c", 0.5)])
    .with_system_presets(vec![
        ("Circle & Parabola", vec!["x^2 + y^2 - r^2", "x^2 - y - c"]),
        (
            "Trigonometric Lattice",
            vec!["sin(x) + y - 1", "x + cos(y) - 1"],
        ),
        ("Hyperbola & Line", vec!["x * y - 1", "x - 2 * y + c"]),
    ])
}

impl Default for RootFinderTool {
    fn default() -> Self {
        let mut tool = Self {
            dimension: RootDimension::Scalar1D,
            scalar_method: ScalarRootMethod::Brent,
            bracket_min: 0.0,
            bracket_max: 2.0,
            init_2d: [1.5, 1.2],
            domain_span_2d: 3.0,
            tolerance: DEFAULT_ROOT_TOL,
            max_iterations: DEFAULT_MAX_ITERS,
            selected_step: 0,
            scalar_input: default_scalar_bar(),
            coupled_bank: default_coupled_2d_bank(),
            converged_root_1d: None,
            converged_root_2d: None,
            final_residual: 0.0,
            status_error: None,
            scalar_steps: Vec::new(),
            system_steps: Vec::new(),
            cached_1d_curve: Vec::new(),
            contours_f1: Vec::new(),
            contours_f2: Vec::new(),
            revision: 1,
        };
        tool.recalculate();
        tool.selected_step = tool.max_step_index();
        tool
    }
}

fn default_tol() -> f64 {
    DEFAULT_ROOT_TOL
}
fn default_iters() -> usize {
    DEFAULT_MAX_ITERS
}
fn default_init_2d() -> [f64; 2] {
    [1.5, 1.2]
}
fn default_span_2d() -> f64 {
    3.0
}
fn default_bmin() -> f64 {
    0.0
}
fn default_bmax() -> f64 {
    2.0
}
fn is_default_dim(d: &RootDimension) -> bool {
    *d == RootDimension::Scalar1D
}
fn is_default_method(m: &ScalarRootMethod) -> bool {
    *m == ScalarRootMethod::Brent
}
fn is_default_tol(v: &f64) -> bool {
    (*v - DEFAULT_ROOT_TOL).abs() < 1e-16
}
fn is_default_iters(v: &usize) -> bool {
    *v == DEFAULT_MAX_ITERS
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RootFinderState {
    #[serde(default, skip_serializing_if = "is_default_dim")]
    dimension: RootDimension,
    #[serde(default, skip_serializing_if = "is_default_method")]
    scalar_method: ScalarRootMethod,
    #[serde(default = "default_bmin")]
    bracket_min: f64,
    #[serde(default = "default_bmax")]
    bracket_max: f64,
    #[serde(default = "default_init_2d")]
    init_2d: [f64; 2],
    #[serde(default = "default_span_2d")]
    domain_span_2d: f64,
    #[serde(default = "default_tol", skip_serializing_if = "is_default_tol")]
    tolerance: f64,
    #[serde(default = "default_iters", skip_serializing_if = "is_default_iters")]
    max_iterations: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scalar_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    scalar_params: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    coupled_exprs: Option<[String; 2]>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    coupled_params: BTreeMap<String, f64>,
}

impl RootFinderTool {
    pub(crate) fn max_step_index(&self) -> usize {
        match self.dimension {
            RootDimension::Scalar1D => self.scalar_steps.len().saturating_sub(1),
            RootDimension::System2D => self.system_steps.len().saturating_sub(1),
        }
    }

    fn compute_state_hash(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut h);
        (self.dimension as u8).hash(&mut h);
        (self.scalar_method as u8).hash(&mut h);
        self.bracket_min.to_bits().hash(&mut h);
        self.bracket_max.to_bits().hash(&mut h);
        self.init_2d[0].to_bits().hash(&mut h);
        self.init_2d[1].to_bits().hash(&mut h);
        self.domain_span_2d.to_bits().hash(&mut h);
        self.tolerance.to_bits().hash(&mut h);
        self.max_iterations.hash(&mut h);
        self.selected_step.hash(&mut h);
        self.scalar_input.raw_text.hash(&mut h);
        self.scalar_input.active_param_names().hash(&mut h);
        for &v in self.scalar_input.param_slice() {
            v.to_bits().hash(&mut h);
        }
        self.coupled_bank.raw_text(0).hash(&mut h);
        self.coupled_bank.raw_text(1).hash(&mut h);
        self.coupled_bank.param_names().hash(&mut h);
        for &v in self.coupled_bank.param_slice() {
            v.to_bits().hash(&mut h);
        }
        h.finish().max(1)
    }
}

impl InteractiveTool for RootFinderTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Root Finder"
    }

    fn domain(&self) -> &'static str {
        "analysis"
    }

    fn state_revision(&self) -> u64 {
        self.compute_state_hash()
    }

    fn has_finite_parameters(&self) -> bool {
        self.bracket_min.is_finite()
            && self.bracket_max.is_finite()
            && (ROOT_DOMAIN_MIN..=ROOT_DOMAIN_MAX).contains(&self.bracket_min)
            && (ROOT_DOMAIN_MIN..=ROOT_DOMAIN_MAX).contains(&self.bracket_max)
            && self.bracket_min < self.bracket_max
            && self
                .init_2d
                .iter()
                .all(|v| v.is_finite() && (ROOT_DOMAIN_MIN..=ROOT_DOMAIN_MAX).contains(v))
            && self.domain_span_2d.is_finite()
            && (0.5..=25.0).contains(&self.domain_span_2d)
            && self.tolerance.is_finite()
            && (MIN_ROOT_TOL..=MAX_ROOT_TOL).contains(&self.tolerance)
            && (4..=200).contains(&self.max_iterations)
            && self.final_residual.is_finite()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.bracket_min = if self.bracket_min.is_finite() {
            self.bracket_min
                .clamp(ROOT_DOMAIN_MIN, ROOT_DOMAIN_MAX - ROOT_MIN_SPAN)
        } else {
            0.0
        };
        self.bracket_max = if self.bracket_max.is_finite() {
            self.bracket_max
                .clamp(self.bracket_min + ROOT_MIN_SPAN, ROOT_DOMAIN_MAX)
        } else {
            (self.bracket_min + 2.0).min(ROOT_DOMAIN_MAX)
        };
        for c in &mut self.init_2d {
            *c = if c.is_finite() {
                c.clamp(ROOT_DOMAIN_MIN, ROOT_DOMAIN_MAX)
            } else {
                1.0
            };
        }
        self.domain_span_2d = if self.domain_span_2d.is_finite() {
            self.domain_span_2d.clamp(0.5, 25.0)
        } else {
            3.0
        };
        self.tolerance = if self.tolerance.is_finite() {
            self.tolerance.clamp(MIN_ROOT_TOL, MAX_ROOT_TOL)
        } else {
            DEFAULT_ROOT_TOL
        };
        self.max_iterations = self.max_iterations.clamp(4, 200);
        self.recalculate();
        self.selected_step = self.max_step_index();
        true
    }

    fn save_state(&self) -> Option<String> {
        let is_2d = self.dimension == RootDimension::System2D;
        let scalar_expr = (!is_2d && self.scalar_input.raw_text != DEFAULT_SCALAR_EXPR)
            .then(|| self.scalar_input.raw_text.clone());
        let scalar_params = if !is_2d {
            self.scalar_input.params_map()
        } else {
            BTreeMap::new()
        };
        let coupled_exprs = is_2d.then(|| {
            [
                self.coupled_bank.raw_text(0).to_string(),
                self.coupled_bank.raw_text(1).to_string(),
            ]
        });
        let coupled_params = if is_2d {
            self.coupled_bank.params_map()
        } else {
            BTreeMap::new()
        };
        serde_json::to_string(&RootFinderState {
            dimension: self.dimension,
            scalar_method: self.scalar_method,
            bracket_min: self.bracket_min,
            bracket_max: self.bracket_max,
            init_2d: self.init_2d,
            domain_span_2d: self.domain_span_2d,
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
            scalar_expr,
            scalar_params,
            coupled_exprs,
            coupled_params,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: RootFinderState =
            serde_json::from_str(state).map_err(|e| format!("Invalid RootFinder state: {e}"))?;
        let (scalar_bar, coupled_bank) = validate_loaded_root_state(&s)?;
        self.dimension = s.dimension;
        self.scalar_method = s.scalar_method;
        self.bracket_min = s.bracket_min;
        self.bracket_max = s.bracket_max;
        self.init_2d = s.init_2d;
        self.domain_span_2d = s.domain_span_2d;
        self.tolerance = s.tolerance;
        self.max_iterations = s.max_iterations;
        self.scalar_input = scalar_bar;
        self.coupled_bank = coupled_bank;
        self.recalculate();
        self.selected_step = self.max_step_index();
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
            self.selected_step = self.max_step_index();
        }
        self.draw_plot(parent_ui);
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "RootFinderTool",
        display_name: "Root Finder",
        domain: "analysis",
        tags: &["roots", "bisection", "brent", "newton", "secant"],
        build: || Box::new(RootFinderTool::default()),
    }
}

fn validate_loaded_root_state(
    s: &RootFinderState,
) -> Result<(MathExpressionInputBar, CoupledExpressionBank), String> {
    if !s.bracket_min.is_finite()
        || !s.bracket_max.is_finite()
        || !(ROOT_DOMAIN_MIN..=ROOT_DOMAIN_MAX).contains(&s.bracket_min)
        || !(ROOT_DOMAIN_MIN..=ROOT_DOMAIN_MAX).contains(&s.bracket_max)
        || s.bracket_min >= s.bracket_max
        || !s
            .init_2d
            .iter()
            .all(|v| v.is_finite() && (ROOT_DOMAIN_MIN..=ROOT_DOMAIN_MAX).contains(v))
        || !s.domain_span_2d.is_finite()
        || !(0.5..=25.0).contains(&s.domain_span_2d)
        || !s.tolerance.is_finite()
        || !(MIN_ROOT_TOL..=MAX_ROOT_TOL).contains(&s.tolerance)
        || !(4..=200).contains(&s.max_iterations)
        || !s.scalar_params.values().all(|v| v.is_finite())
        || !s.coupled_params.values().all(|v| v.is_finite())
    {
        return Err("RootFinder bounds, initial guess, or tolerance out of range".into());
    }
    let mut scalar_bar = default_scalar_bar();
    if let Some(ref expr) = s.scalar_expr {
        scalar_bar.set_expression(expr);
    }
    if !s.scalar_params.is_empty() {
        scalar_bar.set_params_map(&s.scalar_params);
    }
    let mut coupled_bank = default_coupled_2d_bank();
    if let Some(ref exprs) = s.coupled_exprs {
        coupled_bank.set_expressions(exprs);
    }
    if !s.coupled_params.is_empty() {
        coupled_bank.set_params_map(&s.coupled_params);
    }
    let expr_err = (s.dimension == RootDimension::System2D || s.coupled_exprs.is_some())
        .then(|| coupled_bank.first_error())
        .flatten()
        .or_else(|| {
            (s.dimension == RootDimension::Scalar1D || s.scalar_expr.is_some())
                .then(|| scalar_bar.error_message())
                .flatten()
        });
    if let Some(err) = expr_err {
        let msg = format!("Invalid RootFinder custom formula: {err}");
        crate::accessibility::announce_status_with_priority(&msg, "assertive");
        return Err(msg);
    }
    Ok((scalar_bar, coupled_bank))
}

#[cfg(test)]
#[path = "root_finder/tests.rs"]
mod tests;
