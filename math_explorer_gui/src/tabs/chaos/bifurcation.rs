#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{InteractiveTool, MathExpressionInputBar};
use eframe::egui;
use egui_plot::PlotPoint;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

#[path = "bifurcation/compute.rs"]
mod compute;
#[cfg(test)]
use compute::clamp_lyapunov_value;

#[path = "bifurcation/ui.rs"]
mod ui;

#[path = "bifurcation/presets.rs"]
mod presets;
#[cfg(test)]
use presets::{STARTER_PERIOD3_WINDOW_JSON, STARTER_PRESET_JSON};

/// Default initial condition `x_0` avoiding the `x = 0.5` logistic derivative singularity.
pub const DEFAULT_X0: f64 = 0.1;
/// Iteration horizon passed to `logistic_lyapunov` so `r = 3.83, x_0 = 0.1` evaluates to `-0.345`.
pub const LYAPUNOV_ITERATIONS: usize = 2000;
/// Default custom 1D recurrence seeded in `MathExpressionInputBar`.
pub const DEFAULT_CUSTOM_EXPR: &str = "r * sin(pi * x)";

const MAX_CUSTOM_EXPR_BYTES: usize = 256;

fn is_false(flag: &bool) -> bool {
    !*flag
}

/// Interactive 1D bifurcation explorer with coupled Lyapunov, cobweb, and orbit views.
pub struct BifurcationDiagram {
    pub(crate) r_min: f64,
    pub(crate) r_max: f64,
    pub(crate) steps: usize,
    /// Active growth rate `r` tracked by pointer hover, click pin, or keyboard stepping.
    pub selected_r: f64,
    /// Initial state `x_0` in `[0, 1]` used for orbit, cobweb, and Lyapunov evaluation.
    pub x0: f64,
    /// Whether `selected_r` is pinned against pointer hover updates.
    pub r_pinned: bool,
    /// Whether custom `math_commons::ast` recurrence evaluation is active.
    pub use_custom_map: bool,
    /// Reusable AST formula bar for custom `x_{n+1} = f(x, r)` maps.
    pub math_input: MathExpressionInputBar,
    /// Inline evaluation or divergence error surfaced for custom AST maps.
    pub custom_eval_error: Option<String>,
    pub(crate) custom_diverged: bool,
    pub(crate) points: Vec<PlotPoint>,
    pub(crate) lyapunov_points: Vec<PlotPoint>,
    /// Lyapunov exponent `λ` evaluated at `(selected_r, x0)`.
    pub selected_lyapunov: f64,
    pub(crate) cobweb_curve: Vec<PlotPoint>,
    pub(crate) cobweb_diagonal: Vec<PlotPoint>,
    pub(crate) cobweb_transient: Vec<PlotPoint>,
    pub(crate) cobweb_cycle: Vec<PlotPoint>,
    pub(crate) orbit_points: Vec<PlotPoint>,
}

impl Default for BifurcationDiagram {
    fn default() -> Self {
        let r_min = 2.5;
        let r_max = 4.0;
        let mut tool = Self {
            r_min,
            r_max,
            steps: 500,
            selected_r: (r_min + r_max) * 0.5,
            x0: DEFAULT_X0,
            r_pinned: false,
            use_custom_map: false,
            math_input: MathExpressionInputBar::new(
                "x_{n+1}",
                DEFAULT_CUSTOM_EXPR,
                vec![
                    ("Sine Map: r * sin(pi * x)", "r * sin(pi * x)"),
                    ("Logistic Map: r * x * (1 - x)", "r * x * (1 - x)"),
                    ("Cubic Map: r * x * (1 - x^2)", "r * x * (1 - x^2)"),
                    ("Smooth Tent: r * (0.5 - abs(x - 0.5))", "r * (0.5 - abs(x - 0.5))"),
                ],
            ),
            custom_eval_error: None,
            custom_diverged: false,
            points: Vec::new(),
            lyapunov_points: Vec::new(),
            selected_lyapunov: 0.0,
            cobweb_curve: Vec::new(),
            cobweb_diagonal: Vec::new(),
            cobweb_transient: Vec::new(),
            cobweb_cycle: Vec::new(),
            orbit_points: Vec::new(),
        };
        tool.recompute();
        tool
    }
}

impl TheoryDescribable for BifurcationDiagram {
    fn theory_description(&self) -> String {
        format!(
            "Logistic map bifurcation diagram, r range: {:.2} to {:.2}, selected r: {:.3}",
            self.r_min, self.r_max, self.selected_r
        )
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        "May, R. M. (1976). Simple mathematical models with very complicated dynamics. Nature, 261(5560), 459-467; Feigenbaum, M. J. (1978). Quantitative universality for a class of nonlinear transformations. Journal of Statistical Physics, 19(1), 25-52.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "default".to_string(),
            "Logistic map bifurcation diagram".to_string(),
        );
        map
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BifurcationState {
    pub(super) r_min: f64,
    pub(super) r_max: f64,
    pub(super) steps: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) selected_r: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) x0: Option<f64>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub(super) r_pinned: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub(super) use_custom_map: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) custom_expr: Option<String>,
}

impl BifurcationState {
    fn validate(&self) -> Result<(f64, f64, String), String> {
        if !self.r_min.is_finite()
            || !self.r_max.is_finite()
            || !(0.0..=4.0).contains(&self.r_min)
            || !(0.0..=4.0).contains(&self.r_max)
            || self.r_min >= self.r_max
            || !(100..=2000).contains(&self.steps)
        {
            return Err("Bifurcation r bounds or step count invalid".into());
        }
        let sel_r = self.selected_r.unwrap_or((self.r_min + self.r_max) * 0.5);
        if !sel_r.is_finite() || !(0.0..=4.0).contains(&sel_r) {
            return Err("Bifurcation selected_r out of bounds [0, 4]".into());
        }
        let x0 = self.x0.unwrap_or(DEFAULT_X0);
        if !x0.is_finite() || !(0.0..=1.0).contains(&x0) {
            return Err("Bifurcation x0 out of bounds [0, 1]".into());
        }
        let expr_str = self
            .custom_expr
            .as_deref()
            .unwrap_or(DEFAULT_CUSTOM_EXPR)
            .trim()
            .to_string();
        if expr_str.len() > MAX_CUSTOM_EXPR_BYTES {
            return Err("Custom map expression exceeds maximum length".into());
        }
        if self.use_custom_map && math_commons::ast::parse_equation(&expr_str).is_err() {
            return Err("Invalid custom 1D map syntax in state".into());
        }
        Ok((sel_r, x0, expr_str))
    }
}

impl InteractiveTool for BifurcationDiagram {
    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Bifurcation Diagram"
    }

    fn domain(&self) -> &'static str {
        "chaos"
    }

    fn has_finite_parameters(&self) -> bool {
        self.r_min.is_finite()
            && self.r_max.is_finite()
            && (0.0..=4.0).contains(&self.r_min)
            && (0.0..=4.0).contains(&self.r_max)
            && self.r_min < self.r_max
            && (100..=2000).contains(&self.steps)
            && self.selected_r.is_finite()
            && (0.0..=4.0).contains(&self.selected_r)
            && self.x0.is_finite()
            && (0.0..=1.0).contains(&self.x0)
            && !self.custom_diverged
    }

    fn clamp_parameters(&mut self) -> bool {
        self.r_min = if self.r_min.is_finite() {
            self.r_min.clamp(0.0, 3.9)
        } else {
            2.5
        };
        self.r_max = if self.r_max.is_finite() {
            self.r_max.clamp(self.r_min + 0.1, 4.0)
        } else {
            4.0
        };
        self.steps = self.steps.clamp(100, 2000);
        self.selected_r = if self.selected_r.is_finite() {
            self.selected_r.clamp(self.r_min, self.r_max)
        } else {
            (self.r_min + self.r_max) * 0.5
        };
        self.x0 = if self.x0.is_finite() {
            self.x0.clamp(0.001, 0.999)
        } else {
            DEFAULT_X0
        };
        if self.custom_diverged {
            self.use_custom_map = false;
            self.custom_diverged = false;
            self.custom_eval_error = None;
        }
        self.recompute();
        true
    }

    fn save_state(&self) -> Option<String> {
        let midpoint = (self.r_min + self.r_max) * 0.5;
        let selected_r = (self.r_pinned || (self.selected_r - midpoint).abs() > 1e-12)
            .then_some(self.selected_r);
        let x0 = (self.r_pinned || (self.x0 - DEFAULT_X0).abs() > 1e-12).then_some(self.x0);
        let trimmed_expr = self.math_input.raw_text.trim();
        let custom_expr = (self.use_custom_map || trimmed_expr != DEFAULT_CUSTOM_EXPR)
            .then(|| trimmed_expr.to_string());

        serde_json::to_string(&BifurcationState {
            r_min: self.r_min,
            r_max: self.r_max,
            steps: self.steps,
            selected_r,
            x0,
            r_pinned: self.r_pinned,
            use_custom_map: self.use_custom_map,
            custom_expr,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: BifurcationState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bifurcation state: {e}"))?;
        let (sel_r, x0, expr_str) = s.validate()?;
        let mut candidate = Self {
            r_min: s.r_min,
            r_max: s.r_max,
            steps: s.steps,
            selected_r: sel_r,
            x0,
            r_pinned: s.r_pinned,
            use_custom_map: s.use_custom_map,
            ..Self::default()
        };
        candidate.math_input.set_expression(&expr_str);
        candidate.recompute();
        if candidate.custom_diverged || candidate.custom_eval_error.is_some() {
            return Err("Custom bifurcation expression failed finite validation".into());
        }
        *self = candidate;
        Ok(())
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let to_vec = |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect();
        vec![
            ("Attractor".to_string(), to_vec(&self.points)),
            ("Lyapunov Exponent".to_string(), to_vec(&self.lyapunov_points)),
            ("Cobweb Cycle".to_string(), to_vec(&self.cobweb_cycle)),
            ("Orbit x_n".to_string(), to_vec(&self.orbit_points)),
        ]
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.draw_side_panel(ui);
        self.draw_central_panel(ui);
    }
}

// [cite:algorithmic_information_rust]
// [cite:chaos_theory]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BifurcationDiagram",
        display_name: "Bifurcation Diagram",
        domain: "chaos",
        tags: &["logistic", "lyapunov", "cobweb", "bifurcation"],
        build: || Box::new(BifurcationDiagram::default()),
    }
}

#[cfg(test)]
#[path = "bifurcation/tests.rs"]
mod tests;
