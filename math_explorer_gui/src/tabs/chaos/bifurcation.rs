#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::atlas::ParameterAtlasController;
use crate::async_sim::BackgroundWorker;
use crate::framework::{InteractiveTool, MathExpressionInputBar};
use eframe::egui;
use egui_plot::PlotPoint;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};

#[path = "bifurcation/runner.rs"]
pub mod runner;
#[cfg(test)]
pub use runner::BifurcationSweepSnapshot;
pub use runner::{BifurcationRunner, BifurcationSweepRequest, CompiledMap1DJob};

#[path = "bifurcation/atlas.rs"]
pub mod atlas;
#[cfg(test)]
pub use atlas::BifurcationCartographer;
use atlas::{BifurcationState, BIFURCATION_ATLAS_AXIS_PAIRS};

#[path = "bifurcation/compute.rs"]
mod compute;
#[cfg(test)]
use compute::clamp_lyapunov_value;

#[path = "bifurcation/plots.rs"]
mod plots;

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

pub(crate) const CUSTOM_BURN_IN: usize = 200;
pub(crate) const CUSTOM_ORBIT_SAMPLES: usize = 100;
pub(crate) const CUSTOM_ESCAPE_GUARD: f64 = 1e6;
pub(crate) const CUSTOM_R_BOUND: f64 = 20.0;

/// Map family evaluated by [`BifurcationDiagram`].
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum BifurcationMapMode {
    /// Canonical logistic map $x_{n+1} = r x_n (1 - x_n)$ on $r \in [0, 4]$.
    #[default]
    Logistic,
    /// User-defined 1D iteration map $x_{n+1} = f(r, x)$ evaluated via `CompiledExpr::eval_2d`.
    Custom,
}

impl BifurcationMapMode {
    /// Returns `true` when evaluating the canonical logistic map.
    #[must_use]
    pub const fn is_logistic(&self) -> bool {
        matches!(self, Self::Logistic)
    }

    /// Returns `true` when evaluating a user-authored `CompiledExpr` 1D map.
    #[must_use]
    pub const fn is_custom(&self) -> bool {
        matches!(self, Self::Custom)
    }
}

fn default_bifurcation_input_bar() -> MathExpressionInputBar {
    MathExpressionInputBar::new(
        "x_{n+1}",
        DEFAULT_CUSTOM_EXPR,
        vec![
            ("Sine Map", "r * sin(pi * x)"),
            ("Quadratic Map", "x^2 - r"),
            ("Ricker Population", "x * exp(r * (1 - x))"),
            ("Cubic Map", "r * x - x^3"),
            (
                "Perturbed Logistic",
                "r * x * (1 - x) + a * sin(2 * pi * x)",
            ),
        ],
    )
    .with_axes(&["r", "x"])
    .with_param_range(-2.0..=2.0, 0.05)
}

/// Interactive 1D bifurcation explorer with coupled Lyapunov, cobweb, orbit, and 2D atlas views.
pub struct BifurcationDiagram {
    /// Active 1D map family (`Logistic` or `Custom` bytecode expression).
    pub map_mode: BifurcationMapMode,
    pub(crate) r_min: f64,
    pub(crate) r_max: f64,
    pub(crate) steps: usize,
    /// Active growth rate `r` tracked by pointer hover, click pin, keyboard stepping, or audio sweep.
    pub selected_r: f64,
    pub(crate) probe_r: f64,
    /// Initial state `x_0` used for orbit, cobweb, and Lyapunov evaluation.
    pub x0: f64,
    /// Whether `selected_r` is pinned against pointer hover updates.
    pub r_pinned: bool,
    pub(crate) r_from_hover: bool,
    pub(crate) orbit_x: f64,
    pub(crate) orbit_step_count: usize,
    /// Reusable `CompiledExpr` formula bar for custom `x_{n+1} = f(r, x)` maps.
    pub math_input: MathExpressionInputBar,
    /// Inline evaluation or divergence error surfaced for custom AST maps.
    pub custom_eval_error: Option<String>,
    pub(crate) custom_diverged: bool,
    pub(crate) points: Vec<PlotPoint>,
    pub(crate) lyapunov_points: Vec<PlotPoint>,
    /// Lyapunov exponent `λ` evaluated at `(selected_r, x0)`.
    pub selected_lyapunov: f64,
    pub(crate) cobweb_points: Vec<PlotPoint>,
    pub(crate) cobweb_curve: Vec<PlotPoint>,
    pub(crate) cobweb_diagonal: Vec<PlotPoint>,
    pub(crate) cobweb_transient: Vec<PlotPoint>,
    pub(crate) cobweb_cycle: Vec<PlotPoint>,
    pub(crate) orbit_points: Vec<PlotPoint>,
    pub(crate) focused_canvas: Option<egui::Id>,
    pub(crate) atlas: ParameterAtlasController,
    pub(crate) request_generation: u64,
    pub(crate) completed_generation: u64,
    pub(crate) shared_request: Arc<RwLock<BifurcationSweepRequest>>,
    pub(crate) shared_formula: Arc<RwLock<Option<Arc<CompiledMap1DJob>>>>,
    pub(crate) latest_generation: Arc<AtomicU64>,
    pub(crate) worker: BackgroundWorker<BifurcationRunner>,
}

impl Default for BifurcationDiagram {
    fn default() -> Self {
        let r_min = 2.5;
        let r_max = 4.0;
        let selected_r = (r_min + r_max) * 0.5;
        let math_input = default_bifurcation_input_bar();
        let initial_req = BifurcationSweepRequest {
            generation: 0,
            map_mode: BifurcationMapMode::Logistic,
            r_min,
            r_max,
            steps: 500,
            x0: DEFAULT_X0,
        };
        let shared_request = Arc::new(RwLock::new(initial_req));
        let shared_formula = Arc::new(RwLock::new(Some(Arc::new(CompiledMap1DJob::from_bar(
            &math_input,
        )))));
        let latest_generation = Arc::new(AtomicU64::new(0));
        let worker = BackgroundWorker::new(BifurcationRunner::new(
            Arc::clone(&shared_request),
            Arc::clone(&latest_generation),
            Arc::clone(&shared_formula),
        ));

        let mut tool = Self {
            map_mode: BifurcationMapMode::Logistic,
            r_min,
            r_max,
            steps: 500,
            selected_r,
            probe_r: selected_r,
            x0: DEFAULT_X0,
            r_pinned: false,
            r_from_hover: false,
            orbit_x: DEFAULT_X0,
            orbit_step_count: 0,
            math_input,
            custom_eval_error: None,
            custom_diverged: false,
            points: Vec::new(),
            lyapunov_points: Vec::new(),
            selected_lyapunov: 0.0,
            cobweb_points: Vec::new(),
            cobweb_curve: Vec::new(),
            cobweb_diagonal: Vec::new(),
            cobweb_transient: Vec::new(),
            cobweb_cycle: Vec::new(),
            orbit_points: Vec::new(),
            focused_canvas: None,
            atlas: ParameterAtlasController::new(BIFURCATION_ATLAS_AXIS_PAIRS.to_vec()),
            request_generation: 0,
            completed_generation: 0,
            shared_request,
            shared_formula,
            latest_generation,
            worker,
        };
        tool.recompute();
        tool
    }
}

impl BifurcationDiagram {
    #[inline]
    pub(crate) fn eval_active_map(&self, r: f64, x: f64) -> f64 {
        match self.map_mode {
            BifurcationMapMode::Logistic => r * x * (1.0 - x),
            BifurcationMapMode::Custom => {
                if let Some(compiled) = self.math_input.active_compiled() {
                    compiled.eval_2d(r, x, self.math_input.param_slice())
                } else {
                    x
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    pub(crate) fn reset_live_orbit(&mut self) {
        self.selected_r = self.probe_r;
        self.recompute_coupled_views();
    }

    fn build_state_json(&self, include_hover_r: bool) -> Option<String> {
        let is_custom = self.map_mode.is_custom();
        let midpoint = (self.r_min + self.r_max) * 0.5;
        let has_selected_r = self.r_pinned
            || ((include_hover_r || !self.r_from_hover)
                && (self.selected_r - midpoint).abs() > 1e-12);
        let selected_r = has_selected_r.then_some(self.selected_r);
        let x0 =
            (is_custom || self.r_pinned || (self.x0 - DEFAULT_X0).abs() > 1e-12).then_some(self.x0);
        let trimmed_expr = self.math_input.raw_text.trim();
        let custom_expr = (is_custom || trimmed_expr != DEFAULT_CUSTOM_EXPR)
            .then(|| self.math_input.raw_text.clone());
        let custom_params = if is_custom {
            self.math_input.params_map()
        } else {
            BTreeMap::new()
        };
        let (xk, yk) = self.atlas.axis_keys();
        serde_json::to_string(&BifurcationState {
            r_min: self.r_min,
            r_max: self.r_max,
            steps: self.steps,
            map_mode: self.map_mode,
            selected_r,
            x0,
            r_pinned: self.r_pinned,
            use_custom_map: false,
            custom_expr,
            custom_params,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        })
        .ok()
    }
}

impl TheoryDescribable for BifurcationDiagram {
    fn theory_description(&self) -> String {
        match self.map_mode {
            BifurcationMapMode::Logistic => format!(
                "Logistic map bifurcation diagram, r range: {:.2} to {:.2}, selected r: {:.3}",
                self.r_min, self.r_max, self.selected_r
            ),
            BifurcationMapMode::Custom => format!(
                "Custom 1D map bifurcation x_{{n+1}} = {}, r range: {:.2} to {:.2}, selected r: {:.3}",
                self.math_input.raw_text, self.r_min, self.r_max, self.selected_r
            ),
        }
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

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        self.build_theory_parameters()
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

    fn step_simulation(&mut self) {
        let active_r = if (self.probe_r - self.selected_r).abs() > 1e-12 {
            self.probe_r
        } else {
            self.selected_r
        };
        let next = self.eval_active_map(active_r, self.orbit_x);
        if next.is_finite() && next.abs() <= CUSTOM_ESCAPE_GUARD {
            self.orbit_x = next;
            self.orbit_step_count = self.orbit_step_count.saturating_add(1);
        }
    }

    fn has_finite_parameters(&self) -> bool {
        let (r_bounds, x0_bounds) = match self.map_mode {
            BifurcationMapMode::Logistic => (0.0..=4.0, -0.25..=1.25),
            BifurcationMapMode::Custom => (-CUSTOM_R_BOUND..=CUSTOM_R_BOUND, -10.0..=10.0),
        };
        self.r_min.is_finite()
            && self.r_max.is_finite()
            && r_bounds.contains(&self.r_min)
            && r_bounds.contains(&self.r_max)
            && self.r_min < self.r_max
            && (100..=2000).contains(&self.steps)
            && self.selected_r.is_finite()
            && r_bounds.contains(&self.selected_r)
            && self.x0.is_finite()
            && x0_bounds.contains(&self.x0)
            && !self.custom_diverged
    }

    fn clamp_parameters(&mut self) -> bool {
        if self.custom_diverged {
            self.map_mode = BifurcationMapMode::Logistic;
            self.custom_diverged = false;
            self.custom_eval_error = None;
        }
        let (r_lo, r_hi, x_lo, x_hi) = match self.map_mode {
            BifurcationMapMode::Logistic => (0.0, 4.0, 0.001, 0.999),
            BifurcationMapMode::Custom => (-CUSTOM_R_BOUND, CUSTOM_R_BOUND, -10.0, 10.0),
        };
        self.r_min = if self.r_min.is_finite() {
            self.r_min.clamp(r_lo, r_hi - 0.1)
        } else {
            2.5
        };
        self.r_max = if self.r_max.is_finite() {
            self.r_max.clamp(self.r_min + 0.1, r_hi)
        } else {
            4.0
        };
        self.steps = self.steps.clamp(100, 2000);
        self.selected_r = if self.selected_r.is_finite() {
            self.selected_r.clamp(self.r_min, self.r_max)
        } else {
            (self.r_min + self.r_max) * 0.5
        };
        self.probe_r = self.selected_r;
        self.x0 = if self.x0.is_finite() {
            self.x0.clamp(x_lo, x_hi)
        } else {
            DEFAULT_X0
        };
        self.recompute();
        true
    }

    fn save_state(&self) -> Option<String> {
        self.build_state_json(true)
    }

    fn export_url_state(&self) -> Option<String> {
        self.build_state_json(false)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: BifurcationState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bifurcation state: {e}"))?;
        let (mode, sel_r, x0, expr_str) = s.validate()?;
        let mut candidate = Self {
            map_mode: mode,
            r_min: s.r_min,
            r_max: s.r_max,
            steps: s.steps,
            selected_r: sel_r,
            probe_r: sel_r,
            x0,
            r_pinned: s.r_pinned,
            ..Self::default()
        };
        candidate.math_input.set_expression(&expr_str);
        candidate.math_input.set_params_map(&s.custom_params);
        candidate.atlas.set_visible(s.atlas_visible);
        let _ = candidate
            .atlas
            .set_axis_keys(&s.atlas_axes[0], &s.atlas_axes[1]);
        candidate.recompute();
        if candidate.custom_diverged || (mode.is_custom() && !candidate.math_input.is_valid()) {
            return Err("Custom bifurcation expression failed finite validation".into());
        }
        *self = candidate;
        Ok(())
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let to_vec = |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect();
        vec![
            ("Attractor".to_string(), to_vec(&self.points)),
            (
                "Lyapunov Exponent".to_string(),
                to_vec(&self.lyapunov_points),
            ),
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
        let ctx = ui.ctx().clone();
        self.sync_from_worker(&ctx);
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
        tags: &[
            "logistic map",
            "period doubling",
            "feigenbaum",
            "orbit diagram",
            "lyapunov",
            "cobweb",
            "bifurcation",
            "chaos",
        ],
        build: || Box::new(BifurcationDiagram::default()),
    }
}

#[cfg(test)]
#[path = "bifurcation/tests.rs"]
mod tests;
