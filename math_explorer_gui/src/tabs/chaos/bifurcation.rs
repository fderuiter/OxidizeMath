#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::atlas::ParameterAtlasController;
use crate::async_sim::BackgroundWorker;
use crate::framework::{InteractiveTool, MathExpressionInputBar};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::physics::chaos::logistic;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};

#[path = "bifurcation/runner.rs"]
pub mod runner;
#[cfg(test)]
pub use runner::BifurcationSweepSnapshot;
pub use runner::{
    burned_logistic_lyapunov, BifurcationRunner, BifurcationSweepRequest, CompiledMap1DJob,
};

#[path = "bifurcation/atlas.rs"]
pub mod atlas;
#[cfg(test)]
pub use atlas::BifurcationCartographer;
use atlas::{BifurcationState, BIFURCATION_ATLAS_AXIS_PAIRS};

#[path = "bifurcation/ui.rs"]
mod ui;

pub(crate) const DEFAULT_CUSTOM_MAP_EXPR: &str = "r * sin(pi * x)";
pub(crate) const CUSTOM_BURN_IN: usize = 200;
pub(crate) const CUSTOM_ORBIT_SAMPLES: usize = 100;
pub(crate) const CUSTOM_ESCAPE_GUARD: f64 = 1e6;
pub(crate) const CUSTOM_R_BOUND: f64 = 20.0;
pub(crate) const COBWEB_STEPS: usize = 40;
const COBWEB_CURVE_SAMPLES: usize = 128;

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
    const fn is_logistic(&self) -> bool {
        matches!(self, Self::Logistic)
    }
}

fn default_bifurcation_input_bar() -> MathExpressionInputBar {
    MathExpressionInputBar::new(
        "x_{n+1}",
        DEFAULT_CUSTOM_MAP_EXPR,
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

/// Interactive 1D map bifurcation diagram coupled to live `logistic_lyapunov`,
/// cobweb, and $x_n$ iterate time-series inspectors via progressive `BackgroundWorker` sweeps.
pub struct BifurcationDiagram {
    pub(crate) map_mode: BifurcationMapMode,
    pub(crate) r_min: f64,
    pub(crate) r_max: f64,
    pub(crate) probe_r: f64,
    pub(crate) steps: usize,
    pub(crate) x0: f64,
    pub(crate) orbit_x: f64,
    pub(crate) orbit_step_count: usize,
    pub(crate) math_input: MathExpressionInputBar,
    pub(crate) points: Vec<PlotPoint>,
    pub(crate) lyapunov_points: Vec<PlotPoint>,
    pub(crate) cobweb_points: Vec<PlotPoint>,
    pub(crate) cobweb_curve: Vec<PlotPoint>,
    pub(crate) orbit_series: Vec<PlotPoint>,
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
        let math_input = default_bifurcation_input_bar();
        let initial_req = BifurcationSweepRequest {
            generation: 0,
            map_mode: BifurcationMapMode::Logistic,
            r_min: 2.5,
            r_max: 4.0,
            steps: 500,
            x0: 0.5,
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
            r_min: 2.5,
            r_max: 4.0,
            probe_r: 3.50,
            steps: 500,
            x0: 0.5,
            orbit_x: 0.5,
            orbit_step_count: 0,
            math_input,
            points: Vec::new(),
            lyapunov_points: Vec::with_capacity(2002),
            cobweb_points: Vec::with_capacity(COBWEB_STEPS * 2 + 4),
            cobweb_curve: Vec::with_capacity(COBWEB_CURVE_SAMPLES + 1),
            orbit_series: Vec::with_capacity(COBWEB_STEPS + 1),
            atlas: ParameterAtlasController::new(BIFURCATION_ATLAS_AXIS_PAIRS.to_vec()),
            request_generation: 0,
            completed_generation: 0,
            shared_request,
            shared_formula,
            latest_generation,
            worker,
        };
        tool.recompute();
        tool.rebuild_cobweb_buffers();
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

    /// Selects a probe control parameter `r` clamped to `[r_min, r_max]` and refreshes live orbits.
    pub fn select_probe_r(&mut self, r: f64) {
        if r.is_finite() {
            let (lo, hi) = match self.map_mode {
                BifurcationMapMode::Logistic => (0.0, 4.0),
                BifurcationMapMode::Custom => (-CUSTOM_R_BOUND, CUSTOM_R_BOUND),
            };
            self.probe_r = r.clamp(self.r_min, self.r_max).clamp(lo, hi);
            self.reset_live_orbit();
        }
    }

    /// Evaluates the burned-in Lyapunov exponent $\lambda(r)$ at `probe_r`.
    #[must_use]
    pub fn probe_lyapunov(&self) -> Option<f64> {
        match self.map_mode {
            BifurcationMapMode::Logistic => {
                let seed = if (self.x0 - 0.5).abs() < 1e-9 {
                    0.314_159
                } else {
                    self.x0
                };
                burned_logistic_lyapunov(self.probe_r, seed, 96)
            }
            BifurcationMapMode::Custom => {
                let job = CompiledMap1DJob::from_bar(&self.math_input);
                job.lyapunov(self.probe_r, self.x0, None, 80, 96)
            }
        }
    }

    pub(crate) fn reset_live_orbit(&mut self) {
        self.orbit_x = self.x0;
        self.orbit_step_count = 0;
        self.rebuild_cobweb_buffers();
    }

    pub(crate) fn rebuild_cobweb_buffers(&mut self) {
        self.cobweb_points.clear();
        self.cobweb_curve.clear();
        self.orbit_series.clear();
        let (x_lo, x_hi) = match self.map_mode {
            BifurcationMapMode::Logistic => (0.0, 1.0),
            BifurcationMapMode::Custom => (-2.0, 2.0),
        };
        for idx in 0..=COBWEB_CURVE_SAMPLES {
            let x = x_lo + (idx as f64 / COBWEB_CURVE_SAMPLES as f64) * (x_hi - x_lo);
            let y = self.eval_active_map(self.probe_r, x);
            if y.is_finite() && y.abs() <= CUSTOM_ESCAPE_GUARD {
                self.cobweb_curve.push(PlotPoint::new(x, y));
            }
        }
        let mut x = self.x0;
        if x.is_finite() && x.abs() <= CUSTOM_ESCAPE_GUARD {
            self.cobweb_points.push(PlotPoint::new(x, x));
            self.orbit_series.push(PlotPoint::new(0.0, x));
            for step in 1..=COBWEB_STEPS {
                let y = self.eval_active_map(self.probe_r, x);
                if !y.is_finite() || y.abs() > CUSTOM_ESCAPE_GUARD {
                    break;
                }
                self.cobweb_points.push(PlotPoint::new(x, y));
                self.cobweb_points.push(PlotPoint::new(y, y));
                self.orbit_series.push(PlotPoint::new(step as f64, y));
                x = y;
            }
        }
    }

    pub(crate) fn recompute(&mut self) {
        match self.map_mode {
            BifurcationMapMode::Logistic => self.recompute_logistic(),
            BifurcationMapMode::Custom => self.recompute_custom(),
        }
    }

    fn recompute_logistic(&mut self) {
        let steps = self.steps.clamp(2, 2000);
        self.points = logistic::generate_bifurcation_diagram(self.r_min, self.r_max, steps)
            .into_iter()
            .map(|(r, x)| PlotPoint::new(r, x))
            .collect();
        self.lyapunov_points.clear();
        let step_size = (self.r_max - self.r_min) / (steps as f64);
        let seed = if (self.x0 - 0.5).abs() < 1e-9 {
            0.314_159
        } else {
            self.x0.clamp(1e-4, 1.0 - 1e-4)
        };
        for i in 0..=steps {
            let r = self.r_min + step_size * (i as f64);
            if let Some(lam) = burned_logistic_lyapunov(r, seed, 96) {
                self.lyapunov_points.push(PlotPoint::new(r, lam));
            }
        }
    }

    fn recompute_custom(&mut self) {
        let Some(compiled) = self.math_input.active_compiled() else {
            return;
        };
        let params = self.math_input.param_slice();
        let steps = self.steps.clamp(2, 2000);
        let span = self.r_max - self.r_min;
        self.points.clear();
        self.lyapunov_points.clear();
        if !span.is_finite() || span <= 0.0 {
            return;
        }
        let denom = (steps - 1) as f64;
        for step_idx in 0..steps {
            let r = self.r_min + (step_idx as f64 / denom) * span;
            let mut x = self.x0;
            for _ in 0..CUSTOM_BURN_IN {
                if !x.is_finite() || x.abs() > CUSTOM_ESCAPE_GUARD {
                    break;
                }
                x = compiled.eval_2d(r, x, params);
            }
            if !x.is_finite() || x.abs() > CUSTOM_ESCAPE_GUARD {
                continue;
            }
            let mut lyap_sum = 0.0;
            let mut lyap_cnt = 0usize;
            for _ in 0..CUSTOM_ORBIT_SAMPLES {
                let fp = compiled.eval_2d(r, x + 1e-6, params);
                let fm = compiled.eval_2d(r, x - 1e-6, params);
                if fp.is_finite() && fm.is_finite() {
                    lyap_sum += ((fp - fm) / 2e-6).abs().max(1e-15).ln();
                    lyap_cnt += 1;
                }
                x = compiled.eval_2d(r, x, params);
                if !x.is_finite() || x.abs() > CUSTOM_ESCAPE_GUARD {
                    break;
                }
                self.points.push(PlotPoint::new(r, x));
            }
            if lyap_cnt > 0 {
                self.lyapunov_points
                    .push(PlotPoint::new(r, lyap_sum / (lyap_cnt as f64)));
            }
        }
    }
}

impl TheoryDescribable for BifurcationDiagram {
    fn theory_description(&self) -> String {
        match self.map_mode {
            BifurcationMapMode::Logistic => format!(
                "Logistic map bifurcation diagram, r range: {:.2} to {:.2}",
                self.r_min, self.r_max
            ),
            BifurcationMapMode::Custom => format!(
                "Custom 1D map bifurcation x_{{n+1}} = {}, r range: {:.2} to {:.2}",
                self.math_input.raw_text, self.r_min, self.r_max
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
        HashMap::from([(
            "default".to_string(),
            "Logistic map bifurcation diagram".to_string(),
        )])
    }

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        self.build_theory_parameters()
    }
}

impl InteractiveTool for BifurcationDiagram {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Bifurcation Diagram"
    }

    fn domain(&self) -> &'static str {
        "chaos"
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let attractor: Vec<[f64; 2]> = self.points.iter().map(|p| [p.x, p.y]).collect();
        if self.lyapunov_points.is_empty() {
            return vec![("Attractor".to_string(), attractor)];
        }
        let lyapunov: Vec<[f64; 2]> = self.lyapunov_points.iter().map(|p| [p.x, p.y]).collect();
        vec![
            ("Attractor".to_string(), attractor),
            ("Lyapunov Exponent".to_string(), lyapunov),
        ]
    }

    fn step_simulation(&mut self) {
        let next = self.eval_active_map(self.probe_r, self.orbit_x);
        if next.is_finite() && next.abs() <= CUSTOM_ESCAPE_GUARD {
            self.orbit_x = next;
            self.orbit_step_count = self.orbit_step_count.saturating_add(1);
        }
    }

    fn save_state(&self) -> Option<String> {
        let is_custom = self.map_mode == BifurcationMapMode::Custom;
        let custom_expr = (is_custom || self.math_input.raw_text != DEFAULT_CUSTOM_MAP_EXPR)
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
            custom_expr,
            custom_params,
            x0: self.x0,
            probe_r: self.probe_r,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
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
            || !s.probe_r.is_finite()
        {
            return Err("Bifurcation r bounds, initial seed, or step count invalid".into());
        }
        self.map_mode = s.map_mode;
        self.r_min = s.r_min;
        self.r_max = s.r_max;
        self.probe_r = s.probe_r.clamp(self.r_min, self.r_max);
        self.steps = s.steps;
        self.x0 = s.x0;
        self.math_input
            .set_expression(s.custom_expr.as_deref().unwrap_or(DEFAULT_CUSTOM_MAP_EXPR));
        self.math_input.set_params_map(&s.custom_params);
        self.atlas.set_visible(s.atlas_visible);
        let _ = self.atlas.set_axis_keys(&s.atlas_axes[0], &s.atlas_axes[1]);
        self.recompute();
        self.reset_live_orbit();
        self.completed_generation = self.request_generation;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.sync_from_worker(&ctx);
        let panel_id = crate::framework::scoped_panel_id(ui, "bifurcation_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .show_inside(ui, |ui| self.draw_controls(ui));

        egui::CentralPanel::default().show_inside(ui, |ui| {
            if self.atlas.is_visible() && ui.available_width() >= 620.0 {
                ui.columns(2, |cols| {
                    self.draw_bifurcation_and_cobweb(&mut cols[0]);
                    self.render_bifurcation_atlas_panel(&mut cols[1]);
                });
            } else {
                if self.atlas.is_visible() {
                    self.render_bifurcation_atlas_panel(ui);
                    ui.separator();
                }
                self.draw_bifurcation_and_cobweb(ui);
            }
        });
    }
}

// [cite:algorithmic_information_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BifurcationDiagram",
        display_name: "Bifurcation Diagram",
        domain: "chaos",
        tags: &["logistic map", "period doubling", "feigenbaum", "orbit diagram", "chaos"],
        build: || Box::new(BifurcationDiagram::default()),
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": { "schema_version": "1.0", "title": "Bifurcation Diagram", "domain": "chaos", "timestamp": "2026-11-16T00:00:00Z" },
  "config": { "r_min": 2.5, "r_max": 4.0, "steps": 800 }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "bifurcation",
        title: "Bifurcation Diagram",
        description: "x_{n+1} = r·x_n·(1 - x_n): Period-doubling cascade towards deterministic chaos.",
        category: "Non-Linear Dynamics",
        icon: "🌿",
        domain: "chaos",
        tool_name: Some("BifurcationDiagram"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
#[path = "bifurcation/tests.rs"]
mod tests;
