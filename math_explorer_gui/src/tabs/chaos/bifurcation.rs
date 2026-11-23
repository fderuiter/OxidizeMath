#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::{InteractiveTool, MathExpressionInputBar};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::physics::chaos::metrics::logistic_lyapunov;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

pub(crate) const DEFAULT_CUSTOM_MAP_EXPR: &str = "r * sin(pi * x)";
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
    const fn is_logistic(&self) -> bool {
        matches!(self, Self::Logistic)
    }
}

#[path = "bifurcation/runner.rs"]
mod runner;
pub use runner::*;

#[path = "bifurcation/cobweb.rs"]
mod cobweb;
pub use cobweb::*;

#[path = "bifurcation/preset.rs"]
mod preset;

#[path = "bifurcation/ui.rs"]
mod ui;

const fn default_selected_r() -> f64 {
    3.83
}
const fn default_orbit_x0() -> f64 {
    0.1
}
const fn default_show_lyapunov() -> bool {
    true
}
const fn default_x0() -> f64 {
    0.5
}

fn build_default_custom_bar() -> MathExpressionInputBar {
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
/// cobweb, and $x_n$ orbit inspectors via progressive `BackgroundWorker` sweeps.
pub struct BifurcationDiagram {
    pub(crate) map_mode: BifurcationMapMode,
    pub(crate) r_min: f64,
    pub(crate) r_max: f64,
    pub(crate) steps: usize,
    pub(crate) x0: f64,
    pub(crate) selected_r: f64,
    pub(crate) orbit_x0: f64,
    pub(crate) show_lyapunov: bool,
    pub(crate) math_input: MathExpressionInputBar,
    pub(crate) points: Vec<PlotPoint>,
    pub(crate) lyapunov_points: Vec<PlotPoint>,
    pub(crate) raw_lyapunov: Vec<[f64; 2]>,
    pub(crate) sweep_progress: Option<(usize, usize)>,
    pub(crate) dirty: bool,
    pub(crate) request_generation: u64,
    pub(crate) completed_generation: u64,
    pub(crate) shared_request: Arc<RwLock<BifurcationSweepRequest>>,
    pub(crate) shared_custom: Arc<RwLock<Option<Arc<CompiledBifurcationJob>>>>,
    pub(crate) latest_generation: Arc<AtomicU64>,
    pub(crate) worker: BackgroundWorker<BifurcationRunner>,
}

impl Default for BifurcationDiagram {
    fn default() -> Self {
        let r_min = 2.5;
        let r_max = 4.0;
        let steps = 500;
        let math_input = build_default_custom_bar();
        let (pts, lyap, raw) = compute_bifurcation_and_lyapunov_sync(r_min, r_max, steps);
        let initial_req = BifurcationSweepRequest {
            generation: 0,
            r_min,
            r_max,
            steps,
        };
        let shared_request = Arc::new(RwLock::new(initial_req));
        let shared_custom = Arc::new(RwLock::new(None));
        let latest_generation = Arc::new(AtomicU64::new(0));
        let worker = BackgroundWorker::new(BifurcationRunner::with_custom(
            Arc::clone(&shared_request),
            Arc::clone(&shared_custom),
            Arc::clone(&latest_generation),
            pts.clone(),
            lyap.clone(),
            raw.clone(),
        ));
        Self {
            map_mode: BifurcationMapMode::Logistic,
            r_min,
            r_max,
            steps,
            x0: default_x0(),
            selected_r: default_selected_r(),
            orbit_x0: default_orbit_x0(),
            show_lyapunov: true,
            math_input,
            points: pts,
            lyapunov_points: lyap,
            raw_lyapunov: raw,
            sweep_progress: None,
            dirty: false,
            request_generation: 0,
            completed_generation: 0,
            shared_request,
            shared_custom,
            latest_generation,
            worker,
        }
    }
}

impl BifurcationDiagram {
    /// Synchronously recomputes the bifurcation attractor cloud and `logistic_lyapunov` curve.
    pub fn recompute(&mut self) {
        if self.map_mode == BifurcationMapMode::Custom {
            self.recompute_custom_in_place();
            return;
        }
        self.sanitize_bounds();
        let (pts, lyap, raw) =
            compute_bifurcation_and_lyapunov_sync(self.r_min, self.r_max, self.steps);
        self.points = pts;
        self.lyapunov_points = lyap;
        self.raw_lyapunov = raw;
        self.sweep_progress = None;
        self.dirty = false;
        self.completed_generation = self.request_generation;
    }

    fn recompute_custom_in_place(&mut self) {
        let Some(compiled) = self.math_input.active_compiled() else {
            return;
        };
        let params = self.math_input.param_slice();
        let steps = self.steps.clamp(2, 2000);
        let span = self.r_max - self.r_min;
        if !span.is_finite() || span <= 0.0 {
            self.points.clear();
            self.lyapunov_points.clear();
            self.raw_lyapunov.clear();
            return;
        }

        let max_cap = steps.saturating_mul(CUSTOM_ORBIT_SAMPLES);
        if self.points.capacity() < max_cap {
            self.points.reserve(max_cap - self.points.capacity());
        }
        self.points.clear();
        self.lyapunov_points.clear();
        self.raw_lyapunov.clear();

        let denom = (steps - 1) as f64;
        for step_idx in 0..steps {
            let r = self.r_min + (step_idx as f64 / denom) * span;
            let mut x = self.x0;
            let mut bounded = x.is_finite() && x.abs() <= CUSTOM_ESCAPE_GUARD;
            for _ in 0..CUSTOM_BURN_IN {
                if !bounded {
                    break;
                }
                x = compiled.eval_2d(r, x, params);
                bounded = x.is_finite() && x.abs() <= CUSTOM_ESCAPE_GUARD;
            }
            if !bounded {
                continue;
            }
            for _ in 0..CUSTOM_ORBIT_SAMPLES {
                x = compiled.eval_2d(r, x, params);
                if !x.is_finite() || x.abs() > CUSTOM_ESCAPE_GUARD {
                    break;
                }
                self.points.push(PlotPoint::new(r, x));
            }
        }
        self.sweep_progress = None;
        self.dirty = false;
        self.completed_generation = self.request_generation;
    }

    pub(super) fn sanitize_bounds(&mut self) {
        let (lo_cap, hi_cap) = match self.map_mode {
            BifurcationMapMode::Logistic => (0.0, 4.0),
            BifurcationMapMode::Custom => (-CUSTOM_R_BOUND, CUSTOM_R_BOUND),
        };
        if !self.r_min.is_finite() {
            self.r_min = 2.5_f64.clamp(lo_cap, hi_cap - 0.01);
        }
        if !self.r_max.is_finite() {
            self.r_max = 4.0_f64.clamp(lo_cap + 0.01, hi_cap);
        }
        self.r_min = self.r_min.clamp(lo_cap, hi_cap - 0.01);
        self.r_max = self.r_max.clamp((self.r_min + 0.01).min(hi_cap), hi_cap);
        if self.r_min >= self.r_max {
            self.r_min = (self.r_max - 0.01).max(lo_cap);
        }
        self.steps = self.steps.clamp(100, 2000);
        if !self.x0.is_finite() {
            self.x0 = default_x0();
        }
        self.x0 = self.x0.clamp(-10.0, 10.0);
        if !self.selected_r.is_finite() {
            self.selected_r = default_selected_r();
        }
        self.selected_r = self.selected_r.clamp(self.r_min, self.r_max);
        if !self.orbit_x0.is_finite() {
            self.orbit_x0 = default_orbit_x0();
        }
        self.orbit_x0 = self.orbit_x0.clamp(0.01, 0.99);
    }

    /// Selects a control parameter $r$ on the bifurcation axis and clamps it to valid bounds.
    pub fn select_r(&mut self, r: f64) {
        if r.is_finite() {
            let (lo, hi) = match self.map_mode {
                BifurcationMapMode::Logistic => (0.0, 4.0),
                BifurcationMapMode::Custom => (-CUSTOM_R_BOUND, CUSTOM_R_BOUND),
            };
            self.selected_r = r.clamp(self.r_min, self.r_max).clamp(lo, hi);
        }
    }

    /// Evaluates the live Lyapunov exponent $\lambda(r)$ at `selected_r`.
    #[must_use]
    pub fn selected_lyapunov(&self) -> f64 {
        logistic_lyapunov(
            self.selected_r.clamp(0.0, 4.0),
            BIFURCATION_LYAPUNOV_X0,
            BIFURCATION_LYAPUNOV_ITERS,
        )
    }

    fn sync_worker_snapshot(&mut self) {
        let Some(snapshot) = self.worker.update() else {
            return;
        };
        if let Some(data) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<BifurcationSweepSnapshot>())
        {
            if !data.points.is_empty() || data.is_final {
                self.points.clone_from(&data.points);
                self.lyapunov_points.clone_from(&data.lyapunov_points);
                self.raw_lyapunov.clone_from(&data.raw_lyapunov);
            }
            if data.is_final {
                self.completed_generation = data.generation;
                self.sweep_progress = None;
            } else {
                self.sweep_progress = Some((data.completed_steps, data.total_steps + 1));
            }
        }
    }

    fn dispatch_sweep_request(&mut self, ctx: &egui::Context) {
        if self.dirty {
            self.dirty = false;
            if self.map_mode == BifurcationMapMode::Custom
                && self.math_input.active_compiled().is_none()
            {
                return;
            }
            self.sanitize_bounds();
            self.request_generation = self.request_generation.wrapping_add(1).max(1);
            let custom_job = (self.map_mode == BifurcationMapMode::Custom)
                .then(|| Arc::new(CompiledBifurcationJob::from_bar(&self.math_input, self.x0)));
            *self
                .shared_custom
                .write()
                .unwrap_or_else(|e| e.into_inner()) = custom_job;
            *self
                .shared_request
                .write()
                .unwrap_or_else(|e| e.into_inner()) = BifurcationSweepRequest {
                generation: self.request_generation,
                r_min: self.r_min,
                r_max: self.r_max,
                steps: self.steps,
            };
            self.latest_generation
                .store(self.request_generation, Ordering::Release);

            match self.worker.state() {
                SimWorkerState::Ready => {
                    let _ = self.worker.send_command(SimCommand::Start);
                }
                SimWorkerState::Running => {
                    let _ = self
                        .worker
                        .send_command(SimCommand::Custom("cancel_sweep".to_string()));
                }
                SimWorkerState::Faulted | SimWorkerState::Terminated => {
                    self.worker = BackgroundWorker::new(BifurcationRunner::with_custom(
                        Arc::clone(&self.shared_request),
                        Arc::clone(&self.shared_custom),
                        Arc::clone(&self.latest_generation),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                    ));
                    let _ = self.worker.send_command(SimCommand::Start);
                }
                SimWorkerState::Uninitialized | SimWorkerState::Stepping => {}
            }
        }
        if self.completed_generation != self.request_generation && self.worker.state().is_ready() {
            let _ = self.worker.send_command(SimCommand::Start);
        }
        if self.completed_generation != self.request_generation || self.worker.is_running() {
            ctx.request_repaint();
        }
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
        if self.raw_lyapunov.is_empty() {
            return vec![("Attractor".to_string(), attractor)];
        }
        vec![
            ("Attractor".to_string(), attractor),
            ("Lyapunov Exponent".to_string(), self.raw_lyapunov.clone()),
        ]
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.deserialize_state(state)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.sync_worker_snapshot();
        self.draw_side_panel(ui);
        self.draw_central_panel(ui);
        self.dispatch_sweep_request(&ctx);
    }
}

// [cite:algorithmic_information_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BifurcationDiagram",
        display_name: "Bifurcation Diagram",
        domain: "chaos",
        tags: &[],
        build: || Box::new(BifurcationDiagram::default()),
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Bifurcation Diagram",
    "domain": "chaos",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "r_min": 2.5,
    "r_max": 4.0,
    "steps": 800
  }
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
