#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    BifurcationDiagram, BifurcationMapMode, CUSTOM_BURN_IN, CUSTOM_ESCAPE_GUARD,
    CUSTOM_ORBIT_SAMPLES, CUSTOM_R_BOUND,
};
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::{
    recovery::DiagnosticReport, MathExpressionInputBar, MAX_SYNTHESIZED_PARAMS,
};
use egui_plot::PlotPoint;
use math_commons::ast::{parse, CompiledExpr};
use math_explorer::physics::chaos::logistic::{LogisticMap, ATTRACTOR_POINTS, TRANSIENT_STEPS};
use math_explorer::physics::chaos::metrics::logistic_lyapunov;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Iteration count passed to `logistic_lyapunov` per parameter slice.
pub const BIFURCATION_LYAPUNOV_ITERS: usize = 250;
/// Non-critical initial seed $x_0$ passed to `logistic_lyapunov`.
pub const BIFURCATION_LYAPUNOV_X0: f64 = 0.1;
/// Visual clamp floor for plotting $\lambda(r)$ alongside $x \in [0, 1]$.
pub const LYAPUNOV_PLOT_FLOOR: f64 = -1.25;
/// Visual clamp ceiling for plotting $\lambda(r)$ alongside $x \in [0, 1]$.
pub const LYAPUNOV_PLOT_CEIL: f64 = 1.0;

const WORKER_STEP_BUDGET: Duration = Duration::from_millis(8);

/// Immutable, `Arc`-shareable compiled 1D map $x_{n+1} = f(r, x)$ for background sweeps.
#[derive(Clone, Debug)]
pub struct CompiledBifurcationJob {
    compiled: CompiledExpr,
    params: [f64; MAX_SYNTHESIZED_PARAMS],
    param_len: usize,
    x0: f64,
}

impl CompiledBifurcationJob {
    /// Captures the compiled bytecode and inline parameter slice from `bar`.
    #[must_use]
    pub fn from_bar(bar: &MathExpressionInputBar, x0: f64) -> Self {
        let compiled = bar.active_compiled().cloned().unwrap_or_else(|| {
            let fallback = parse("r * sin(pi * x)").expect("fallback parses");
            CompiledExpr::compile_with_vars(&fallback, &["r", "x"]).expect("fallback compiles")
        });
        let slice = bar.param_slice();
        let param_len = slice.len().min(MAX_SYNTHESIZED_PARAMS);
        let mut params = [0.0; MAX_SYNTHESIZED_PARAMS];
        params[..param_len].copy_from_slice(&slice[..param_len]);
        Self {
            compiled,
            params,
            param_len,
            x0,
        }
    }

    #[inline]
    pub(super) fn eval(&self, r: f64, x: f64) -> f64 {
        self.compiled.eval_2d(r, x, &self.params[..self.param_len])
    }
}

/// Immutable parameter bundle describing a progressive bifurcation + Lyapunov sweep request.
#[derive(Clone, Copy, Debug)]
pub struct BifurcationSweepRequest {
    /// Monotonic request generation for mid-sweep cancellation.
    pub generation: u64,
    /// Lower control-parameter bound (`r_min < r_max`).
    pub r_min: f64,
    /// Upper control-parameter bound (`r_min < r_max`).
    pub r_max: f64,
    /// Number of parameter subdivisions in `100..=2000`.
    pub steps: usize,
}

/// Snapshot payload emitted by [`BifurcationRunner`] after each cooperative `8ms` slice.
#[derive(Clone)]
pub struct BifurcationSweepSnapshot {
    /// Request generation that produced this snapshot.
    pub generation: u64,
    /// Accumulated `(r, x)` attractor points up to `completed_steps`.
    pub points: Arc<Vec<PlotPoint>>,
    /// Accumulated `(r, clamp(lambda))` overlay curve points.
    pub lyapunov_points: Arc<Vec<PlotPoint>>,
    /// Unclamped `[r, lambda]` samples for exact stability inspection.
    pub raw_lyapunov: Arc<Vec<[f64; 2]>>,
    /// Number of `r` columns processed so far (`0..=total_steps + 1`).
    pub completed_steps: usize,
    /// Total `steps` requested (`total_steps + 1` columns).
    pub total_steps: usize,
    /// `true` once all `0..=steps` columns have finished for `generation`.
    pub is_final: bool,
}

fn sample_logistic_column(
    r: f64,
    points: &mut Vec<PlotPoint>,
    lyapunov_points: &mut Vec<PlotPoint>,
    raw_lyapunov: &mut Vec<[f64; 2]>,
) {
    let safe_r = r.clamp(0.0, 4.0);
    let mut map = LogisticMap::new(safe_r, 0.5);
    for _ in 0..TRANSIENT_STEPS {
        map.next();
    }
    let col_start = points.len();
    for _ in 0..ATTRACTOR_POINTS {
        if let Some(x) = map.next().filter(|v| v.is_finite()) {
            let y = x.clamp(0.0, 1.0);
            if !points[col_start..].iter().any(|p| (p.y - y).abs() < 2e-3) {
                points.push(PlotPoint::new(safe_r, y));
            }
        }
    }
    let lambda = logistic_lyapunov(safe_r, BIFURCATION_LYAPUNOV_X0, BIFURCATION_LYAPUNOV_ITERS);
    if lambda.is_finite() {
        raw_lyapunov.push([safe_r, lambda]);
        lyapunov_points.push(PlotPoint::new(
            safe_r,
            lambda.clamp(LYAPUNOV_PLOT_FLOOR, LYAPUNOV_PLOT_CEIL),
        ));
    }
}

fn sample_custom_column(r: f64, job: &CompiledBifurcationJob, points: &mut Vec<PlotPoint>) {
    let mut x = job.x0;
    let mut bounded = x.is_finite() && x.abs() <= CUSTOM_ESCAPE_GUARD;
    for _ in 0..CUSTOM_BURN_IN {
        if !bounded {
            break;
        }
        x = job.eval(r, x);
        bounded = x.is_finite() && x.abs() <= CUSTOM_ESCAPE_GUARD;
    }
    if !bounded {
        return;
    }
    for _ in 0..CUSTOM_ORBIT_SAMPLES {
        x = job.eval(r, x);
        if !x.is_finite() || x.abs() > CUSTOM_ESCAPE_GUARD {
            break;
        }
        points.push(PlotPoint::new(r, x));
    }
}

/// Synchronously evaluates the bifurcation diagram and `logistic_lyapunov` curve (used for initial seeding).
#[must_use]
pub fn compute_bifurcation_and_lyapunov_sync(
    r_min: f64,
    r_max: f64,
    steps: usize,
) -> (Vec<PlotPoint>, Vec<PlotPoint>, Vec<[f64; 2]>) {
    let clamped_steps = steps.clamp(1, 2000);
    let lo = r_min.clamp(0.0, 4.0);
    let hi = r_max.clamp((lo + 1e-6).min(4.0), 4.0);
    let step_size = (hi - lo) / clamped_steps as f64;

    let mut points = Vec::with_capacity((clamped_steps + 1) * ATTRACTOR_POINTS);
    let mut lyapunov_points = Vec::with_capacity(clamped_steps + 1);
    let mut raw_lyapunov = Vec::with_capacity(clamped_steps + 1);
    for i in 0..=clamped_steps {
        let r = lo + step_size * i as f64;
        sample_logistic_column(r, &mut points, &mut lyapunov_points, &mut raw_lyapunov);
    }
    (points, lyapunov_points, raw_lyapunov)
}

/// Cooperative `BackgroundWorker` runner computing bifurcation and Lyapunov sweeps in `8ms` slices.
pub struct BifurcationRunner {
    shared_request: Arc<RwLock<BifurcationSweepRequest>>,
    shared_custom: Arc<RwLock<Option<Arc<CompiledBifurcationJob>>>>,
    latest_generation: Arc<AtomicU64>,
    active: BifurcationSweepRequest,
    active_custom: Option<Arc<CompiledBifurcationJob>>,
    next_step_idx: usize,
    points_buf: Vec<PlotPoint>,
    lyapunov_buf: Vec<PlotPoint>,
    raw_lyapunov_buf: Vec<[f64; 2]>,
}

impl BifurcationRunner {
    /// Creates a new [`BifurcationRunner`] initialized in an idle state matching `shared_request`.
    #[cfg(test)]
    #[must_use]
    pub fn new(
        shared_request: Arc<RwLock<BifurcationSweepRequest>>,
        latest_generation: Arc<AtomicU64>,
        seed_points: Vec<PlotPoint>,
        seed_lyapunov: Vec<PlotPoint>,
        seed_raw: Vec<[f64; 2]>,
    ) -> Self {
        Self::with_custom(
            shared_request,
            Arc::new(RwLock::new(None)),
            latest_generation,
            seed_points,
            seed_lyapunov,
            seed_raw,
        )
    }

    /// Creates a [`BifurcationRunner`] sharing an optional [`CompiledBifurcationJob`] slot.
    #[must_use]
    pub fn with_custom(
        shared_request: Arc<RwLock<BifurcationSweepRequest>>,
        shared_custom: Arc<RwLock<Option<Arc<CompiledBifurcationJob>>>>,
        latest_generation: Arc<AtomicU64>,
        seed_points: Vec<PlotPoint>,
        seed_lyapunov: Vec<PlotPoint>,
        seed_raw: Vec<[f64; 2]>,
    ) -> Self {
        let initial = *shared_request.read().unwrap_or_else(|e| e.into_inner());
        let active_custom = shared_custom
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        Self {
            shared_request,
            shared_custom,
            latest_generation,
            active: initial,
            active_custom,
            next_step_idx: initial.steps.saturating_add(1),
            points_buf: seed_points,
            lyapunov_buf: seed_lyapunov,
            raw_lyapunov_buf: seed_raw,
        }
    }

    fn sync_request(&mut self) -> bool {
        let target_gen = self.latest_generation.load(Ordering::Acquire);
        if target_gen == self.active.generation {
            return false;
        }
        let mut req = *self
            .shared_request
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let custom = self
            .shared_custom
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        req.steps = req.steps.clamp(100, 2000);
        if custom.is_some() {
            req.r_min = req.r_min.clamp(-CUSTOM_R_BOUND, CUSTOM_R_BOUND - 1e-6);
            req.r_max = req.r_max.clamp(req.r_min + 1e-6, CUSTOM_R_BOUND);
        } else {
            req.r_min = req.r_min.clamp(0.0, 3.999_999);
            req.r_max = req.r_max.clamp(req.r_min + 1e-6, 4.0);
        }
        self.active = req;
        self.active_custom = custom;
        self.next_step_idx = 0;
        self.points_buf.clear();
        self.lyapunov_buf.clear();
        self.raw_lyapunov_buf.clear();
        true
    }
}

impl SimulationRunner for BifurcationRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if matches!(cmd, SimCommand::Reset | SimCommand::Custom(_)) {
            let _ = self.sync_request();
        }
    }

    fn step(&mut self) {
        let _ = self.sync_request();
        if self.next_step_idx > self.active.steps || !self.is_finite_state() {
            return;
        }

        let total = self.active.steps.max(1);
        let step_size = (self.active.r_max - self.active.r_min) / total as f64;
        let slice_start = crate::web_io::PlatformInstant::now();

        while self.next_step_idx <= total {
            if self.latest_generation.load(Ordering::Relaxed) != self.active.generation {
                let _ = self.sync_request();
                return;
            }
            let r = self.active.r_min + step_size * self.next_step_idx as f64;
            if let Some(ref job) = self.active_custom {
                sample_custom_column(r, job, &mut self.points_buf);
            } else {
                sample_logistic_column(
                    r,
                    &mut self.points_buf,
                    &mut self.lyapunov_buf,
                    &mut self.raw_lyapunov_buf,
                );
            }
            self.next_step_idx += 1;
            if self.next_step_idx <= total && slice_start.elapsed() >= WORKER_STEP_BUDGET {
                return;
            }
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let is_final = self.next_step_idx > self.active.steps
            && self.latest_generation.load(Ordering::Acquire) == self.active.generation;
        let snap = BifurcationSweepSnapshot {
            generation: self.active.generation,
            points: Arc::new(self.points_buf.clone()),
            lyapunov_points: Arc::new(self.lyapunov_buf.clone()),
            raw_lyapunov: Arc::new(self.raw_lyapunov_buf.clone()),
            completed_steps: self.next_step_idx,
            total_steps: self.active.steps,
            is_final,
        };
        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(RwLock::new(Vec::new())),
            custom_data: vec![
                self.active.r_min,
                self.active.r_max,
                self.next_step_idx as f64,
            ],
            structured_data: Some(Box::new(snap)),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_complete(&self) -> bool {
        self.latest_generation.load(Ordering::Acquire) == self.active.generation
            && self.next_step_idx > self.active.steps
    }

    fn is_finite_state(&self) -> bool {
        let bounds = if self.active_custom.is_some() {
            -CUSTOM_R_BOUND..=CUSTOM_R_BOUND
        } else {
            0.0..=4.0
        };
        self.active.r_min.is_finite()
            && self.active.r_max.is_finite()
            && bounds.contains(&self.active.r_min)
            && bounds.contains(&self.active.r_max)
            && self.active.r_min < self.active.r_max
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        (!self.is_finite_state()).then(|| {
            DiagnosticReport::new(
                0.0,
                0,
                "bifurcation_bounds",
                "Invalid or non-finite bifurcation parameter bounds",
            )
        })
    }
}

impl TheoryDescribable for BifurcationDiagram {
    fn theory_description(&self) -> String {
        match self.map_mode {
            BifurcationMapMode::Logistic => format!(
                "Logistic map bifurcation diagram, r range: {:.2} to {:.2}, selected r={:.3} (λ={:+.3})",
                self.r_min,
                self.r_max,
                self.selected_r,
                self.selected_lyapunov()
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
        let mut map = HashMap::new();
        map.insert(
            "default".to_string(),
            "Logistic map bifurcation diagram".to_string(),
        );
        map
    }
}
