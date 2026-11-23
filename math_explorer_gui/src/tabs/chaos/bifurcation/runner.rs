#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    BifurcationDiagram, BifurcationMapMode, CUSTOM_BURN_IN, CUSTOM_ESCAPE_GUARD,
    CUSTOM_ORBIT_SAMPLES, CUSTOM_R_BOUND,
};
use crate::async_sim::{
    BackgroundWorker, SimCommand, SimWorkerState, SimulationRunner, StateSnapshot,
};
use crate::framework::recovery::DiagnosticReport;
use crate::framework::{MathExpressionInputBar, MAX_SYNTHESIZED_PARAMS};
use eframe::egui;
use egui_plot::PlotPoint;
use math_commons::ast::{parse, CompiledExpr};
use math_explorer::physics::chaos::logistic::{
    self, LogisticMap, ATTRACTOR_POINTS, TRANSIENT_STEPS,
};
use math_explorer::physics::chaos::metrics::logistic_lyapunov;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

const WORKER_STEP_BUDGET: Duration = Duration::from_millis(8);
const LOGISTIC_SLICE_STEPS: usize = 64;
const LYAPUNOV_STEPS: usize = 96;

/// Immutable, `Arc`-shareable compiled 1D map $x_{n+1} = f(r, x)$ evaluated without heap allocation.
#[derive(Clone, Debug)]
pub struct CompiledMap1DJob {
    compiled: CompiledExpr,
    params: [f64; MAX_SYNTHESIZED_PARAMS],
    param_len: usize,
}

impl CompiledMap1DJob {
    /// Snapshots the active 1D map [`CompiledExpr`] and synthesized parameters from `bar`.
    #[must_use]
    pub fn from_bar(bar: &MathExpressionInputBar) -> Self {
        let compiled = bar.active_compiled().cloned().unwrap_or_else(|| {
            let ast = parse("r * x * (1 - x)").expect("fallback parses");
            CompiledExpr::compile_with_vars(&ast, &["r", "x"]).expect("fallback compiles")
        });
        let slice = bar.param_slice();
        let param_len = slice.len().min(MAX_SYNTHESIZED_PARAMS);
        let mut params = [0.0; MAX_SYNTHESIZED_PARAMS];
        params[..param_len].copy_from_slice(&slice[..param_len]);
        Self {
            compiled,
            params,
            param_len,
        }
    }

    /// Evaluates $f(r, x)$ on stack buffers with an optional override for `param_0`.
    #[inline]
    #[must_use]
    pub fn eval(&self, r: f64, x: f64, param0_override: Option<f64>) -> f64 {
        let mut p = self.params;
        if let Some(p0) = param0_override {
            if self.param_len > 0 {
                p[0] = p0;
            }
        }
        self.compiled.eval_2d(r, x, &p[..self.param_len])
    }

    /// Estimates the Lyapunov exponent $\lambda$ of $x_{n+1} = f(r, x_n)$ on stack buffers.
    #[must_use]
    pub fn lyapunov(
        &self,
        r: f64,
        x0: f64,
        param0_override: Option<f64>,
        burn_in: usize,
        n: usize,
    ) -> Option<f64> {
        if n == 0 || !r.is_finite() || !x0.is_finite() || x0.abs() > CUSTOM_ESCAPE_GUARD {
            return None;
        }
        let mut x = x0;
        for _ in 0..burn_in {
            x = self.eval(r, x, param0_override);
            if !x.is_finite() || x.abs() > CUSTOM_ESCAPE_GUARD {
                return None;
            }
        }
        let h = 1e-6;
        let mut sum_logs = 0.0;
        for _ in 0..n {
            let f_plus = self.eval(r, x + h, param0_override);
            let f_minus = self.eval(r, x - h, param0_override);
            if !f_plus.is_finite() || !f_minus.is_finite() {
                return None;
            }
            let deriv = ((f_plus - f_minus) / (2.0 * h)).abs().max(1e-15);
            sum_logs += deriv.ln();
            x = self.eval(r, x, param0_override);
            if !x.is_finite() || x.abs() > CUSTOM_ESCAPE_GUARD {
                return None;
            }
        }
        Some(sum_logs / (n as f64))
    }
}

/// Evaluates burned-in logistic Lyapunov exponent $\lambda(r, x_0)$ without heap allocation.
#[inline]
#[must_use]
pub fn burned_logistic_lyapunov(r: f64, x0: f64, n: usize) -> Option<f64> {
    if !r.is_finite() || !x0.is_finite() || x0.abs() > CUSTOM_ESCAPE_GUARD {
        return None;
    }
    let mut map = LogisticMap::new(r, x0);
    for _ in 0..TRANSIENT_STEPS {
        let _ = map.next();
        if !map.state.is_finite() || map.state.abs() > CUSTOM_ESCAPE_GUARD {
            return None;
        }
    }
    let lam = logistic_lyapunov(r, map.state, n);
    lam.is_finite().then_some(lam)
}

/// Request payload dispatched from [`super::BifurcationDiagram`] to [`BifurcationRunner`].
#[derive(Clone, Copy, Debug)]
pub struct BifurcationSweepRequest {
    pub generation: u64,
    pub map_mode: BifurcationMapMode,
    pub r_min: f64,
    pub r_max: f64,
    pub steps: usize,
    pub x0: f64,
}

/// Progressive sweep payload emitted by [`BifurcationRunner`].
#[derive(Clone, Debug)]
pub struct BifurcationSweepSnapshot {
    pub generation: u64,
    pub is_final: bool,
    pub points: Vec<PlotPoint>,
    pub lyapunov_points: Vec<PlotPoint>,
}

/// Background worker runner executing `8ms` slices over logistic and custom 1D bifurcation sweeps.
pub struct BifurcationRunner {
    shared_request: Arc<RwLock<BifurcationSweepRequest>>,
    shared_formula: Arc<RwLock<Option<Arc<CompiledMap1DJob>>>>,
    latest_generation: Arc<AtomicU64>,
    active: BifurcationSweepRequest,
    active_formula: Option<Arc<CompiledMap1DJob>>,
    next_step_idx: usize,
    points_buf: Vec<PlotPoint>,
    lyapunov_buf: Vec<PlotPoint>,
}

impl BifurcationRunner {
    /// Creates a [`BifurcationRunner`] sharing request and compiled formula slots with the UI.
    #[must_use]
    pub fn new(
        shared_request: Arc<RwLock<BifurcationSweepRequest>>,
        latest_generation: Arc<AtomicU64>,
        shared_formula: Arc<RwLock<Option<Arc<CompiledMap1DJob>>>>,
    ) -> Self {
        let active = *shared_request.read().unwrap_or_else(|e| e.into_inner());
        let active_formula = shared_formula
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let total = active.steps.saturating_add(1);
        Self {
            shared_request,
            shared_formula,
            latest_generation,
            active,
            active_formula,
            next_step_idx: total,
            points_buf: Vec::new(),
            lyapunov_buf: Vec::new(),
        }
    }

    fn sync_request(&mut self) -> bool {
        let target_gen = self.latest_generation.load(Ordering::Acquire);
        if target_gen == self.active.generation {
            return false;
        }
        let req = *self
            .shared_request
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let formula = self
            .shared_formula
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        self.active = req;
        self.active_formula = formula;
        self.next_step_idx = 0;
        self.points_buf.clear();
        self.lyapunov_buf.clear();
        true
    }

    fn step_logistic_chunk(&mut self, total_r_samples: usize) {
        let steps = self.active.steps.max(1);
        let step_size = (self.active.r_max - self.active.r_min) / (steps as f64);
        let remaining = total_r_samples.saturating_sub(self.next_step_idx);
        let count = remaining.min(LOGISTIC_SLICE_STEPS);
        if count == 0 {
            return;
        }
        let start_idx = self.next_step_idx;
        if count >= 2 {
            let sub_steps = count - 1;
            let sub_r_min = self.active.r_min + step_size * (start_idx as f64);
            let sub_r_max = self.active.r_min + step_size * ((start_idx + sub_steps) as f64);
            let raw = logistic::generate_bifurcation_diagram(sub_r_min, sub_r_max, sub_steps);
            for (r, x) in raw {
                self.points_buf.push(PlotPoint::new(r, x));
            }
        } else {
            let r = self.active.r_min + step_size * (start_idx as f64);
            let raw = logistic::generate_bifurcation_diagram(r, r, 1);
            for (pr, px) in raw.into_iter().take(ATTRACTOR_POINTS) {
                self.points_buf.push(PlotPoint::new(pr, px));
            }
        }
        let seed = if (self.active.x0 - 0.5).abs() < 1e-9 {
            0.314_159
        } else {
            self.active.x0.clamp(1e-4, 1.0 - 1e-4)
        };
        for offset in 0..count {
            let r = self.active.r_min + step_size * ((start_idx + offset) as f64);
            if let Some(lam) = burned_logistic_lyapunov(r, seed, LYAPUNOV_STEPS) {
                self.lyapunov_buf.push(PlotPoint::new(r, lam));
            }
        }
        self.next_step_idx += count;
    }

    fn step_custom_single(&mut self, step_idx: usize, steps: usize) {
        let Some(job) = self.active_formula.as_deref() else {
            self.next_step_idx = steps;
            return;
        };
        let denom = (steps.saturating_sub(1).max(1)) as f64;
        let r =
            self.active.r_min + (step_idx as f64 / denom) * (self.active.r_max - self.active.r_min);
        let mut x = self.active.x0;
        let mut bounded = x.is_finite() && x.abs() <= CUSTOM_ESCAPE_GUARD;
        for _ in 0..CUSTOM_BURN_IN {
            if !bounded {
                break;
            }
            x = job.eval(r, x, None);
            bounded = x.is_finite() && x.abs() <= CUSTOM_ESCAPE_GUARD;
        }
        if bounded {
            let lyap_seed = x;
            for _ in 0..CUSTOM_ORBIT_SAMPLES {
                x = job.eval(r, x, None);
                if !x.is_finite() || x.abs() > CUSTOM_ESCAPE_GUARD {
                    break;
                }
                self.points_buf.push(PlotPoint::new(r, x));
            }
            if let Some(lam) = job.lyapunov(r, lyap_seed, None, 0, LYAPUNOV_STEPS) {
                self.lyapunov_buf.push(PlotPoint::new(r, lam));
            }
        }
        self.next_step_idx = step_idx + 1;
    }

    fn target_sample_count(&self) -> usize {
        match self.active.map_mode {
            BifurcationMapMode::Logistic => self.active.steps.clamp(2, 2000) + 1,
            BifurcationMapMode::Custom => self.active.steps.clamp(2, 2000),
        }
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
        let total = self.target_sample_count();
        if self.next_step_idx >= total || !self.is_finite_state() {
            return;
        }
        let slice_start = crate::web_io::PlatformInstant::now();
        while self.next_step_idx < total {
            if self.latest_generation.load(Ordering::Relaxed) != self.active.generation {
                let _ = self.sync_request();
                return;
            }
            match self.active.map_mode {
                BifurcationMapMode::Logistic => self.step_logistic_chunk(total),
                BifurcationMapMode::Custom => {
                    let steps = self.active.steps.clamp(2, 2000);
                    self.step_custom_single(self.next_step_idx, steps);
                }
            }
            if self.next_step_idx < total && slice_start.elapsed() >= WORKER_STEP_BUDGET {
                return;
            }
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let total = self.target_sample_count();
        let is_final = self.next_step_idx >= total
            && self.latest_generation.load(Ordering::Acquire) == self.active.generation;
        let payload = BifurcationSweepSnapshot {
            generation: self.active.generation,
            is_final,
            points: self.points_buf.clone(),
            lyapunov_points: self.lyapunov_buf.clone(),
        };
        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(RwLock::new(Vec::new())),
            custom_data: vec![self.active.r_min, self.active.r_max],
            structured_data: Some(Box::new(payload)),
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_complete(&self) -> bool {
        self.latest_generation.load(Ordering::Acquire) == self.active.generation
            && self.next_step_idx >= self.target_sample_count()
    }

    fn is_finite_state(&self) -> bool {
        let r_bounds = match self.active.map_mode {
            BifurcationMapMode::Logistic => 0.0..=4.0,
            BifurcationMapMode::Custom => -CUSTOM_R_BOUND..=CUSTOM_R_BOUND,
        };
        self.active.r_min.is_finite()
            && self.active.r_max.is_finite()
            && self.active.r_min < self.active.r_max
            && r_bounds.contains(&self.active.r_min)
            && r_bounds.contains(&self.active.r_max)
            && self.active.x0.is_finite()
            && (-10.0..=10.0).contains(&self.active.x0)
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        (!self.is_finite_state()).then(|| {
            DiagnosticReport::new(
                0.0,
                0,
                "bifurcation_parameters",
                "Non-finite or out-of-range bifurcation sweep parameter encountered",
            )
        })
    }
}

impl BifurcationDiagram {
    pub(crate) fn dispatch_sweep_request(&mut self, ctx: &egui::Context) {
        if self.map_mode == BifurcationMapMode::Custom
            && self.math_input.active_compiled().is_none()
        {
            return;
        }
        self.request_generation = self.request_generation.wrapping_add(1).max(1);
        let formula = (self.map_mode == BifurcationMapMode::Custom)
            .then(|| Arc::new(CompiledMap1DJob::from_bar(&self.math_input)));
        *self
            .shared_formula
            .write()
            .unwrap_or_else(|e| e.into_inner()) = formula;
        *self
            .shared_request
            .write()
            .unwrap_or_else(|e| e.into_inner()) = BifurcationSweepRequest {
            generation: self.request_generation,
            map_mode: self.map_mode,
            r_min: self.r_min,
            r_max: self.r_max,
            steps: self.steps.clamp(2, 2000),
            x0: self.x0,
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
                self.worker = BackgroundWorker::new(BifurcationRunner::new(
                    Arc::clone(&self.shared_request),
                    Arc::clone(&self.latest_generation),
                    Arc::clone(&self.shared_formula),
                ));
                let _ = self.worker.send_command(SimCommand::Start);
            }
            SimWorkerState::Uninitialized | SimWorkerState::Stepping => {}
        }
        ctx.request_repaint();
    }

    pub(super) fn sync_from_worker(&mut self, ctx: &egui::Context) {
        if let Some(snap) = self.worker.update() {
            if let Some(data) = snap
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<BifurcationSweepSnapshot>())
            {
                if data.generation == self.request_generation {
                    self.points.clone_from(&data.points);
                    self.lyapunov_points.clone_from(&data.lyapunov_points);
                    if data.is_final {
                        self.completed_generation = data.generation;
                        self.recompute_coupled_views();
                    }
                }
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
