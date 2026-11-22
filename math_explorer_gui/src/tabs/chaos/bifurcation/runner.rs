#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::BifurcationDiagram;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::DiagnosticReport;
use egui_plot::PlotPoint;
use math_explorer::physics::chaos::logistic::{ATTRACTOR_POINTS, LogisticMap, TRANSIENT_STEPS};
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
/// Number of discrete map iterations displayed in the cobweb and orbit inspectors.
pub const COBWEB_ORBIT_STEPS: usize = 50;
/// Visual clamp floor for plotting $\lambda(r)$ alongside $x \in [0, 1]$.
pub const LYAPUNOV_PLOT_FLOOR: f64 = -1.25;
/// Visual clamp ceiling for plotting $\lambda(r)$ alongside $x \in [0, 1]$.
pub const LYAPUNOV_PLOT_CEIL: f64 = 1.0;

const WORKER_STEP_BUDGET: Duration = Duration::from_millis(8);

/// Immutable parameter bundle describing a progressive bifurcation + Lyapunov sweep request.
#[derive(Clone, Copy, Debug)]
pub struct BifurcationSweepRequest {
    /// Monotonic request generation for mid-sweep cancellation.
    pub generation: u64,
    /// Lower growth-rate bound in `0.0..=4.0`.
    pub r_min: f64,
    /// Upper growth-rate bound in `0.0..=4.0` (`r_min < r_max`).
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

fn sample_column(
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
        sample_column(r, &mut points, &mut lyapunov_points, &mut raw_lyapunov);
    }
    (points, lyapunov_points, raw_lyapunov)
}

/// Synchronized parabola, cobweb staircase, and $x_n$ orbit series for a chosen $r$ and $x_0$.
pub struct CobwebOrbitSeries {
    /// Sampled parabola $y = r x (1 - x)$ on $[0, 1]$.
    pub parabola: Vec<[f64; 2]>,
    /// Alternating vertical/horizontal cobweb staircase segments.
    pub cobweb: Vec<[f64; 2]>,
    /// Discrete iteration time series $(n, x_n)$.
    pub orbit: Vec<[f64; 2]>,
}

/// Builds synchronized parabola, cobweb staircase, and $x_n$ orbit series from [`LogisticMap`].
#[must_use]
pub fn build_cobweb_and_orbit(r: f64, x0: f64, n_steps: usize) -> CobwebOrbitSeries {
    let safe_r = if r.is_finite() {
        r.clamp(0.0, 4.0)
    } else {
        3.5
    };
    let safe_x0 = if x0.is_finite() {
        x0.clamp(0.001, 0.999)
    } else {
        0.1
    };
    let mut parabola = Vec::with_capacity(101);
    for idx in 0..=100 {
        let x = idx as f64 / 100.0;
        parabola.push([x, safe_r * x * (1.0 - x)]);
    }

    let mut map = LogisticMap::new(safe_r, safe_x0);
    let mut cobweb = Vec::with_capacity(1 + n_steps * 2);
    let mut orbit = Vec::with_capacity(n_steps + 1);
    let mut x_curr = safe_x0;
    cobweb.push([x_curr, 0.0]);
    orbit.push([0.0, x_curr]);

    for step in 1..=n_steps {
        let x_next = map.next().unwrap_or(0.0);
        if !x_next.is_finite() {
            break;
        }
        let clamped_next = x_next.clamp(0.0, 1.0);
        cobweb.push([x_curr, clamped_next]);
        cobweb.push([clamped_next, clamped_next]);
        orbit.push([step as f64, clamped_next]);
        x_curr = clamped_next;
    }

    CobwebOrbitSeries {
        parabola,
        cobweb,
        orbit,
    }
}

/// Cooperative `BackgroundWorker` runner computing bifurcation and Lyapunov sweeps in `8ms` slices.
pub struct BifurcationRunner {
    shared_request: Arc<RwLock<BifurcationSweepRequest>>,
    latest_generation: Arc<AtomicU64>,
    active: BifurcationSweepRequest,
    next_step_idx: usize,
    points_buf: Vec<PlotPoint>,
    lyapunov_buf: Vec<PlotPoint>,
    raw_lyapunov_buf: Vec<[f64; 2]>,
}

impl BifurcationRunner {
    /// Creates a new [`BifurcationRunner`] initialized in an idle state matching `shared_request`.
    #[must_use]
    pub fn new(
        shared_request: Arc<RwLock<BifurcationSweepRequest>>,
        latest_generation: Arc<AtomicU64>,
        seed_points: Vec<PlotPoint>,
        seed_lyapunov: Vec<PlotPoint>,
        seed_raw: Vec<[f64; 2]>,
    ) -> Self {
        let initial = *shared_request.read().unwrap_or_else(|e| e.into_inner());
        Self {
            shared_request,
            latest_generation,
            active: initial,
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
        req.steps = req.steps.clamp(100, 2000);
        req.r_min = req.r_min.clamp(0.0, 3.999_999);
        req.r_max = req.r_max.clamp(req.r_min + 1e-6, 4.0);
        self.active = req;
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
            sample_column(
                r,
                &mut self.points_buf,
                &mut self.lyapunov_buf,
                &mut self.raw_lyapunov_buf,
            );
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
        self.active.r_min.is_finite()
            && self.active.r_max.is_finite()
            && (0.0..=4.0).contains(&self.active.r_min)
            && (0.0..=4.0).contains(&self.active.r_max)
            && self.active.r_min < self.active.r_max
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        (!self.is_finite_state()).then(|| {
            DiagnosticReport::new(
                0.0,
                0,
                "bifurcation_bounds",
                "Invalid or non-finite logistic growth rate bounds",
            )
        })
    }
}

impl TheoryDescribable for BifurcationDiagram {
    fn theory_description(&self) -> String {
        format!(
            "Logistic map bifurcation diagram, r range: {:.2} to {:.2}, selected r={:.3} (λ={:+.3})",
            self.r_min,
            self.r_max,
            self.selected_r,
            self.selected_lyapunov()
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
