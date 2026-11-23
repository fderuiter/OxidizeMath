#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{BackgroundWorker, SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::DiagnosticReport;
use eframe::egui;
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::chaos::fractals::correlation_dimension_in_place;
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
use math_explorer::physics::chaos::metrics::lorenz_lyapunov;
use math_explorer::pure_math::analysis::ode::RungeKutta4;
use nalgebra::Vector3;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Maximum sliding-window sample count used for live in-place correlation sum readouts.
pub const CORRELATION_WINDOW_CAP: usize = 256;
/// Neighborhood radius $\epsilon$ passed to `correlation_dimension_in_place`.
pub const CORRELATION_EPSILON: f64 = 5.0;
/// Dimensions `(cols_rho, rows_sigma)` of the 2D Lorenz Lyapunov regime heatmap.
pub const REGIME_GRID_SIZE: (usize, usize) = (36, 24);
/// Progressive block strides for [`LorenzRegimeMapRunner`].
pub const REGIME_STRIDES: [usize; 3] = [4, 2, 1];
/// Rayleigh parameter domain `[0.0, 100.0]` on the horizontal regime axis.
pub const REGIME_RHO_RANGE: (f64, f64) = (0.0, 100.0);
/// Prandtl parameter domain `[0.0, 50.0]` on the vertical regime axis.
pub const REGIME_SIGMA_RANGE: (f64, f64) = (0.0, 50.0);

const WORKER_STEP_BUDGET: Duration = Duration::from_millis(8);

#[derive(Clone, Copy)]
struct LyapunovCacheEntry {
    key: [u64; 4],
    value: f64,
}

struct AttractorMetricsScratch {
    solver: RungeKutta4<Vector3<f64>>,
    trajectory_buf: Vec<Vector3<f64>>,
    lyapunov_cache: [Option<LyapunovCacheEntry>; 4],
    cache_cursor: usize,
}

impl Default for AttractorMetricsScratch {
    fn default() -> Self {
        Self {
            solver: RungeKutta4::new(&Vector3::zeros()),
            trajectory_buf: vec![Vector3::zeros(); CORRELATION_WINDOW_CAP],
            lyapunov_cache: [None; 4],
            cache_cursor: 0,
        }
    }
}

thread_local! {
    static ATTRACTOR_METRICS_SCRATCH: RefCell<AttractorMetricsScratch> =
        RefCell::new(AttractorMetricsScratch::default());
}

/// Eagerly initializes the thread-local RK4 and trajectory scratch storage outside `MEMORY_LOCK`.
pub fn warmup_attractor_metrics_scratch() {
    ATTRACTOR_METRICS_SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        if s.trajectory_buf.len() < CORRELATION_WINDOW_CAP {
            s.trajectory_buf
                .resize(CORRELATION_WINDOW_CAP, Vector3::zeros());
        }
    });
}

fn compute_correlation_in_place(
    history: &VecDeque<Vector3<f64>>,
    scratch: &mut [Vector3<f64>],
) -> Option<f64> {
    let take_n = history.len().min(scratch.len());
    let mut written = 0_usize;
    for pt in history.iter().rev().take(take_n) {
        if pt.x.is_finite() && pt.y.is_finite() && pt.z.is_finite() {
            scratch[written] = *pt;
            written += 1;
        }
    }
    (written >= 2)
        .then(|| correlation_dimension_in_place(&mut scratch[..written], CORRELATION_EPSILON))
}

/// Zero-allocation evaluator computing `lorenz_lyapunov` and `correlation_dimension_in_place`
/// using caller-provided `RungeKutta4` and trajectory scratch buffers.
#[must_use]
pub fn evaluate_attractor_metrics_in_place(
    system: &LorenzSystem,
    dt: f64,
    history: &VecDeque<Vector3<f64>>,
    solver: &mut RungeKutta4<Vector3<f64>>,
    scratch: &mut [Vector3<f64>],
) -> (Option<f64>, Option<f64>) {
    let v = system.state.vec;
    if !v.x.is_finite()
        || !v.y.is_finite()
        || !v.z.is_finite()
        || !system.sigma.is_finite()
        || !system.rho.is_finite()
        || !system.beta.is_finite()
        || !dt.is_finite()
        || dt <= 0.0
    {
        return (None, None);
    }

    let safe_dt = dt.clamp(0.005, 0.02);
    let safe_evol = (18.0 * safe_dt).max(safe_dt);
    let lyapunov = lorenz_lyapunov(
        system,
        solver,
        Vector3::new(10.0, 10.0, 10.0),
        safe_dt,
        16,
        safe_evol,
    )
    .ok()
    .filter(|val| val.is_finite());

    (lyapunov, compute_correlation_in_place(history, scratch))
}

pub(super) fn compute_cached_correlation_only(history: &VecDeque<Vector3<f64>>) -> Option<f64> {
    ATTRACTOR_METRICS_SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        compute_correlation_in_place(history, &mut s.trajectory_buf)
    })
}

pub(super) fn compute_cached_attractor_metrics(
    system: &LorenzSystem,
    dt: f64,
    history: &VecDeque<Vector3<f64>>,
) -> (Option<f64>, Option<f64>) {
    ATTRACTOR_METRICS_SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        let v = system.state.vec;
        if !v.x.is_finite()
            || !v.y.is_finite()
            || !v.z.is_finite()
            || !system.sigma.is_finite()
            || !system.rho.is_finite()
            || !system.beta.is_finite()
            || !dt.is_finite()
            || dt <= 0.0
        {
            return (None, None);
        }
        let key = [
            system.sigma.to_bits(),
            system.rho.to_bits(),
            system.beta.to_bits(),
            dt.to_bits(),
        ];
        if let Some(cached) = s
            .lyapunov_cache
            .iter()
            .flatten()
            .find_map(|entry| (entry.key == key).then_some(entry.value))
        {
            let corr = compute_correlation_in_place(history, &mut s.trajectory_buf);
            return (Some(cached), corr);
        }
        let AttractorMetricsScratch {
            solver,
            trajectory_buf,
            lyapunov_cache,
            cache_cursor,
        } = &mut *s;
        let (lyapunov, corr) =
            evaluate_attractor_metrics_in_place(system, dt, history, solver, trajectory_buf);
        if let Some(val) = lyapunov {
            let idx = *cache_cursor % lyapunov_cache.len();
            lyapunov_cache[idx] = Some(LyapunovCacheEntry { key, value: val });
            *cache_cursor = cache_cursor.wrapping_add(1);
        }
        (lyapunov, corr)
    })
}

/// Maps a Lyapunov exponent $\lambda_{\max}$ to an opaque RGB color via `math_commons::colormap`.
#[must_use]
pub fn map_lyapunov_to_regime_color(lambda: f64, high_contrast: bool) -> egui::Color32 {
    let safe_l = if lambda.is_finite() { lambda } else { 0.0 };
    let normalized = if safe_l <= 0.0 {
        0.5 * ((safe_l + 2.0) / 2.0).clamp(0.0, 1.0)
    } else {
        0.5 + 0.5 * (safe_l / 1.5).clamp(0.0, 1.0)
    };
    let rgb = if high_contrast {
        heatmap_color_high_contrast(normalized)
    } else {
        heatmap_color(normalized)
    };
    egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
}

/// Immutable request bundle for a progressive 2D `(rho, sigma)` Lyapunov regime sweep.
#[derive(Clone, Copy, Debug)]
pub struct LorenzRegimeRequest {
    /// Monotonic generation counter for cancellation.
    pub generation: u64,
    /// Fixed geometric parameter $\beta$ for the 2D slice.
    pub beta: f64,
    /// Grid width (`rho` columns).
    pub width: usize,
    /// Grid height (`sigma` rows).
    pub height: usize,
    /// Whether WCAG high-contrast palette lifting is active.
    pub high_contrast: bool,
}

/// Snapshot metadata emitted after each progressive slice of [`LorenzRegimeMapRunner`].
#[derive(Clone)]
pub struct LorenzRegimeSnapshot {
    /// Generation that produced this snapshot.
    pub generation: u64,
    /// Current progressive block stride (`4`, `2`, or `1`).
    pub stride: usize,
    /// `true` once full `1x` resolution is complete.
    pub is_final: bool,
    /// Row-major grid of evaluated $\lambda_{\max}$ values.
    pub lambdas: Arc<Vec<f64>>,
}

/// Progressive `BackgroundWorker` runner evaluating a 2D `(rho, sigma)` `lorenz_lyapunov` heatmap.
pub struct LorenzRegimeMapRunner {
    shared_request: Arc<RwLock<LorenzRegimeRequest>>,
    latest_generation: Arc<AtomicU64>,
    active: LorenzRegimeRequest,
    stride_idx: usize,
    next_cell_idx: usize,
    completed_stride: usize,
    solver: RungeKutta4<Vector3<f64>>,
    pixels: Vec<egui::Color32>,
    lambdas: Vec<f64>,
}

impl LorenzRegimeMapRunner {
    /// Creates a new [`LorenzRegimeMapRunner`] initialized from `shared_request`.
    #[must_use]
    pub fn new(
        shared_request: Arc<RwLock<LorenzRegimeRequest>>,
        latest_generation: Arc<AtomicU64>,
    ) -> Self {
        let req = *shared_request.read().unwrap_or_else(|e| e.into_inner());
        let w = req.width.max(1);
        let h = req.height.max(1);
        let initial_color = map_lyapunov_to_regime_color(0.0, req.high_contrast);
        Self {
            shared_request,
            latest_generation,
            active: LorenzRegimeRequest {
                width: w,
                height: h,
                ..req
            },
            stride_idx: 0,
            next_cell_idx: 0,
            completed_stride: REGIME_STRIDES[0],
            solver: RungeKutta4::new(&Vector3::zeros()),
            pixels: vec![initial_color; w * h],
            lambdas: vec![0.0; w * h],
        }
    }

    fn sync_request(&mut self) -> bool {
        let target = self.latest_generation.load(Ordering::Acquire);
        if target == self.active.generation {
            return false;
        }
        let mut req = *self
            .shared_request
            .read()
            .unwrap_or_else(|e| e.into_inner());
        req.width = req.width.clamp(4, 128);
        req.height = req.height.clamp(4, 128);
        let total = req.width * req.height;
        if self.pixels.len() != total {
            self.pixels.resize(total, egui::Color32::BLACK);
            self.lambdas.resize(total, 0.0);
        }
        self.active = req;
        self.stride_idx = 0;
        self.next_cell_idx = 0;
        self.completed_stride = REGIME_STRIDES[0];
        true
    }

    fn eval_cell(&mut self, gx: usize, gy: usize) -> f64 {
        let w = self.active.width.max(2);
        let h = self.active.height.max(2);
        let tx = gx as f64 / (w - 1) as f64;
        let ty = 1.0 - (gy as f64 / (h - 1) as f64);
        let rho = REGIME_RHO_RANGE.0 + tx * (REGIME_RHO_RANGE.1 - REGIME_RHO_RANGE.0);
        let sigma = REGIME_SIGMA_RANGE.0 + ty * (REGIME_SIGMA_RANGE.1 - REGIME_SIGMA_RANGE.0);
        let sys = LorenzBuilder::new()
            .sigma(sigma)
            .rho(rho)
            .beta(self.active.beta.clamp(0.0, 10.0))
            .build(LorenzState::new(10.0, 10.0, 10.0));
        lorenz_lyapunov(
            &sys,
            &mut self.solver,
            Vector3::new(10.0, 10.0, 10.0),
            0.02,
            12,
            0.3,
        )
        .unwrap_or(0.0)
    }

    fn fill_stride_block(&mut self, sx: usize, sy: usize, stride: usize) {
        let w = self.active.width;
        let h = self.active.height;
        let prev = stride.saturating_mul(2);
        let reuse =
            stride < REGIME_STRIDES[0] && sx.is_multiple_of(prev) && sy.is_multiple_of(prev);
        let (lambda, color) = if reuse {
            let idx = flatten_2d_index(sx, sy, w);
            (self.lambdas[idx], self.pixels[idx])
        } else {
            let lam = self.eval_cell(sx, sy);
            (
                lam,
                map_lyapunov_to_regime_color(lam, self.active.high_contrast),
            )
        };
        let y_end = sy.saturating_add(stride).min(h);
        let x_end = sx.saturating_add(stride).min(w);
        for by in sy..y_end {
            for bx in sx..x_end {
                let idx = flatten_2d_index(bx, by, w);
                self.lambdas[idx] = lambda;
                self.pixels[idx] = color;
            }
        }
    }
}

impl SimulationRunner for LorenzRegimeMapRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if matches!(cmd, SimCommand::Reset | SimCommand::Custom(_)) {
            let _ = self.sync_request();
        }
    }

    fn step(&mut self) {
        let _ = self.sync_request();
        if self.stride_idx >= REGIME_STRIDES.len() || !self.is_finite_state() {
            return;
        }
        let stride = REGIME_STRIDES[self.stride_idx];
        let cols = self.active.width.div_ceil(stride);
        let rows = self.active.height.div_ceil(stride);
        let total_blocks = cols * rows;
        let slice_start = crate::web_io::PlatformInstant::now();

        while self.next_cell_idx < total_blocks {
            if self.latest_generation.load(Ordering::Relaxed) != self.active.generation {
                let _ = self.sync_request();
                return;
            }
            let bx = (self.next_cell_idx % cols) * stride;
            let by = (self.next_cell_idx / cols) * stride;
            self.fill_stride_block(bx, by, stride);
            self.next_cell_idx += 1;
            if self.next_cell_idx < total_blocks && slice_start.elapsed() >= WORKER_STEP_BUDGET {
                self.completed_stride = stride;
                return;
            }
        }

        self.completed_stride = stride;
        self.stride_idx += 1;
        self.next_cell_idx = 0;
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let is_final = self.stride_idx >= REGIME_STRIDES.len()
            && self.latest_generation.load(Ordering::Acquire) == self.active.generation;
        let meta = LorenzRegimeSnapshot {
            generation: self.active.generation,
            stride: self.completed_stride,
            is_final,
            lambdas: Arc::new(self.lambdas.clone()),
        };
        StateSnapshot {
            width: self.active.width,
            height: self.active.height,
            pixels: Arc::new(RwLock::new(self.pixels.clone())),
            custom_data: vec![self.active.beta],
            structured_data: Some(Box::new(meta)),
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_complete(&self) -> bool {
        self.latest_generation.load(Ordering::Acquire) == self.active.generation
            && self.stride_idx >= REGIME_STRIDES.len()
    }

    fn is_finite_state(&self) -> bool {
        self.active.beta.is_finite() && (0.0..=10.0).contains(&self.active.beta)
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        (!self.is_finite_state()).then(|| {
            DiagnosticReport::new(
                0.0,
                0,
                "lorenz_regime_beta",
                "Non-finite beta for regime map",
            )
        })
    }
}

pub(crate) struct LorenzRegimeAtlasState {
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) lambdas: Arc<Vec<f64>>,
    pub(crate) current_stride: usize,
    pub(crate) request_generation: u64,
    pub(crate) completed_generation: u64,
    pub(crate) last_beta_bits: u64,
    pub(crate) high_contrast: bool,
    pub(crate) shared_request: Arc<RwLock<LorenzRegimeRequest>>,
    pub(crate) latest_generation: Arc<AtomicU64>,
    pub(crate) worker: BackgroundWorker<LorenzRegimeMapRunner>,
}

impl LorenzRegimeAtlasState {
    pub(crate) fn new(beta: f64) -> Self {
        let safe_beta = if beta.is_finite() {
            beta.clamp(0.0, 10.0)
        } else {
            8.0 / 3.0
        };
        let req = LorenzRegimeRequest {
            generation: 1,
            beta: safe_beta,
            width: REGIME_GRID_SIZE.0,
            height: REGIME_GRID_SIZE.1,
            high_contrast: false,
        };
        let shared_request = Arc::new(RwLock::new(req));
        let latest_generation = Arc::new(AtomicU64::new(1));
        let worker = BackgroundWorker::new(LorenzRegimeMapRunner::new(
            Arc::clone(&shared_request),
            Arc::clone(&latest_generation),
        ));
        Self {
            texture: None,
            lambdas: Arc::new(vec![0.0; REGIME_GRID_SIZE.0 * REGIME_GRID_SIZE.1]),
            current_stride: REGIME_STRIDES[0],
            request_generation: 1,
            completed_generation: 0,
            last_beta_bits: safe_beta.to_bits(),
            high_contrast: false,
            shared_request,
            latest_generation,
            worker,
        }
    }
}
