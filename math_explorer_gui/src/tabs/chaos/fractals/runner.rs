#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "render.rs"]
mod render;
pub use render::*;

use super::formula::CompiledFractalJob;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::DiagnosticReport;
use eframe::egui;
use math_commons::math_kernel::colormap::ShaderPalette;
use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Minimum allowed real/imaginary coordinate for `center` and `julia_c`.
pub const COORD_MIN: f64 = -4.0;
/// Maximum allowed real/imaginary coordinate for `center` and `julia_c`.
pub const COORD_MAX: f64 = 4.0;
/// Minimum allowed magnification factor for `FractalViewer`.
pub const ZOOM_MIN: f64 = 0.1;
/// Maximum allowed magnification factor for `FractalViewer`.
pub const ZOOM_MAX: f64 = 10_000.0;

/// Maximum GPU/CPU texture side permitted for fractal raster uploads and exports.
pub const MAX_FRACTAL_TEXTURE_SIDE: usize = 2048;

/// Default full-viewport export size used before a real window screen_rect is measured.
pub const DEFAULT_EXPORT_VIEWPORT_SIZE: (usize, usize) = (640, 480);

/// Maximum UTF-8 byte length permitted for serialized custom fractal/mapping formulas.
pub const MAX_CUSTOM_SHADER_EXPR_BYTES: usize = 512;

/// Progressive block-stride schedule from coarse (`8x`) to full resolution (`1x`).
pub const FRACTAL_STRIDES: [usize; 4] = [8, 4, 2, 1];

/// Initial lightweight preview dimensions rendered on tool construction.
pub const INITIAL_PREVIEW_SIZE: (usize, usize) = (80, 60);

const WORKER_STEP_BUDGET: Duration = Duration::from_millis(8);

/// Fractal family selector.
#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum FractalMode {
    /// Mandelbrot set $z_{n+1} = z_n^2 + c$ from $z_0 = 0$.
    Mandelbrot,
    /// Julia set $z_{n+1} = z_n^2 + c$ for fixed complex constant $c$.
    Julia,
    /// Custom c-plane escape fractal $z_{n+1} = f(z_n, c)$ from $z_0 = 0$.
    CustomMandelbrot,
    /// Custom Julia escape fractal $z_{n+1} = f(z_n, c)$ for fixed complex constant $c$.
    CustomJulia,
}

impl FractalMode {
    /// Returns `true` when evaluating a user-authored [`CompiledFractalJob`].
    #[must_use]
    pub const fn is_custom(self) -> bool {
        matches!(self, Self::CustomMandelbrot | Self::CustomJulia)
    }

    /// Returns `true` when the complex constant $c$ slider applies.
    #[must_use]
    pub const fn uses_julia_c(self) -> bool {
        matches!(self, Self::Julia | Self::CustomJulia)
    }
}

/// Coloring configuration shared between [`super::FractalViewer`] and [`FractalRunner`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FractalColorStyle {
    /// Whether continuous Douady-Hubbard log-log escape renormalization is enabled.
    pub smooth_coloring: bool,
    /// Selected analytical colormap family.
    pub palette: ShaderPalette,
}

impl Default for FractalColorStyle {
    fn default() -> Self {
        Self {
            smooth_coloring: true,
            palette: ShaderPalette::CyclicCosine,
        }
    }
}

/// Immutable parameter bundle describing a single progressive fractal render request.
#[derive(Clone, Copy, Debug)]
pub struct FractalRenderRequest {
    /// Monotonic request generation used for mid-pass cancellation.
    pub generation: u64,
    /// Clamped output raster width in pixels.
    pub width: usize,
    /// Clamped output raster height in pixels.
    pub height: usize,
    /// Complex-plane viewport center.
    pub center: Complex<f64>,
    /// Magnification factor in `0.1..=10_000.0`.
    pub zoom: f64,
    /// Escape-time iteration ceiling.
    pub max_iter: u32,
    /// Mandelbrot, Julia, or symbolic Custom mode.
    pub mode: FractalMode,
    /// Complex constant $c$ for Julia or Custom Julia-plane mode.
    pub julia_c: Complex<f64>,
    /// Whether `ThemeMode::HighContrast` dark-canvas contrast guards are active.
    pub high_contrast: bool,
}

/// Snapshot metadata attached to each progressive pass emitted by [`FractalRunner`].
#[derive(Clone, Copy, Debug)]
pub struct FractalPassMeta {
    /// Request generation that produced this snapshot.
    pub generation: u64,
    /// Stride (`8`, `4`, `2`, or `1`) of the pass reflected in this snapshot.
    pub stride: usize,
    /// `true` once the full-resolution `1x` pass has finished for `generation`.
    pub is_final: bool,
}

/// Background worker runner executing cancellable `8x -> 4x -> 2x -> 1x` fractal passes.
pub struct FractalRunner {
    shared_request: Arc<RwLock<FractalRenderRequest>>,
    shared_formula: Arc<RwLock<Option<Arc<CompiledFractalJob>>>>,
    shared_style: Arc<RwLock<FractalColorStyle>>,
    latest_generation: Arc<AtomicU64>,
    active: FractalRenderRequest,
    active_formula: Option<Arc<CompiledFractalJob>>,
    active_style: FractalColorStyle,
    stride_idx: usize,
    pub(super) next_sample_y: usize,
    completed_stride: usize,
    buffer: Vec<egui::Color32>,
}

impl FractalRunner {
    /// Creates a new [`FractalRunner`] seeded with an initial lightweight coarse preview.
    #[cfg(test)]
    #[must_use]
    pub fn new(
        shared_request: Arc<RwLock<FractalRenderRequest>>,
        latest_generation: Arc<AtomicU64>,
    ) -> Self {
        Self::with_formula(
            shared_request,
            latest_generation,
            Arc::new(RwLock::new(None)),
        )
    }

    /// Creates a [`FractalRunner`] sharing an `Arc`-backed custom [`CompiledFractalJob`] slot.
    #[cfg(test)]
    #[must_use]
    pub fn with_formula(
        shared_request: Arc<RwLock<FractalRenderRequest>>,
        latest_generation: Arc<AtomicU64>,
        shared_formula: Arc<RwLock<Option<Arc<CompiledFractalJob>>>>,
    ) -> Self {
        Self::with_formula_and_style(
            shared_request,
            latest_generation,
            shared_formula,
            Arc::new(RwLock::new(FractalColorStyle::default())),
        )
    }

    /// Creates a [`FractalRunner`] sharing both [`CompiledFractalJob`] and [`FractalColorStyle`] slots.
    #[must_use]
    pub fn with_formula_and_style(
        shared_request: Arc<RwLock<FractalRenderRequest>>,
        latest_generation: Arc<AtomicU64>,
        shared_formula: Arc<RwLock<Option<Arc<CompiledFractalJob>>>>,
        shared_style: Arc<RwLock<FractalColorStyle>>,
    ) -> Self {
        let initial = *shared_request.read().unwrap_or_else(|e| e.into_inner());
        let formula = shared_formula
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let style = *shared_style.read().unwrap_or_else(|e| e.into_inner());
        let seed_img = render_fractal_color_image_styled(&initial, formula.as_deref(), style);
        Self {
            shared_request,
            shared_formula,
            shared_style,
            latest_generation,
            active: FractalRenderRequest {
                width: seed_img.size[0],
                height: seed_img.size[1],
                ..initial
            },
            active_formula: formula,
            active_style: style,
            stride_idx: FRACTAL_STRIDES.len(),
            next_sample_y: 0,
            completed_stride: 1,
            buffer: seed_img.pixels,
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
        let formula = self
            .shared_formula
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let style = *self.shared_style.read().unwrap_or_else(|e| e.into_inner());
        let (w, h) = clamp_fractal_dimensions(req.width, req.height, MAX_FRACTAL_TEXTURE_SIDE);
        req.width = w;
        req.height = h;
        let needed = w.saturating_mul(h);
        if self.buffer.len() != needed {
            self.buffer.resize(needed, egui::Color32::BLACK);
        }
        self.active = req;
        self.active_formula = formula;
        self.active_style = style;
        self.stride_idx = 0;
        self.next_sample_y = 0;
        self.completed_stride = FRACTAL_STRIDES[0];
        true
    }

    fn render_sample_row(&mut self, sy: usize, stride: usize, scale_x: f64, scale_y: f64) {
        let width = self.active.width;
        let height = self.active.height;
        let x_start = self.active.center.re - scale_x * 0.5;
        let y_start = self.active.center.im + scale_y * 0.5;
        let im = y_start - (sy as f64 / height as f64) * scale_y;
        let prev_stride = stride.saturating_mul(2);
        let y_end = sy.saturating_add(stride).min(height);

        let mut sx = 0;
        while sx < width {
            let reuse_coarse = stride < FRACTAL_STRIDES[0]
                && sx.is_multiple_of(prev_stride)
                && sy.is_multiple_of(prev_stride);
            let color = if reuse_coarse {
                self.buffer[flatten_2d_index(sx, sy, width)]
            } else {
                let re = x_start + (sx as f64 / width as f64) * scale_x;
                eval_pixel_color(
                    re,
                    im,
                    &self.active,
                    self.active_formula.as_deref(),
                    self.active_style,
                )
            };

            if stride == 1 {
                self.buffer[flatten_2d_index(sx, sy, width)] = color;
            } else {
                let x_end = sx.saturating_add(stride).min(width);
                for by in sy..y_end {
                    for bx in sx..x_end {
                        self.buffer[flatten_2d_index(bx, by, width)] = color;
                    }
                }
            }
            sx = sx.saturating_add(stride);
        }
    }
}

impl SimulationRunner for FractalRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if matches!(cmd, SimCommand::Reset | SimCommand::Custom(_)) {
            let _ = self.sync_request();
        }
    }

    fn step(&mut self) {
        let _ = self.sync_request();
        if self.stride_idx >= FRACTAL_STRIDES.len()
            || self.active.width == 0
            || self.active.height == 0
            || !self.is_finite_state()
        {
            return;
        }

        let stride = FRACTAL_STRIDES[self.stride_idx];
        let height = self.active.height;
        let scale_y = 3.0 / self.active.zoom;
        let scale_x = scale_y * (self.active.width as f64 / height as f64);
        let slice_start = crate::web_io::PlatformInstant::now();

        let mut sy = self.next_sample_y;
        while sy < height {
            if self.latest_generation.load(Ordering::Relaxed) != self.active.generation {
                let _ = self.sync_request();
                return;
            }
            self.render_sample_row(sy, stride, scale_x, scale_y);
            sy = sy.saturating_add(stride);
            self.next_sample_y = sy;
            if sy < height && slice_start.elapsed() >= WORKER_STEP_BUDGET {
                self.completed_stride = stride;
                return;
            }
        }

        self.completed_stride = stride;
        self.stride_idx += 1;
        self.next_sample_y = 0;
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let is_final = self.stride_idx >= FRACTAL_STRIDES.len()
            && self.latest_generation.load(Ordering::Acquire) == self.active.generation;
        let meta = FractalPassMeta {
            generation: self.active.generation,
            stride: self.completed_stride,
            is_final,
        };
        StateSnapshot {
            width: self.active.width,
            height: self.active.height,
            pixels: Arc::new(RwLock::new(self.buffer.clone())),
            custom_data: vec![self.active.zoom, f64::from(self.active.max_iter)],
            structured_data: Some(Box::new(meta)),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_complete(&self) -> bool {
        self.latest_generation.load(Ordering::Acquire) == self.active.generation
            && self.stride_idx >= FRACTAL_STRIDES.len()
    }

    fn is_finite_state(&self) -> bool {
        self.active.center.re.is_finite()
            && self.active.center.im.is_finite()
            && (COORD_MIN..=COORD_MAX).contains(&self.active.center.re)
            && (COORD_MIN..=COORD_MAX).contains(&self.active.center.im)
            && self.active.zoom.is_finite()
            && (ZOOM_MIN..=ZOOM_MAX).contains(&self.active.zoom)
            && self.active.julia_c.re.is_finite()
            && self.active.julia_c.im.is_finite()
            && (COORD_MIN..=COORD_MAX).contains(&self.active.julia_c.re)
            && (COORD_MIN..=COORD_MAX).contains(&self.active.julia_c.im)
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        (!self.is_finite_state()).then(|| {
            DiagnosticReport::new(
                0.0,
                0,
                "fractal_parameters",
                "Non-finite fractal viewport parameter encountered",
            )
        })
    }
}
