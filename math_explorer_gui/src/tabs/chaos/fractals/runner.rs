#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::formula::CompiledFractalJob;
use super::state::FractalMode;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::DiagnosticReport;
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::chaos::fractals::{escape_time_julia, escape_time_mandelbrot};
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

/// Maximum GPU/CPU texture side permitted for fractal raster uploads.
pub const MAX_FRACTAL_TEXTURE_SIDE: usize = 2048;

/// Progressive block-stride schedule from coarse (`8x`) to full resolution (`1x`).
pub const FRACTAL_STRIDES: [usize; 4] = [8, 4, 2, 1];

/// Initial lightweight preview dimensions rendered on tool construction.
pub const INITIAL_PREVIEW_SIZE: (usize, usize) = (80, 60);

/// Dimensions of the live Julia hover preview inset rendered over the Mandelbrot plane.
pub const JULIA_INSET_SIZE: (usize, usize) = (96, 72);

const WORKER_STEP_BUDGET: Duration = Duration::from_millis(8);

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
    /// Mandelbrot or Julia mode.
    pub mode: FractalMode,
    /// Complex constant $c$ for Julia mode.
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

/// Maps a canvas screen position inside `rect` to a clamped complex coordinate `(re, im)`.
#[must_use]
pub fn map_canvas_pos_to_complex(
    pos: egui::Pos2,
    rect: egui::Rect,
    center: Complex<f64>,
    zoom: f64,
) -> Complex<f64> {
    let w = f64::from(rect.width()).max(1.0);
    let h = f64::from(rect.height()).max(1.0);
    let nx = (f64::from(pos.x - rect.min.x) / w).clamp(0.0, 1.0);
    let ny = (f64::from(pos.y - rect.min.y) / h).clamp(0.0, 1.0);
    let safe_zoom = if zoom.is_finite() {
        zoom.clamp(ZOOM_MIN, ZOOM_MAX)
    } else {
        1.0
    };
    let scale_y = 3.0 / safe_zoom;
    let scale_x = scale_y * (w / h);
    Complex::new(
        (center.re + (nx - 0.5) * scale_x).clamp(COORD_MIN, COORD_MAX),
        (center.im + (0.5 - ny) * scale_y).clamp(COORD_MIN, COORD_MAX),
    )
}

/// Clamps `(width, height)` proportionally so neither side exceeds `min(max_texture_side, 2048)`.
#[must_use]
pub fn clamp_fractal_dimensions(
    width: usize,
    height: usize,
    max_texture_side: usize,
) -> (usize, usize) {
    if width == 0 || height == 0 {
        return (0, 0);
    }
    let limit = if max_texture_side == 0 {
        MAX_FRACTAL_TEXTURE_SIDE
    } else {
        max_texture_side.min(MAX_FRACTAL_TEXTURE_SIDE)
    };
    let max_dim = width.max(height);
    if max_dim <= limit {
        return (width, height);
    }
    let scale = limit as f64 / max_dim as f64;
    let clamped_w = ((width as f64 * scale).round() as usize).clamp(1, limit);
    let clamped_h = ((height as f64 * scale).round() as usize).clamp(1, limit);
    (clamped_w, clamped_h)
}

pub(super) fn snapshot_to_clamped_color_image(
    snapshot: &StateSnapshot,
    max_texture_side: usize,
) -> Option<egui::ColorImage> {
    let (clamped_w, clamped_h) =
        clamp_fractal_dimensions(snapshot.width, snapshot.height, max_texture_side);
    if clamped_w == 0 || clamped_h == 0 || snapshot.width == 0 || snapshot.height == 0 {
        return None;
    }
    let px = snapshot.pixels.try_read().ok()?;
    if px.len() != snapshot.width.saturating_mul(snapshot.height) {
        return None;
    }
    if (clamped_w, clamped_h) == (snapshot.width, snapshot.height) {
        return Some(egui::ColorImage::new([clamped_w, clamped_h], px.clone()));
    }
    let mut out = vec![egui::Color32::BLACK; clamped_w * clamped_h];
    for y in 0..clamped_h {
        let src_y = (y * snapshot.height / clamped_h).min(snapshot.height - 1);
        for x in 0..clamped_w {
            let src_x = (x * snapshot.width / clamped_w).min(snapshot.width - 1);
            out[flatten_2d_index(x, y, clamped_w)] =
                px[flatten_2d_index(src_x, src_y, snapshot.width)];
        }
    }
    Some(egui::ColorImage::new([clamped_w, clamped_h], out))
}

/// Maps an escape-time iteration count to an opaque `Color32`, lifting dark escape bands
/// above `3.0:1` on `BLACK` when `high_contrast` is `true`.
#[must_use]
pub fn map_iter_to_color32(iter: u32, max_iter: u32, high_contrast: bool) -> egui::Color32 {
    if iter >= max_iter {
        egui::Color32::BLACK
    } else if high_contrast {
        let rgb = math_commons::math_kernel::colormap::cyclic_cosine_palette_high_contrast(
            f64::from(iter),
        );
        egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
    } else {
        let rgb = math_commons::math_kernel::colormap::cyclic_cosine_palette(f64::from(iter));
        egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
    }
}

fn eval_pixel_color(
    re: f64,
    im: f64,
    req: &FractalRenderRequest,
    formula: Option<&CompiledFractalJob>,
) -> egui::Color32 {
    let pt = Complex::new(re, im);
    let iter = match req.mode {
        FractalMode::Mandelbrot => escape_time_mandelbrot(pt, req.max_iter),
        FractalMode::Julia => escape_time_julia(pt, req.julia_c, req.max_iter),
        FractalMode::CustomMandelbrot => formula.map_or_else(
            || escape_time_mandelbrot(pt, req.max_iter),
            |job| job.escape_time(Complex::new(0.0, 0.0), pt, req.max_iter),
        ),
        FractalMode::CustomJulia => formula.map_or_else(
            || escape_time_julia(pt, req.julia_c, req.max_iter),
            |job| job.escape_time(pt, req.julia_c, req.max_iter),
        ),
    };
    map_iter_to_color32(iter, req.max_iter, req.high_contrast)
}

/// Synchronously renders a full-resolution fractal image (used for small seed previews and exports).
#[cfg(test)]
#[must_use]
pub fn render_fractal_color_image(req: &FractalRenderRequest) -> egui::ColorImage {
    render_fractal_color_image_with_formula(req, None)
}

/// Synchronously renders a full-resolution fractal image with an optional [`CompiledFractalJob`].
#[must_use]
pub fn render_fractal_color_image_with_formula(
    req: &FractalRenderRequest,
    formula: Option<&CompiledFractalJob>,
) -> egui::ColorImage {
    let (width, height) = clamp_fractal_dimensions(req.width, req.height, MAX_FRACTAL_TEXTURE_SIDE);
    if width == 0 || height == 0 {
        return egui::ColorImage::new([1, 1], vec![egui::Color32::BLACK]);
    }
    let zoom = if req.zoom.is_finite() && req.zoom >= ZOOM_MIN {
        req.zoom.min(ZOOM_MAX)
    } else {
        1.0
    };
    let scale_y = 3.0 / zoom;
    let scale_x = scale_y * (width as f64 / height as f64);
    let x_start = req.center.re - scale_x * 0.5;
    let y_start = req.center.im + scale_y * 0.5;
    let mut pixels = vec![egui::Color32::BLACK; width * height];

    for y in 0..height {
        let im = y_start - (y as f64 / height as f64) * scale_y;
        for x in 0..width {
            let re = x_start + (x as f64 / width as f64) * scale_x;
            pixels[flatten_2d_index(x, y, width)] = eval_pixel_color(re, im, req, formula);
        }
    }
    egui::ColorImage::new([width, height], pixels)
}

/// Background worker runner executing cancellable `8x -> 4x -> 2x -> 1x` fractal passes.
pub struct FractalRunner {
    shared_request: Arc<RwLock<FractalRenderRequest>>,
    shared_formula: Arc<RwLock<Option<Arc<CompiledFractalJob>>>>,
    latest_generation: Arc<AtomicU64>,
    active: FractalRenderRequest,
    active_formula: Option<Arc<CompiledFractalJob>>,
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
    #[must_use]
    pub fn with_formula(
        shared_request: Arc<RwLock<FractalRenderRequest>>,
        latest_generation: Arc<AtomicU64>,
        shared_formula: Arc<RwLock<Option<Arc<CompiledFractalJob>>>>,
    ) -> Self {
        let initial = *shared_request.read().unwrap_or_else(|e| e.into_inner());
        let formula = shared_formula
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let seed_img = render_fractal_color_image_with_formula(&initial, formula.as_deref());
        Self {
            shared_request,
            shared_formula,
            latest_generation,
            active: FractalRenderRequest {
                width: seed_img.size[0],
                height: seed_img.size[1],
                ..initial
            },
            active_formula: formula,
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
        let (w, h) = clamp_fractal_dimensions(req.width, req.height, MAX_FRACTAL_TEXTURE_SIDE);
        req.width = w;
        req.height = h;
        let needed = w.saturating_mul(h);
        if self.buffer.len() != needed {
            self.buffer.resize(needed, egui::Color32::BLACK);
        }
        self.active = req;
        self.active_formula = formula;
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
                eval_pixel_color(re, im, &self.active, self.active_formula.as_deref())
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
            sonification: crate::audio::SonificationFrame::silent(),
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
