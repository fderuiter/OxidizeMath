#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AtlasAxisPair, AtlasCell, RegimeCartographer};
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::DiagnosticReport;
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Default and maximum raster side (`128 x 128`) keeping cooperative slices lightweight.
pub const DEFAULT_ATLAS_SIDE: usize = 128;

/// Hard upper cap on atlas raster dimensions (`128 x 128`).
pub const MAX_ATLAS_SIDE: usize = 128;

/// Progressive coarse-to-fine block-stride schedule (`8x -> 4x -> 2x -> 1x`).
pub const ATLAS_STRIDES: [usize; 4] = [8, 4, 2, 1];

/// Cooperative per-step wall-clock slice budget (`8ms`), matching `WASM_FRAME_SLICE_BUDGET`.
pub const ATLAS_SLICE_BUDGET: Duration = Duration::from_millis(8);

const SLICE_CHECK_INTERVAL: usize = 16;

/// Clamps requested `(width, height)` into `1..=MAX_ATLAS_SIDE` (`(0, 0)` when either side is `0`).
#[must_use]
pub fn clamp_atlas_dimensions(width: usize, height: usize) -> (usize, usize) {
    if width == 0 || height == 0 {
        return (0, 0);
    }
    let max_dim = width.max(height);
    if max_dim <= MAX_ATLAS_SIDE {
        return (width, height);
    }
    let scale = MAX_ATLAS_SIDE as f64 / max_dim as f64;
    let cw = ((width as f64 * scale).round() as usize).clamp(1, MAX_ATLAS_SIDE);
    let ch = ((height as f64 * scale).round() as usize).clamp(1, MAX_ATLAS_SIDE);
    (cw, ch)
}

/// Shared render request consumed by [`AtlasRunner`].
#[derive(Clone)]
pub struct AtlasRenderRequest {
    /// Monotonic generation token matched against `latest_generation` for mid-pass cancellation.
    pub generation: u64,
    /// Requested raster width (`<= 128`).
    pub width: usize,
    /// Requested raster height (`<= 128`).
    pub height: usize,
    /// Whether `ThemeMode::HighContrast` (`>= 3.0:1` on `BLACK`) is active.
    pub high_contrast: bool,
    /// Active domain cartographer evaluating `(p1, p2)` cells.
    pub cartographer: Arc<dyn RegimeCartographer>,
}

/// Metadata attached to `StateSnapshot::structured_data` for each progressive pass.
#[derive(Clone)]
pub struct AtlasPassMeta {
    /// Request generation that produced this snapshot.
    pub generation: u64,
    /// Stride (`8`, `4`, `2`, or `1`) reflected in this snapshot.
    pub stride: usize,
    /// `true` once the `1x` pass has completed for `generation`.
    pub is_final: bool,
    /// Clamped raster width.
    pub width: usize,
    /// Clamped raster height.
    pub height: usize,
    /// Axis metadata for coordinate projection and hover lookup.
    pub axes: AtlasAxisPair,
    /// Evaluated cell grid matching `width * height`.
    pub cells: Arc<Vec<AtlasCell>>,
}

/// Progressive background runner executing `[8, 4, 2, 1]` parameter-atlas passes.
pub struct AtlasRunner {
    shared_request: Arc<RwLock<AtlasRenderRequest>>,
    latest_generation: Arc<AtomicU64>,
    active: AtlasRenderRequest,
    axes: AtlasAxisPair,
    stride_idx: usize,
    next_sample_x: usize,
    next_sample_y: usize,
    completed_stride: usize,
    pixels: Vec<egui::Color32>,
    cells: Vec<AtlasCell>,
}

impl AtlasRunner {
    /// Constructs a new [`AtlasRunner`] with preallocated scratch buffers ready for the `8x` pass.
    #[must_use]
    pub fn new(
        shared_request: Arc<RwLock<AtlasRenderRequest>>,
        latest_generation: Arc<AtomicU64>,
    ) -> Self {
        let mut initial = shared_request
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let (w, h) = clamp_atlas_dimensions(initial.width, initial.height);
        initial.width = w;
        initial.height = h;
        let axes = initial.cartographer.axes();
        let total = w.saturating_mul(h);
        Self {
            shared_request,
            latest_generation,
            active: initial,
            axes,
            stride_idx: 0,
            next_sample_x: 0,
            next_sample_y: 0,
            completed_stride: ATLAS_STRIDES[0],
            pixels: vec![egui::Color32::BLACK; total],
            cells: vec![AtlasCell::default(); total],
        }
    }

    /// Synchronizes scratch buffers with `shared_request` if `latest_generation` advanced.
    pub fn sync_request(&mut self) -> bool {
        let target_gen = self.latest_generation.load(Ordering::Acquire);
        if target_gen == self.active.generation {
            return false;
        }
        let mut req = self
            .shared_request
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let (w, h) = clamp_atlas_dimensions(req.width, req.height);
        req.width = w;
        req.height = h;
        let needed = w.saturating_mul(h);
        if self.pixels.len() != needed {
            self.pixels.resize(needed, egui::Color32::BLACK);
        }
        if self.cells.len() != needed {
            self.cells.resize(needed, AtlasCell::default());
        }
        self.axes = req.cartographer.axes();
        self.active = req;
        self.stride_idx = 0;
        self.next_sample_x = 0;
        self.next_sample_y = 0;
        self.completed_stride = ATLAS_STRIDES[0];
        true
    }

    #[inline]
    fn write_sample_block(
        &mut self,
        sx: usize,
        sy: usize,
        stride: usize,
        cell: AtlasCell,
        color: egui::Color32,
    ) {
        let width = self.active.width;
        let height = self.active.height;
        if stride == 1 {
            let idx = flatten_2d_index(sx, sy, width);
            self.cells[idx] = cell;
            self.pixels[idx] = color;
            return;
        }
        let x_end = sx.saturating_add(stride).min(width);
        let y_end = sy.saturating_add(stride).min(height);
        for by in sy..y_end {
            for bx in sx..x_end {
                let idx = flatten_2d_index(bx, by, width);
                self.cells[idx] = cell;
                self.pixels[idx] = color;
            }
        }
    }

    fn advance_pass_slice(&mut self, stride: usize, slice_start: crate::web_io::PlatformInstant) {
        let width = self.active.width;
        let height = self.active.height;
        let prev_stride = stride.saturating_mul(2);
        let mut evals_since_check = 0usize;
        let mut sy = self.next_sample_y;
        let mut sx = self.next_sample_x;

        while sy < height {
            while sx < width {
                if evals_since_check >= SLICE_CHECK_INTERVAL {
                    evals_since_check = 0;
                    if self.latest_generation.load(Ordering::Acquire) != self.active.generation {
                        let _ = self.sync_request();
                        return;
                    }
                    if slice_start.elapsed() >= ATLAS_SLICE_BUDGET {
                        self.next_sample_x = sx;
                        self.next_sample_y = sy;
                        self.completed_stride = stride;
                        return;
                    }
                }
                let reuse_coarse = stride < ATLAS_STRIDES[0]
                    && sx.is_multiple_of(prev_stride)
                    && sy.is_multiple_of(prev_stride);
                let idx = flatten_2d_index(sx, sy, width);
                let (cell, color) = if reuse_coarse {
                    (self.cells[idx], self.pixels[idx])
                } else {
                    evals_since_check += 1;
                    let (p1, p2) = self.axes.pixel_to_coords(sx, sy, width, height);
                    let c = self.active.cartographer.evaluate_cell(p1, p2);
                    let col = self
                        .active
                        .cartographer
                        .color_for_cell(c, self.active.high_contrast);
                    (c, col)
                };
                self.write_sample_block(sx, sy, stride, cell, color);
                sx = sx.saturating_add(stride);
            }
            sx = 0;
            sy = sy.saturating_add(stride);
        }

        self.completed_stride = stride;
        self.stride_idx += 1;
        self.next_sample_x = 0;
        self.next_sample_y = 0;
    }

    /// Steps the current progressive pass without time-slice truncation (used in deterministic tests).
    #[cfg(test)]
    pub fn step_without_budget(&mut self) {
        let _ = self.sync_request();
        if self.stride_idx >= ATLAS_STRIDES.len()
            || self.active.width == 0
            || self.active.height == 0
        {
            return;
        }
        let stride = ATLAS_STRIDES[self.stride_idx];
        let width = self.active.width;
        let height = self.active.height;
        let prev_stride = stride.saturating_mul(2);
        let mut sy = 0;
        while sy < height {
            let mut sx = 0;
            while sx < width {
                let reuse = stride < ATLAS_STRIDES[0]
                    && sx.is_multiple_of(prev_stride)
                    && sy.is_multiple_of(prev_stride);
                let idx = flatten_2d_index(sx, sy, width);
                let (cell, color) = if reuse {
                    (self.cells[idx], self.pixels[idx])
                } else {
                    let (p1, p2) = self.axes.pixel_to_coords(sx, sy, width, height);
                    let c = self.active.cartographer.evaluate_cell(p1, p2);
                    let col = self
                        .active
                        .cartographer
                        .color_for_cell(c, self.active.high_contrast);
                    (c, col)
                };
                self.write_sample_block(sx, sy, stride, cell, color);
                sx = sx.saturating_add(stride);
            }
            sy = sy.saturating_add(stride);
        }
        self.completed_stride = stride;
        self.stride_idx += 1;
        self.next_sample_x = 0;
        self.next_sample_y = 0;
    }
}

impl SimulationRunner for AtlasRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if matches!(cmd, SimCommand::Reset | SimCommand::Custom(_)) {
            let _ = self.sync_request();
        }
    }

    fn step(&mut self) {
        let _ = self.sync_request();
        if self.stride_idx >= ATLAS_STRIDES.len()
            || self.active.width == 0
            || self.active.height == 0
            || !self.is_finite_state()
        {
            return;
        }
        let stride = ATLAS_STRIDES[self.stride_idx];
        let slice_start = crate::web_io::PlatformInstant::now();
        self.advance_pass_slice(stride, slice_start);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let is_final = self.stride_idx >= ATLAS_STRIDES.len()
            && self.latest_generation.load(Ordering::Acquire) == self.active.generation;
        let meta = AtlasPassMeta {
            generation: self.active.generation,
            stride: self.completed_stride,
            is_final,
            width: self.active.width,
            height: self.active.height,
            axes: self.axes.clone(),
            cells: Arc::new(self.cells.clone()),
        };
        StateSnapshot {
            width: self.active.width,
            height: self.active.height,
            pixels: Arc::new(RwLock::new(self.pixels.clone())),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(meta)),
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_complete(&self) -> bool {
        self.latest_generation.load(Ordering::Acquire) == self.active.generation
            && self.stride_idx >= ATLAS_STRIDES.len()
    }

    fn is_finite_state(&self) -> bool {
        self.axes.x.constraint.min.is_finite()
            && self.axes.x.constraint.max.is_finite()
            && self.axes.x.constraint.max > self.axes.x.constraint.min
            && self.axes.y.constraint.min.is_finite()
            && self.axes.y.constraint.max.is_finite()
            && self.axes.y.constraint.max > self.axes.y.constraint.min
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        (!self.is_finite_state()).then(|| {
            DiagnosticReport::new(
                0.0,
                0,
                "atlas_axes",
                "Non-finite or degenerate parameter atlas axis bounds encountered",
            )
        })
    }
}
