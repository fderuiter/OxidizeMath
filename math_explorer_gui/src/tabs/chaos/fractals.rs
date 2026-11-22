#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState, StateSnapshot};
use crate::framework::InteractiveTool;
use eframe::egui;
use num_complex::Complex;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

#[path = "fractals/runner.rs"]
mod runner;
pub use runner::*;

#[path = "fractals/overlay.rs"]
mod overlay;
pub use overlay::*;

#[path = "fractals/ui.rs"]
mod ui;

#[path = "fractals/presets.rs"]
mod presets;
#[cfg(test)]
use presets::{STARTER_CUSP_CONNECTED_JSON, STARTER_CUSP_DUST_JSON, STARTER_PRESET_JSON};

/// Default complex-plane center for `FractalMode::Mandelbrot`.
pub const DEFAULT_MANDELBROT_CENTER: Complex<f64> = Complex::new(-0.75, 0.0);
/// Default complex-plane center for `FractalMode::Julia`.
pub const DEFAULT_JULIA_CENTER: Complex<f64> = Complex::new(0.0, 0.0);

/// Interactive Mandelbrot and Julia fractal viewer backed by progressive background passes.
pub struct FractalViewer {
    pub(crate) center: Complex<f64>,
    pub(crate) zoom: f64,
    pub(crate) mandelbrot_center: Complex<f64>,
    pub(crate) mandelbrot_zoom: f64,
    pub(crate) julia_center: Complex<f64>,
    pub(crate) julia_zoom: f64,
    pub(crate) max_iter: u32,
    pub(crate) mode: FractalMode,
    pub(crate) julia_c: Complex<f64>,
    /// Whether `julia_c` is pinned against Mandelbrot pointer hover updates.
    pub c_pinned: bool,
    pub(crate) high_contrast: bool,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) julia_inset_texture: Option<egui::TextureHandle>,
    pub(crate) julia_inset_image: Option<egui::ColorImage>,
    pub(crate) julia_inset_key: Option<(u64, u64, u32, bool)>,
    pub(crate) dirty: bool,
    pub(crate) requested_size: (usize, usize),
    pub(crate) request_generation: u64,
    pub(crate) completed_generation: u64,
    pub(crate) current_stride: usize,
    pub(crate) shared_request: Arc<RwLock<FractalRenderRequest>>,
    pub(crate) latest_generation: Arc<AtomicU64>,
    pub(crate) worker: BackgroundWorker<FractalRunner>,
}

impl Default for FractalViewer {
    fn default() -> Self {
        let center = DEFAULT_MANDELBROT_CENTER;
        let zoom = 1.0;
        let max_iter = 100;
        let mode = FractalMode::Mandelbrot;
        let julia_c = Complex::new(-0.4, 0.6);
        let initial_req = FractalRenderRequest {
            generation: 0,
            width: INITIAL_PREVIEW_SIZE.0,
            height: INITIAL_PREVIEW_SIZE.1,
            center,
            zoom,
            max_iter,
            mode,
            julia_c,
            high_contrast: false,
        };
        let shared_request = Arc::new(RwLock::new(initial_req));
        let latest_generation = Arc::new(AtomicU64::new(0));
        let worker = BackgroundWorker::new(FractalRunner::new(
            Arc::clone(&shared_request),
            Arc::clone(&latest_generation),
        ));
        let inset_iter = max_iter.clamp(10, JULIA_INSET_MAX_ITER);
        let inset_img = render_julia_inset_image(julia_c, inset_iter, false);
        let inset_key = (julia_c.re.to_bits(), julia_c.im.to_bits(), inset_iter, false);
        Self {
            center,
            zoom,
            mandelbrot_center: DEFAULT_MANDELBROT_CENTER,
            mandelbrot_zoom: 1.0,
            julia_center: DEFAULT_JULIA_CENTER,
            julia_zoom: 1.0,
            max_iter,
            mode,
            julia_c,
            c_pinned: false,
            high_contrast: false,
            texture: None,
            julia_inset_texture: None,
            julia_inset_image: Some(inset_img),
            julia_inset_key: Some(inset_key),
            dirty: true,
            requested_size: INITIAL_PREVIEW_SIZE,
            request_generation: 0,
            completed_generation: 0,
            current_stride: 1,
            shared_request,
            latest_generation,
            worker,
        }
    }
}

impl FractalViewer {
    /// Saves the active viewport and switches `FractalMode` while preserving per-mode viewports.
    pub fn switch_mode(&mut self, next_mode: FractalMode) {
        if self.mode == next_mode {
            return;
        }
        self.sync_active_viewport();
        self.mode = next_mode;
        match next_mode {
            FractalMode::Mandelbrot => {
                self.center = self.mandelbrot_center;
                self.zoom = self.mandelbrot_zoom;
            }
            FractalMode::Julia => {
                self.center = self.julia_center;
                self.zoom = self.julia_zoom;
            }
        }
        self.dirty = true;
    }

    pub(crate) fn sync_active_viewport(&mut self) {
        match self.mode {
            FractalMode::Mandelbrot => {
                self.mandelbrot_center = self.center;
                self.mandelbrot_zoom = self.zoom;
            }
            FractalMode::Julia => {
                self.julia_center = self.center;
                self.julia_zoom = self.zoom;
            }
        }
    }

    fn build_state(&self) -> FractalState {
        let m_non_default = (self.mandelbrot_center.re - DEFAULT_MANDELBROT_CENTER.re).abs()
            > 1e-12
            || (self.mandelbrot_center.im - DEFAULT_MANDELBROT_CENTER.im).abs() > 1e-12
            || (self.mandelbrot_zoom - 1.0).abs() > 1e-12;
        let j_non_default = (self.julia_center.re - DEFAULT_JULIA_CENTER.re).abs() > 1e-12
            || (self.julia_center.im - DEFAULT_JULIA_CENTER.im).abs() > 1e-12
            || (self.julia_zoom - 1.0).abs() > 1e-12;

        let (m_re, m_im, m_z) = if self.mode == FractalMode::Julia && m_non_default {
            (
                Some(self.mandelbrot_center.re),
                Some(self.mandelbrot_center.im),
                Some(self.mandelbrot_zoom),
            )
        } else {
            (None, None, None)
        };
        let (j_re, j_im, j_z) = if self.mode == FractalMode::Mandelbrot && j_non_default {
            (
                Some(self.julia_center.re),
                Some(self.julia_center.im),
                Some(self.julia_zoom),
            )
        } else {
            (None, None, None)
        };

        FractalState {
            center_re: self.center.re,
            center_im: self.center.im,
            zoom: self.zoom,
            max_iter: self.max_iter,
            mode: self.mode,
            julia_re: self.julia_c.re,
            julia_im: self.julia_c.im,
            c_pinned: self.c_pinned,
            mandelbrot_center_re: m_re,
            mandelbrot_center_im: m_im,
            mandelbrot_zoom: m_z,
            julia_center_re: j_re,
            julia_center_im: j_im,
            julia_zoom: j_z,
        }
    }
}

impl TheoryDescribable for FractalViewer {
    fn theory_description(&self) -> String {
        let name = match self.mode {
            FractalMode::Mandelbrot => "Mandelbrot set".to_string(),
            FractalMode::Julia => format!(
                "Julia set with c=({:.2}, {:.2})",
                self.julia_c.re, self.julia_c.im
            ),
        };
        format!(
            "Fractal viewer: {}, center: ({:.2}, {:.2}), zoom: {:.2}",
            name, self.center.re, self.center.im, self.zoom
        )
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        "Mandelbrot, B. B. (1982). The Fractal Geometry of Nature. W. H. Freeman; Julia, G. (1918). Mémoire sur l'itération des fonctions rationnelles. Journal de Mathématiques Pures et Appliquées, 8, 47-245.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("default".to_string(), "Fractal viewer".to_string());
        map
    }
}

impl InteractiveTool for FractalViewer {
    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Fractal Viewer"
    }

    fn domain(&self) -> &'static str {
        "chaos"
    }

    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::raster_only()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        let zoom = if self.zoom.is_finite() && (ZOOM_MIN..=ZOOM_MAX).contains(&self.zoom) {
            self.zoom
        } else {
            1.0
        };
        let req = FractalRenderRequest {
            generation: self.request_generation,
            width: 480,
            height: 320,
            center: Complex::new(
                self.center.re.clamp(COORD_MIN, COORD_MAX),
                self.center.im.clamp(COORD_MIN, COORD_MAX),
            ),
            zoom,
            max_iter: self
                .max_iter
                .clamp(10, math_commons::registry::MAX_ITERATIONS as u32),
            mode: self.mode,
            julia_c: Complex::new(
                self.julia_c.re.clamp(COORD_MIN, COORD_MAX),
                self.julia_c.im.clamp(COORD_MIN, COORD_MAX),
            ),
            high_contrast: self.high_contrast,
        };
        Some(render_fractal_color_image(&req))
    }

    fn step_simulation(&mut self) {
        self.zoom = (self.zoom * 1.06).clamp(ZOOM_MIN, ZOOM_MAX);
        self.sync_active_viewport();
        self.dirty = true;
    }

    fn has_finite_parameters(&self) -> bool {
        self.build_state().validate().is_ok()
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.build_state()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: FractalState =
            serde_json::from_str(state).map_err(|e| format!("Invalid fractal state: {e}"))?;
        s.validate()?;
        self.center = Complex::new(s.center_re, s.center_im);
        self.zoom = s.zoom;
        self.max_iter = s.max_iter;
        self.mode = s.mode;
        self.julia_c = Complex::new(s.julia_re, s.julia_im);
        self.c_pinned = s.c_pinned;
        match s.mode {
            FractalMode::Mandelbrot => {
                self.mandelbrot_center = self.center;
                self.mandelbrot_zoom = self.zoom;
                self.julia_center = Complex::new(
                    s.julia_center_re.unwrap_or(DEFAULT_JULIA_CENTER.re),
                    s.julia_center_im.unwrap_or(DEFAULT_JULIA_CENTER.im),
                );
                self.julia_zoom = s.julia_zoom.unwrap_or(1.0);
            }
            FractalMode::Julia => {
                self.julia_center = self.center;
                self.julia_zoom = self.zoom;
                self.mandelbrot_center = Complex::new(
                    s.mandelbrot_center_re.unwrap_or(DEFAULT_MANDELBROT_CENTER.re),
                    s.mandelbrot_center_im.unwrap_or(DEFAULT_MANDELBROT_CENTER.im),
                );
                self.mandelbrot_zoom = s.mandelbrot_zoom.unwrap_or(1.0);
            }
        }
        self.dirty = true;
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let ctx = parent_ui.ctx().clone();
        let is_high_contrast = crate::app::ThemeMode::from_visuals(parent_ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        if self.high_contrast != is_high_contrast {
            self.high_contrast = is_high_contrast;
            self.dirty = true;
        }
        let max_side = ctx.input(|i| i.max_texture_side);
        self.sync_texture_from_worker(&ctx, max_side);
        self.draw_side_panel(parent_ui);
        self.draw_central_panel(parent_ui, &ctx, max_side);
    }
}

impl FractalViewer {
    fn upload_clamped_snapshot(
        texture_slot: &mut Option<egui::TextureHandle>,
        ctx: &egui::Context,
        snapshot: &StateSnapshot,
        max_texture_side: usize,
    ) -> Option<FractalPassMeta> {
        let image = snapshot_to_clamped_color_image(snapshot, max_texture_side)?;
        let opts = egui::TextureOptions::LINEAR;
        if let Some(tex) = texture_slot {
            tex.set(image, opts);
        } else {
            *texture_slot = Some(ctx.load_texture("fractal_texture", image, opts));
        }
        snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<FractalPassMeta>())
            .copied()
    }

    fn sync_texture_from_worker(&mut self, ctx: &egui::Context, max_texture_side: usize) {
        let uploaded_meta = if let Some(snapshot) = self.worker.update() {
            Self::upload_clamped_snapshot(&mut self.texture, ctx, snapshot, max_texture_side)
        } else if self.texture.is_none() {
            self.worker.latest_snapshot().and_then(|snapshot| {
                Self::upload_clamped_snapshot(&mut self.texture, ctx, snapshot, max_texture_side)
            })
        } else {
            None
        };
        if let Some(meta) = uploaded_meta {
            self.current_stride = meta.stride;
            if meta.is_final {
                self.completed_generation = meta.generation;
            }
        }
    }

    pub(super) fn dispatch_render_request(
        &mut self,
        render_w: usize,
        render_h: usize,
        ctx: &egui::Context,
    ) {
        if !self.has_finite_parameters() {
            return;
        }
        if self.dirty || self.requested_size != (render_w, render_h) {
            self.dirty = false;
            self.requested_size = (render_w, render_h);
            self.request_generation = self.request_generation.wrapping_add(1).max(1);
            let req = FractalRenderRequest {
                generation: self.request_generation,
                width: render_w,
                height: render_h,
                center: self.center,
                zoom: self.zoom,
                max_iter: self.max_iter,
                mode: self.mode,
                julia_c: self.julia_c,
                high_contrast: self.high_contrast,
            };
            *self
                .shared_request
                .write()
                .unwrap_or_else(|e| e.into_inner()) = req;
            self.latest_generation
                .store(self.request_generation, Ordering::Release);

            match self.worker.state() {
                SimWorkerState::Ready => {
                    let _ = self.worker.send_command(SimCommand::Start);
                }
                SimWorkerState::Running => {
                    let _ = self
                        .worker
                        .send_command(SimCommand::Custom("cancel_pass".to_string()));
                }
                SimWorkerState::Faulted | SimWorkerState::Terminated => {
                    let shared = Arc::clone(&self.shared_request);
                    let latest = Arc::clone(&self.latest_generation);
                    self.worker = BackgroundWorker::new(FractalRunner::new(shared, latest));
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

// [cite:graph_parameters_rust]
// [cite:chaos_theory]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "FractalViewer",
        display_name: "Fractal Viewer",
        domain: "chaos",
        tags: &["mandelbrot", "julia", "fractal", "orbit"],
        build: || Box::new(FractalViewer::default()),
    }
}

#[cfg(test)]
#[path = "fractals/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "fractals/coupling_tests.rs"]
mod coupling_tests;
