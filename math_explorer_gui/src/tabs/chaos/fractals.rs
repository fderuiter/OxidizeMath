#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState, StateSnapshot};
use crate::framework::shader_foundry::{
    emit_fractal_shader, ShaderPreviewState, ShaderUniformDefaults,
};
use crate::framework::{InteractiveTool, MathExpressionInputBar, ShaderExportFormat};
use eframe::egui;
use math_commons::ast::{parse, Expr};
use math_commons::math_kernel::colormap::ShaderPalette;
use num_complex::Complex;
use scientific_metadata::theory::TheoryDescribable;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

#[path = "fractals/runner.rs"]
mod runner;
pub use runner::*;

#[path = "fractals/state.rs"]
mod state;
use state::FractalState;

#[path = "fractals/ui.rs"]
mod ui;

/// Interactive Mandelbrot, Julia, and custom complex recurrence fractal viewer.
pub struct FractalViewer {
    pub(crate) center: Complex<f64>,
    pub(crate) zoom: f64,
    pub(crate) max_iter: u32,
    pub(crate) mode: FractalMode,
    pub(crate) julia_c: Complex<f64>,
    pub(crate) high_contrast: bool,
    pub(crate) smooth_coloring: bool,
    pub(crate) palette: ShaderPalette,
    pub(crate) custom_julia_seed: bool,
    pub(crate) math_input: MathExpressionInputBar,
    pub(crate) shader_preview: ShaderPreviewState,
    pub(crate) viewport_size: (usize, usize),
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) dirty: bool,
    pub(crate) requested_size: (usize, usize),
    pub(crate) request_generation: u64,
    pub(crate) completed_generation: u64,
    pub(crate) current_stride: usize,
    pub(crate) shared_request: Arc<RwLock<FractalRenderRequest>>,
    pub(crate) latest_generation: Arc<AtomicU64>,
    pub(crate) worker: BackgroundWorker<FractalRunner>,
}

fn build_fractal_math_input() -> MathExpressionInputBar {
    MathExpressionInputBar::new(
        "z'",
        DEFAULT_CUSTOM_FRACTAL_EXPR,
        vec![
            ("z' = z^3 + c (Multibrot 3)", "z^3 + c"),
            ("z' = z^4 + c (Multibrot 4)", "z^4 + c"),
            ("z' = z^2 + c (Quadratic)", "z^2 + c"),
            ("z' = sin(z) + c", "sin(z) + c"),
            ("z' = cos(z) * c", "cos(z) * c"),
        ],
    )
}

impl Default for FractalViewer {
    fn default() -> Self {
        let center = Complex::new(-0.75, 0.0);
        let zoom = 1.0;
        let max_iter = 100;
        let mode = FractalMode::Mandelbrot;
        let julia_c = Complex::new(-0.4, 0.6);
        let math_input = build_fractal_math_input();
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
            smooth_coloring: true,
            palette: ShaderPalette::CyclicCosine,
            custom_expr: math_input.expr().cloned().map(Arc::new),
            custom_julia_seed: false,
        };
        let shared_request = Arc::new(RwLock::new(initial_req));
        let latest_generation = Arc::new(AtomicU64::new(0));
        let worker = BackgroundWorker::new(FractalRunner::new(
            Arc::clone(&shared_request),
            Arc::clone(&latest_generation),
        ));
        Self {
            center,
            zoom,
            max_iter,
            mode,
            julia_c,
            high_contrast: false,
            smooth_coloring: true,
            palette: ShaderPalette::CyclicCosine,
            custom_julia_seed: false,
            math_input,
            shader_preview: ShaderPreviewState::default(),
            viewport_size: DEFAULT_EXPORT_VIEWPORT_SIZE,
            texture: None,
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
        crate::framework::ExportCapabilities::raster_with_shader()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        let raw = if self.viewport_size.0 > 0 && self.viewport_size.1 > 0 {
            self.viewport_size
        } else {
            DEFAULT_EXPORT_VIEWPORT_SIZE
        };
        let (width, height) = clamp_fractal_dimensions(raw.0, raw.1, MAX_FRACTAL_TEXTURE_SIDE);
        if !self.dirty && self.completed_generation == self.request_generation {
            if let Some(snap) = self.worker.latest_snapshot() {
                if (snap.width, snap.height) == (width, height) {
                    if let Some(img) =
                        snapshot_to_clamped_color_image(snap, MAX_FRACTAL_TEXTURE_SIDE)
                    {
                        return Some(img);
                    }
                }
            }
        }
        let req = self.build_clamped_render_request(width, height);
        Some(render_fractal_color_image(&req))
    }

    fn export_shader_source(&self, format: ShaderExportFormat) -> Result<String, String> {
        let (expr, use_julia_seed) = self.resolve_active_shader_expr()?;
        emit_fractal_shader(
            format,
            &expr,
            use_julia_seed,
            &self.shader_uniform_defaults(),
        )
    }

    fn step_simulation(&mut self) {
        self.zoom = (self.zoom * 1.06).clamp(ZOOM_MIN, ZOOM_MAX);
        self.mark_dirty();
    }

    fn has_finite_parameters(&self) -> bool {
        self.to_serializable_state().validate().is_ok()
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.to_serializable_state()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: FractalState =
            serde_json::from_str(state).map_err(|e| format!("Invalid fractal state: {e}"))?;
        let palette = s.validate()?;
        self.center = Complex::new(s.center_re, s.center_im);
        self.zoom = s.zoom;
        self.max_iter = s.max_iter;
        self.mode = s.mode;
        self.julia_c = Complex::new(s.julia_re, s.julia_im);
        self.smooth_coloring = s.smooth_coloring;
        self.palette = palette;
        self.custom_julia_seed = s.custom_julia_seed;
        self.math_input.set_expression(
            s.custom_expr
                .as_deref()
                .unwrap_or(DEFAULT_CUSTOM_FRACTAL_EXPR),
        );
        self.mark_dirty();
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
            self.mark_dirty();
        }
        let max_side = ctx.input(|i| i.max_texture_side);
        self.sync_texture_from_worker(&ctx, max_side);
        self.draw_side_panel(parent_ui);
        self.draw_central_panel(parent_ui, &ctx, max_side);
    }
}

impl FractalViewer {
    pub(super) fn mark_dirty(&mut self) {
        self.dirty = true;
        self.shader_preview.invalidate();
    }

    pub(super) fn resolve_active_shader_expr(&self) -> Result<(Expr, bool), String> {
        match self.mode {
            FractalMode::Mandelbrot => parse("z^2 + c")
                .map(|e| (e, false))
                .map_err(|e| e.to_string()),
            FractalMode::Julia => parse("z^2 + c")
                .map(|e| (e, true))
                .map_err(|e| e.to_string()),
            FractalMode::Custom => match self.math_input.expr() {
                Some(expr) => Ok((expr.clone(), self.custom_julia_seed)),
                None => Err(self
                    .math_input
                    .error_message()
                    .unwrap_or_else(|| "Invalid custom fractal expression".to_string())),
            },
        }
    }

    pub(super) fn shader_uniform_defaults(&self) -> ShaderUniformDefaults {
        let raw = if self.viewport_size.0 > 0 && self.viewport_size.1 > 0 {
            self.viewport_size
        } else {
            DEFAULT_EXPORT_VIEWPORT_SIZE
        };
        let (w, h) = clamp_fractal_dimensions(raw.0, raw.1, MAX_FRACTAL_TEXTURE_SIDE);
        let safe_zoom = if self.zoom.is_finite() && (ZOOM_MIN..=ZOOM_MAX).contains(&self.zoom) {
            self.zoom
        } else {
            1.0
        };
        let scale_y = 3.0 / safe_zoom;
        let scale_x = scale_y * (w.max(1) as f64 / h.max(1) as f64);
        let cx = self.center.re.clamp(COORD_MIN, COORD_MAX);
        let cy = self.center.im.clamp(COORD_MIN, COORD_MAX);
        ShaderUniformDefaults {
            resolution: [w.max(1) as u32, h.max(1) as u32],
            viewport: [
                cx - scale_x * 0.5,
                cy - scale_y * 0.5,
                cx + scale_x * 0.5,
                cy + scale_y * 0.5,
            ],
            max_iter: self
                .max_iter
                .clamp(10, math_commons::registry::MAX_ITERATIONS as u32),
            julia_c: [
                self.julia_c.re.clamp(COORD_MIN, COORD_MAX),
                self.julia_c.im.clamp(COORD_MIN, COORD_MAX),
            ],
            palette: self.palette,
            high_contrast: self.high_contrast,
            smooth_coloring: self.smooth_coloring,
        }
    }

    fn build_clamped_render_request(&self, width: usize, height: usize) -> FractalRenderRequest {
        let zoom = if self.zoom.is_finite() && (ZOOM_MIN..=ZOOM_MAX).contains(&self.zoom) {
            self.zoom
        } else {
            1.0
        };
        FractalRenderRequest {
            generation: self.request_generation,
            width,
            height,
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
            smooth_coloring: self.smooth_coloring,
            palette: self.palette,
            custom_expr: self.math_input.expr().cloned().map(Arc::new),
            custom_julia_seed: self.custom_julia_seed,
        }
    }

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
            let req = self.build_clamped_render_request(render_w, render_h);
            *self
                .shared_request
                .write()
                .unwrap_or_else(|e| e.into_inner()) = req;
            self.latest_generation
                .store(self.request_generation, Ordering::Release);
            self.notify_worker_after_request_update();
        }
        if self.completed_generation != self.request_generation && self.worker.state().is_ready() {
            let _ = self.worker.send_command(SimCommand::Start);
        }
        if self.completed_generation != self.request_generation || self.worker.is_running() {
            ctx.request_repaint();
        }
    }

    fn notify_worker_after_request_update(&mut self) {
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
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "FractalViewer",
        display_name: "Fractal Viewer",
        domain: "chaos",
        tags: &[],
        build: || Box::new(FractalViewer::default()),
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Mandelbrot Explorer",
    "domain": "chaos",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "center_re": -0.75,
    "center_im": 0.0,
    "zoom": 1.0,
    "max_iter": 150,
    "mode": "Mandelbrot",
    "julia_re": -0.7,
    "julia_im": 0.27015
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "fractal_viewer_starter",
        title: "Mandelbrot Explorer",
        description: "z_{n+1} = z_n^2 + c: Complex quadratic polynomial iteration and boundary fractal escape times.",
        category: "Fractals",
        icon: "❄️",
        domain: "chaos",
        tool_name: Some("FractalViewer"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
#[path = "fractals/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "fractals/shader_tests.rs"]
mod shader_tests;
