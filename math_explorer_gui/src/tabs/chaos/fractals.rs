#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::atlas::ParameterAtlasController;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState, StateSnapshot};
use crate::framework::shader_foundry::{emit_fractal_shader, ShaderPreviewState};
use crate::framework::{InteractiveTool, MathExpressionInputBar, ShaderExportFormat};
use eframe::egui;
use math_commons::math_kernel::colormap::ShaderPalette;
use num_complex::Complex;
use scientific_metadata::theory::TheoryDescribable;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

#[path = "fractals/atlas.rs"]
pub mod atlas;
#[cfg(test)]
pub use atlas::FractalCartographer;
use atlas::FRACTAL_ATLAS_AXIS_PAIRS;

#[path = "fractals/formula.rs"]
pub mod formula;
pub use formula::{default_fractal_input_bar, CompiledFractalJob, DEFAULT_FRACTAL_EXPR};

#[path = "fractals/runner.rs"]
mod runner;
pub use runner::*;

#[path = "fractals/state.rs"]
mod state;

#[path = "fractals/overlay.rs"]
mod overlay;
pub use overlay::*;

#[path = "fractals/shader.rs"]
mod shader;

#[path = "fractals/ui.rs"]
mod ui;

#[path = "fractals/presets.rs"]
mod presets;
#[cfg(test)]
use presets::{STARTER_CUSP_CONNECTED_JSON, STARTER_CUSP_DUST_JSON, STARTER_PRESET_JSON};

/// Default complex-plane center for `FractalMode::Mandelbrot`.
pub const DEFAULT_MANDELBROT_CENTER: Complex<f64> = Complex::new(-0.75, 0.0);
/// Default complex-plane center for `FractalMode::Julia` and custom modes.
pub const DEFAULT_JULIA_CENTER: Complex<f64> = Complex::new(0.0, 0.0);

/// Interactive Mandelbrot, Julia, and custom complex escape fractal viewer backed by progressive background passes.
pub struct FractalViewer {
    pub(crate) center: Complex<f64>,
    pub(crate) zoom: f64,
    pub(crate) c_plane_view: (Complex<f64>, f64),
    pub(crate) z_plane_view: (Complex<f64>, f64),
    pub(crate) max_iter: u32,
    pub(crate) mode: FractalMode,
    pub(crate) julia_c: Complex<f64>,
    /// Whether `julia_c` is pinned against Mandelbrot pointer hover updates.
    pub c_pinned: bool,
    /// Whether clicking or pressing `Enter` on the Mandelbrot plane teleports directly to `J_c`.
    pub linked_explorer: bool,
    pub(crate) math_input: MathExpressionInputBar,
    pub(crate) quadratic_job: CompiledFractalJob,
    pub(crate) atlas: ParameterAtlasController,
    pub(crate) high_contrast: bool,
    pub(crate) smooth_coloring: bool,
    pub(crate) palette: ShaderPalette,
    pub(crate) shader_preview: ShaderPreviewState,
    pub(crate) viewport_size: (usize, usize),
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) julia_inset_texture: Option<egui::TextureHandle>,
    pub(crate) julia_inset_image: Option<egui::ColorImage>,
    pub(crate) julia_inset_key: Option<JuliaInsetKey>,
    pub(crate) focused_canvas: Option<egui::Id>,
    pub(crate) dirty: bool,
    pub(crate) requested_size: (usize, usize),
    pub(crate) request_generation: u64,
    pub(crate) completed_generation: u64,
    pub(crate) current_stride: usize,
    pub(crate) shared_request: Arc<RwLock<FractalRenderRequest>>,
    pub(crate) shared_formula: Arc<RwLock<Option<Arc<CompiledFractalJob>>>>,
    pub(crate) shared_style: Arc<RwLock<FractalColorStyle>>,
    pub(crate) latest_generation: Arc<AtomicU64>,
    pub(crate) worker: BackgroundWorker<FractalRunner>,
}

type SharedFormulaSlot = Arc<RwLock<Option<Arc<CompiledFractalJob>>>>;

struct InitialWorkerBundle {
    shared_request: Arc<RwLock<FractalRenderRequest>>,
    shared_formula: SharedFormulaSlot,
    shared_style: Arc<RwLock<FractalColorStyle>>,
    latest_generation: Arc<AtomicU64>,
    worker: BackgroundWorker<FractalRunner>,
}

fn build_initial_worker(
    initial_req: FractalRenderRequest,
    math_input: &MathExpressionInputBar,
    initial_style: FractalColorStyle,
) -> InitialWorkerBundle {
    let initial_formula = Some(Arc::new(CompiledFractalJob::from_bar(math_input)));
    let shared_request = Arc::new(RwLock::new(initial_req));
    let shared_formula = Arc::new(RwLock::new(initial_formula));
    let shared_style = Arc::new(RwLock::new(initial_style));
    let latest_generation = Arc::new(AtomicU64::new(0));
    let worker = BackgroundWorker::new(FractalRunner::with_formula_and_style(
        Arc::clone(&shared_request),
        Arc::clone(&latest_generation),
        Arc::clone(&shared_formula),
        Arc::clone(&shared_style),
    ));
    InitialWorkerBundle {
        shared_request,
        shared_formula,
        shared_style,
        latest_generation,
        worker,
    }
}

impl Default for FractalViewer {
    fn default() -> Self {
        let (center, zoom, max_iter) = (DEFAULT_MANDELBROT_CENTER, 1.0, 100);
        let (mode, julia_c) = (FractalMode::Mandelbrot, Complex::new(-0.4, 0.6));
        let math_input = default_fractal_input_bar();
        let initial_style = FractalColorStyle::default();
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
        let bundle = build_initial_worker(initial_req, &math_input, initial_style);
        let inset_iter = max_iter.clamp(10, JULIA_INSET_MAX_ITER);
        let inset_img = render_julia_inset_image(julia_c, inset_iter, false);
        let inset_key = (
            julia_c.re.to_bits(),
            julia_c.im.to_bits(),
            inset_iter,
            false,
            0,
            initial_style,
        );
        Self {
            center,
            zoom,
            c_plane_view: (DEFAULT_MANDELBROT_CENTER, zoom),
            z_plane_view: (DEFAULT_JULIA_CENTER, 1.0),
            max_iter,
            mode,
            julia_c,
            c_pinned: false,
            linked_explorer: false,
            math_input,
            quadratic_job: CompiledFractalJob::quadratic(),
            atlas: ParameterAtlasController::new(FRACTAL_ATLAS_AXIS_PAIRS.to_vec()),
            high_contrast: false,
            smooth_coloring: initial_style.smooth_coloring,
            palette: initial_style.palette,
            shader_preview: ShaderPreviewState::default(),
            viewport_size: DEFAULT_EXPORT_VIEWPORT_SIZE,
            texture: None,
            julia_inset_texture: None,
            julia_inset_image: Some(inset_img),
            julia_inset_key: Some(inset_key),
            focused_canvas: None,
            dirty: true,
            requested_size: INITIAL_PREVIEW_SIZE,
            request_generation: 0,
            completed_generation: 0,
            current_stride: 1,
            shared_request: bundle.shared_request,
            shared_formula: bundle.shared_formula,
            shared_style: bundle.shared_style,
            latest_generation: bundle.latest_generation,
            worker: bundle.worker,
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

    fn set_suspended(&mut self, suspended: bool) {
        self.worker.set_suspended(suspended);
        self.atlas.set_suspended(suspended);
    }

    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::raster_with_shader()
    }

    fn export_glsl_shader(
        &self,
        dialect: crate::framework::ShaderDialect,
    ) -> Result<String, String> {
        self.build_glsl_shader(dialect)
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
        let job = self
            .mode
            .is_custom()
            .then(|| CompiledFractalJob::from_bar(&self.math_input));
        Some(render_fractal_color_image_styled(
            &req,
            job.as_ref(),
            self.color_style(),
        ))
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
        self.sync_active_plane_view();
        self.mark_dirty();
    }

    fn has_finite_parameters(&self) -> bool {
        self.build_state(false).validate_bounds().is_ok()
            && self
                .math_input
                .param_slice()
                .iter()
                .all(|v| v.is_finite() && (-1.0e4..=1.0e4).contains(v))
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.build_state(true)).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.apply_loaded_state(state)
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
    #[cfg(test)]
    pub(crate) fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    pub(super) const fn color_style(&self) -> FractalColorStyle {
        FractalColorStyle {
            smooth_coloring: self.smooth_coloring,
            palette: self.palette,
        }
    }

    pub(super) fn mark_dirty(&mut self) {
        self.dirty = true;
        self.shader_preview.invalidate();
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
            let formula_job = self
                .mode
                .is_custom()
                .then(|| Arc::new(CompiledFractalJob::from_bar(&self.math_input)));
            *self
                .shared_formula
                .write()
                .unwrap_or_else(|e| e.into_inner()) = formula_job;
            *self.shared_style.write().unwrap_or_else(|e| e.into_inner()) = self.color_style();
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
                self.worker
                    .replace_runner(FractalRunner::with_formula_and_style(
                        Arc::clone(&self.shared_request),
                        Arc::clone(&self.latest_generation),
                        Arc::clone(&self.shared_formula),
                        Arc::clone(&self.shared_style),
                    ));
                let _ = self.worker.send_command(SimCommand::Start);
            }
            SimWorkerState::Uninitialized | SimWorkerState::Stepping => {}
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
        tags: &[
            "mandelbrot",
            "julia",
            "escape time",
            "complex dynamics",
            "self-similarity",
            "critical orbit",
            "fractal",
        ],
        build: || Box::new(FractalViewer::default()),
    }
}

#[cfg(test)]
#[path = "fractals/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "fractals/coupling_tests.rs"]
mod coupling_tests;

#[cfg(test)]
#[path = "fractals/custom_tests.rs"]
mod custom_tests;

#[cfg(test)]
#[path = "fractals/shader_tests.rs"]
mod shader_tests;
