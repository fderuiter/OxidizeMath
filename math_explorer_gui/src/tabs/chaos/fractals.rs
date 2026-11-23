#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState, StateSnapshot};
use crate::framework::{InteractiveTool, MathExpressionInputBar};
use eframe::egui;
use num_complex::Complex;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

#[path = "fractals/formula.rs"]
pub mod formula;
pub use formula::{default_fractal_input_bar, CompiledFractalJob, DEFAULT_FRACTAL_EXPR};

#[path = "fractals/runner.rs"]
mod runner;
pub use runner::*;

#[path = "fractals/ui.rs"]
mod ui;

/// Interactive Mandelbrot, Julia, and custom complex escape fractal viewer backed by progressive background passes.
pub struct FractalViewer {
    pub(crate) center: Complex<f64>,
    pub(crate) zoom: f64,
    pub(crate) max_iter: u32,
    pub(crate) mode: FractalMode,
    pub(crate) julia_c: Complex<f64>,
    pub(crate) math_input: MathExpressionInputBar,
    pub(crate) high_contrast: bool,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) dirty: bool,
    pub(crate) requested_size: (usize, usize),
    pub(crate) request_generation: u64,
    pub(crate) completed_generation: u64,
    pub(crate) current_stride: usize,
    pub(crate) shared_request: Arc<RwLock<FractalRenderRequest>>,
    pub(crate) shared_formula: Arc<RwLock<Option<Arc<CompiledFractalJob>>>>,
    pub(crate) latest_generation: Arc<AtomicU64>,
    pub(crate) worker: BackgroundWorker<FractalRunner>,
}

impl Default for FractalViewer {
    fn default() -> Self {
        let center = Complex::new(-0.75, 0.0);
        let zoom = 1.0;
        let max_iter = 100;
        let mode = FractalMode::Mandelbrot;
        let julia_c = Complex::new(-0.4, 0.6);
        let math_input = default_fractal_input_bar();
        let initial_formula = Some(Arc::new(CompiledFractalJob::from_bar(&math_input)));
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
        let shared_formula = Arc::new(RwLock::new(initial_formula));
        let latest_generation = Arc::new(AtomicU64::new(0));
        let worker = BackgroundWorker::new(FractalRunner::with_formula(
            Arc::clone(&shared_request),
            Arc::clone(&latest_generation),
            Arc::clone(&shared_formula),
        ));
        Self {
            center,
            zoom,
            max_iter,
            mode,
            julia_c,
            math_input,
            high_contrast: false,
            texture: None,
            dirty: true,
            requested_size: INITIAL_PREVIEW_SIZE,
            request_generation: 0,
            completed_generation: 0,
            current_stride: 1,
            shared_request,
            shared_formula,
            latest_generation,
            worker,
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
            FractalMode::CustomMandelbrot => {
                format!("Custom escape fractal z_{{n+1}} = {}", self.math_input.raw_text)
            }
            FractalMode::CustomJulia => format!(
                "Custom Julia fractal z_{{n+1}} = {} with c=({:.2}, {:.2})",
                self.math_input.raw_text, self.julia_c.re, self.julia_c.im
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
        let job = self
            .mode
            .is_custom()
            .then(|| CompiledFractalJob::from_bar(&self.math_input));
        Some(render_fractal_color_image_with_formula(&req, job.as_ref()))
    }

    fn step_simulation(&mut self) {
        self.zoom = (self.zoom * 1.06).clamp(ZOOM_MIN, ZOOM_MAX);
        self.dirty = true;
    }

    fn has_finite_parameters(&self) -> bool {
        FractalState {
            center_re: self.center.re,
            center_im: self.center.im,
            zoom: self.zoom,
            max_iter: self.max_iter,
            mode: self.mode,
            julia_re: self.julia_c.re,
            julia_im: self.julia_c.im,
            custom_expr: None,
            custom_params: self.math_input.params_map(),
        }
        .validate()
        .is_ok()
    }

    fn save_state(&self) -> Option<String> {
        let is_custom = self.mode.is_custom();
        let custom_expr = (is_custom || self.math_input.raw_text != DEFAULT_FRACTAL_EXPR)
            .then(|| self.math_input.raw_text.clone());
        let custom_params = if is_custom {
            self.math_input.params_map()
        } else {
            BTreeMap::new()
        };
        serde_json::to_string(&FractalState {
            center_re: self.center.re,
            center_im: self.center.im,
            zoom: self.zoom,
            max_iter: self.max_iter,
            mode: self.mode,
            julia_re: self.julia_c.re,
            julia_im: self.julia_c.im,
            custom_expr,
            custom_params,
        })
        .ok()
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
        self.math_input
            .set_expression(s.custom_expr.as_deref().unwrap_or(DEFAULT_FRACTAL_EXPR));
        self.math_input.set_params_map(&s.custom_params);
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
            let formula_job = self
                .mode
                .is_custom()
                .then(|| Arc::new(CompiledFractalJob::from_bar(&self.math_input)));
            *self
                .shared_formula
                .write()
                .unwrap_or_else(|e| e.into_inner()) = formula_job;
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
                    let formula = Arc::clone(&self.shared_formula);
                    self.worker = BackgroundWorker::new(FractalRunner::with_formula(
                        shared, latest, formula,
                    ));
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
#[path = "fractals/custom_tests.rs"]
mod custom_tests;
