//! Interactive tool controller for 2D Schnakenberg Turing and Gray-Scott morphogenesis.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::AtlasCache;
use super::cartographer::MORPHO_ATLAS_AXIS_PAIRS;
use super::presets::MorphogenesisTheory;
use super::runner::{MorphogenesisRunner, MorphogenesisSharedParams};
use super::state::{
    initialize_for_kinetics, CanvasInteractionMode, MorphogenesisKinetics, MorphogenesisToolState,
};
use crate::async_sim::atlas::ParameterAtlasController;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui::{self, ColorImage};
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{
    DiffusionCoeff, GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
use math_explorer::math_kernel::types::{Dimension, StepSize};
use scientific_metadata::theory::TheoryDescribable;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Interactive tool for 2D Schnakenberg Turing and Gray-Scott reaction-diffusion morphogenesis.
pub struct MorphogenesisTool {
    /// Asynchronous simulation worker controller.
    pub worker: BackgroundWorker<MorphogenesisRunner>,
    /// Progressive 2D parameter regime atlas controller.
    pub atlas: ParameterAtlasController,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) pixels: Vec<egui::Color32>,
    pub(crate) dirty: bool,
    pub(crate) stepped_pixels: Option<(usize, usize, Vec<egui::Color32>)>,
    pub(crate) step_runner: Option<MorphogenesisRunner>,
    pub(crate) pending_initial_start: bool,
    pub(crate) last_high_contrast: Option<bool>,
    pub(crate) atlas_cache: AtlasCache,
    pub(crate) canvas_mode: CanvasInteractionMode,
    pub(crate) brush_radius: i32,
    pub(crate) pending_brush_frames: u8,
    pub(crate) theory: MorphogenesisTheory,
}

impl Default for MorphogenesisTool {
    fn default() -> Self {
        let width = 100;
        let height = 100;
        let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams::default()));
        let runner = Self::create_runner(&params, width, height);
        let worker = BackgroundWorker::new(runner);
        let theory = MorphogenesisTheory::new(Arc::clone(&params));

        Self {
            worker,
            atlas: ParameterAtlasController::new(MORPHO_ATLAS_AXIS_PAIRS.to_vec()),
            params,
            width,
            height,
            texture: None,
            pixels: Vec::new(),
            dirty: true,
            stepped_pixels: None,
            step_runner: None,
            pending_initial_start: true,
            last_high_contrast: None,
            atlas_cache: AtlasCache::default(),
            canvas_mode: CanvasInteractionMode::Brush,
            brush_radius: 4,
            pending_brush_frames: 0,
            theory,
        }
    }
}

impl MorphogenesisTool {
    pub(crate) fn create_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisRunner {
        let p = params.read().unwrap_or_else(|e| e.into_inner()).clone();
        let kinetics = SchnakenbergKinetics { a: p.a, b: p.b };
        let gray_scott = GrayScottKinetics::new(p.f, p.k);
        let diffusion = FiniteDifference2D::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        );
        let mut system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(p.d_u),
            DiffusionCoeff(p.d_v),
            kinetics,
            diffusion,
        );
        initialize_for_kinetics(&mut system, p.kinetics, width, height);
        MorphogenesisRunner {
            system,
            kinetics: p.kinetics,
            gray_scott,
            params: Arc::clone(params),
            width,
            height,
        }
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn restart_worker(&mut self) {
        let runner = Self::create_runner(&self.params, self.width, self.height);
        self.worker = BackgroundWorker::new(runner);
        self.stepped_pixels = None;
        self.step_runner = None;
        self.pending_initial_start = false;
        self.dirty = true;
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn retry(&mut self) {
        self.restart_worker();
    }

    /// Resets the worker in place or respawns a disconnected worker in `Ready` state.
    pub fn reset(&mut self) {
        let params = Arc::clone(&self.params);
        let width = self.width;
        let height = self.height;
        self.worker
            .reset_or_respawn(|| Self::create_runner(&params, width, height));
        self.stepped_pixels = None;
        self.step_runner = None;
        self.pending_initial_start = false;
        self.dirty = true;
    }

    pub(crate) fn reset_system(&mut self) {
        let was_pending = self.pending_initial_start;
        self.reset();
        if was_pending {
            self.pending_initial_start = true;
        } else {
            let _ = self.worker.send_command(SimCommand::Start);
        }
    }

    pub(crate) fn switch_kinetics(&mut self, next: MorphogenesisKinetics) {
        let was_running = self.worker.is_running();
        let was_pending = self.pending_initial_start;
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.kinetics = next;
            match next {
                MorphogenesisKinetics::Schnakenberg => {
                    p.d_u = 1.0;
                    p.d_v = 100.0;
                    p.dt = 0.05;
                    p.auto_dt = true;
                }
                MorphogenesisKinetics::GrayScott => {
                    p.d_u = 0.2;
                    p.d_v = 0.1;
                    p.dt = 1.0;
                    p.auto_dt = true;
                }
            }
        }
        if self.atlas.is_visible() {
            match (next, self.atlas.axis_keys()) {
                (MorphogenesisKinetics::GrayScott, ("a", "b")) => {
                    let _ = self.atlas.set_axis_keys("f", "k");
                }
                (MorphogenesisKinetics::Schnakenberg, ("f", "k")) => {
                    let _ = self.atlas.set_axis_keys("a", "b");
                }
                _ => {}
            }
        }
        scientific_metadata::citation_registry::CitationRegistry::register(
            self.name().to_string(),
            self.theory.theory_citation(),
        );
        self.reset();
        if was_running {
            let _ = self.worker.send_command(SimCommand::Start);
        } else if was_pending {
            self.pending_initial_start = true;
        }
    }

    /// Applies `(x, y)` coordinates clicked on the 2D stability minimap (`(a, b)` or `(F, k)`).
    pub fn apply_atlas_coordinates(&mut self, kinetics: MorphogenesisKinetics, x: f64, y: f64) {
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            match kinetics {
                MorphogenesisKinetics::Schnakenberg => {
                    p.a = x.clamp(0.0, 1.0);
                    p.b = y.clamp(0.0, 2.0);
                }
                MorphogenesisKinetics::GrayScott => {
                    p.f = x.clamp(0.0, 0.2);
                    p.k = y.clamp(0.0, 0.2);
                }
            }
        }
        let _ = self
            .worker
            .send_command(SimCommand::Custom("ensure_nucleated".to_string()));
        self.pending_brush_frames = 2;
        self.dirty = true;
    }
}

impl InteractiveTool for MorphogenesisTool {
    fn name(&self) -> &'static str {
        "Turing Morphogenesis"
    }

    fn domain(&self) -> &'static str {
        "biology"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::raster_only()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        if let Some((w, h, ref px)) = self.stepped_pixels {
            if w > 0 && h > 0 && px.len() == w * h {
                return Some(ColorImage {
                    size: [w, h],
                    pixels: px.clone(),
                    ..Default::default()
                });
            }
        }
        if !self.dirty && self.pixels.len() == self.width * self.height && !self.pixels.is_empty() {
            return Some(ColorImage {
                size: [self.width, self.height],
                pixels: self.pixels.clone(),
                ..Default::default()
            });
        }
        let snapshot = self.worker.latest_snapshot()?;
        let px = snapshot.pixels.read().unwrap_or_else(|e| e.into_inner());
        if snapshot.width == 0
            || snapshot.height == 0
            || px.len() != snapshot.width * snapshot.height
        {
            return None;
        }
        Some(ColorImage {
            size: [snapshot.width, snapshot.height],
            pixels: px.clone(),
            ..Default::default()
        })
    }

    fn step_simulation(&mut self) {
        let runner = self
            .step_runner
            .get_or_insert_with(|| Self::create_runner(&self.params, self.width, self.height));
        for _ in 0..5 {
            runner.step();
        }
        let snap = runner.get_snapshot();
        let px = snap.pixels.read().unwrap_or_else(|e| e.into_inner());
        self.stepped_pixels = Some((snap.width, snap.height, px.clone()));
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let high_contrast = crate::app::ThemeMode::from_visuals(ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        if self.last_high_contrast != Some(high_contrast) {
            self.last_high_contrast = Some(high_contrast);
            self.params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .high_contrast = high_contrast;
            self.dirty = true;
        }
        if std::mem::take(&mut self.pending_initial_start) && self.worker.state().is_ready() {
            let _ = self.worker.send_command(SimCommand::Start);
        }
        if self.worker.update().is_some() {
            self.dirty = true;
            self.pending_brush_frames = 0;
        }
        self.draw_controls(ui);
        self.draw_canvas(ui);
        if matches!(
            self.worker.state(),
            SimWorkerState::Running | SimWorkerState::Stepping
        ) {
            ctx.request_repaint();
        } else if self.pending_brush_frames > 0 {
            self.pending_brush_frames -= 1;
            ctx.request_repaint();
        }
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            p.kinetics.hash(&mut h);
            p.a.to_bits().hash(&mut h);
            p.b.to_bits().hash(&mut h);
            p.f.to_bits().hash(&mut h);
            p.k.to_bits().hash(&mut h);
            p.d_u.to_bits().hash(&mut h);
            p.d_v.to_bits().hash(&mut h);
            p.dt.to_bits().hash(&mut h);
            p.auto_dt.hash(&mut h);
        }
        self.width.hash(&mut h);
        self.height.hash(&mut h);
        std::mem::discriminant(&self.worker.state()).hash(&mut h);
        self.pending_initial_start.hash(&mut h);
        self.atlas.is_visible().hash(&mut h);
        self.atlas.axis_keys().hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        let p = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let (xk, yk) = self.atlas.axis_keys();
        serde_json::to_string(&MorphogenesisToolState {
            kinetics_mode: p.kinetics,
            a: p.a,
            b: p.b,
            f: p.f,
            k: p.k,
            d_u: p.d_u,
            d_v: p.d_v,
            dt: p.dt,
            auto_dt: p.auto_dt,
            width: self.width,
            height: self.height,
            state: Some(self.worker.state()),
            paused: None,
            pending_initial_start: self.pending_initial_start,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: MorphogenesisToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Morphogenesis state: {e}"))?;
        parsed.validate()?;
        self.atlas
            .set_axis_keys(&parsed.atlas_axes[0], &parsed.atlas_axes[1])?;
        self.atlas.set_visible(parsed.atlas_visible);
        let (target_state, next_pending) = parsed.resolve_target_state(self.pending_initial_start);
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.kinetics = parsed.kinetics_mode;
            p.a = parsed.a;
            p.b = parsed.b;
            p.f = parsed.f;
            p.k = parsed.k;
            p.d_u = parsed.d_u;
            p.d_v = parsed.d_v;
            p.dt = parsed.dt;
            p.auto_dt = parsed.auto_dt;
        }
        scientific_metadata::citation_registry::CitationRegistry::register(
            self.name().to_string(),
            self.theory.theory_citation(),
        );
        if parsed.width != self.width || parsed.height != self.height {
            self.width = parsed.width;
            self.height = parsed.height;
            let runner = Self::create_runner(&self.params, self.width, self.height);
            self.worker = BackgroundWorker::new(runner);
            self.stepped_pixels = None;
            self.step_runner = None;
            self.dirty = true;
        } else {
            self.reset();
        }
        if target_state == SimWorkerState::Running {
            self.pending_initial_start = false;
            if !self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Start);
            }
        } else {
            self.pending_initial_start = next_pending;
            if self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Pause);
            }
        }
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

    fn has_finite_parameters(&self) -> bool {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        p.a.is_finite()
            && p.b.is_finite()
            && p.f.is_finite()
            && p.k.is_finite()
            && p.d_u.is_finite()
            && p.d_v.is_finite()
            && p.dt.is_finite()
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "MorphogenesisTool",
        display_name: "Turing Morphogenesis",
        domain: "biology",
        tags: &[
            "turing",
            "morphogenesis",
            "schnakenberg",
            "gray-scott",
            "reaction-diffusion",
            "spots",
            "stripes",
            "mitosis",
            "coral",
            "solitons",
        ],
        build: || Box::new(MorphogenesisTool::default()),
    }
}
