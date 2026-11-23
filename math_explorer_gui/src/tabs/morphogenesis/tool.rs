#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::MORPHO_ATLAS_AXIS_PAIRS;
use super::presets::MorphogenesisTheory;
pub use super::presets::MorphogenesisToolState;
use super::runner::{
    MorphogenesisKineticsAdapter, MorphogenesisKineticsRunner, MorphogenesisKineticsState,
    MorphogenesisSharedParams,
};
use super::sculpt::default_morpho_brush_kit;
use crate::async_sim::atlas::ParameterAtlasController;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{
    decode_rle_mask, ExportCapabilities, InteractionContext, InteractiveTool, SpatialBrushKit,
};
use eframe::egui;
use eframe::egui::ColorImage;
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::TuringSystem;
use scientific_metadata::theory::TheoryDescribable;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Interactive tool for 2D Schnakenberg and Gray-Scott reaction-diffusion morphogenesis.
pub struct MorphogenesisTool {
    /// Asynchronous simulation worker controller.
    pub worker: BackgroundWorker<MorphogenesisKineticsRunner>,
    /// Progressive 2D parameter regime atlas controller.
    pub atlas: ParameterAtlasController,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) kinetics_state: Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
    pub(crate) mask_rle: Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
    pub(crate) brush_kit: SpatialBrushKit,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) pixels: Vec<egui::Color32>,
    pub(crate) dirty: bool,
    pub(crate) stepped_pixels: Option<(usize, usize, Vec<egui::Color32>)>,
    pub(crate) step_runner: Option<MorphogenesisKineticsRunner>,
    pub(crate) pending_initial_start: bool,
    pub(crate) last_high_contrast: Option<bool>,
    pub(crate) theory: MorphogenesisTheory,
}

impl Default for MorphogenesisTool {
    fn default() -> Self {
        let width = 100;
        let height = 100;
        let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams {
            a: 0.1,
            b: 0.9,
            d_u: 1.0,
            d_v: 100.0,
            dt: 0.05,
            high_contrast: false,
        }));
        let kinetics_state =
            Arc::new(std::sync::RwLock::new(MorphogenesisKineticsState::default()));

        let mask_rle = Arc::new(std::sync::RwLock::new(Vec::new()));
        let runner =
            Self::create_kinetics_runner(&params, &kinetics_state, &mask_rle, width, height);
        let worker = BackgroundWorker::new(runner);

        Self {
            worker,
            atlas: ParameterAtlasController::new(MORPHO_ATLAS_AXIS_PAIRS.to_vec()),
            params,
            kinetics_state,
            mask_rle,
            brush_kit: default_morpho_brush_kit(),
            width,
            height,
            texture: None,
            pixels: Vec::new(),
            dirty: true,
            stepped_pixels: None,
            step_runner: None,
            pending_initial_start: true,
            last_high_contrast: None,
            theory: MorphogenesisTheory::default(),
        }
    }
}

impl MorphogenesisTool {
    #[cfg(test)]
    pub(crate) fn create_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisKineticsRunner {
        let ks = Arc::new(std::sync::RwLock::new(MorphogenesisKineticsState::default()));
        let mask = Arc::new(std::sync::RwLock::new(Vec::new()));
        Self::create_kinetics_runner(params, &ks, &mask, width, height)
    }

    pub(crate) fn create_kinetics_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        kinetics_state: &Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisKineticsRunner {
        use math_explorer::biology::morphogenesis::DiffusionCoeff;
        use math_explorer::math_kernel::types::{Dimension, StepSize};
        let (a, b, d_u, d_v) = {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v)
        };
        let (mode, f, k) = {
            let ks = kinetics_state.read().unwrap_or_else(|e| e.into_inner());
            (ks.mode, ks.f, ks.k)
        };
        let kinetics = MorphogenesisKineticsAdapter { mode, a, b, f, k };
        let diffusion = FiniteDifference2D::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        )
        .with_mask();
        let mut system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(d_u),
            DiffusionCoeff(d_v),
            kinetics,
            diffusion,
        );
        Self::initialize_adapter_system(&mut system, width, height);
        let rle = mask_rle.read().unwrap_or_else(|e| e.into_inner()).clone();
        if !rle.is_empty() {
            if let Ok(bytes) = decode_rle_mask(&rle, width * height) {
                let _ = system.diffusion.mask.set_roles_slice(&bytes);
            }
        }
        let mut runner = MorphogenesisKineticsRunner {
            system,
            params: Arc::clone(params),
            kinetics_state: Arc::clone(kinetics_state),
            width,
            height,
            step_count: 0,
            rng: oxidize_core::rng::OxidizeRng::new(super::runner::DEFAULT_MORPHOGENESIS_SEED),
            noise_amplitude: 0.0,
        };
        runner.enforce_mask_concentrations();
        runner
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn restart_worker(&mut self) {
        let runner = Self::create_kinetics_runner(
            &self.params,
            &self.kinetics_state,
            &self.mask_rle,
            self.width,
            self.height,
        );
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
        let ks = Arc::clone(&self.kinetics_state);
        let mask = Arc::clone(&self.mask_rle);
        let width = self.width;
        let height = self.height;
        self.worker
            .reset_or_respawn(|| Self::create_kinetics_runner(&params, &ks, &mask, width, height));
        self.stepped_pixels = None;
        self.step_runner = None;
        self.pending_initial_start = false;
        self.dirty = true;
    }

    pub(crate) fn reset_system(&mut self) {
        self.reset();
        let _ = self.worker.send_command(SimCommand::Start);
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
        let runner = self.step_runner.get_or_insert_with(|| {
            Self::create_kinetics_runner(
                &self.params,
                &self.kinetics_state,
                &self.mask_rle,
                self.width,
                self.height,
            )
        });
        for _ in 0..5 {
            runner.step();
        }
        let snap = runner.get_snapshot();
        let px = snap.pixels.read().unwrap_or_else(|e| e.into_inner());
        self.stepped_pixels = Some((snap.width, snap.height, px.clone()));
    }

    fn on_brush(&mut self, ctx: &InteractionContext) {
        if let Some(stroke) = ctx.brush_stroke {
            self.dispatch_brush_stroke(stroke, Some(&ctx.response.ctx));
        }
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.brush_kit.focused_plot_id()
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
        }
        self.draw_controls(ui);
        self.draw_canvas(ui);
        if matches!(
            self.worker.state(),
            SimWorkerState::Running | SimWorkerState::Stepping
        ) {
            ctx.request_repaint();
        }
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        {
            let ks = self
                .kinetics_state
                .read()
                .unwrap_or_else(|e| e.into_inner());
            ks.mode.hash(&mut h);
            ks.f.to_bits().hash(&mut h);
            ks.k.to_bits().hash(&mut h);
        }
        {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            p.a.to_bits().hash(&mut h);
            p.b.to_bits().hash(&mut h);
            p.d_u.to_bits().hash(&mut h);
            p.d_v.to_bits().hash(&mut h);
            p.dt.to_bits().hash(&mut h);
        }
        self.width.hash(&mut h);
        self.height.hash(&mut h);
        self.mask_rle
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .hash(&mut h);
        std::mem::discriminant(&self.worker.state()).hash(&mut h);
        self.pending_initial_start.hash(&mut h);
        self.atlas.is_visible().hash(&mut h);
        self.atlas.axis_keys().hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        let (kinetics_mode, f, k) = {
            let ks = self
                .kinetics_state
                .read()
                .unwrap_or_else(|e| e.into_inner());
            (ks.mode, ks.f, ks.k)
        };
        let (a, b, d_u, d_v, dt) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v, p.dt)
        };
        let mask_rle = self
            .mask_rle
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let (xk, yk) = self.atlas.axis_keys();
        serde_json::to_string(&MorphogenesisToolState {
            a,
            b,
            d_u,
            d_v,
            dt,
            width: self.width,
            height: self.height,
            kinetics_mode,
            f,
            k,
            state: Some(self.worker.state()),
            paused: None,
            pending_initial_start: self.pending_initial_start,
            mask_rle,
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
        let target_state = parsed.resolve_target_state();
        self.theory.mode = parsed.kinetics_mode;
        {
            let mut ks = self
                .kinetics_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            ks.mode = parsed.kinetics_mode;
            ks.f = parsed.f;
            ks.k = parsed.k;
        }
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.a = parsed.a;
            p.b = parsed.b;
            p.d_u = parsed.d_u;
            p.d_v = parsed.d_v;
            p.dt = parsed.dt;
        }
        *self.mask_rle.write().unwrap_or_else(|e| e.into_inner()) = parsed.mask_rle;
        self.apply_dimensions_and_target_state(
            parsed.width,
            parsed.height,
            target_state,
            parsed.pending_initial_start,
        );
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
        let ks = self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        ks.f.is_finite()
            && ks.k.is_finite()
            && p.a.is_finite()
            && p.b.is_finite()
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
            "mitosis",
            "coral",
            "reaction-diffusion",
            "spots",
            "stripes",
        ],
        build: || Box::new(MorphogenesisTool::default()),
    }
}
