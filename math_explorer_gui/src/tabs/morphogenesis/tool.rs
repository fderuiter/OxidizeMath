//! Interactive tool lifecycle and framework integration for 2D morphogenesis.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::MORPHO_ATLAS_AXIS_PAIRS;
use super::diffusion::CoupledMorphoDiffusion;
use super::presets::MorphogenesisTheory;
use super::runner::{
    MorphogenesisKineticsAdapter, MorphogenesisKineticsRunner, MorphogenesisKineticsState,
    DEFAULT_MORPHOGENESIS_SEED,
};
use super::sculpt::default_morpho_brush_kit;
use super::types::{
    MorphogenesisFlowMode, MorphogenesisFlowState, MorphogenesisOverlayMode,
    MorphogenesisSharedParams,
};
use crate::app::ThemeMode;
use crate::async_sim::atlas::ParameterAtlasController;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{
    decode_rle_mask, ExportCapabilities, InteractionContext, InteractiveTool, SpatialBrushKit,
};
use eframe::egui;
use eframe::egui::ColorImage;
use math_explorer::biology::morphogenesis::{DiffusionCoeff, TuringSystem};
use math_explorer::math_kernel::types::{Dimension, StepSize};
#[cfg(feature = "physics")]
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;
#[cfg(feature = "physics")]
use math_explorer::physics::fluid_dynamics::potential_flow::PotentialFlowField;
use scientific_metadata::theory::TheoryDescribable;
use std::sync::Arc;

/// Interactive tool for 2D upwind advected Turing & Gray-Scott morphogenesis with microfluidic sculpting.
pub struct MorphogenesisTool {
    /// Asynchronous simulation worker controller.
    pub worker: BackgroundWorker<MorphogenesisKineticsRunner>,
    /// Progressive 2D parameter regime atlas controller.
    pub atlas: ParameterAtlasController,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) kinetics_state: Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
    pub(crate) flow_state: Arc<std::sync::RwLock<MorphogenesisFlowState>>,
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
    pub(crate) last_theme_mode: Option<ThemeMode>,
    pub(crate) last_overlay_mode: Option<MorphogenesisOverlayMode>,
    pub(crate) live_peclet: f64,
    pub(crate) live_damkohler: f64,
    pub(crate) theory: MorphogenesisTheory,
}

impl Default for MorphogenesisTool {
    fn default() -> Self {
        let width = 100;
        let height = 100;
        let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams::default()));
        let kinetics_state =
            Arc::new(std::sync::RwLock::new(MorphogenesisKineticsState::default()));
        let flow_state = Arc::new(std::sync::RwLock::new(MorphogenesisFlowState::default()));
        let mask_rle = Arc::new(std::sync::RwLock::new(Vec::new()));
        let runner = Self::create_coupled_runner(
            &params,
            &kinetics_state,
            &flow_state,
            &mask_rle,
            width,
            height,
        );
        let peclet = runner.peclet;
        let damkohler = runner.damkohler;
        let worker = BackgroundWorker::new(runner);

        Self {
            worker,
            atlas: ParameterAtlasController::new(MORPHO_ATLAS_AXIS_PAIRS.to_vec()),
            params,
            kinetics_state,
            flow_state,
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
            last_theme_mode: None,
            last_overlay_mode: None,
            live_peclet: peclet,
            live_damkohler: damkohler,
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

    #[cfg(test)]
    pub(crate) fn create_kinetics_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        kinetics_state: &Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisKineticsRunner {
        let flow_state = Arc::new(std::sync::RwLock::new(MorphogenesisFlowState::default()));
        Self::create_coupled_runner(params, kinetics_state, &flow_state, mask_rle, width, height)
    }

    fn build_coupled_system(
        p: MorphogenesisSharedParams,
        ks: MorphogenesisKineticsState,
        flow: MorphogenesisFlowState,
        rle: &[[u32; 2]],
        width: usize,
        height: usize,
    ) -> TuringSystem<2, MorphogenesisKineticsAdapter, CoupledMorphoDiffusion> {
        let kinetics = MorphogenesisKineticsAdapter {
            mode: ks.mode,
            a: p.a,
            b: p.b,
            f: ks.f,
            k: ks.k,
        };
        let mut diffusion = CoupledMorphoDiffusion::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        );
        diffusion.flow_active = flow.flow_mode != MorphogenesisFlowMode::None;
        let mut system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(p.d_u),
            DiffusionCoeff(p.d_v),
            kinetics,
            diffusion,
        );
        Self::initialize_adapter_system(&mut system, width, height);
        if !rle.is_empty() {
            if let Ok(bytes) = decode_rle_mask(rle, width * height) {
                let _ = system.diffusion.mask.set_roles_slice(&bytes);
            }
        }
        system
    }

    pub(crate) fn create_coupled_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        kinetics_state: &Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
        flow_state: &Arc<std::sync::RwLock<MorphogenesisFlowState>>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisKineticsRunner {
        let p = *params.read().unwrap_or_else(|e| e.into_inner());
        let ks = *kinetics_state.read().unwrap_or_else(|e| e.into_inner());
        let flow = *flow_state.read().unwrap_or_else(|e| e.into_inner());
        let rle = mask_rle.read().unwrap_or_else(|e| e.into_inner()).clone();
        let system = Self::build_coupled_system(p, ks, flow, &rle, width, height);
        #[cfg(feature = "physics")]
        let tau = (3.0 * flow.viscosity + 0.5).max(0.51);
        let mut runner = MorphogenesisKineticsRunner {
            system,
            params: Arc::clone(params),
            kinetics_state: Arc::clone(kinetics_state),
            flow_state: Arc::clone(flow_state),
            width,
            height,
            step_count: 0,
            rng: oxidize_core::rng::OxidizeRng::new(DEFAULT_MORPHOGENESIS_SEED),
            noise_amplitude: 0.0,
            vorticity_buf: vec![0.0; width * height],
            peclet: 0.0,
            damkohler: 0.0,
            last_flow_mode: flow.flow_mode,
            #[cfg(feature = "physics")]
            lbm: LatticeBoltzmannD2Q9::new(width.max(1), height.max(1), tau),
            #[cfg(feature = "physics")]
            potential_flow: PotentialFlowField::new(),
            #[cfg(feature = "physics")]
            last_potential_key: (u64::MAX, u64::MAX),
            #[cfg(feature = "physics")]
            potential_dirty: true,
        };
        runner.enforce_mask_concentrations();
        runner.sync_mask_to_hydrodynamics();
        if flow.flow_mode != MorphogenesisFlowMode::None
            && !runner.system.diffusion.obstacles().iter().any(|&b| b)
        {
            runner.seed_default_bluff_obstacle();
        }
        #[cfg(feature = "physics")]
        if flow.flow_mode == MorphogenesisFlowMode::PotentialFlow {
            runner.rebuild_potential_elements(flow.inlet_velocity, flow.circulation);
        }
        runner.refresh_diagnostics();
        runner
    }

    /// Respawns a fresh worker instance in `Ready` state for testing.
    #[cfg(test)]
    pub fn restart_worker(&mut self) {
        let runner = Self::create_coupled_runner(
            &self.params,
            &self.kinetics_state,
            &self.flow_state,
            &self.mask_rle,
            self.width,
            self.height,
        );
        self.live_peclet = runner.peclet;
        self.live_damkohler = runner.damkohler;
        self.worker = BackgroundWorker::new(runner);
        self.stepped_pixels = None;
        self.step_runner = None;
        self.pending_initial_start = false;
        self.dirty = true;
    }

    /// Alias for [`Self::restart_worker`] used in fault-recovery tests.
    #[cfg(test)]
    pub fn retry(&mut self) {
        self.restart_worker();
    }

    /// Resets the worker in place or respawns a disconnected worker in `Ready` state.
    pub fn reset(&mut self) {
        let params = Arc::clone(&self.params);
        let ks = Arc::clone(&self.kinetics_state);
        let flow = Arc::clone(&self.flow_state);
        let mask = Arc::clone(&self.mask_rle);
        let width = self.width;
        let height = self.height;
        self.worker.reset_or_respawn(|| {
            Self::create_coupled_runner(&params, &ks, &flow, &mask, width, height)
        });
        self.stepped_pixels = None;
        self.step_runner = None;
        self.pending_initial_start = false;
        self.dirty = true;
    }

    pub(crate) fn reset_system(&mut self) {
        self.reset();
        let _ = self.worker.send_command(SimCommand::Start);
    }

    fn sync_visual_preferences(&mut self, visuals: &egui::Visuals) {
        let theme_mode = ThemeMode::from_visuals(visuals);
        let high_contrast = theme_mode == ThemeMode::HighContrast;
        let kinetics_mode = self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .mode;
        let overlay_mode = self
            .flow_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .overlay_mode;
        self.theory.set_mode(kinetics_mode);
        if self.last_theme_mode != Some(theme_mode)
            || self.last_high_contrast != Some(high_contrast)
            || self.last_overlay_mode != Some(overlay_mode)
        {
            self.last_theme_mode = Some(theme_mode);
            self.last_high_contrast = Some(high_contrast);
            self.last_overlay_mode = Some(overlay_mode);
            self.params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .high_contrast = high_contrast;
            self.flow_state
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .theme_mode = theme_mode;
            self.dirty = true;
        }
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
            Self::create_coupled_runner(
                &self.params,
                &self.kinetics_state,
                &self.flow_state,
                &self.mask_rle,
                self.width,
                self.height,
            )
        });
        for _ in 0..5 {
            runner.step();
        }
        self.live_peclet = runner.peclet;
        self.live_damkohler = runner.damkohler;
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
        self.sync_visual_preferences(ui.visuals());
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
        self.compute_state_revision()
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_tool_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.load_tool_state(state)
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
        self.params_are_finite()
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
            "advection",
            "vorticity",
            "reaction-diffusion",
            "spots",
            "stripes",
        ],
        build: || Box::new(MorphogenesisTool::default()),
    }
}
