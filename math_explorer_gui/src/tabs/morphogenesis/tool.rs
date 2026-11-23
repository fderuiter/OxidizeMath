//! Interactive tool controller for 2D Schnakenberg Turing, Gray-Scott, counterfactual twins,
//! and microfluidic morphogenesis.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::AtlasCache;
use super::cartographer::MORPHO_ATLAS_AXIS_PAIRS;
use super::presets::MorphogenesisTheory;
use super::runner::{MorphogenesisKineticsState, MorphogenesisRunner, MorphogenesisSharedParams};
use super::sculpt::default_morpho_brush_kit;
use super::state::MorphogenesisKinetics;
use crate::app::ThemeMode;
use crate::async_sim::atlas::ParameterAtlasController;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{
    BranchControlAction, CounterfactualBranchEngine, DivergenceSnapshot, ExportCapabilities,
    InteractionContext, InteractiveTool, SpatialBrushKit,
};
use eframe::egui::{self, ColorImage};
use scientific_metadata::theory::TheoryDescribable;
use std::sync::Arc;

/// Interactive tool for 2D Schnakenberg Turing and Gray-Scott reaction-diffusion morphogenesis.
pub struct MorphogenesisTool {
    /// Asynchronous simulation worker controller.
    pub worker: BackgroundWorker<MorphogenesisRunner>,
    /// Progressive 2D parameter regime atlas controller.
    pub atlas: ParameterAtlasController,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
    pub(crate) kinetics_state: Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
    pub(crate) mask_rle: Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
    pub(crate) brush_kit: SpatialBrushKit,
    pub(crate) reagent_brush: bool,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) pixels: Vec<egui::Color32>,
    pub(crate) dirty: bool,
    pub(crate) stepped_pixels: Option<(usize, usize, Vec<egui::Color32>)>,
    pub(crate) step_runner: Option<MorphogenesisRunner>,
    pub(crate) local_runner: MorphogenesisRunner,
    pub(crate) pending_initial_start: bool,
    pub(crate) last_theme_mode: Option<ThemeMode>,
    pub(crate) atlas_cache: AtlasCache,
    pub(crate) pending_brush_frames: u8,
    /// Counterfactual shadow twin branch state and perturbation settings.
    pub branch_engine: CounterfactualBranchEngine,
    pub(crate) theory: MorphogenesisTheory,
    pub(crate) field_stream: super::dataflow::MorphoFieldStreamState,
}

impl Default for MorphogenesisTool {
    fn default() -> Self {
        let width = 100;
        let height = 100;
        let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams::default()));
        let kinetics_state =
            Arc::new(std::sync::RwLock::new(MorphogenesisKineticsState::default()));
        let mask_rle = Arc::new(std::sync::RwLock::new(Vec::new()));
        let runner =
            Self::create_kinetics_runner(&params, &kinetics_state, &mask_rle, width, height);
        let local_runner =
            Self::create_kinetics_runner(&params, &kinetics_state, &mask_rle, width, height);
        let worker = BackgroundWorker::new(runner);
        let theory = MorphogenesisTheory::new(Arc::clone(&params));

        Self {
            worker,
            atlas: ParameterAtlasController::new(MORPHO_ATLAS_AXIS_PAIRS.to_vec()),
            params,
            kinetics_state,
            mask_rle,
            brush_kit: default_morpho_brush_kit(),
            reagent_brush: false,
            width,
            height,
            texture: None,
            pixels: Vec::new(),
            dirty: true,
            stepped_pixels: None,
            step_runner: None,
            local_runner,
            pending_initial_start: true,
            last_theme_mode: None,
            atlas_cache: AtlasCache::default(),
            pending_brush_frames: 0,
            branch_engine: CounterfactualBranchEngine::default(),
            theory,
            field_stream: super::dataflow::MorphoFieldStreamState::default(),
        }
    }
}

impl MorphogenesisTool {
    #[cfg(test)]
    pub(crate) fn create_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisRunner {
        let mask = Arc::new(std::sync::RwLock::new(Vec::new()));
        Self::create_masked_runner(params, &mask, width, height)
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn restart_worker(&mut self) {
        self.local_runner = Self::create_kinetics_runner(
            &self.params,
            &self.kinetics_state,
            &self.mask_rle,
            self.width,
            self.height,
        );
        let runner = Self::create_kinetics_runner(
            &self.params,
            &self.kinetics_state,
            &self.mask_rle,
            self.width,
            self.height,
        );
        self.worker = BackgroundWorker::new(runner);
        self.branch_engine.clear();
        self.stepped_pixels = None;
        self.step_runner = None;
        self.pending_initial_start = false;
        self.dirty = true;
        self.invalidate_scalar_field_cache();
    }

    #[cfg(test)]
    #[allow(missing_docs)]
    pub fn retry(&mut self) {
        self.restart_worker();
    }

    /// Resets the worker in place (or respawns if disconnected) and clears any active fork.
    pub fn reset(&mut self) {
        let params = Arc::clone(&self.params);
        let ks = Arc::clone(&self.kinetics_state);
        let mask = Arc::clone(&self.mask_rle);
        let width = self.width;
        let height = self.height;
        self.local_runner = Self::create_kinetics_runner(&params, &ks, &mask, width, height);
        self.branch_engine.clear();
        self.worker
            .reset_or_respawn(|| Self::create_kinetics_runner(&params, &ks, &mask, width, height));
        self.stepped_pixels = None;
        self.step_runner = None;
        self.pending_initial_start = false;
        self.dirty = true;
        self.invalidate_scalar_field_cache();
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
        self.theory.mode = next;
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
        self.kinetics_state
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .mode = next;
        self.sync_atlas_options_for_mode(next);
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
        if kinetics == MorphogenesisKinetics::GrayScott {
            let mut ks = self
                .kinetics_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            ks.f = x.clamp(0.0, 0.2);
            ks.k = y.clamp(0.0, 0.2);
        }
        if !self.is_forked() {
            let _ = self
                .worker
                .send_command(SimCommand::Custom("ensure_nucleated".to_string()));
        }
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
        if self.is_forked() {
            ExportCapabilities::plot()
        } else {
            ExportCapabilities::raster_only()
        }
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.export_morpho_twin_datasets()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        self.export_morpho_raster_image()
    }

    fn step_simulation(&mut self) {
        self.advance_morpho_step();
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
        let theme_mode = ThemeMode::from_visuals(ui.visuals());
        if self.last_theme_mode != Some(theme_mode) {
            self.last_theme_mode = Some(theme_mode);
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.theme_mode = theme_mode;
            p.high_contrast = theme_mode == ThemeMode::HighContrast;
            drop(p);
            self.dirty = true;
        }
        if std::mem::take(&mut self.pending_initial_start) && self.worker.state().is_ready() {
            let _ = self.worker.send_command(SimCommand::Start);
        }
        if let Some(snap) = self.worker.update() {
            self.field_stream.record_snapshot(snap);
            self.pull_latest_worker_snapshot();
            self.dirty = true;
            self.pending_brush_frames = 0;
        }
        self.publish_bus_scalar_field(&ctx);
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

    fn counterfactual_snapshot(&self) -> Option<DivergenceSnapshot> {
        let dt = self.params.read().unwrap_or_else(|e| e.into_inner()).dt;
        self.local_runner
            .twin
            .as_ref()
            .map(|t| t.snapshot(self.local_runner.step_count, dt))
    }

    fn apply_counterfactual_action(&mut self, action: BranchControlAction) -> bool {
        self.apply_morpho_branch_action(action)
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn state_revision(&self) -> u64 {
        self.compute_state_revision()
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_morpho_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.deserialize_morpho_state(state)
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
        let (_, f, k) = self.effective_kinetics_triplet();
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        p.a.is_finite()
            && p.b.is_finite()
            && f.is_finite()
            && k.is_finite()
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
