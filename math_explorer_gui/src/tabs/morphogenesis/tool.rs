//! Interactive tool lifecycle and framework integration for 2D morphogenesis.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::MORPHO_ATLAS_AXIS_PAIRS;
use super::counterfactual::MorphogenesisHeatmapMode;
use super::presets::MorphogenesisTheory;
use super::runner::{
    MorphogenesisKineticsRunner, MorphogenesisKineticsState, MorphogenesisSharedParams,
};
use super::sculpt::default_morpho_brush_kit;
use super::types::{MorphogenesisFlowState, MorphogenesisOverlayMode};
use crate::app::ThemeMode;
use crate::async_sim::atlas::ParameterAtlasController;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{
    BranchControlAction, CounterfactualBranchEngine, DivergenceSnapshot, ExportCapabilities,
    InteractionContext, InteractiveTool, SpatialBrushKit,
};
use eframe::egui;
use eframe::egui::ColorImage;
use scientific_metadata::theory::TheoryDescribable;
use std::sync::Arc;

/// Interactive tool for 2D upwind advected Turing & Gray-Scott morphogenesis with microfluidic
/// sculpting and counterfactual shadow twins.
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
    pub(crate) local_runner: MorphogenesisKineticsRunner,
    pub(crate) pending_initial_start: bool,
    pub(crate) last_high_contrast: Option<bool>,
    pub(crate) last_theme_mode: Option<ThemeMode>,
    pub(crate) last_overlay_mode: Option<MorphogenesisOverlayMode>,
    pub(crate) last_heatmap_mode: Option<MorphogenesisHeatmapMode>,
    pub(crate) live_peclet: f64,
    pub(crate) live_damkohler: f64,
    /// Counterfactual shadow twin branch state & perturbation settings.
    pub branch_engine: CounterfactualBranchEngine,
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
        let local_runner = Self::create_coupled_runner(
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
            local_runner,
            pending_initial_start: true,
            last_high_contrast: None,
            last_theme_mode: None,
            last_overlay_mode: None,
            last_heatmap_mode: None,
            live_peclet: peclet,
            live_damkohler: damkohler,
            branch_engine: CounterfactualBranchEngine::default(),
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
        MorphogenesisKineticsRunner::new_seeded(params, width, height)
    }

    #[cfg(test)]
    pub(crate) fn create_kinetics_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        kinetics_state: &Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisKineticsRunner {
        MorphogenesisKineticsRunner::new_with_kinetics(
            params,
            kinetics_state,
            mask_rle,
            width,
            height,
        )
    }

    pub(crate) fn create_coupled_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        kinetics_state: &Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
        flow_state: &Arc<std::sync::RwLock<MorphogenesisFlowState>>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisKineticsRunner {
        MorphogenesisKineticsRunner::new_with_coupled_flow(
            params,
            kinetics_state,
            flow_state,
            mask_rle,
            width,
            height,
        )
    }

    /// Respawns a fresh worker instance in `Ready` state for testing.
    #[cfg(test)]
    pub fn restart_worker(&mut self) {
        self.local_runner = Self::create_coupled_runner(
            &self.params,
            &self.kinetics_state,
            &self.flow_state,
            &self.mask_rle,
            self.width,
            self.height,
        );
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
        self.branch_engine.clear();
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

    /// Resets the worker in place (or respawns if disconnected) and clears any active fork.
    pub fn reset(&mut self) {
        let params = Arc::clone(&self.params);
        let ks = Arc::clone(&self.kinetics_state);
        let flow = Arc::clone(&self.flow_state);
        let mask = Arc::clone(&self.mask_rle);
        let width = self.width;
        let height = self.height;
        self.local_runner = Self::create_coupled_runner(&params, &ks, &flow, &mask, width, height);
        self.live_peclet = self.local_runner.peclet;
        self.live_damkohler = self.local_runner.damkohler;
        self.branch_engine.clear();
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
        let heatmap_mode = self.heatmap_mode();
        self.theory.set_mode(kinetics_mode);
        if self.last_theme_mode != Some(theme_mode)
            || self.last_high_contrast != Some(high_contrast)
            || self.last_overlay_mode != Some(overlay_mode)
            || self.last_heatmap_mode != Some(heatmap_mode)
        {
            self.last_theme_mode = Some(theme_mode);
            self.last_high_contrast = Some(high_contrast);
            self.last_overlay_mode = Some(overlay_mode);
            self.last_heatmap_mode = Some(heatmap_mode);
            {
                let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
                p.theme_mode = theme_mode;
                p.high_contrast = high_contrast;
            }
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
        if self.is_forked() {
            return Some(self.render_current_heatmap_image());
        }
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
        if self.is_forked() {
            self.step_lockstep(5);
            return;
        }
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

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.sync_visual_preferences(ui.visuals());
        if std::mem::take(&mut self.pending_initial_start) && self.worker.state().is_ready() {
            let _ = self.worker.send_command(SimCommand::Start);
        }
        if self.worker.update().is_some() {
            self.pull_latest_worker_snapshot();
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
