#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::MORPHO_ATLAS_AXIS_PAIRS;
use super::counterfactual::MorphogenesisHeatmapMode;
use super::presets::MorphogenesisTheory;
pub use super::presets::MorphogenesisToolState;
use super::runner::{
    MorphogenesisKineticsRunner, MorphogenesisKineticsState, MorphogenesisSharedParams,
};
use super::sculpt::default_morpho_brush_kit;
use crate::app::ThemeMode;
use crate::async_sim::atlas::ParameterAtlasController;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{
    BranchControlAction, CounterfactualBranchEngine, DivergenceSnapshot, ExportCapabilities,
    InteractionContext, InteractiveTool, SpatialBrushKit,
};
use eframe::egui;
use eframe::egui::ColorImage;
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
    pub(crate) stepped_scalar: Option<(usize, usize, Vec<f64>)>,
    pub(crate) step_runner: Option<MorphogenesisKineticsRunner>,
    pub(crate) local_runner: MorphogenesisKineticsRunner,
    pub(crate) pending_initial_start: bool,
    pub(crate) last_theme_mode: Option<ThemeMode>,
    /// Counterfactual shadow twin branch state & perturbation settings.
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
            stepped_scalar: None,
            step_runner: None,
            local_runner,
            pending_initial_start: true,
            last_theme_mode: None,
            branch_engine: CounterfactualBranchEngine::default(),
            theory: MorphogenesisTheory::default(),
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
    ) -> MorphogenesisKineticsRunner {
        MorphogenesisKineticsRunner::new_seeded(params, width, height)
    }

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
        self.stepped_scalar = None;
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
        self.stepped_scalar = None;
        self.step_runner = None;
        self.pending_initial_start = false;
        self.dirty = true;
        self.invalidate_scalar_field_cache();
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
        if self.is_forked() {
            ExportCapabilities {
                dataset: true,
                dataset_disabled_reason: "",
                ..ExportCapabilities::scalar_field()
            }
        } else {
            ExportCapabilities::scalar_field()
        }
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.export_morpho_twin_datasets()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        self.build_export_raster_image()
    }

    fn export_scalar_field(
        &self,
    ) -> Option<math_commons::math_kernel::marching_squares::ScalarField2D> {
        self.build_export_scalar_field()
    }

    fn step_simulation(&mut self) {
        self.step_simulation_offscreen();
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
        }
        self.publish_bus_scalar_field(&ctx);
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
        let mut h = std::collections::hash_map::DefaultHasher::new();
        {
            let ks = self
                .kinetics_state
                .read()
                .unwrap_or_else(|e| e.into_inner());
            ks.mode.hash(&mut h);
            ks.boundary.hash(&mut h);
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
            p.heatmap_mode.hash(&mut h);
        }
        self.width.hash(&mut h);
        self.height.hash(&mut h);
        self.is_forked().hash(&mut h);
        self.local_runner.step_count.hash(&mut h);
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
        let (kinetics_mode, boundary, f, k) = {
            let ks = self
                .kinetics_state
                .read()
                .unwrap_or_else(|e| e.into_inner());
            (ks.mode, ks.boundary, ks.f, ks.k)
        };
        let (a, b, d_u, d_v, dt, heatmap_mode) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v, p.dt, p.heatmap_mode)
        };
        let fork = self
            .local_runner
            .twin
            .as_ref()
            .map(|t| t.to_descriptor(heatmap_mode));
        let mask_rle = self
            .mask_rle
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let atlas_axes = if self.atlas.has_default_persisted_state() {
            super::atlas::default_morpho_atlas_axes()
        } else {
            let (xk, yk) = self.atlas.axis_keys();
            [xk.to_string(), yk.to_string()]
        };
        serde_json::to_string(&MorphogenesisToolState {
            a,
            b,
            d_u,
            d_v,
            dt,
            width: self.width,
            height: self.height,
            kinetics_mode,
            boundary,
            f,
            k,
            state: Some(self.worker.state()),
            paused: None,
            pending_initial_start: self.pending_initial_start,
            fork,
            mask_rle,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: MorphogenesisToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Morphogenesis state: {e}"))?;
        parsed.validate()?;
        self.sync_atlas_options_for_mode(parsed.kinetics_mode);
        let use_mode_default = parsed.kinetics_mode
            == crate::presets::MorphogenesisKineticsMode::GrayScott
            && super::atlas::is_default_morpho_atlas_axes(&parsed.atlas_axes);
        if !use_mode_default {
            self.atlas
                .set_axis_keys(&parsed.atlas_axes[0], &parsed.atlas_axes[1])?;
        }
        self.atlas.set_visible(parsed.atlas_visible);
        let target_state = parsed.resolve_target_state();
        self.theory.mode = parsed.kinetics_mode;
        {
            let mut ks = self
                .kinetics_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            ks.mode = parsed.kinetics_mode;
            ks.boundary = parsed.boundary;
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
            p.heatmap_mode = parsed
                .fork
                .map_or(MorphogenesisHeatmapMode::ActiveUb, |fk| fk.heatmap_mode);
        }
        *self.mask_rle.write().unwrap_or_else(|e| e.into_inner()) = parsed.mask_rle;
        if let Some(fork_desc) = parsed.fork {
            self.apply_loaded_fork_state(
                parsed.width,
                parsed.height,
                fork_desc,
                target_state,
                parsed.pending_initial_start,
            )?;
        } else {
            self.apply_dimensions_and_target_state(
                parsed.width,
                parsed.height,
                target_state,
                parsed.pending_initial_start,
            );
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
