#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::persistence::build_formula_bar;
use super::{CompiledFormulaPayload, UnifiedModel, UnifiedSimRunner, UnifiedUiExtension};
use crate::async_sim::{
    BackgroundWorker, SculptStatePayload, SimCommand, SimWorkerState, StateSnapshot,
};
use crate::framework::{
    BrushStroke, CalloutManager, ChallengeEvaluation, InteractionContext, InteractiveTool,
    MathExpressionInputBar, SpatialBrushKit,
};
use eframe::egui::{self, ColorImage};
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[allow(missing_docs)]
pub struct UnifiedSimTool<M: UnifiedModel> {
    pub(crate) controller: BackgroundWorker<UnifiedSimRunner<M>>,
    pub(crate) params: Arc<RwLock<HashMap<String, f64>>>,
    pub(crate) sculpt_state: Arc<RwLock<SculptStatePayload>>,
    pub(crate) brush_kit: SpatialBrushKit,
    pub(crate) shared_formula: Arc<RwLock<Option<CompiledFormulaPayload>>>,
    pub(crate) formula_bar: Option<MathExpressionInputBar>,
    pub(crate) custom_formula_enabled: bool,
    pub(super) formula_revision: u64,
    pub(super) param_metadata: Vec<(String, ParameterConstraint)>,
    pub(super) steps_per_frame: usize,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(super) dirty: bool,
    pub(crate) step_model: Option<M>,
    pub(super) stepped_snapshot: Option<StateSnapshot>,
    cached_theory_desc: String,
    cached_latex_eq: Option<String>,
    cached_phonetic: String,
    cached_citation: String,
    pub(super) cached_descs: HashMap<String, String>,
    pub(super) cached_param_labels: Vec<String>,
    pub(super) cached_observable_label: Option<&'static str>,
    pub(crate) sensitivity_engine: crate::sensitivity::SensitivityEngine,
    pub(super) brush_state: crate::framework::RasterBrushState,
    pub(super) obstacle_rle: Arc<RwLock<Option<String>>>,
    pub(super) local_obstacles: Option<Vec<bool>>,
    pub(crate) callout_manager: CalloutManager,
    pub(super) ui_extension: Option<Box<dyn UnifiedUiExtension>>,
    pub(crate) atlas: Option<crate::async_sim::atlas::ParameterAtlasController>,
    _marker: std::marker::PhantomData<M>,
}

impl<M: UnifiedModel> Default for UnifiedSimTool<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: UnifiedModel> UnifiedSimTool<M> {
    fn build_initial_params(
        param_metadata: &[(String, ParameterConstraint)],
    ) -> HashMap<String, f64> {
        param_metadata
            .iter()
            .map(|(name, constraint)| {
                let val = M::parameter_choices(name)
                    .first()
                    .map_or((constraint.min + constraint.max) / 2.0, |&(_, v)| v);
                (name.clone(), val)
            })
            .collect()
    }

    #[allow(missing_docs)]
    pub fn new() -> Self {
        let mut param_metadata: Vec<_> = M::parameters().into_iter().collect();
        param_metadata.sort_by(|a, b| a.0.cmp(&b.0));
        let initial_params = Self::build_initial_params(&param_metadata);
        let local_obstacles = M::default_obstacle_mask(&initial_params);
        let obstacle_rle = Arc::new(RwLock::new(None));
        let formula_bar = M::formula_spec().as_ref().map(build_formula_bar);
        let params = Arc::new(RwLock::new(initial_params.clone()));
        let sculpt_state = Arc::new(RwLock::new(SculptStatePayload::default()));
        let shared_formula = Arc::new(RwLock::new(None));
        let runner = UnifiedSimRunner::<M>::with_shared_and_obstacles(
            Arc::clone(&params),
            Arc::clone(&sculpt_state),
            Arc::clone(&shared_formula),
            Arc::clone(&obstacle_rle),
        );
        let controller = BackgroundWorker::new(runner);
        let temp_model = M::new(&initial_params);
        let cached_param_labels = param_metadata
            .iter()
            .map(|(name, _)| M::parameter_label(name))
            .collect();
        let mut callout_manager = CalloutManager::new();
        let ui_extension = M::create_ui_extension();
        if let Some(ext) = &ui_extension {
            ext.sync_callouts(controller.latest_snapshot(), &mut callout_manager);
        }
        Self {
            controller,
            params,
            sculpt_state,
            brush_kit: M::default_brush_kit(),
            shared_formula,
            formula_bar,
            custom_formula_enabled: false,
            formula_revision: 1,
            param_metadata,
            steps_per_frame: 5,
            texture: None,
            dirty: true,
            step_model: None,
            stepped_snapshot: None,
            cached_theory_desc: temp_model.theory_description(),
            cached_latex_eq: temp_model.theory_latex_equation(),
            cached_phonetic: temp_model.phonetic_description(),
            cached_citation: temp_model.theory_citation(),
            cached_descs: temp_model.available_descriptions(),
            cached_param_labels,
            cached_observable_label: None,
            sensitivity_engine: crate::sensitivity::SensitivityEngine::new(),
            brush_state: crate::framework::RasterBrushState::default(),
            obstacle_rle,
            local_obstacles,
            callout_manager,
            ui_extension,
            atlas: Self::create_optional_atlas(),
            _marker: std::marker::PhantomData,
        }
    }

    #[allow(missing_docs)]
    pub fn state(&self) -> SimWorkerState {
        self.controller.state()
    }

    pub(super) fn clear_stepped_state(&mut self) {
        self.step_model = None;
        self.stepped_snapshot = None;
    }

    pub(super) fn refresh_paused_simulation(&mut self, ctx: Option<&egui::Context>) {
        if self.controller.state().is_ready() {
            self.clear_stepped_state();
            let _ = self.controller.send_command(SimCommand::Custom(
                super::SYNC_PARAMETERS_COMMAND.to_string(),
            ));
            self.dirty = true;
            if let Some(c) = ctx {
                c.request_repaint();
            }
        }
    }

    pub(super) fn dispatch_extension_commands(
        &mut self,
        commands: Vec<SimCommand>,
        ctx: Option<&egui::Context>,
    ) {
        if commands.is_empty() {
            return;
        }
        *self.obstacle_rle.write().unwrap_or_else(|e| e.into_inner()) = None;
        self.clear_stepped_state();
        for cmd in commands {
            let _ = self.controller.send_command(cmd);
        }
        self.dirty = true;
        if let Some(ext) = &self.ui_extension {
            if M::supports_spatial_brush() {
                let mut sculpt = self.sculpt_state.write().unwrap_or_else(|e| e.into_inner());
                ext.sync_sculpt_payload(&mut sculpt);
            }
            ext.sync_callouts(self.controller.latest_snapshot(), &mut self.callout_manager);
        }
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    pub(crate) fn reset_simulation(&mut self, ctx: Option<&egui::Context>) {
        if self
            .ui_extension
            .as_ref()
            .is_none_or(|ext| ext.active_challenge_id().is_none())
        {
            self.sculpt_state
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .clear();
        }
        self.respawn_preserving_sculpt(ctx);
    }

    pub(super) fn respawn_preserving_sculpt(&mut self, ctx: Option<&egui::Context>) {
        self.clear_stepped_state();
        if let Some(ext) = &mut self.ui_extension {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            ext.on_reset(&p);
            ext.sync_callouts(self.controller.latest_snapshot(), &mut self.callout_manager);
        }
        let params = Arc::clone(&self.params);
        let sculpt = Arc::clone(&self.sculpt_state);
        let formula = Arc::clone(&self.shared_formula);
        let obstacle_rle = Arc::clone(&self.obstacle_rle);
        self.controller.reset_or_respawn(|| {
            UnifiedSimRunner::<M>::with_shared_and_obstacles(params, sculpt, formula, obstacle_rle)
        });
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    /// Dispatches an interpolated [`BrushStroke`] to both UI sculpt state and the background worker.
    pub fn dispatch_brush_stroke(&mut self, stroke: BrushStroke, ctx: Option<&egui::Context>) {
        if let Some(ext) = &mut self.ui_extension {
            if !ext.on_brush_stroke(&stroke) {
                return;
            }
        }
        if M::supports_spatial_brush() {
            *self.obstacle_rle.write().unwrap_or_else(|e| e.into_inner()) = None;
            let params_snap = self
                .params
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let formula_snap = self
                .shared_formula
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let mut sculpt = self.sculpt_state.write().unwrap_or_else(|e| e.into_inner());
            M::record_ui_brush_stroke(&stroke, &params_snap, formula_snap.as_ref(), &mut sculpt);
        }
        if self
            .controller
            .send_command(SimCommand::ApplyBrushStroke(stroke))
            .is_ok()
        {
            self.clear_stepped_state();
            self.dirty = true;
            if let Some(c) = ctx {
                c.request_repaint();
            }
        }
    }
}

impl<M: UnifiedModel> InteractiveTool for UnifiedSimTool<M> {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        M::name()
    }
    fn domain(&self) -> &'static str {
        M::domain()
    }

    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        if M::is_raster_model() {
            crate::framework::ExportCapabilities::raster_only()
        } else {
            crate::framework::ExportCapabilities::plot()
        }
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        if !M::is_raster_model() {
            if let Some(ref snap) = self.stepped_snapshot {
                let ds = M::snapshot_datasets(snap);
                if !ds.is_empty() {
                    return ds;
                }
            }
            if let Some(snap) = self.controller.latest_snapshot() {
                let ds = M::snapshot_datasets(snap);
                if !ds.is_empty() {
                    return ds;
                }
            }
        }
        egui_plot::get_last_plot_datasets()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        let snapshot = self
            .stepped_snapshot
            .as_ref()
            .or_else(|| self.controller.latest_snapshot())?;
        let guard = snapshot.pixels.try_read().ok()?;
        if guard.is_empty() || snapshot.width == 0 || snapshot.height == 0 {
            return None;
        }
        Some(ColorImage::new(
            [snapshot.width, snapshot.height],
            guard.clone(),
        ))
    }

    fn step_simulation(&mut self) {
        if let Ok(params) = self.params.read() {
            let custom_formula_enabled = self.custom_formula_enabled;
            let shared_formula = &self.shared_formula;
            let sculpt_state = &self.sculpt_state;
            let obstacle_rle = &self.obstacle_rle;
            let model = self.step_model.get_or_insert_with(|| {
                let mut m = M::new(&params);
                if let Some(ref rle) = *obstacle_rle.read().unwrap_or_else(|e| e.into_inner()) {
                    m.apply_obstacle_rle(rle);
                }
                if custom_formula_enabled {
                    let payload = shared_formula
                        .read()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone();
                    m.apply_compiled_formula(payload.as_ref(), &params);
                }
                if M::supports_spatial_brush() {
                    let sculpt = sculpt_state.read().unwrap_or_else(|e| e.into_inner());
                    if !sculpt.is_empty() {
                        m.apply_sculpt_payload(&sculpt, &params);
                    }
                }
                m
            });
            for _ in 0..self.steps_per_frame.max(1) {
                model.step(&params);
            }
            self.stepped_snapshot = Some(model.get_snapshot());
        }
    }

    fn on_brush(&mut self, ctx: &InteractionContext) {
        if let Some(stroke) = ctx.brush_stroke {
            self.dispatch_brush_stroke(stroke, Some(&ctx.response.ctx));
        }
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.brush_state
            .focused_id
            .or_else(|| self.brush_kit.focused_plot_id())
    }

    fn state_revision(&self) -> u64 {
        self.compute_state_revision()
    }

    fn has_finite_parameters(&self) -> bool {
        let params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let formula_ok = self
            .formula_bar
            .as_ref()
            .is_none_or(|b| b.param_slice().iter().all(|v| v.is_finite()));
        params.values().all(|v| v.is_finite())
            && formula_ok
            && !matches!(
                self.controller.state(),
                crate::async_sim::SimWorkerState::Faulted
            )
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_tool_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.deserialize_tool_state(state)
    }

    fn apply_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        let Some(ext) = &mut self.ui_extension else {
            return if let Some(id) = challenge_id {
                Err(format!(
                    "Tool '{}' does not support challenge '{id}'",
                    M::name()
                ))
            } else {
                Ok(())
            };
        };
        let mut next_params = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let cmds = ext.apply_challenge(challenge_id, &mut next_params)?;
        *self.params.write().unwrap_or_else(|e| e.into_inner()) = next_params;
        self.clear_stepped_state();
        self.dispatch_extension_commands(cmds, None);
        Ok(())
    }

    fn active_challenge_id(&self) -> Option<&'static str> {
        self.ui_extension
            .as_ref()
            .and_then(|ext| ext.active_challenge_id())
    }

    fn evaluate_challenge(&self) -> Option<ChallengeEvaluation> {
        let snap = self
            .stepped_snapshot
            .as_ref()
            .or_else(|| self.controller.latest_snapshot());
        self.ui_extension
            .as_ref()
            .and_then(|ext| ext.evaluate_challenge(snap))
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn clamp_parameters(&mut self) -> bool {
        self.handle_recovery_action(
            crate::framework::recovery::RecoveryAction::ClampParameters,
            None,
        );
        true
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let has_update = self.controller.update().is_some();
        if has_update {
            self.clear_stepped_state();
            self.dirty = true;
        }
        self.draw_left_panel(ui);
        self.draw_central_panel(ui);
        if has_update
            || matches!(
                self.controller.state(),
                SimWorkerState::Running | SimWorkerState::Stepping
            )
        {
            ui.ctx().request_repaint();
        }
    }
}

impl<M: UnifiedModel> scientific_metadata::theory::TheoryDescribable for UnifiedSimTool<M> {
    fn theory_description(&self) -> String {
        self.cached_theory_desc.clone()
    }
    fn theory_latex_equation(&self) -> Option<String> {
        self.cached_latex_eq.clone()
    }
    fn phonetic_description(&self) -> String {
        self.cached_phonetic.clone()
    }
    fn theory_citation(&self) -> String {
        self.cached_citation.clone()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        self.cached_descs.clone()
    }
    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        self.param_metadata.iter().cloned().collect()
    }
}
