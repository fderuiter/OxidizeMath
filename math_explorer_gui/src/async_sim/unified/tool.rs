#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::persistence::build_formula_bar;
use super::{CompiledFormulaPayload, UnifiedModel, UnifiedSimRunner};
use crate::async_sim::ui::{render_worker_fault_banner, WorkerFaultAction};
use crate::async_sim::{
    BackgroundWorker, SculptStatePayload, SimCommand, SimWorkerState, StateSnapshot,
};
use crate::framework::{
    BrushStroke, InteractionContext, InteractiveTool, MathExpressionInputBar, PlaybackAction,
    PlaybackTransport, SpatialBrushKit,
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
    steps_per_frame: usize,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(super) dirty: bool,
    stepped_snapshot: Option<StateSnapshot>,
    cached_theory_desc: String,
    cached_phonetic: String,
    cached_citation: String,
    pub(super) cached_descs: HashMap<String, String>,
    pub(super) cached_param_labels: Vec<String>,
    pub(super) cached_observable_label: Option<&'static str>,
    pub(crate) sensitivity_engine: crate::sensitivity::SensitivityEngine,
    _marker: std::marker::PhantomData<M>,
}

impl<M: UnifiedModel> Default for UnifiedSimTool<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: UnifiedModel> UnifiedSimTool<M> {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        let param_map = M::parameters();
        let mut param_metadata: Vec<_> = param_map.into_iter().collect();
        param_metadata.sort_by(|a, b| a.0.cmp(&b.0));

        let mut initial_params = HashMap::new();
        for (name, constraint) in &param_metadata {
            let default_val = M::parameter_choices(name)
                .first()
                .map_or((constraint.min + constraint.max) / 2.0, |&(_, v)| v);
            initial_params.insert(name.clone(), default_val);
        }

        let formula_bar = M::formula_spec().as_ref().map(build_formula_bar);
        let params = Arc::new(RwLock::new(initial_params.clone()));
        let sculpt_state = Arc::new(RwLock::new(SculptStatePayload::default()));
        let shared_formula = Arc::new(RwLock::new(None));
        let runner = UnifiedSimRunner::<M>::with_shared(
            Arc::clone(&params),
            Arc::clone(&sculpt_state),
            Arc::clone(&shared_formula),
        );
        let controller = BackgroundWorker::new(runner);
        let temp_model = M::new(&initial_params);
        let brush_kit = M::default_brush_kit();
        let cached_param_labels = param_metadata
            .iter()
            .map(|(name, _)| M::parameter_label(name))
            .collect();

        Self {
            controller,
            params,
            sculpt_state,
            brush_kit,
            shared_formula,
            formula_bar,
            custom_formula_enabled: false,
            formula_revision: 1,
            param_metadata,
            steps_per_frame: 5,
            texture: None,
            dirty: true,
            stepped_snapshot: None,
            cached_theory_desc: temp_model.theory_description(),
            cached_phonetic: temp_model.phonetic_description(),
            cached_citation: temp_model.theory_citation(),
            cached_descs: temp_model.available_descriptions(),
            cached_param_labels,
            cached_observable_label: None,
            sensitivity_engine: crate::sensitivity::SensitivityEngine::new(),
            _marker: std::marker::PhantomData,
        }
    }

    #[allow(missing_docs)]
    pub fn state(&self) -> SimWorkerState {
        self.controller.state()
    }

    pub(super) fn reset_simulation(&mut self, ctx: Option<&egui::Context>) {
        self.sculpt_state
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.respawn_preserving_sculpt(ctx);
    }

    pub(super) fn respawn_preserving_sculpt(&mut self, ctx: Option<&egui::Context>) {
        self.stepped_snapshot = None;
        let params = Arc::clone(&self.params);
        let sculpt = Arc::clone(&self.sculpt_state);
        let formula = Arc::clone(&self.shared_formula);
        self.controller
            .reset_or_respawn(|| UnifiedSimRunner::<M>::with_shared(params, sculpt, formula));
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    /// Dispatches an interpolated [`BrushStroke`] to both UI sculpt state and the background worker.
    pub fn dispatch_brush_stroke(&mut self, stroke: BrushStroke, ctx: Option<&egui::Context>) {
        if M::supports_spatial_brush() {
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
            self.dirty = true;
            if let Some(c) = ctx {
                c.request_repaint();
            }
        }
    }

    fn handle_recovery_action(
        &mut self,
        action: crate::framework::recovery::RecoveryAction,
        ctx: Option<&egui::Context>,
    ) {
        use crate::framework::recovery::RecoveryAction;
        match action {
            RecoveryAction::AutoScaleStepSize => {
                let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
                if let Some(dt) = p.get_mut("dt") {
                    *dt = (*dt * 0.5).max(1e-4);
                } else if let Some(visc) = p.get_mut("viscosity") {
                    *visc = (*visc * 0.5).max(0.00333);
                }
                drop(p);
                self.reset_simulation(ctx);
            }
            RecoveryAction::ClampParameters => {
                let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
                for (name, c) in &self.param_metadata {
                    if let Some(v) = p.get_mut(name) {
                        *v = if !v.is_finite() {
                            M::parameter_choices(name)
                                .first()
                                .map_or((c.min + c.max) / 2.0, |&(_, choice)| choice)
                        } else {
                            v.clamp(c.min, c.max)
                        };
                    }
                }
                drop(p);
                self.reset_simulation(ctx);
            }
            RecoveryAction::ResetParameters => {
                let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
                for (name, c) in &self.param_metadata {
                    let d = match name.as_str() {
                        "viscosity" => 0.02,
                        "dt" => 0.05,
                        "potential_type" => 0.0,
                        _ => M::parameter_choices(name)
                            .first()
                            .map_or((c.min + c.max) / 2.0, |&(_, choice)| choice),
                    };
                    p.insert(name.clone(), d);
                }
                drop(p);
                if let Some(spec) = M::formula_spec() {
                    self.custom_formula_enabled = false;
                    self.formula_bar = Some(build_formula_bar(&spec));
                    self.publish_formula_update();
                }
                self.reset_simulation(ctx);
                let _ = self.controller.send_command(SimCommand::Start);
            }
        }
    }

    fn draw_status_and_transport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let state = self.controller.state();
        if let Some(fault_action) =
            render_worker_fault_banner(ui, state, self.controller.diagnostic_report(), true)
        {
            match fault_action {
                WorkerFaultAction::Reset | WorkerFaultAction::ResetParameters => {
                    self.reset_simulation(Some(ctx));
                }
                WorkerFaultAction::Recovery(action) => {
                    self.handle_recovery_action(action, Some(ctx));
                }
            }
            ui.separator();
        }

        let state = self.controller.state();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.horizontal(|ui| {
            ui.label("Status:");
            let (status_text, color) = match state {
                SimWorkerState::Uninitialized => ("Uninitialized", palette.muted_text),
                SimWorkerState::Ready => ("Ready", palette.success),
                SimWorkerState::Running => ("Running", palette.info),
                SimWorkerState::Stepping => ("Stepping", palette.warning),
                SimWorkerState::Faulted => ("Faulted", palette.error),
                SimWorkerState::Terminated => ("Terminated", palette.muted_text),
            };
            ui.colored_label(color, status_text);
        });
        ui.separator();

        if let Some(action) =
            PlaybackTransport::from_worker_state("unified_transport", M::name(), state)
                .can_reset(true)
                .show(ui)
        {
            match action {
                PlaybackAction::Play => {
                    let _ = self.controller.send_command(SimCommand::Start);
                }
                PlaybackAction::Pause => {
                    let _ = self.controller.send_command(SimCommand::Pause);
                }
                PlaybackAction::Step => {
                    let _ = self.controller.send_command(SimCommand::Step(1));
                }
                PlaybackAction::Reset => {
                    self.reset_simulation(Some(ctx));
                }
            }
        }
    }

    fn draw_left_panel(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let panel_id = crate::framework::scoped_panel_id(ui, format!("{}_controls", M::name()));
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading(M::name());
            ui.separator();
            self.draw_status_and_transport(ui, &ctx);
            ui.separator();
            ui.label("Simulation Constants");
            if ui
                .add(
                    egui::Slider::new(&mut self.steps_per_frame, 1..=100)
                        .text("Speed (Steps/Frame)"),
                )
                .changed()
            {
                let _ = self
                    .controller
                    .send_command(SimCommand::SetSpeed(self.steps_per_frame));
            }
            self.draw_model_parameters(ui, &ctx);
        });
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
            let mut model = M::new(&params);
            if self.custom_formula_enabled {
                let payload = self
                    .shared_formula
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                model.apply_compiled_formula(payload.as_ref(), &params);
            }
            if M::supports_spatial_brush() {
                let sculpt = self.sculpt_state.read().unwrap_or_else(|e| e.into_inner());
                if !sculpt.is_empty() {
                    model.apply_sculpt_payload(&sculpt, &params);
                }
            }
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
        self.brush_kit.focused_plot_id()
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let sculpt = self.sculpt_state.read().unwrap_or_else(|e| e.into_inner());
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for (key, _) in &self.param_metadata {
            key.hash(&mut h);
            if let Some(val) = params.get(key) {
                val.to_bits().hash(&mut h);
            }
        }
        sculpt.hash_into(&mut h);
        self.custom_formula_enabled.hash(&mut h);
        if let Some(ref bar) = self.formula_bar {
            bar.raw_text.hash(&mut h);
            for &pv in bar.param_slice() {
                pv.to_bits().hash(&mut h);
            }
        }
        h.finish().max(1)
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
            self.stepped_snapshot = None;
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
