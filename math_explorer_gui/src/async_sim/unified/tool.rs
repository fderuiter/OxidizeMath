#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{UnifiedModel, UnifiedSimRunner};
use crate::accessibility::AccessibleHoverText;
use crate::async_sim::ui::{render_worker_fault_banner, WorkerFaultAction};
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::InteractiveTool;
use eframe::egui;
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[allow(missing_docs)]
pub struct UnifiedSimTool<M: UnifiedModel> {
    pub(crate) controller: BackgroundWorker<UnifiedSimRunner<M>>,
    pub(crate) params: Arc<RwLock<HashMap<String, f64>>>,
    param_metadata: Vec<(String, ParameterConstraint)>,
    steps_per_frame: usize,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(super) dirty: bool,
    cached_theory_desc: String,
    cached_phonetic: String,
    cached_citation: String,
    cached_descs: HashMap<String, String>,
    sensitivity_engine: crate::sensitivity::SensitivityEngine,
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
            initial_params.insert(name.clone(), (constraint.min + constraint.max) / 2.0);
        }

        let params = Arc::new(RwLock::new(initial_params.clone()));
        let runner = UnifiedSimRunner::<M>::new(Arc::clone(&params));
        let controller = BackgroundWorker::new(runner);
        let temp_model = M::new(&initial_params);

        Self {
            controller,
            params,
            param_metadata,
            steps_per_frame: 5,
            texture: None,
            dirty: true,
            cached_theory_desc: temp_model.theory_description(),
            cached_phonetic: temp_model.phonetic_description(),
            cached_citation: temp_model.theory_citation(),
            cached_descs: temp_model.available_descriptions(),
            sensitivity_engine: crate::sensitivity::SensitivityEngine::new(),
            _marker: std::marker::PhantomData,
        }
    }

    #[allow(missing_docs)]
    pub fn state(&self) -> SimWorkerState {
        self.controller.state()
    }

    fn reset_simulation(&mut self, ctx: Option<&egui::Context>) {
        let params = Arc::clone(&self.params);
        self.controller
            .reset_or_respawn(|| UnifiedSimRunner::<M>::new(params));
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
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
                            (c.min + c.max) / 2.0
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
                        _ => (c.min + c.max) / 2.0,
                    };
                    p.insert(name.clone(), d);
                }
                drop(p);
                self.reset_simulation(ctx);
                let _ = self.controller.send_command(SimCommand::Start);
            }
        }
    }

    fn draw_presets_and_actions(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) -> bool {
        let mut reset_needed = false;
        let presets = M::presets();
        if !presets.is_empty() {
            let mut params_lock = self.params.write().unwrap_or_else(|e| e.into_inner());
            ui.horizontal(|ui| {
                ui.label("Presets:");
                for (name, preset_params) in presets {
                    if ui.button(name).clicked() {
                        for (k, v) in &preset_params {
                            params_lock.insert(k.clone(), *v);
                        }
                        reset_needed = true;
                    }
                }
            });
            drop(params_lock);
            ui.separator();
        }

        let custom_actions = M::custom_actions();
        if !custom_actions.is_empty() {
            ui.horizontal(|ui| {
                ui.label("Actions:");
                for action in custom_actions {
                    if ui.button(action).clicked()
                        && self
                            .controller
                            .send_command(SimCommand::Custom(action.to_string()))
                            .is_ok()
                    {
                        ctx.request_repaint();
                    }
                }
            });
            ui.separator();
        }
        reset_needed
    }

    fn draw_model_parameters(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.separator();
        ui.label("Model Parameters");

        let reset_needed = self.draw_presets_and_actions(ui, ctx);
        let mut params_lock = self.params.write().unwrap_or_else(|e| e.into_inner());

        for (name, constraint) in &self.param_metadata {
            if let Some(val) = params_lock.get_mut(name) {
                let slider = egui::Slider::new(val, constraint.min..=constraint.max)
                    .step_by(constraint.step)
                    .text(name);
                let mut resp = ui.add(slider);
                if let Some(desc) = self.cached_descs.get(name) {
                    resp = resp.accessible_hover_text(desc);
                }
            }
        }

        drop(params_lock);
        self.draw_observable_sensitivity(ui);
        if reset_needed {
            self.reset_simulation(Some(ctx));
        }
    }

    fn draw_observable_sensitivity(&mut self, ui: &mut egui::Ui) {
        let observable_setup = {
            let current_params = self.params.read().unwrap_or_else(|e| e.into_inner());
            M::evaluate_observable(&current_params).map(|(metric_label, baseline_val)| {
                let param_list: Vec<(String, f64)> = self
                    .param_metadata
                    .iter()
                    .filter_map(|(name, _)| current_params.get(name).map(|v| (name.clone(), *v)))
                    .collect();
                (metric_label, baseline_val, param_list)
            })
        };

        let Some((metric_label, baseline_val, param_list)) = observable_setup else {
            return;
        };

        self.sensitivity_engine.baseline_output = baseline_val;
        let mut initial_sample = Some(baseline_val);
        self.sensitivity_engine.compute(&param_list, |p_map| {
            initial_sample.take().unwrap_or_else(|| {
                M::evaluate_observable(p_map)
                    .map(|(_, val)| val)
                    .unwrap_or(0.0)
            })
        });

        crate::sensitivity::render_labeled_sensitivity_panel_ui(
            ui,
            &self.sensitivity_engine,
            M::name(),
            metric_label,
        );
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

        ui.horizontal(|ui| {
            ui.label("Status:");
            let (status_text, color) = match state {
                SimWorkerState::Uninitialized => ("Uninitialized", egui::Color32::GRAY),
                SimWorkerState::Ready => ("Ready", egui::Color32::from_rgb(80, 200, 120)),
                SimWorkerState::Running => ("Running", egui::Color32::from_rgb(60, 160, 240)),
                SimWorkerState::Stepping => ("Stepping", egui::Color32::from_rgb(240, 200, 60)),
                SimWorkerState::Faulted => ("Faulted", egui::Color32::RED),
                SimWorkerState::Terminated => ("Terminated", egui::Color32::DARK_GRAY),
            };
            ui.colored_label(color, status_text);
        });
        ui.separator();

        let state = self.controller.state();
        let run_pause_enabled = matches!(state, SimWorkerState::Ready | SimWorkerState::Running);
        let button_text = if state.is_running() {
            "⏸ Pause"
        } else {
            "▶ Run"
        };
        let tooltip_text = match state {
            SimWorkerState::Running => "Pause simulation",
            SimWorkerState::Ready => "Start simulation",
            SimWorkerState::Faulted => "Worker thread faulted. Click Reset to restore simulation.",
            _ => "Simulation worker unavailable",
        };

        if ui
            .add_enabled(run_pause_enabled, egui::Button::new(button_text))
            .accessible_hover_text(tooltip_text)
            .clicked()
        {
            let cmd = if state.is_running() {
                SimCommand::Pause
            } else {
                SimCommand::Start
            };
            let _ = self.controller.send_command(cmd);
        }

        if ui
            .button("↻ Reset")
            .accessible_hover_text("Reset simulation state")
            .clicked()
        {
            self.reset_simulation(Some(ctx));
        }
    }

    fn draw_left_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left(format!("{}_controls", M::name())).show(ctx, |ui| {
            ui.heading(M::name());
            ui.separator();
            self.draw_status_and_transport(ui, ctx);
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
            self.draw_model_parameters(ui, ctx);
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

    fn save_state(&self) -> Option<String> {
        let params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let sorted: std::collections::BTreeMap<&String, &f64> = params.iter().collect();
        serde_json::to_string(&sorted).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let loaded = serde_json::from_str::<HashMap<String, f64>>(state)
            .map_err(|e| format!("Invalid parameter map JSON: {}", e))?;
        if !self.param_metadata.is_empty() && loaded.is_empty() {
            return Err(format!("Parameter map cannot be empty for {}", M::name()));
        }
        for (key, &val) in &loaded {
            if !self.param_metadata.iter().any(|(n, _)| n == key) {
                return Err(format!("Unknown parameter '{}' for {}", key, M::name()));
            }
            if !val.is_finite() {
                return Err(format!("Parameter '{}' value {} must be finite", key, val));
            }
        }
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            for (k, v) in loaded {
                p.insert(k, v);
            }
        }
        self.texture = None;
        self.reset_simulation(None);
        Ok(())
    }

    fn clamp_parameters(&mut self) -> bool {
        self.handle_recovery_action(
            crate::framework::recovery::RecoveryAction::ClampParameters,
            None,
        );
        true
    }

    fn show(&mut self, ctx: &egui::Context) {
        let has_update = self.controller.update().is_some();
        if has_update {
            self.dirty = true;
        }
        if has_update || self.controller.state().is_running() {
            ctx.request_repaint();
        }

        self.draw_left_panel(ctx);
        self.draw_central_panel(ctx);
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
