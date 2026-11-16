use crate::accessibility::AccessibleHoverText;
use crate::async_sim::{
    BackgroundWorker, SimCommand, SimWorkerState, StateSnapshot,
};
use crate::framework::InteractiveTool;
use eframe::egui;
use egui::{ColorImage, TextureOptions};
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[allow(missing_docs)]
pub trait UnifiedModel: Send + 'static + scientific_metadata::theory::TheoryDescribable {
    /// Initialize the model given the starting parameters.
    fn new(params: &HashMap<String, f64>) -> Self
    where
        Self: Sized;

    /// Step the simulation forward one tick.
    fn step(&mut self, params: &HashMap<String, f64>);

    /// Extract a visual snapshot of the state.
    fn get_snapshot(&self) -> StateSnapshot;

    /// Process any custom commands (e.g. ApplyBrush). Default is no-op.
    fn process_command(&mut self, _cmd: SimCommand, _params: &HashMap<String, f64>) {}

    /// Return parameter definitions for automatic UI generation.
    fn parameters() -> HashMap<String, ParameterConstraint>
    where
        Self: Sized;

    /// The name of the tool.
    fn name() -> &'static str
    where
        Self: Sized;

    /// Return the domain category of the model.
    fn domain() -> &'static str
    where
        Self: Sized,
    {
        ""
    }

    /// Return presets that the model supports.
    fn presets() -> Vec<(&'static str, HashMap<String, f64>)>
    where
        Self: Sized,
    {
        Vec::new()
    }

    /// Return custom actions that the model supports.
    fn custom_actions() -> Vec<&'static str>
    where
        Self: Sized,
    {
        Vec::new()
    }

    /// Allow the model to draw custom visualization in the central panel.
    /// Return true if custom rendering was performed.
    fn custom_central_panel(_ui: &mut eframe::egui::Ui, _snapshot: Option<&StateSnapshot>) -> bool
    where
        Self: Sized,
    {
        false
    }

    /// Returns true if the model has diverged numerically.
    fn diverged(&self) -> bool {
        false
    }

    /// Returns the diagnostic report if diverged.
    fn diagnostic_report(&self) -> Option<crate::async_sim::recovery::DiagnosticReport> {
        None
    }

    /// Default parameters for baseline configuration.
    fn default_parameters() -> HashMap<String, f64>
    where
        Self: Sized,
    {
        Self::parameters()
            .into_iter()
            .map(|(n, c)| (n, (c.min + c.max) / 2.0))
            .collect()
    }

    /// Clamp parameters to stable bounds.
    fn clamp_parameters(params: &mut HashMap<String, f64>)
    where
        Self: Sized,
    {
        for (name, constraint) in Self::parameters() {
            if let Some(val) = params.get_mut(&name) {
                *val = if !val.is_finite() {
                    (constraint.min + constraint.max) / 2.0
                } else {
                    val.clamp(constraint.min, constraint.max)
                };
            }
        }
    }

    /// Telemetry tool source name.
    fn tool_source() -> &'static str
    where
        Self: Sized,
    {
        Self::name()
    }
}

pub use super::unified_runner::UnifiedSimRunner;

#[allow(missing_docs)]
pub struct UnifiedSimTool<M: UnifiedModel> {
    controller: BackgroundWorker<UnifiedSimRunner<M>>,
    params: Arc<RwLock<HashMap<String, f64>>>,
    param_metadata: Vec<(String, ParameterConstraint)>,
    steps_per_frame: usize,
    texture: Option<egui::TextureHandle>,
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
        param_metadata.sort_by(|a, b| a.0.cmp(&b.0)); // Sort for consistent UI order

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

    #[allow(missing_docs)]
    pub fn is_diverged(&self) -> bool {
        self.controller.is_diverged()
    }

    #[allow(missing_docs)]
    pub fn diagnostic_report(&self) -> Option<&crate::async_sim::recovery::DiagnosticReport> {
        self.controller.diagnostic_report()
    }

    fn reset_simulation(&mut self) {
        if self.controller.state() == SimWorkerState::Faulted
            || self.controller.state() == SimWorkerState::Terminated
        {
            let runner = UnifiedSimRunner::<M>::new(Arc::clone(&self.params));
            self.controller = BackgroundWorker::new(runner);
        } else {
            let _ = self.controller.send_command(SimCommand::Reset);
        }
    }

    fn draw_model_parameters(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Model Parameters");

        let mut reset_needed = false;
        let mut params_lock = self.params.write().unwrap();

        let presets = M::presets();
        if !presets.is_empty() {
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
            ui.separator();
        }

        let custom_actions = M::custom_actions();
        if !custom_actions.is_empty() {
            ui.horizontal(|ui| {
                ui.label("Actions:");
                for action in custom_actions {
                    if ui.button(action).clicked() {
                        let _ = self
                            .controller
                            .send_command(SimCommand::Custom(action.to_string()));
                    }
                }
            });
            ui.separator();
        }

        let param_list: Vec<(String, f64)> = self
            .param_metadata
            .iter()
            .filter_map(|(name, _)| params_lock.get(name).map(|v| (name.clone(), *v)))
            .collect();

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

        self.sensitivity_engine
            .compute(&param_list, |p_map| p_map.values().map(|v| v.abs()).sum());

        crate::sensitivity::render_sensitivity_panel_ui(ui, &self.sensitivity_engine, M::name());

        drop(params_lock);
        if reset_needed {
            self.reset_simulation();
        }
    }

    #[allow(clippy::too_many_lines)]
    fn draw_left_panel(&mut self, ctx: &egui::Context) {
        let state = self.controller.state();

        egui::SidePanel::left(format!("{}_controls", M::name())).show(ctx, |ui| {
            ui.heading(M::name());
            ui.separator();

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

            let run_pause_enabled =
                matches!(state, SimWorkerState::Ready | SimWorkerState::Running)
                    && !self.controller.is_diverged();
            let button_text = if state.is_running() {
                "⏸ Pause"
            } else {
                "▶ Run"
            };
            let tooltip_text = if self.controller.is_diverged() {
                "Simulation diverged. Trigger recovery or reset before resuming."
            } else {
                match state {
                    SimWorkerState::Running => "Pause simulation",
                    SimWorkerState::Ready => "Start simulation",
                    SimWorkerState::Faulted => {
                        "Worker thread faulted. Click Reset to restore simulation."
                    }
                    _ => "Simulation worker unavailable",
                }
            };

            if ui
                .add_enabled(run_pause_enabled, egui::Button::new(button_text))
                .accessible_hover_text(tooltip_text)
                .clicked()
            {
                if state.is_running() {
                    let _ = self.controller.send_command(SimCommand::Pause);
                } else {
                    let _ = self.controller.send_command(SimCommand::Start);
                }
            }

            if ui
                .button("↻ Reset")
                .accessible_hover_text("Reset simulation state")
                .clicked()
            {
                self.reset_simulation();
            }

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

            self.draw_model_parameters(ui);
        });
    }

    fn draw_image_plot(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        texture: &egui::TextureHandle,
    ) {
        crate::async_sim::unified_plot::draw_image_plot_helper(
            M::name(),
            ui,
            ctx,
            texture,
            |cmd| {
                let _ = self.controller.send_command(cmd);
            },
        );
    }

    fn draw_line_plot(&self, ui: &mut egui::Ui, snapshot: &StateSnapshot) {
        crate::async_sim::unified_plot::draw_line_plot_helper(M::name(), ui, snapshot);
    }

    fn draw_central_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(report) = self.controller.diagnostic_report().cloned() {
                if let Some(action) =
                    crate::async_sim::recovery::render_recovery_banner(ui, Some(&report))
                {
                    crate::async_sim::recovery::apply_recovery_action_helper::<M>(
                        action,
                        &self.params,
                        &mut self.controller,
                        &mut self.steps_per_frame,
                    );
                }
                ui.add_space(8.0);
            }

            if M::custom_central_panel(ui, self.controller.latest_snapshot()) {
                return;
            }

            let mut has_image = false;
            if let Some(snapshot) = self.controller.latest_snapshot() {
                if let Ok(guard) = snapshot.pixels.try_read() {
                    if !guard.is_empty() {
                        let image =
                            ColorImage::new([snapshot.width, snapshot.height], guard.clone());
                        let texture = ctx.load_texture(M::name(), image, TextureOptions::NEAREST);
                        self.texture = Some(texture);
                        has_image = true;
                    }
                }
            }

            if has_image || self.texture.is_some() {
                if let Some(texture) = self.texture.clone() {
                    self.draw_image_plot(ui, ctx, &texture);
                }
            } else if let Some(snapshot) = self.controller.latest_snapshot() {
                self.draw_line_plot(ui, snapshot);
            }
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
        let params = self.params.read().ok()?.clone();
        serde_json::to_string(&params).ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(params) = serde_json::from_str::<HashMap<String, f64>>(state) {
            if let Ok(mut p) = self.params.write() {
                *p = params;
            }
            let _ = self.controller.send_command(SimCommand::Reset);
        }
    }

    fn show(&mut self, ctx: &egui::Context) {
        let has_update = self.controller.update().is_some();
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
}
