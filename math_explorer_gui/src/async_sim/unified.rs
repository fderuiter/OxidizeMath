use crate::accessibility::AccessibleHoverText;
use crate::async_sim::{SimCommand, SimulationController, SimulationRunner, StateSnapshot};
use crate::framework::InteractiveTool;
use eframe::egui;
use egui::{ColorImage, TextureOptions};
use egui_plot::{Line, Plot, PlotImage, PlotPoint, PlotPoints};
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
}

#[allow(missing_docs)]
pub struct UnifiedSimRunner<M: UnifiedModel> {
    model: M,
    params: Arc<RwLock<HashMap<String, f64>>>,
    steps_per_frame: usize,
}

impl<M: UnifiedModel> UnifiedSimRunner<M> {
    #[allow(missing_docs)]
    pub fn new(params: Arc<RwLock<HashMap<String, f64>>>) -> Self {
        let initial_params = params.read().unwrap().clone();
        Self {
            model: M::new(&initial_params),
            params,
            steps_per_frame: 5,
        }
    }
}

impl<M: UnifiedModel> SimulationRunner for UnifiedSimRunner<M> {
    fn process_command(&mut self, cmd: SimCommand) {
        let current_params = self.params.read().unwrap().clone();
        match cmd {
            SimCommand::SetSpeed(speed) => self.steps_per_frame = speed,
            SimCommand::Reset => {
                self.model = M::new(&current_params);
            }
            _ => self.model.process_command(cmd, &current_params),
        }
    }

    fn step(&mut self) {
        let current_params = self.params.read().unwrap().clone();
        self.model.step(&current_params);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        self.model.get_snapshot()
    }

    fn get_steps_per_frame(&self) -> usize {
        self.steps_per_frame
    }
}

#[allow(missing_docs)]
pub struct UnifiedSimTool<M: UnifiedModel> {
    controller: SimulationController,
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
        let controller = SimulationController::new(runner);

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

    fn draw_model_parameters(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Model Parameters");

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
                        self.controller.send_command(SimCommand::Reset);
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
                        self.controller
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
    }

    fn draw_left_panel(&mut self, ctx: &egui::Context) {
        let is_running = self.controller.running;

        egui::SidePanel::left(format!("{}_controls", M::name())).show(ctx, |ui| {
            ui.heading(M::name());
            ui.separator();

            if ui
                .button(if is_running { "⏸ Pause" } else { "▶ Run" })
                .accessible_hover_text(if is_running {
                    "Pause simulation"
                } else {
                    "Start simulation"
                })
                .clicked()
            {
                if is_running {
                    self.controller.send_command(SimCommand::Pause);
                } else {
                    self.controller.send_command(SimCommand::Start);
                }
            }

            if ui
                .button("↻ Reset")
                .accessible_hover_text("Reset simulation state")
                .clicked()
            {
                self.controller.send_command(SimCommand::Reset);
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
                self.controller
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
        let width = texture.size()[0] as f32;
        let height = texture.size()[1] as f32;

        Plot::new(format!("{}_plot_img", M::name()))
            .data_aspect(1.0)
            .view_aspect(1.0)
            .show_grid(false)
            .show_axes([false, false])
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .show(ui, |plot_ui| {
                plot_ui.image(PlotImage::new(
                    format!("{}_image", M::name()),
                    texture.id(),
                    PlotPoint::new(width as f64 / 2.0, height as f64 / 2.0),
                    [width, height],
                ));

                if plot_ui.response().hovered()
                    && ctx.input(|i| i.pointer.primary_down() || i.pointer.secondary_down())
                {
                    if let Some(pos) = plot_ui.pointer_coordinate() {
                        let grid_x = pos.x.round() as i32;
                        let grid_y = pos.y.round() as i32;
                        self.controller.send_command(SimCommand::ApplyBrush {
                            cx: grid_x,
                            cy: grid_y,
                            r: 2,
                            is_obstacle: ctx.input(|i| i.pointer.primary_down()),
                        });
                    }
                }
            });
    }

    fn draw_line_plot(&self, ui: &mut egui::Ui, snapshot: &StateSnapshot) {
        if snapshot.custom_data.is_empty() {
            return;
        }
        let mut points = Vec::new();
        for i in (0..snapshot.custom_data.len()).step_by(2) {
            if i + 1 < snapshot.custom_data.len() {
                points.push([snapshot.custom_data[i], snapshot.custom_data[i + 1]]);
            }
        }
        let line = Line::new(M::name(), PlotPoints::from_iter(points))
            .color(egui::Color32::from_rgb(100, 200, 255));

        Plot::new(format!("{}_plot_line", M::name()))
            .view_aspect(2.0)
            .show(ui, |plot_ui| {
                plot_ui.line(line);
            });
    }

    fn draw_central_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
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

    fn show(&mut self, ctx: &egui::Context) {
        let has_update = self.controller.update().is_some();
        if has_update || self.controller.running {
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
