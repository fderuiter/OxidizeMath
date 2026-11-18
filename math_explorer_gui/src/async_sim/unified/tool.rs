#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{UnifiedModel, UnifiedSimRunner};
use crate::accessibility::AccessibleHoverText;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState, StateSnapshot};
use crate::framework::InteractiveTool;
use eframe::egui;
use egui::{ColorImage, TextureOptions};
use egui_plot::{Line, Plot, PlotImage, PlotPoint, PlotPoints};
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[allow(missing_docs)]
pub struct UnifiedSimTool<M: UnifiedModel> {
    controller: BackgroundWorker<UnifiedSimRunner<M>>,
    params: Arc<RwLock<HashMap<String, f64>>>,
    param_metadata: Vec<(String, ParameterConstraint)>,
    steps_per_frame: usize,
    fallback_snapshot: StateSnapshot,
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
        let fallback_snapshot = temp_model.get_snapshot();

        Self {
            controller,
            params,
            param_metadata,
            steps_per_frame: 5,
            fallback_snapshot,
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

    fn active_snapshot(&self) -> &StateSnapshot {
        self.controller
            .latest_snapshot()
            .unwrap_or(&self.fallback_snapshot)
    }

    fn reset_simulation(&mut self) {
        if let Ok(params) = self.params.read() {
            self.fallback_snapshot = M::new(&params).get_snapshot();
        }
        if self.controller.state() == SimWorkerState::Faulted
            || self.controller.state() == SimWorkerState::Terminated
        {
            let runner = UnifiedSimRunner::<M>::new(Arc::clone(&self.params));
            self.controller = BackgroundWorker::new(runner);
        } else {
            let _ = self.controller.send_command(SimCommand::Reset);
        }
    }

    fn handle_recovery_action(&mut self, action: crate::framework::recovery::RecoveryAction) {
        use crate::framework::recovery::RecoveryAction;
        match action {
            RecoveryAction::AutoScaleStepSize => {
                let mut p = self.params.write().unwrap();
                if let Some(dt) = p.get_mut("dt") {
                    *dt = (*dt * 0.5).max(1e-4);
                } else if let Some(visc) = p.get_mut("viscosity") {
                    *visc = (*visc * 0.5).max(0.00333);
                }
                drop(p);
                self.reset_simulation();
            }
            RecoveryAction::ClampParameters => {
                let mut p = self.params.write().unwrap();
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
                self.reset_simulation();
            }
            RecoveryAction::ResetParameters => {
                let mut p = self.params.write().unwrap();
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
                self.reset_simulation();
                let _ = self.controller.send_command(SimCommand::Start);
            }
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
            self.reset_simulation();
        }
    }

    fn draw_observable_sensitivity(&mut self, ui: &mut egui::Ui) {
        let observable_setup = {
            let current_params = self.params.read().unwrap();
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

    #[allow(clippy::too_many_lines)]
    fn draw_left_panel(&mut self, ctx: &egui::Context) {
        let state = self.controller.state();

        egui::SidePanel::left(format!("{}_controls", M::name())).show(ctx, |ui| {
            ui.heading(M::name());
            ui.separator();

            if state == SimWorkerState::Faulted || self.controller.diagnostic_report().is_some() {
                if let Some(action) = crate::framework::recovery::render_recovery_banner(
                    ui,
                    self.controller.diagnostic_report(),
                ) {
                    self.handle_recovery_action(action);
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

            let run_pause_enabled =
                matches!(state, SimWorkerState::Ready | SimWorkerState::Running);
            let button_text = if state.is_running() {
                "⏸ Pause"
            } else {
                "▶ Run"
            };
            let tooltip_text = match state {
                SimWorkerState::Running => "Pause simulation",
                SimWorkerState::Ready => "Start simulation",
                SimWorkerState::Faulted => {
                    "Worker thread faulted. Click Reset to restore simulation."
                }
                _ => "Simulation worker unavailable",
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
                        let _ = self.controller.send_command(SimCommand::ApplyBrush {
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
            let snap_ref = self.active_snapshot();
            if M::custom_central_panel(ui, Some(snap_ref)) {
                return;
            }

            let mut loaded_image = None;
            if let Ok(guard) = snap_ref.pixels.try_read() {
                if !guard.is_empty() && snap_ref.width > 0 && snap_ref.height > 0 {
                    loaded_image = Some(ColorImage::new(
                        [snap_ref.width, snap_ref.height],
                        guard.clone(),
                    ));
                }
            }

            if let Some(image) = loaded_image {
                let texture = ctx.load_texture(M::name(), image, TextureOptions::NEAREST);
                self.texture = Some(texture.clone());
                self.draw_image_plot(ui, ctx, &texture);
            } else if let Some(texture) = self.texture.clone() {
                self.draw_image_plot(ui, ctx, &texture);
            } else {
                let snapshot = self.active_snapshot();
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

    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        if M::is_raster_model() {
            crate::framework::ExportCapabilities::raster_only()
        } else {
            crate::framework::ExportCapabilities::plot()
        }
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        let snapshot = self.active_snapshot();
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
            for _ in 0..self.steps_per_frame.max(1) {
                model.step(&params);
            }
            self.fallback_snapshot = model.get_snapshot();
        }
    }

    fn save_state(&self) -> Option<String> {
        let params = self.params.read().ok()?.clone();
        serde_json::to_string(&params).ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(params) = serde_json::from_str::<HashMap<String, f64>>(state) {
            self.fallback_snapshot = M::new(&params).get_snapshot();
            if let Ok(mut p) = self.params.write() {
                *p = params;
            }
            let _ = self.controller.send_command(SimCommand::Reset);
        }
    }

    fn clamp_parameters(&mut self) -> bool {
        self.handle_recovery_action(crate::framework::recovery::RecoveryAction::ClampParameters);
        true
    }

    fn show(&mut self, ctx: &egui::Context) {
        if self.controller.update().is_some() || self.controller.state().is_running() {
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
