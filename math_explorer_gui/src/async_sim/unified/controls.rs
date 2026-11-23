#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::persistence::build_formula_bar;
use super::{UnifiedModel, UnifiedSimTool};
use crate::accessibility::AccessibleHoverText;
use crate::async_sim::ui::{render_worker_fault_banner, WorkerFaultAction};
use crate::async_sim::{SimCommand, SimWorkerState};
use crate::framework::{PlaybackAction, PlaybackTransport};
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use scientific_metadata::theory::ParameterConstraint;

impl<M: UnifiedModel> UnifiedSimTool<M> {
    pub(super) fn handle_recovery_action(
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

    pub(super) fn draw_left_panel(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let panel_id = crate::framework::scoped_panel_id(ui, format!("{}_controls", M::name()));
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading(M::name());
            ui.separator();
            self.draw_status_and_transport(ui, &ctx);
            if let Some(tool_id) = M::sonification_tool_id() {
                let worker_state = self.controller.state();
                if matches!(worker_state, SimWorkerState::Faulted) {
                    crate::audio::notify_solver_diverged();
                }
                let frame = self
                    .stepped_snapshot
                    .as_ref()
                    .or_else(|| self.controller.latest_snapshot())
                    .map_or_else(crate::audio::SonificationFrame::silent, |s| s.sonification);
                let is_idle = !matches!(
                    worker_state,
                    SimWorkerState::Running | SimWorkerState::Stepping
                ) && self.stepped_snapshot.is_none();
                ui.separator();
                crate::audio::render_sonification_panel(ui, tool_id, &frame, is_idle);
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
            self.draw_model_parameters(ui, &ctx);
        });
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
                        if action.contains("Clear") {
                            self.sculpt_state
                                .write()
                                .unwrap_or_else(|e| e.into_inner())
                                .clear();
                        }
                        if let Some(ext) = &mut self.ui_extension {
                            ext.on_custom_action(action);
                        }
                        self.dirty = true;
                        ctx.request_repaint();
                    }
                }
            });
            ui.separator();
        }
        reset_needed
    }

    pub(super) fn draw_model_parameters(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.separator();
        ui.label("Model Parameters");

        let mut reset_needed = self.draw_presets_and_actions(ui, ctx);
        let ext_cmds = if let Some(ext) = &mut self.ui_extension {
            ext.draw_side_panel_controls(ui, self.controller.latest_snapshot())
        } else {
            Vec::new()
        };
        if !ext_cmds.is_empty() {
            self.dispatch_extension_commands(ext_cmds, Some(ctx));
        }

        let (param_reset, mut slider_changed) = self.draw_parameter_entries(ui);
        reset_needed |= param_reset;
        let (formula_reset, formula_changed) = self.draw_custom_formula_controls(ui, ctx);
        reset_needed |= formula_reset;
        slider_changed |= formula_changed;
        self.draw_spatial_brush_section(ui, ctx);
        self.draw_atlas_toggle_if_supported(ui);
        self.draw_observable_sensitivity(ui);
        if reset_needed {
            self.reset_simulation(Some(ctx));
        } else if slider_changed {
            self.refresh_paused_simulation(Some(ctx));
        }
    }

    fn draw_parameter_entries(&self, ui: &mut egui::Ui) -> (bool, bool) {
        let mut reset_needed = false;
        let mut slider_changed = false;
        let mut params_lock = self.params.write().unwrap_or_else(|e| e.into_inner());
        for (name, constraint) in &self.param_metadata {
            let locked = self
                .ui_extension
                .as_ref()
                .is_some_and(|ext| ext.is_parameter_locked(name));
            if let Some(val) = params_lock.get_mut(name) {
                let is_choice = !M::parameter_choices(name).is_empty();
                let desc = self.cached_descs.get(name);
                let mut changed = false;
                ui.add_enabled_ui(!locked, |ui| {
                    changed = Self::draw_single_parameter(ui, name, constraint, val, desc);
                });
                if is_choice {
                    reset_needed |= changed;
                } else {
                    slider_changed |= changed;
                }
            }
        }
        (reset_needed, slider_changed)
    }

    fn draw_custom_formula_controls(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
    ) -> (bool, bool) {
        let Some(spec) = M::formula_spec() else {
            return (false, false);
        };
        let mut reset_needed = false;
        let mut formula_changed = false;
        ui.add_space(4.0);
        let toggle_label = format!("Custom {} Formula", spec.lhs_label);
        if ui
            .checkbox(&mut self.custom_formula_enabled, toggle_label)
            .changed()
        {
            self.publish_formula_update();
            reset_needed = true;
        }
        if self.custom_formula_enabled {
            if let Some(ref mut bar) = self.formula_bar {
                ui.add_space(4.0);
                formula_changed = bar.ui(ui);
            }
            if formula_changed {
                self.publish_formula_update();
                self.dirty = true;
                ctx.request_repaint();
            }
        }
        (reset_needed, formula_changed)
    }

    fn draw_spatial_brush_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if !M::supports_spatial_brush() {
            return;
        }
        ui.separator();
        let cmds = M::draw_brush_controls(
            ui,
            &mut self.brush_kit,
            &self.params,
            &self.sculpt_state,
            self.controller.latest_snapshot(),
        );
        for cmd in cmds {
            if self.controller.send_command(cmd).is_ok() {
                self.dirty = true;
                ctx.request_repaint();
            }
        }
    }
    fn draw_single_parameter(
        ui: &mut egui::Ui,
        name: &str,
        constraint: &ParameterConstraint,
        val: &mut f64,
        hover_desc: Option<&String>,
    ) -> bool {
        let label = M::parameter_label(name);
        let choices = M::parameter_choices(name);
        if !choices.is_empty() {
            let current_label = choices
                .iter()
                .min_by(|a, b| {
                    (a.1 - *val)
                        .abs()
                        .partial_cmp(&(b.1 - *val).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map_or(label.as_str(), |(l, _)| *l);
            let mut changed = false;
            let combo_resp = egui::ComboBox::from_label(&label)
                .selected_text(current_label)
                .show_ui(ui, |ui| {
                    for &(choice_label, choice_val) in choices {
                        let selected = (*val - choice_val).abs() < 1e-6;
                        if ui.selectable_label(selected, choice_label).clicked() && !selected {
                            *val = choice_val;
                            changed = true;
                        }
                    }
                });
            if let Some(desc) = hover_desc {
                let _ = combo_resp.response.accessible_hover_text(desc);
            }
            changed
        } else {
            let slider = AnimatedSlider::new(val, constraint.min..=constraint.max)
                .hard_bounds(constraint.min..=constraint.max)
                .step_by(constraint.step)
                .text(&label)
                .id_salt(("unified_sim_param", M::name(), name));
            let resp = ui.add(slider);
            let changed = resp.changed();
            let resp = if let Some(desc) = hover_desc {
                resp.accessible_hover_text(desc)
            } else {
                resp
            };
            changed || resp.changed()
        }
    }

    fn draw_observable_sensitivity(&mut self, ui: &mut egui::Ui) {
        const STACK_CAP: usize = 16;
        let mut stack_pairs: [(&str, f64); STACK_CAP] = [("", 0.0); STACK_CAP];
        let mut heap_pairs: Vec<(&str, f64)> = Vec::new();
        let param_slice: &[(&str, f64)] =
            {
                let current_params = self.params.read().unwrap_or_else(|e| e.into_inner());
                if self.param_metadata.len() <= STACK_CAP {
                    let mut len = 0;
                    for (name, _) in &self.param_metadata {
                        if let Some(&val) = current_params.get(name) {
                            stack_pairs[len] = (name.as_str(), val);
                            len += 1;
                        }
                    }
                    &stack_pairs[..len]
                } else {
                    heap_pairs.extend(self.param_metadata.iter().filter_map(|(name, _)| {
                        current_params.get(name).map(|&v| (name.as_str(), v))
                    }));
                    &heap_pairs
                }
            };

        if let Some(metric_label) = Self::update_observable_sensitivity_engine(
            &mut self.sensitivity_engine,
            &mut self.cached_observable_label,
            &self.cached_param_labels,
            param_slice,
        ) {
            crate::sensitivity::render_labeled_sensitivity_panel_ui(
                ui,
                &self.sensitivity_engine,
                M::name(),
                metric_label,
            );
        }
    }

    fn update_observable_sensitivity_engine(
        engine: &mut crate::sensitivity::SensitivityEngine,
        cached_observable_label: &mut Option<&'static str>,
        cached_param_labels: &[String],
        param_slice: &[(&str, f64)],
    ) -> Option<&'static str> {
        let is_hit = engine.matches_cache(param_slice, &());
        let was_converged = is_hit && engine.is_converged();
        let mut observed_label = if is_hit {
            *cached_observable_label
        } else {
            None
        };

        engine.compute(param_slice, |p_map| {
            if let Some((label, val)) = M::evaluate_observable(p_map) {
                observed_label = Some(label);
                val
            } else {
                0.0
            }
        });

        if !was_converged {
            *cached_observable_label = observed_label;
            if observed_label.is_some() {
                for (metric, label) in engine.metrics.iter_mut().zip(cached_param_labels) {
                    if metric.param_name != *label {
                        metric.param_name.clone_from(label);
                    }
                }
            }
        }

        *cached_observable_label
    }

    fn refresh_paused_simulation(&mut self, ctx: Option<&egui::Context>) {
        if self.controller.state().is_ready() {
            self.stepped_snapshot = None;
            let _ = self.controller.send_command(SimCommand::Custom(
                super::SYNC_PARAMETERS_COMMAND.to_string(),
            ));
            self.dirty = true;
            if let Some(c) = ctx {
                c.request_repaint();
            }
        }
    }
}
