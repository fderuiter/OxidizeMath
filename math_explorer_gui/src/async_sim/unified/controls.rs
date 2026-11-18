#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{UnifiedModel, UnifiedSimTool};
use crate::accessibility::AccessibleHoverText;
use crate::async_sim::SimCommand;
use eframe::egui;
use scientific_metadata::theory::ParameterConstraint;

impl<M: UnifiedModel> UnifiedSimTool<M> {
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

    pub(super) fn draw_model_parameters(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.separator();
        ui.label("Model Parameters");

        let mut reset_needed = self.draw_presets_and_actions(ui, ctx);
        let mut params_lock = self.params.write().unwrap_or_else(|e| e.into_inner());

        for (name, constraint) in &self.param_metadata {
            if let Some(val) = params_lock.get_mut(name) {
                reset_needed |= Self::draw_single_parameter(
                    ui,
                    name,
                    constraint,
                    val,
                    self.cached_descs.get(name),
                );
            }
        }

        drop(params_lock);
        self.draw_observable_sensitivity(ui);
        if reset_needed {
            self.reset_simulation(Some(ctx));
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
            let slider = egui::Slider::new(val, constraint.min..=constraint.max)
                .step_by(constraint.step)
                .text(label);
            let resp = ui.add(slider);
            if let Some(desc) = hover_desc {
                let _ = resp.accessible_hover_text(desc);
            }
            false
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

        // Decouple internal parameter keys used by evaluate_observable from display labels in UI cards
        for (metric, (param_id, _)) in self.sensitivity_engine.metrics.iter_mut().zip(&param_list) {
            metric.param_name = M::parameter_label(param_id);
        }

        crate::sensitivity::render_labeled_sensitivity_panel_ui(
            ui,
            &self.sensitivity_engine,
            M::name(),
            metric_label,
        );
    }
}
