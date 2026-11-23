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
                        if let Some(ext) = &mut self.ui_extension {
                            ext.on_custom_action(action);
                        }
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

        let mut params_lock = self.params.write().unwrap_or_else(|e| e.into_inner());

        for (name, constraint) in &self.param_metadata {
            let locked = self
                .ui_extension
                .as_ref()
                .is_some_and(|ext| ext.is_parameter_locked(name));
            if let Some(val) = params_lock.get_mut(name) {
                ui.add_enabled_ui(!locked, |ui| {
                    reset_needed |= Self::draw_single_parameter(
                        ui,
                        name,
                        constraint,
                        val,
                        self.cached_descs.get(name),
                    );
                });
            }
        }
        drop(params_lock);
        reset_needed |= self.draw_custom_formula_controls(ui, ctx);
        self.draw_atlas_toggle_if_supported(ui);
        self.draw_observable_sensitivity(ui);
        if reset_needed {
            self.reset_simulation(Some(ctx));
        }
    }

    fn draw_custom_formula_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) -> bool {
        let Some(spec) = M::formula_spec() else {
            return false;
        };
        let mut reset_needed = false;
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
            let mut formula_changed = false;
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
        reset_needed
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

    /// Advances one cooperative batch (`<= 8` model evaluations) of local finite-difference
    /// sensitivity, wide-range Tornado bounds, or `SplitMix64`-seeded Monte Carlo propagation.
    pub fn step_cooperative_sensitivity(&mut self) -> Option<&'static str> {
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

        let is_hit = self.sensitivity_engine.matches_cache(param_slice, &());
        if !is_hit || !self.sensitivity_engine.is_converged() {
            return Self::update_observable_sensitivity_engine(
                &mut self.sensitivity_engine,
                &mut self.uncertainty.cached_label,
                &self.cached_param_labels,
                param_slice,
            );
        }

        if self.uncertainty.enabled && self.uncertainty.cached_label.is_some() {
            Self::step_unified_uncertainty_engine(
                &mut self.sensitivity_engine,
                &self.param_metadata,
                param_slice,
                self.uncertainty.seed,
                self.uncertainty.target_samples,
            );
        }
        self.uncertainty.cached_label
    }

    fn step_unified_uncertainty_engine(
        engine: &mut crate::sensitivity::SensitivityEngine,
        param_metadata: &[(String, ParameterConstraint)],
        param_slice: &[(&str, f64)],
        seed: u64,
        target_samples: usize,
    ) {
        use crate::sensitivity::{
            ParameterUncertainty, TornadoBounds, MAX_SENSITIVITY_BATCH_EVALS,
        };
        if !engine.is_tornado_converged() {
            let bounds: Vec<TornadoBounds> = param_metadata
                .iter()
                .map(|(_, c)| TornadoBounds::new(c.min, c.max))
                .collect();
            engine.step_tornado(param_slice, &bounds, MAX_SENSITIVITY_BATCH_EVALS, |p_map| {
                M::evaluate_observable(p_map).map_or(0.0, |(_, v)| v)
            });
        } else if !engine.is_monte_carlo_converged() {
            let uncertainties: Vec<ParameterUncertainty> = param_metadata
                .iter()
                .map(|(_, c)| {
                    let sigma = ((c.max - c.min).abs() * 0.05).max(1.0e-6);
                    ParameterUncertainty::new(sigma, c.min, c.max)
                })
                .collect();
            engine.step_monte_carlo(
                param_slice,
                &uncertainties,
                seed,
                target_samples,
                MAX_SENSITIVITY_BATCH_EVALS,
                |p_map| M::evaluate_observable(p_map).map_or(0.0, |(_, v)| v),
            );
        }
    }

    fn draw_observable_sensitivity(&mut self, ui: &mut egui::Ui) {
        let Some(metric_label) = self.step_cooperative_sensitivity() else {
            return;
        };
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(
                &mut self.uncertainty.enabled,
                "Tornado & Monte Carlo Sweep",
            )
            .accessible_hover_text(
                "Cooperatively step wide-range Tornado bounds and SplitMix64 Monte Carlo quantiles (<= 8 evals/frame)",
            );
            if self.uncertainty.enabled {
                ui.label("Seed:");
                ui.add(egui::DragValue::new(&mut self.uncertainty.seed).speed(1.0))
                    .accessible_hover_text("Explicit u64 SplitMix64 PRNG seed");
            }
        });
        if self.uncertainty.enabled
            && (!self.sensitivity_engine.is_tornado_converged()
                || !self.sensitivity_engine.is_monte_carlo_converged())
        {
            ui.ctx().request_repaint();
        }
        crate::sensitivity::render_labeled_sensitivity_panel_ui(
            ui,
            &self.sensitivity_engine,
            M::name(),
            metric_label,
        );
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
}
