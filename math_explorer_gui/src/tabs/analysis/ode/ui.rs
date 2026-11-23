#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::systems::eval_ode_final_state;
use super::OdeSolverTool;
use crate::framework::TraceSeries;
use crate::presets::OdePreset;
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use math_explorer::pure_math::analysis::ode::IntegrationMethod;

impl OdeSolverTool {
    pub(super) fn render_sidebar_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= self.render_preset_selector(ui);
        changed |= self.render_parameter_inputs(ui);
        changed |= self.render_initial_condition_inputs(ui);
        changed |= self.render_solver_settings_controls(ui);
        self.render_snapshots_and_sensitivity(ui);
        if self.callout_manager.render_ui(ui) {
            changed = true;
        }
        changed
    }

    fn render_preset_selector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("ODE System");
        ui.separator();

        egui::ComboBox::from_id_salt("ode_preset")
            .selected_text(self.preset.name())
            .show_ui(ui, |ui| {
                for candidate in [
                    OdePreset::Exponential,
                    OdePreset::HarmonicOscillator,
                    OdePreset::LogisticGrowth,
                    OdePreset::Custom,
                ] {
                    if ui
                        .selectable_value(&mut self.preset, candidate, candidate.name())
                        .changed()
                    {
                        changed = true;
                    }
                }
            });

        if self.preset == OdePreset::Custom {
            changed |= self.render_custom_ode_editor(ui);
        } else {
            let (latex_eq, plain_eq) = match self.preset {
                OdePreset::Exponential => (r"\frac{dy}{dt} = k y", "dy/dt = k*y"),
                OdePreset::HarmonicOscillator => (r"\frac{d^2 y}{dt^2} = -k y", "d^2y/dt^2 = -k*y"),
                OdePreset::LogisticGrowth => (
                    r"\frac{dy}{dt} = r y (1 - \frac{y}{K})",
                    "dy/dt = r*y*(1 - y/K)",
                ),
                OdePreset::Custom => unreachable!(),
            };
            crate::reflective_ui::render_formula_with_export(ui, latex_eq, Some(plain_eq));
        }

        changed
    }

    fn render_custom_ode_editor(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("Dimension:");
            if ui
                .selectable_value(&mut self.custom_coupled, false, "Scalar y'(t, y)")
                .changed()
                || ui
                    .selectable_value(&mut self.custom_coupled, true, "Coupled (y', v')")
                    .changed()
            {
                changed = true;
            }
        });
        ui.add_space(4.0);
        if self.custom_coupled {
            changed |= self.coupled_bank.ui_inputs(ui, "ode_coupled_bank");
        } else if self.math_input.ui(ui) {
            self.sync_legacy_params_from_math_input();
            changed = true;
        }
        changed
    }

    fn render_parameter_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        if self.preset == OdePreset::Custom && !self.custom_coupled {
            return false;
        }
        let mut changed = false;
        ui.add_space(10.0);
        ui.heading("Parameters");
        match self.preset {
            OdePreset::Exponential => {
                changed |= ui
                    .add(
                        AnimatedSlider::new(&mut self.param_k, -5.0..=5.0)
                            .text("k (Rate / Parameter)")
                            .id_salt("ode_param_k_exp"),
                    )
                    .changed();
            }
            OdePreset::Custom => {
                changed |= self.coupled_bank.sliders_ui(ui, "ode_coupled_params");
            }
            OdePreset::HarmonicOscillator => {
                changed |= ui
                    .add(
                        AnimatedSlider::new(&mut self.param_k, 0.1..=10.0)
                            .text("k (Spring Constant)")
                            .id_salt("ode_param_k_osc"),
                    )
                    .changed();
            }
            OdePreset::LogisticGrowth => {
                changed |= ui
                    .add(
                        AnimatedSlider::new(&mut self.param_r, 0.0..=5.0)
                            .text("r (Growth Rate)")
                            .id_salt("ode_param_r"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        AnimatedSlider::new(&mut self.param_cap_k, 1.0..=100.0)
                            .text("K (Carrying Capacity)")
                            .id_salt("ode_param_cap_k"),
                    )
                    .changed();
            }
        }
        changed
    }

    fn render_initial_condition_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.add_space(10.0);
        ui.heading("Initial Conditions");
        ui.horizontal(|ui| {
            ui.label("y(0):");
            if ui
                .add(egui::DragValue::new(&mut self.ic_y0).speed(0.1))
                .changed()
            {
                changed = true;
            }
        });
        if self.has_second_state() {
            let label = if self.preset == OdePreset::HarmonicOscillator {
                "y'(0):"
            } else {
                "v(0):"
            };
            ui.horizontal(|ui| {
                ui.label(label);
                if ui
                    .add(egui::DragValue::new(&mut self.ic_v0).speed(0.1))
                    .changed()
                {
                    changed = true;
                }
            });
        }
        changed
    }

    fn render_solver_settings_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.add_space(10.0);
        ui.heading("Solver Settings");
        ui.horizontal(|ui| {
            ui.label("Solver Algorithm:");
            egui::ComboBox::from_id_salt("ode_solver_method")
                .selected_text(self.method.name())
                .show_ui(ui, |ui| {
                    for m in IntegrationMethod::ALL {
                        if ui.selectable_value(&mut self.method, m, m.name()).changed() {
                            changed = true;
                        }
                    }
                });
        });
        if ui
            .checkbox(&mut self.comparison_mode, "Multi-Solver Comparison")
            .changed()
        {
            changed = true;
        }

        let min_dt = (self.total_time.max(0.1) / crate::presets::MAX_ODE_STEPS as f64).max(1e-6);
        ui.horizontal(|ui| {
            ui.label("dt (Step Size):");
            if ui
                .add(
                    egui::DragValue::new(&mut self.dt)
                        .speed(0.01)
                        .range(min_dt..=1.0),
                )
                .changed()
            {
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Total Time:");
            if ui
                .add(
                    egui::DragValue::new(&mut self.total_time)
                        .speed(1.0)
                        .range(1.0..=100.0),
                )
                .changed()
            {
                changed = true;
            }
        });

        ui.add_space(5.0);
        if ui
            .checkbox(&mut self.show_annotations, "Show Annotations")
            .changed()
        {
            changed = true;
        }

        if self.comparison_mode {
            self.render_discrepancy_panel(ui);
        }
        changed
    }

    fn render_snapshots_and_sensitivity(&mut self, ui: &mut egui::Ui) {
        ui.add_space(10.0);
        let param_summary = self.param_summary();
        let has_v = self.has_second_state();
        let v_label = self.second_state_label();
        let current_len = self.snapshot_manager.len();
        let time_series = &self.time_series;
        let y_series = &self.y_series;
        let v_series = &self.v_series;

        if self.snapshot_manager.render_ui(ui, || {
            let label = format!("Baseline {}", current_len + 1);
            let py: Vec<[f64; 2]> = time_series
                .iter()
                .zip(y_series)
                .map(|(&t, &y)| [t, y])
                .collect();
            let mut series = vec![TraceSeries::new("y(t)", py)];
            if has_v && !v_series.is_empty() {
                let pv: Vec<[f64; 2]> = time_series
                    .iter()
                    .zip(v_series)
                    .map(|(&t, &v)| [t, v])
                    .collect();
                series.push(TraceSeries::new(v_label, pv));
            }
            Some((label, param_summary, series))
        }) {
            self.bump_revision();
        }

        self.run_sensitivity_analysis();

        crate::sensitivity::render_sensitivity_panel_ui(ui, &self.sensitivity_engine, "ode_tool");
    }

    fn collect_sensitivity_params(&self) -> Vec<(String, f64)> {
        let mut dynamic_params: Vec<(String, f64)> = Vec::with_capacity(8);
        match self.preset {
            OdePreset::Exponential => {
                dynamic_params.push(("k".to_string(), self.param_k));
                dynamic_params.push(("y0".to_string(), self.ic_y0));
            }
            OdePreset::HarmonicOscillator => {
                dynamic_params.push(("k".to_string(), self.param_k));
                dynamic_params.push(("y0".to_string(), self.ic_y0));
                dynamic_params.push(("v0".to_string(), self.ic_v0));
            }
            OdePreset::LogisticGrowth => {
                dynamic_params.push(("r".to_string(), self.param_r));
                dynamic_params.push(("cap_k".to_string(), self.param_cap_k));
                dynamic_params.push(("y0".to_string(), self.ic_y0));
            }
            OdePreset::Custom if self.custom_coupled => {
                for (name, &val) in self
                    .coupled_bank
                    .param_names()
                    .iter()
                    .zip(self.coupled_bank.param_slice())
                {
                    dynamic_params.push((name.clone(), val));
                }
                if !dynamic_params.iter().any(|(name, _)| name == "y0") {
                    dynamic_params.push(("y0".to_string(), self.ic_y0));
                }
                if !dynamic_params.iter().any(|(name, _)| name == "v0") {
                    dynamic_params.push(("v0".to_string(), self.ic_v0));
                }
            }
            OdePreset::Custom => {
                if let Some(compiled) = self.math_input.active_compiled() {
                    for (name, &val) in compiled
                        .param_names()
                        .iter()
                        .zip(self.math_input.param_slice())
                    {
                        dynamic_params.push((name.clone(), val));
                    }
                }
                if !dynamic_params.iter().any(|(name, _)| name == "y0") {
                    dynamic_params.push(("y0".to_string(), self.ic_y0));
                }
            }
        }
        dynamic_params
    }

    fn run_sensitivity_analysis(&mut self) {
        use crate::sensitivity::canonical_f64_bits;

        let dynamic_params = self.collect_sensitivity_params();
        let param_refs: Vec<(&str, f64)> = dynamic_params
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();

        let current_preset = self.preset;
        let current_method = self.method;
        let current_dt = self.dt;
        let current_total_time = self.total_time;
        let custom_spec = self.build_custom_ode_spec();
        let custom_ctx = (current_preset == OdePreset::Custom)
            .then_some(super::systems::CustomOdeContext(custom_spec.as_ref()));
        let context = (
            current_preset as u8,
            current_method as u8,
            canonical_f64_bits(current_dt),
            canonical_f64_bits(current_total_time),
            custom_ctx,
        );

        self.sensitivity_engine
            .compute_with_context(&param_refs, &context, |p_map| {
                eval_ode_final_state(
                    current_preset,
                    current_method,
                    current_dt,
                    current_total_time,
                    p_map,
                    custom_spec.as_ref(),
                )
            });
    }
}
