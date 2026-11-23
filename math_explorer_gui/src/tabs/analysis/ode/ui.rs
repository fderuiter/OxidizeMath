#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::recovery;
use super::systems::eval_ode_final_state;
use super::OdeSolverTool;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::TraceSeries;
use crate::presets::OdePreset;
use crate::reflective_ui::{render_copyable_metric, AnimatedSlider};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::pure_math::analysis::ode::IntegrationMethod;
use std::sync::Arc;

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
                if ui
                    .selectable_value(
                        &mut self.preset,
                        OdePreset::Exponential,
                        OdePreset::Exponential.name(),
                    )
                    .changed()
                    || ui
                        .selectable_value(
                            &mut self.preset,
                            OdePreset::HarmonicOscillator,
                            OdePreset::HarmonicOscillator.name(),
                        )
                        .changed()
                    || ui
                        .selectable_value(
                            &mut self.preset,
                            OdePreset::LogisticGrowth,
                            OdePreset::LogisticGrowth.name(),
                        )
                        .changed()
                    || ui
                        .selectable_value(
                            &mut self.preset,
                            OdePreset::Custom,
                            OdePreset::Custom.name(),
                        )
                        .changed()
                {
                    changed = true;
                }
            });

        if self.preset == OdePreset::Custom {
            ui.add_space(6.0);
            if self.math_input.ui(ui) {
                self.sync_legacy_params_from_math_input();
                changed = true;
            }
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

    fn render_parameter_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        if self.preset == OdePreset::Custom {
            return false;
        }
        let mut changed = false;
        ui.add_space(10.0);
        ui.heading("Parameters");
        match self.preset {
            OdePreset::Exponential => {
                if ui
                    .add(
                        AnimatedSlider::new(&mut self.param_k, -5.0..=5.0)
                            .text("k (Rate / Parameter)")
                            .id_salt("ode_param_k_exp"),
                    )
                    .changed()
                {
                    changed = true;
                }
            }
            OdePreset::HarmonicOscillator => {
                if ui
                    .add(
                        AnimatedSlider::new(&mut self.param_k, 0.1..=10.0)
                            .text("k (Spring Constant)")
                            .id_salt("ode_param_k_osc"),
                    )
                    .changed()
                {
                    changed = true;
                }
            }
            OdePreset::LogisticGrowth => {
                if ui
                    .add(
                        AnimatedSlider::new(&mut self.param_r, 0.0..=5.0)
                            .text("r (Growth Rate)")
                            .id_salt("ode_param_r"),
                    )
                    .changed()
                {
                    changed = true;
                }
                if ui
                    .add(
                        AnimatedSlider::new(&mut self.param_cap_k, 1.0..=100.0)
                            .text("K (Carrying Capacity)")
                            .id_salt("ode_param_cap_k"),
                    )
                    .changed()
                {
                    changed = true;
                }
            }
            OdePreset::Custom => unreachable!(),
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
        if self.preset == OdePreset::HarmonicOscillator {
            ui.horizontal(|ui| {
                ui.label("y'(0):");
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
        let is_harmonic = self.preset == OdePreset::HarmonicOscillator;
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
            if is_harmonic && !v_series.is_empty() {
                let pv: Vec<[f64; 2]> = time_series
                    .iter()
                    .zip(v_series)
                    .map(|(&t, &v)| [t, v])
                    .collect();
                series.push(TraceSeries::new("y'(t)", pv));
            }
            Some((label, param_summary, series))
        }) {
            self.bump_revision();
        }

        self.run_sensitivity_analysis();

        crate::sensitivity::render_sensitivity_panel_ui(ui, &self.sensitivity_engine, "ode_tool");
    }

    fn collect_custom_sensitivity_params<'a>(
        preset: OdePreset,
        custom_compiled: Option<&'a math_commons::ast::CompiledExpr>,
        param_slice: &[f64],
        ic_y0: f64,
    ) -> Vec<(&'a str, f64)> {
        let mut custom_params = Vec::new();
        if preset == OdePreset::Custom {
            if let Some(compiled) = custom_compiled {
                for (name, &val) in compiled.param_names().iter().zip(param_slice) {
                    custom_params.push((name.as_str(), val));
                }
            }
            if !custom_params.iter().any(|(name, _)| *name == "y0") {
                custom_params.push(("y0", ic_y0));
            }
        }
        custom_params
    }

    fn run_sensitivity_analysis(&mut self) {
        use crate::sensitivity::canonical_f64_bits;

        let current_preset = self.preset;
        let current_method = self.method;
        let current_dt = self.dt;
        let current_total_time = self.total_time;
        let custom_compiled = self.math_input.active_compiled();

        let exp_params = [("k", self.param_k), ("y0", self.ic_y0)];
        let osc_params = [("k", self.param_k), ("y0", self.ic_y0), ("v0", self.ic_v0)];
        let log_params = [
            ("r", self.param_r),
            ("cap_k", self.param_cap_k),
            ("y0", self.ic_y0),
        ];
        let custom_params = Self::collect_custom_sensitivity_params(
            current_preset,
            custom_compiled,
            self.math_input.param_slice(),
            self.ic_y0,
        );

        let params: &[(&str, f64)] = match current_preset {
            OdePreset::Exponential => &exp_params,
            OdePreset::HarmonicOscillator => &osc_params,
            OdePreset::LogisticGrowth => &log_params,
            OdePreset::Custom => &custom_params,
        };

        let custom_ctx = (current_preset == OdePreset::Custom)
            .then_some(super::systems::CompiledContext(custom_compiled));
        let context = (
            current_preset as u8,
            current_method as u8,
            canonical_f64_bits(current_dt),
            canonical_f64_bits(current_total_time),
            custom_ctx,
        );

        self.sensitivity_engine
            .compute_with_context(params, &context, |p_map| {
                eval_ode_final_state(
                    current_preset,
                    current_method,
                    current_dt,
                    current_total_time,
                    p_map,
                    custom_compiled,
                )
            });
    }

    pub(super) fn render_discrepancy_panel(&self, ui: &mut egui::Ui) {
        ui.add_space(10.0);
        ui.heading("Discrepancy Metrics");
        ui.group(|ui| {
            render_copyable_metric(
                ui,
                "Euler vs RK4 (Max)",
                &format!("{:.4e}", self.discrepancy_euler_rk4.max_error),
            );
            render_copyable_metric(
                ui,
                "Euler vs RK4 (RMS)",
                &format!("{:.4e}", self.discrepancy_euler_rk4.rms_error),
            );
            render_copyable_metric(
                ui,
                "Heun vs RK4 (Max)",
                &format!("{:.4e}", self.discrepancy_heun_rk4.max_error),
            );
            render_copyable_metric(
                ui,
                "Heun vs RK4 (RMS)",
                &format!("{:.4e}", self.discrepancy_heun_rk4.rms_error),
            );
            render_copyable_metric(
                ui,
                "Euler vs Heun (Max)",
                &format!("{:.4e}", self.discrepancy_euler_heun.max_error),
            );
            render_copyable_metric(
                ui,
                "Euler vs Heun (RMS)",
                &format!("{:.4e}", self.discrepancy_euler_heun.rms_error),
            );
        });
    }

    pub(super) fn render_plot_canvas(&mut self, ui: &mut egui::Ui, changed: &mut bool) {
        ui.heading("Solution Trajectory");

        if FiniteStateGuard::diverged(self) {
            let mut scratch_y = Vec::new();
            let mut scratch_v = Vec::new();
            recovery::render_recovery_ui(ui, self, changed, &mut scratch_y, &mut scratch_v);
        }

        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let line_y = Line::new("y(t)", PlotPoints::from_shared(Arc::clone(&self.display_y)))
            .color(palette.series_color(0));

        Plot::new("ode_plot")
            .view_aspect(2.0)
            .x_axis_label("t \\text{ (Time)}")
            .y_axis_label("y(t)")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                if !self.comparison_mode {
                    plot_ui.line(line_y);
                    if self.preset == OdePreset::HarmonicOscillator && !self.display_v.is_empty() {
                        let line_v = Line::new(
                            "y'(t)",
                            PlotPoints::from_shared(Arc::clone(&self.display_v)),
                        )
                        .color(palette.series_color(1));
                        plot_ui.line(line_v);
                    }
                } else {
                    self.comparison_manager.render_on_plot(plot_ui);
                }
                if self.show_annotations && !FiniteStateGuard::diverged(self) {
                    super::annotations::render_ode_callouts(plot_ui, self);
                }
                self.snapshot_manager.render_on_plot(plot_ui);
                self.callout_manager.render_on_plot(plot_ui);
            });
    }
}
