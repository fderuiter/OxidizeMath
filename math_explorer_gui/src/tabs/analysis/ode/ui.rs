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

impl OdeSolverTool {
    pub(super) fn render_sidebar_controls(
        &mut self,
        ui: &mut egui::Ui,
        plot_points_y: &[[f64; 2]],
        plot_points_v: &[[f64; 2]],
    ) -> bool {
        let mut changed = false;
        changed |= self.render_preset_selector(ui);
        changed |= self.render_parameter_inputs(ui);
        changed |= self.render_initial_condition_inputs(ui);
        changed |= self.render_solver_settings_controls(ui);
        self.render_snapshots_and_sensitivity(ui, plot_points_y, plot_points_v);
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
            let params = self.active_param_map();
            if self.math_input.ui_with_params(ui, &params) {
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
        let mut changed = false;
        ui.add_space(10.0);
        ui.heading("Parameters");
        match self.preset {
            OdePreset::Exponential | OdePreset::Custom => {
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

        ui.horizontal(|ui| {
            ui.label("dt (Step Size):");
            if ui
                .add(
                    egui::DragValue::new(&mut self.dt)
                        .speed(0.01)
                        .range(0.001..=1.0),
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
        if ui.checkbox(&mut self.show_annotations, "Show Annotations").changed() {
            changed = true;
        }

        if self.comparison_mode {
            self.render_discrepancy_panel(ui);
        }
        changed
    }

    fn render_snapshots_and_sensitivity(
        &mut self,
        ui: &mut egui::Ui,
        plot_points_y: &[[f64; 2]],
        plot_points_v: &[[f64; 2]],
    ) {
        ui.add_space(10.0);
        let param_summary = self.param_summary();
        let is_harmonic = self.preset == OdePreset::HarmonicOscillator;
        let py = plot_points_y.to_vec();
        let pv = plot_points_v.to_vec();
        let current_len = self.snapshot_manager.len();

        if self.snapshot_manager.render_ui(ui, || {
            let label = format!("Baseline {}", current_len + 1);
            let mut series = vec![TraceSeries::new("y(t)", py)];
            if is_harmonic && !pv.is_empty() {
                series.push(TraceSeries::new("y'(t)", pv));
            }
            Some((label, param_summary, series))
        }) {
            self.bump_revision();
        }

        self.run_sensitivity_analysis();

        crate::sensitivity::render_sensitivity_panel_ui(
            ui,
            &self.sensitivity_engine,
            "ode_tool",
        );
    }

    fn run_sensitivity_analysis(&mut self) {
        use crate::sensitivity::canonical_f64_bits;

        let exp_params = [("k", self.param_k), ("y0", self.ic_y0)];
        let osc_params = [
            ("k", self.param_k),
            ("y0", self.ic_y0),
            ("v0", self.ic_v0),
        ];
        let log_params = [
            ("r", self.param_r),
            ("cap_k", self.param_cap_k),
            ("y0", self.ic_y0),
        ];
        let params: &[(&str, f64)] = match self.preset {
            OdePreset::Exponential | OdePreset::Custom => &exp_params,
            OdePreset::HarmonicOscillator => &osc_params,
            OdePreset::LogisticGrowth => &log_params,
        };

        let current_preset = self.preset;
        let current_method = self.method;
        let current_dt = self.dt;
        let current_total_time = self.total_time;
        let custom_expr = self.math_input.expr();
        let custom_ctx =
            (current_preset == OdePreset::Custom).then_some(super::systems::AstContext(custom_expr));
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
                    custom_expr,
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

    pub(super) fn render_plot_canvas(
        &mut self,
        ui: &mut egui::Ui,
        mut plot_points_y: Vec<[f64; 2]>,
        mut plot_points_v: Vec<[f64; 2]>,
        changed: &mut bool,
    ) {
        ui.heading("Solution Trajectory");

        if FiniteStateGuard::diverged(self) {
            recovery::render_recovery_ui(
                ui,
                self,
                changed,
                &mut plot_points_y,
                &mut plot_points_v,
            );
        }

        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let line_y =
            Line::new("y(t)", PlotPoints::new(plot_points_y)).color(palette.series_color(0));

        Plot::new("ode_plot")
            .view_aspect(2.0)
            .x_axis_label("t \\text{ (Time)}")
            .y_axis_label("y(t)")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                if !self.comparison_mode {
                    plot_ui.line(line_y);
                    if self.preset == OdePreset::HarmonicOscillator && !plot_points_v.is_empty() {
                        let line_v = Line::new("y'(t)", PlotPoints::new(plot_points_v))
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
