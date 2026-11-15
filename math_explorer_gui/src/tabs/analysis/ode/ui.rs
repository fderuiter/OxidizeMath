#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::systems::eval_ode_final_state;
use super::OdeSolverTool;
use crate::framework::TraceSeries;
use crate::presets::OdePreset;
use crate::reflective_ui::render_copyable_metric;
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
        changed |= self.render_snapshots_and_sensitivity(ui, plot_points_y, plot_points_v);
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
                {
                    changed = true;
                }
            });
        changed
    }

    fn render_parameter_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.add_space(10.0);
        ui.heading("Parameters");
        match self.preset {
            OdePreset::Exponential => {
                ui.horizontal(|ui| {
                    ui.label("k (Rate):");
                    if ui
                        .add(egui::DragValue::new(&mut self.param_k).speed(0.1))
                        .changed()
                    {
                        changed = true;
                    }
                });
            }
            OdePreset::HarmonicOscillator => {
                ui.horizontal(|ui| {
                    ui.label("k (Spring Constant):");
                    if ui
                        .add(egui::DragValue::new(&mut self.param_k).speed(0.1))
                        .changed()
                    {
                        changed = true;
                    }
                });
            }
            OdePreset::LogisticGrowth => {
                ui.horizontal(|ui| {
                    ui.label("r (Growth Rate):");
                    if ui
                        .add(egui::DragValue::new(&mut self.param_r).speed(0.1))
                        .changed()
                    {
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("K (Carrying Capacity):");
                    if ui
                        .add(egui::DragValue::new(&mut self.param_cap_k).speed(0.1))
                        .changed()
                    {
                        changed = true;
                    }
                });
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
    ) -> bool {
        let mut changed = false;
        ui.add_space(10.0);
        let param_summary = self.param_summary();
        let is_harmonic = self.preset == OdePreset::HarmonicOscillator;
        let py = plot_points_y.to_vec();
        let pv = plot_points_v.to_vec();
        let current_len = self.snapshot_manager.len();

        self.snapshot_manager.render_ui(ui, || {
            let label = format!("Baseline {}", current_len + 1);
            let mut series = vec![TraceSeries::new("y(t)", py)];
            if is_harmonic && !pv.is_empty() {
                series.push(TraceSeries::new("y'(t)", pv));
            }
            Some((label, param_summary, series))
        });

        let mut dialog_state = std::mem::take(&mut self.dialog_state);
        if crate::presets::render_preset_buttons(&mut dialog_state, ui, self) {
            changed = true;
        }
        self.dialog_state = dialog_state;

        let params = match self.preset {
            OdePreset::Exponential => vec![
                ("k".to_string(), self.param_k),
                ("y0".to_string(), self.ic_y0),
            ],
            OdePreset::HarmonicOscillator => vec![
                ("k".to_string(), self.param_k),
                ("y0".to_string(), self.ic_y0),
                ("v0".to_string(), self.ic_v0),
            ],
            OdePreset::LogisticGrowth => vec![
                ("r".to_string(), self.param_r),
                ("cap_k".to_string(), self.param_cap_k),
                ("y0".to_string(), self.ic_y0),
            ],
        };

        let current_preset = self.preset;
        let current_method = self.method;
        let current_dt = self.dt;
        let current_total_time = self.total_time;

        self.sensitivity_engine.compute(&params, |p_map| {
            eval_ode_final_state(
                current_preset,
                current_method,
                current_dt,
                current_total_time,
                p_map,
            )
        });

        crate::sensitivity::render_sensitivity_panel_ui(
            ui,
            &self.sensitivity_engine,
            "ode_tool",
        );
        changed
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
        plot_points_y: Vec<[f64; 2]>,
        plot_points_v: Vec<[f64; 2]>,
        changed: &mut bool,
    ) {
        ui.heading("Solution Trajectory");

        if self.diverged {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(
                            "⚠️ Solver Divergence Warning: Non-finite state detected (NaN or Inf). Integration terminated.",
                        )
                        .color(egui::Color32::RED)
                        .strong(),
                    );
                    if ui.button("↻ Reset Parameters").clicked() {
                        self.reset_parameters();
                        *changed = true;
                    }
                });
            });
        }

        let line_y = Line::new("y(t)", PlotPoints::new(plot_points_y))
            .color(egui::Color32::LIGHT_BLUE);

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
                            .color(egui::Color32::LIGHT_RED);
                        plot_ui.line(line_v);
                    }
                } else {
                    self.comparison_manager.render_on_plot(plot_ui);
                }
                self.snapshot_manager.render_on_plot(plot_ui);
            });
    }
}
