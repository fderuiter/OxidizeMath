#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::phase_plane::PLANAR_COUPLED_PRESETS;
use super::recovery;
use super::systems::eval_ode_final_state_planar;
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
        if self.preset.is_2d() {
            changed |= self.render_phase_plane_controls(ui);
        }
        changed |= self.render_parameter_inputs(ui);
        if self.preset == OdePreset::LinearSystem2D || self.is_phase_plane_active() {
            changed |= self.render_poincare_minimap(ui);
        }
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

        let prev_preset = self.preset;
        egui::ComboBox::from_id_salt("ode_preset")
            .selected_text(self.preset.name())
            .show_ui(ui, |ui| {
                for p in OdePreset::ALL {
                    if ui.selectable_value(&mut self.preset, p, p.name()).changed() {
                        changed = true;
                    }
                }
            });
        if self.preset != prev_preset
            && matches!(
                self.preset,
                OdePreset::LinearSystem2D | OdePreset::PlanarCustom
            )
        {
            self.phase_plane_mode = true;
        }

        changed |= self.render_preset_equations_ui(ui);
        changed
    }

    fn render_preset_equations_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        match self.preset {
            OdePreset::Custom => {
                ui.add_space(6.0);
                changed |= self.math_input.ui(ui);
            }
            OdePreset::PlanarCustom => {
                ui.add_space(6.0);
                egui::ComboBox::from_id_salt("ode_planar_coupled_preset")
                    .selected_text("Coupled System Templates...")
                    .show_ui(ui, |ui| {
                        for (title, fx, fy) in PLANAR_COUPLED_PRESETS {
                            if ui.button(*title).clicked() {
                                self.planar_x_input.set_expression(fx);
                                self.planar_y_input.set_expression(fy);
                                changed = true;
                                ui.close();
                            }
                        }
                    });
                ui.add_space(4.0);
                changed |= self.planar_x_input.ui(ui);
                ui.add_space(4.0);
                changed |= self.planar_y_input.ui(ui);
            }
            OdePreset::Exponential => {
                crate::reflective_ui::render_formula_with_export(
                    ui,
                    r"\frac{dy}{dt} = k y",
                    Some("dy/dt = k*y"),
                );
            }
            OdePreset::HarmonicOscillator => {
                crate::reflective_ui::render_formula_with_export(
                    ui,
                    r"\frac{d^2 y}{dt^2} = -k y",
                    Some("d^2y/dt^2 = -k*y"),
                );
            }
            OdePreset::LogisticGrowth => {
                crate::reflective_ui::render_formula_with_export(
                    ui,
                    r"\frac{dy}{dt} = r y (1 - \frac{y}{K})",
                    Some("dy/dt = r*y*(1 - y/K)"),
                );
            }
            OdePreset::LinearSystem2D => {
                crate::reflective_ui::render_formula_with_export(
                    ui,
                    r"\dot{\mathbf{x}} = A \mathbf{x}",
                    Some("x' = a11*x + a12*y, y' = a21*x + a22*y"),
                );
            }
        }
        changed
    }

    fn render_parameter_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.add_space(10.0);
        ui.heading("Parameters");
        match self.preset {
            OdePreset::Exponential | OdePreset::Custom => {
                let s = AnimatedSlider::new(&mut self.param_k, -5.0..=5.0)
                    .text("k (Rate / Parameter)")
                    .id_salt("ode_param_k_exp");
                changed |= ui.add(s).changed();
            }
            OdePreset::HarmonicOscillator => {
                let s = AnimatedSlider::new(&mut self.param_k, 0.1..=10.0)
                    .text("k (Spring Constant)")
                    .id_salt("ode_param_k_osc");
                changed |= ui.add(s).changed();
            }
            OdePreset::LogisticGrowth => {
                let sr = AnimatedSlider::new(&mut self.param_r, 0.0..=5.0)
                    .text("r (Growth Rate)")
                    .id_salt("ode_param_r");
                let sk = AnimatedSlider::new(&mut self.param_cap_k, 1.0..=100.0)
                    .text("K (Carrying Capacity)")
                    .id_salt("ode_param_cap_k");
                changed |= ui.add(sr).changed();
                changed |= ui.add(sk).changed();
            }
            OdePreset::LinearSystem2D => {
                changed |= self.render_matrix_inputs(ui);
            }
            OdePreset::PlanarCustom => {
                let sk = AnimatedSlider::new(&mut self.param_k, -5.0..=5.0)
                    .text("k (Coupling / Damping)")
                    .id_salt("ode_param_k_planar");
                let sr = AnimatedSlider::new(&mut self.param_r, -5.0..=5.0)
                    .text("r (Growth / Forcing)")
                    .id_salt("ode_param_r_planar");
                changed |= ui.add(sk).changed();
                changed |= ui.add(sr).changed();
            }
        }
        changed
    }

    fn render_initial_condition_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let (_, _, px_lbl, py_lbl) = self.preset.state_labels();
        ui.add_space(10.0);
        ui.heading("Initial Conditions");
        let first_lbl = if self.preset.is_2d() {
            format!("{px_lbl}(0):")
        } else {
            "y(0):".to_string()
        };
        ui.horizontal(|ui| {
            ui.label(first_lbl);
            changed |= ui
                .add(egui::DragValue::new(&mut self.ic_y0).speed(0.1))
                .changed();
        });
        if self.preset.is_2d() {
            ui.horizontal(|ui| {
                ui.label(format!("{py_lbl}(0):"));
                changed |= ui
                    .add(egui::DragValue::new(&mut self.ic_v0).speed(0.1))
                    .changed();
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
        changed |= ui
            .checkbox(&mut self.comparison_mode, "Multi-Solver Comparison")
            .changed();

        let min_dt = (self.total_time.max(0.1) / crate::presets::MAX_ODE_STEPS as f64).max(1e-6);
        ui.horizontal(|ui| {
            ui.label("dt (Step Size):");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.dt)
                        .speed(0.01)
                        .range(min_dt..=1.0),
                )
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Total Time:");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.total_time)
                        .speed(1.0)
                        .range(1.0..=100.0),
                )
                .changed();
        });

        ui.add_space(5.0);
        changed |= ui
            .checkbox(&mut self.show_annotations, "Show Annotations")
            .changed();

        if self.comparison_mode {
            self.render_discrepancy_panel(ui);
        }
        changed
    }

    fn render_snapshots_and_sensitivity(&mut self, ui: &mut egui::Ui) {
        ui.add_space(10.0);
        let param_summary = self.param_summary();
        let (t1_lbl, t2_lbl, _, _) = self.preset.state_labels();
        let is_2d = self.preset.is_2d();
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
            let mut series = vec![TraceSeries::new(t1_lbl, py)];
            if is_2d && !v_series.is_empty() {
                let pv: Vec<[f64; 2]> = time_series
                    .iter()
                    .zip(v_series)
                    .map(|(&t, &v)| [t, v])
                    .collect();
                series.push(TraceSeries::new(t2_lbl, pv));
            }
            Some((label, param_summary, series))
        }) {
            self.bump_revision();
        }

        self.run_sensitivity_analysis();
        crate::sensitivity::render_sensitivity_panel_ui(ui, &self.sensitivity_engine, "ode_tool");
    }

    fn sensitivity_params(&self) -> Vec<(&'static str, f64)> {
        match self.preset {
            OdePreset::Exponential | OdePreset::Custom => {
                vec![("k", self.param_k), ("y0", self.ic_y0)]
            }
            OdePreset::HarmonicOscillator => {
                vec![("k", self.param_k), ("y0", self.ic_y0), ("v0", self.ic_v0)]
            }
            OdePreset::LogisticGrowth => {
                vec![
                    ("r", self.param_r),
                    ("cap_k", self.param_cap_k),
                    ("y0", self.ic_y0),
                ]
            }
            OdePreset::LinearSystem2D => vec![
                ("a11", self.matrix_a[0][0]),
                ("a12", self.matrix_a[0][1]),
                ("a21", self.matrix_a[1][0]),
                ("a22", self.matrix_a[1][1]),
                ("y0", self.ic_y0),
                ("v0", self.ic_v0),
            ],
            OdePreset::PlanarCustom => vec![
                ("k", self.param_k),
                ("r", self.param_r),
                ("y0", self.ic_y0),
                ("v0", self.ic_v0),
            ],
        }
    }

    fn run_sensitivity_analysis(&mut self) {
        use crate::sensitivity::canonical_f64_bits;

        let params = self.sensitivity_params();
        let custom_expr = self.math_input.cached_expr();
        let px_expr = self.planar_x_input.cached_expr();
        let py_expr = self.planar_y_input.cached_expr();
        let base_spec = super::comparison::OdeSimSpec {
            preset: self.preset,
            method: self.method,
            dt: self.dt,
            total_time: self.total_time,
            param_k: self.param_k,
            param_r: self.param_r,
            param_cap_k: self.param_cap_k,
            ic_y0: self.ic_y0,
            ic_v0: self.ic_v0,
            matrix_a: self.matrix_a,
            custom_expr,
            planar_x_expr: px_expr,
            planar_y_expr: py_expr,
        };
        let context = (
            base_spec.preset as u8,
            base_spec.method as u8,
            canonical_f64_bits(base_spec.dt),
            canonical_f64_bits(base_spec.total_time),
            (base_spec.preset == OdePreset::Custom)
                .then_some(super::systems::AstContext(custom_expr)),
            (base_spec.preset == OdePreset::PlanarCustom).then_some((
                super::systems::AstContext(px_expr),
                super::systems::AstContext(py_expr),
            )),
        );

        self.sensitivity_engine
            .compute_with_context(&params, &context, |p_map| {
                eval_ode_final_state_planar(&base_spec, p_map)
            });
    }

    pub(super) fn render_discrepancy_panel(&self, ui: &mut egui::Ui) {
        ui.add_space(10.0);
        ui.heading("Discrepancy Metrics");
        ui.group(|ui| {
            for (label, disc) in [
                ("Euler vs RK4", self.discrepancy_euler_rk4),
                ("Heun vs RK4", self.discrepancy_heun_rk4),
                ("Euler vs Heun", self.discrepancy_euler_heun),
            ] {
                render_copyable_metric(
                    ui,
                    &format!("{label} (Max)"),
                    &format!("{:.4e}", disc.max_error),
                );
                render_copyable_metric(
                    ui,
                    &format!("{label} (RMS)"),
                    &format!("{:.4e}", disc.rms_error),
                );
            }
        });
    }

    pub(super) fn render_plot_canvas(&mut self, ui: &mut egui::Ui, changed: &mut bool) {
        if self.is_phase_plane_active() {
            self.render_phase_plane_canvas(ui, changed);
            return;
        }

        let (t1_lbl, t2_lbl, _, _) = self.preset.state_labels();
        ui.heading("Solution Trajectory");

        if FiniteStateGuard::diverged(self) {
            let mut scratch_y = Vec::new();
            let mut scratch_v = Vec::new();
            recovery::render_recovery_ui(ui, self, changed, &mut scratch_y, &mut scratch_v);
        }

        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let line_y = Line::new(t1_lbl, PlotPoints::from_shared(Arc::clone(&self.display_y)))
            .color(palette.series_color(0));

        Plot::new("ode_plot")
            .view_aspect(2.0)
            .x_axis_label("t \\text{ (Time)}")
            .y_axis_label(t1_lbl)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                if !self.comparison_mode {
                    plot_ui.line(line_y);
                    if self.preset.is_2d() && !self.display_v.is_empty() {
                        let line_v =
                            Line::new(t2_lbl, PlotPoints::from_shared(Arc::clone(&self.display_v)))
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
