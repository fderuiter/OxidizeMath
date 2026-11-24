#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::phase_plane::PLANAR_COUPLED_PRESETS;
use super::OdeSolverTool;
use crate::framework::TraceSeries;
use crate::presets::OdePreset;
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use math_explorer::pure_math::analysis::ode::IntegrationMethod;

impl OdeSolverTool {
    pub(super) fn render_sidebar_controls(
        &mut self,
        ui: &mut egui::Ui,
        sweeps_before: usize,
    ) -> bool {
        let mut changed = false;
        changed |= self.render_preset_selector(ui);
        changed |= self.render_view_and_atlas_controls(ui);
        if self.has_second_state() {
            changed |= self.render_phase_plane_controls(ui);
        }
        changed |= self.render_parameter_inputs(ui);
        if self.preset == OdePreset::LinearSystem2D
            || (self.preset.is_2d() && self.is_phase_plane_active())
        {
            changed |= self.render_poincare_minimap(ui);
        }
        changed |= self.render_initial_condition_inputs(ui);
        changed |= self.render_solver_settings_controls(ui);

        let sweep_active =
            crate::reflective_ui::slider::current_pass_active_sweep_count(ui.ctx()) > sweeps_before;
        self.apply_sidebar_changes(changed, sweep_active);
        self.sync_series_cache();
        self.render_budget_notices(ui);

        if self.comparison_mode {
            self.render_discrepancy_panel(ui);
        }
        let defer_sensitivity =
            crate::sensitivity::is_transient_parameter_gesture(ui.ctx(), sweep_active);
        self.render_snapshots_and_sensitivity(ui, defer_sensitivity);
        if self.callout_manager.render_ui(ui) {
            self.bump_revision();
            changed = true;
        }
        changed
    }

    fn render_view_and_atlas_controls(&mut self, ui: &mut egui::Ui) -> bool {
        use super::phase::PlotViewMode;

        let mut view_updated = false;
        ui.add_space(8.0);
        ui.heading("Visualization Mode");
        ui.horizontal_wrapped(|ui| {
            for mode in [PlotViewMode::TimeSeries, PlotViewMode::PhasePortrait] {
                if ui
                    .selectable_label(self.view_mode == mode, mode.label())
                    .clicked()
                    && self.view_mode != mode
                {
                    self.set_view_mode(mode);
                }
            }
            let _ = self.atlas.show_toggle_button(ui);
        });
        if self.view_mode == PlotViewMode::PhasePortrait {
            let max_t = self.total_time.max(0.1);
            let mut slice = self.t_slice.clamp(0.0, max_t);
            if ui
                .add(egui::Slider::new(&mut slice, 0.0..=max_t).text("t_slice (Quiver Time)"))
                .changed()
            {
                self.set_t_slice(slice);
                view_updated = false;
            }
        }
        view_updated
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
                changed |= self.render_custom_ode_editor(ui);
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
        let (_, _, px_lbl, py_lbl) = self.active_state_labels();
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
        if self.has_second_state() {
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
        changed
    }

    fn render_snapshots_and_sensitivity(&mut self, ui: &mut egui::Ui, defer_gesture: bool) {
        ui.add_space(10.0);
        let param_summary = self.param_summary();
        let (t1_lbl, t2_lbl, _, _) = self.active_state_labels();
        let has_v = self.has_second_state();
        let current_len = self.snapshot_manager.len();
        let (py, pv) = self.series_cache.shared_points();

        if self.snapshot_manager.render_ui(ui, move || {
            let label = format!("Baseline {}", current_len + 1);
            let mut series = vec![TraceSeries::new(t1_lbl, py)];
            if has_v && !pv.is_empty() {
                series.push(TraceSeries::new(t2_lbl, pv));
            }
            Some((label, param_summary, series))
        }) {
            self.bump_revision();
        }

        self.run_sensitivity_analysis(defer_gesture);
        crate::sensitivity::render_sensitivity_panel_ui(ui, &self.sensitivity_engine, "ode_tool");
    }

    fn push_coupled_sensitivity_params(&self, out: &mut Vec<(String, f64)>) {
        for (name, &val) in self
            .coupled_bank
            .param_names()
            .iter()
            .zip(self.coupled_bank.param_slice())
        {
            out.push((name.clone(), val));
        }
        if !out.iter().any(|(name, _)| name == "y0") {
            out.push(("y0".to_string(), self.ic_y0));
        }
        if !out.iter().any(|(name, _)| name == "v0") {
            out.push(("v0".to_string(), self.ic_v0));
        }
    }

    fn push_scalar_custom_sensitivity_params(&self, out: &mut Vec<(String, f64)>) {
        if let Some(compiled) = self.math_input.active_compiled() {
            for (name, &val) in compiled
                .param_names()
                .iter()
                .zip(self.math_input.param_slice())
            {
                out.push((name.clone(), val));
            }
        }
        if !out.iter().any(|(name, _)| name == "y0") {
            out.push(("y0".to_string(), self.ic_y0));
        }
    }

    pub(super) fn collect_sensitivity_params(&self) -> Vec<(String, f64)> {
        let mut out: Vec<(String, f64)> = Vec::with_capacity(8);
        let push_pairs = |dst: &mut Vec<(String, f64)>, pairs: &[(&str, f64)]| {
            dst.extend(pairs.iter().map(|&(k, v)| (k.to_string(), v)));
        };
        match self.preset {
            OdePreset::Exponential => {
                push_pairs(&mut out, &[("k", self.param_k), ("y0", self.ic_y0)]);
            }
            OdePreset::HarmonicOscillator => {
                push_pairs(
                    &mut out,
                    &[("k", self.param_k), ("y0", self.ic_y0), ("v0", self.ic_v0)],
                );
            }
            OdePreset::LogisticGrowth => push_pairs(
                &mut out,
                &[
                    ("r", self.param_r),
                    ("cap_k", self.param_cap_k),
                    ("y0", self.ic_y0),
                ],
            ),
            OdePreset::LinearSystem2D => push_pairs(
                &mut out,
                &[
                    ("a11", self.matrix_a[0][0]),
                    ("a12", self.matrix_a[0][1]),
                    ("a21", self.matrix_a[1][0]),
                    ("a22", self.matrix_a[1][1]),
                    ("y0", self.ic_y0),
                    ("v0", self.ic_v0),
                ],
            ),
            OdePreset::PlanarCustom => push_pairs(
                &mut out,
                &[
                    ("k", self.param_k),
                    ("r", self.param_r),
                    ("y0", self.ic_y0),
                    ("v0", self.ic_v0),
                ],
            ),
            OdePreset::Custom if self.custom_coupled => {
                self.push_coupled_sensitivity_params(&mut out);
            }
            OdePreset::Custom => self.push_scalar_custom_sensitivity_params(&mut out),
        }
        out
    }
}
