#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:advanced_linear_algebra]
// [cite:graph_parameters_rust]

use eframe::egui;
use egui_plot::commands::{CommandRegistryData, CommandTrigger};
use math_explorer::pure_math::analysis::pde::kdv::{kdv_two_soliton_phase_shifts, KdvEquationMode};

use super::*;
use crate::framework::bounded_drag_value;
use crate::reflective_ui::render_copyable_metric;

impl SolitonCollisionTool {
    pub(super) fn draw_controls_panel(&mut self, parent_ui: &mut egui::Ui) {
        self.register_shortcuts(parent_ui.ctx());
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "soliton_collision_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("KdV & NLS Controls");
                ui.separator();
                self.draw_preset_and_transport_controls(ui);
                ui.separator();
                self.draw_equation_and_nonlinearity_controls(ui);
                ui.separator();
                self.draw_soliton_seed_controls(ui);
                ui.separator();
                self.draw_visual_and_metric_cards(ui);
            });
        });
    }

    fn register_shortcuts(&self, ctx: &egui::Context) {
        CommandRegistryData::register(
            ctx,
            "KdV Soliton Studio: Inject Soliton Pulse",
            "Inject an additive sech^2 soliton pulse at the cursor coordinate x0",
            CommandTrigger::Shortcut(egui::Modifiers::NONE, egui::Key::Enter),
            true,
            "Analysis & Calculus",
        );
    }

    fn draw_preset_and_transport_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Starter Preset:");
        let mut selected = self.state.preset;
        egui::ComboBox::from_id_salt("soliton_preset_selector")
            .selected_text(selected.label())
            .show_ui(ui, |ui| {
                for p in [
                    SolitonPreset::TwoSolitonOvertaking,
                    SolitonPreset::LinearAiryCollapse,
                    SolitonPreset::ZabuskyKruskalFission,
                    SolitonPreset::CubicNlsEnvelopeCollision,
                ] {
                    ui.selectable_value(&mut selected, p, p.label());
                }
            });
        if selected != self.state.preset {
            self.apply_preset(selected);
        }

        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            let play_label = if self.paused { "▶ Play" } else { "⏸ Pause" };
            if ui.button(play_label).clicked() {
                self.paused = !self.paused;
            }
            if ui.button("⏭ Step").clicked() {
                self.advance_physics_step();
                self.refresh_diagnostics();
            }
            if ui.button("💥 Inject").clicked() {
                self.solver.inject_soliton_pulse(
                    self.state.pulse_x0,
                    self.state.pulse_speed_c,
                    self.state.mode,
                );
                self.reanchor_invariant_baseline();
                self.push_waterfall_row();
                self.refresh_diagnostics();
            }
            if ui.button("↺ Reset").clicked() {
                self.reseed_from_state();
            }
        });
    }

    fn draw_equation_and_nonlinearity_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Governing 1D Dispersive PDE:");
        let prev_mode = self.state.mode;
        egui::ComboBox::from_id_salt("soliton_pde_mode")
            .selected_text(match self.state.mode {
                KdvEquationMode::KdvSoliton => "Korteweg-de Vries (KdV / Airy)",
                KdvEquationMode::CubicNls1D => "1D Cubic Nonlinear Schrödinger",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.state.mode,
                    KdvEquationMode::KdvSoliton,
                    "Korteweg-de Vries (KdV / Airy)",
                );
                ui.selectable_value(
                    &mut self.state.mode,
                    KdvEquationMode::CubicNls1D,
                    "1D Cubic Nonlinear Schrödinger",
                );
            });

        let nl_changed = ui
            .checkbox(
                &mut self.state.nonlinear_enabled,
                "Nonlinear Steepening (6 u u_x / |ψ|²ψ)",
            )
            .on_hover_text(
                "Toggle live between elastic soliton balance and pure linear Airy / Schrödinger dispersion",
            )
            .changed();

        let mut param_changed = false;
        ui.horizontal(|ui| {
            ui.label("Step Δt:");
            param_changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.time_step,
                    DT_MIN..=DT_MAX,
                    0.001,
                ))
                .changed();
            if self.state.mode == KdvEquationMode::CubicNls1D {
                ui.label("Coupling σ:");
                param_changed |= ui
                    .add(bounded_drag_value(
                        &mut self.state.nls_coupling,
                        NLS_COUPLING_MIN..=NLS_COUPLING_MAX,
                        0.05,
                    ))
                    .changed();
            }
        });

        if self.state.mode != prev_mode {
            self.state_rev = self.state_rev.wrapping_add(1);
            self.reseed_from_state();
        } else if nl_changed || param_changed {
            self.state_rev = self.state_rev.wrapping_add(1);
            self.reanchor_invariant_baseline();
            self.refresh_diagnostics();
        }
    }

    fn draw_soliton_seed_controls(&mut self, ui: &mut egui::Ui) {
        let mut reseed = false;
        let half_l = (0.5 * self.state.domain_length - 0.5).max(4.0);
        ui.horizontal(|ui| {
            ui.label("Domain L:");
            reseed |= ui
                .add(bounded_drag_value(
                    &mut self.state.domain_length,
                    DOMAIN_LEN_MIN..=DOMAIN_LEN_MAX,
                    0.5,
                ))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Fast c1 / x1:");
            reseed |= ui
                .add(bounded_drag_value(
                    &mut self.state.fast_speed_c1,
                    SPEED_C_MIN..=SPEED_C_MAX,
                    0.2,
                ))
                .changed();
            reseed |= ui
                .add(bounded_drag_value(
                    &mut self.state.fast_x0,
                    -half_l..=half_l,
                    0.2,
                ))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Slow c2 / x2:");
            reseed |= ui
                .add(bounded_drag_value(
                    &mut self.state.slow_speed_c2,
                    SPEED_C_MIN..=SPEED_C_MAX,
                    0.2,
                ))
                .changed();
            reseed |= ui
                .add(bounded_drag_value(
                    &mut self.state.slow_x0,
                    -half_l..=half_l,
                    0.2,
                ))
                .changed();
        });
        self.draw_pulse_cursor_controls(ui, half_l);
        if reseed {
            presets::clamp_state_fields(&mut self.state);
            self.state_rev = self.state_rev.wrapping_add(1);
            self.reseed_from_state();
        }
    }

    fn draw_pulse_cursor_controls(&mut self, ui: &mut egui::Ui, half_l: f64) {
        let mut cursor_changed = false;
        ui.horizontal(|ui| {
            ui.label("Inject x0 / c:");
            cursor_changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.pulse_x0,
                    -half_l..=half_l,
                    0.2,
                ))
                .changed();
            cursor_changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.pulse_speed_c,
                    SPEED_C_MIN..=SPEED_C_MAX,
                    0.2,
                ))
                .changed();
        });
        if cursor_changed {
            self.state_rev = self.state_rev.wrapping_add(1);
        }
    }

    fn draw_visual_and_metric_cards(&mut self, ui: &mut egui::Ui) {
        let mut vis_changed = false;
        egui::ComboBox::from_id_salt("soliton_colormap_selector")
            .selected_text(self.state.colormap.label())
            .show_ui(ui, |ui| {
                for cm in [
                    SolitonColormap::PerceptualThermal,
                    SolitonColormap::BipolarWave,
                    SolitonColormap::WcagHighContrast,
                ] {
                    vis_changed |= ui
                        .selectable_value(&mut self.state.colormap, cm, cm.label())
                        .changed();
                }
            });
        ui.horizontal(|ui| {
            vis_changed |= ui
                .checkbox(&mut self.state.show_initial_overlay, "Initial u(x,0)")
                .changed();
            vis_changed |= ui
                .checkbox(&mut self.state.show_ballistic_markers, "Ballistic Markers")
                .changed();
        });
        if vis_changed {
            self.state_rev = self.state_rev.wrapping_add(1);
        }

        ui.separator();
        self.render_invariant_and_phase_metrics(ui);
    }

    fn render_invariant_and_phase_metrics(&self, ui: &mut egui::Ui) {
        let drift = self
            .current_invariants
            .relative_drift(self.initial_invariants);
        let peak_ratio = self.top_peaks.0.amplitude / self.initial_peak.max(1.0e-9);
        render_copyable_metric(
            ui,
            "Elapsed Time / Substeps",
            &format!("t = {:.3} ({}x)", self.solver.time(), self.last_substeps),
        );
        render_copyable_metric(
            ui,
            "Mass M (rel drift)",
            &format!("{:.4} ({:.2e})", self.current_invariants.mass, drift[0]),
        );
        render_copyable_metric(
            ui,
            "Momentum P (rel drift)",
            &format!("{:.4} ({:.2e})", self.current_invariants.momentum, drift[1]),
        );
        render_copyable_metric(
            ui,
            "Hamiltonian H (rel drift)",
            &format!(
                "{:.4} ({:.2e})",
                self.current_invariants.hamiltonian, drift[2]
            ),
        );
        render_copyable_metric(ui, "Peak Ratio u_max/u_0", &format!("{peak_ratio:.3}"));

        if self.state.mode == KdvEquationMode::KdvSoliton {
            let exact =
                kdv_two_soliton_phase_shifts(self.state.fast_speed_c1, self.state.slow_speed_c2);
            render_copyable_metric(
                ui,
                "Analytic Δ1 / Δ2",
                &format!("{:+.4} / {:+.4}", exact.delta_fast, exact.delta_slow),
            );
            render_copyable_metric(
                ui,
                "Measured Δ1 / Δ2",
                &format!(
                    "{:+.4} / {:+.4}",
                    self.measured_shifts.0, self.measured_shifts.1
                ),
            );
        }
    }
}
