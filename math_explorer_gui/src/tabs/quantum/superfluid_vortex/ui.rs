#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:quantum_mechanics]

use eframe::egui;
use egui_plot::commands::{CommandRegistryData, CommandTrigger};
use math_explorer::physics::quantum::gross_pitaevskii::GpeTrapKind;

use super::*;
use crate::framework::bounded_drag_value;
use crate::reflective_ui::render_copyable_metric;

impl SuperfluidVortexTool {
    pub(super) fn draw_controls_panel(&mut self, parent_ui: &mut egui::Ui) {
        self.register_shortcuts(parent_ui.ctx());
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "superfluid_vortex_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Superfluid & GPE");
                ui.separator();
                self.draw_preset_and_transport_controls(ui);
                ui.separator();
                self.draw_interaction_and_trap_controls(ui);
                ui.separator();
                self.draw_vortex_imprint_controls(ui);
                ui.separator();
                self.draw_view_and_metric_cards(ui);
            });
        });
    }

    fn register_shortcuts(&self, ctx: &egui::Context) {
        CommandRegistryData::register(
            ctx,
            "Superfluid Vortex Studio: Imprint Vortex",
            "Imprint a quantized +/-2pi topological vortex core at the active cursor",
            CommandTrigger::Shortcut(egui::Modifiers::NONE, egui::Key::Enter),
            true,
            "Quantum Mechanics",
        );
    }

    fn draw_preset_and_transport_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Starter Preset:");
        let mut selected = self.state.preset;
        egui::ComboBox::from_id_salt("superfluid_preset_selector")
            .selected_text(selected.label())
            .show_ui(ui, |ui| {
                for p in [
                    SuperfluidPreset::VortexDipole,
                    SuperfluidPreset::VortexQuadrupole,
                    SuperfluidPreset::CoRotatingPair,
                    SuperfluidPreset::ObstaclePinningCooling,
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
            if ui.button("🌀 Dipole").clicked() {
                let d = 0.5 * self.state.dipole_separation;
                self.solver.imprint_vortex_dipole(
                    self.state.cursor_x - d,
                    self.state.cursor_y,
                    self.state.cursor_x + d,
                    self.state.cursor_y,
                    self.state.healing_length,
                );
                self.initial_invariants = self.solver.compute_invariants(self.state.interaction_g);
                self.refresh_diagnostics();
            }
            if ui.button("↺ Reset").clicked() {
                self.reseed_condensate();
            }
        });
    }

    fn draw_interaction_and_trap_controls(&mut self, ui: &mut egui::Ui) {
        let mut trap_dirty = false;
        let mut param_changed = false;

        param_changed |= ui
            .checkbox(
                &mut self.state.imaginary_time_cooling,
                "Imaginary-Time Cooling (τ = i t)",
            )
            .on_hover_text(
                "Damps Bogoliubov phonon excitations via Wick rotation while preserving unit norm",
            )
            .changed();

        ui.horizontal(|ui| {
            ui.label("Coupling g:");
            param_changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.interaction_g,
                    COUPLING_G_MIN..=COUPLING_G_MAX,
                    2.0,
                ))
                .changed();
            ui.label("Δt:");
            param_changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.time_step,
                    GPE_DT_MIN..=GPE_DT_MAX,
                    5.0e-4,
                ))
                .changed();
        });

        let prev_trap = self.state.trap_kind;
        egui::ComboBox::from_id_salt("superfluid_trap_kind")
            .selected_text(format!("{:?}", self.state.trap_kind))
            .show_ui(ui, |ui| {
                for tk in [
                    GpeTrapKind::HarmonicIsotropic,
                    GpeTrapKind::HarmonicAnisotropic,
                    GpeTrapKind::CentralObstacle,
                    GpeTrapKind::BoxPeriodic,
                ] {
                    ui.selectable_value(&mut self.state.trap_kind, tk, format!("{tk:?}"));
                }
            });
        trap_dirty |= self.state.trap_kind != prev_trap;
        trap_dirty |= self.draw_trap_frequency_drags(ui);

        if self.state.trap_kind != prev_trap {
            self.state_rev = self.state_rev.wrapping_add(1);
            self.reseed_condensate();
        } else if trap_dirty {
            self.state_rev = self.state_rev.wrapping_add(1);
            self.update_trap_in_place();
        } else if param_changed {
            self.state_rev = self.state_rev.wrapping_add(1);
            self.initial_invariants = self.solver.compute_invariants(self.state.interaction_g);
            self.refresh_diagnostics();
        }
    }

    fn draw_trap_frequency_drags(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Trap ωx / ωy:");
            changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.omega_x,
                    OMEGA_MIN..=OMEGA_MAX,
                    0.05,
                ))
                .changed();
            changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.omega_y,
                    OMEGA_MIN..=OMEGA_MAX,
                    0.05,
                ))
                .changed();
        });
        if self.state.trap_kind == GpeTrapKind::CentralObstacle {
            ui.horizontal(|ui| {
                ui.label("Barrier V0 / w0:");
                changed |= ui
                    .add(bounded_drag_value(
                        &mut self.state.obstacle_height,
                        OBSTACLE_H_MIN..=OBSTACLE_H_MAX,
                        0.5,
                    ))
                    .changed();
                changed |= ui
                    .add(bounded_drag_value(
                        &mut self.state.obstacle_width,
                        OBSTACLE_W_MIN..=OBSTACLE_W_MAX,
                        0.05,
                    ))
                    .changed();
            });
        }
        changed
    }

    fn draw_vortex_imprint_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Stamp Charge:");
            changed |= ui
                .selectable_value(&mut self.state.stamp_charge, 1, "+1 (+2π)")
                .changed();
            changed |= ui
                .selectable_value(&mut self.state.stamp_charge, -1, "-1 (-2π)")
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Core ξ / Sep d:");
            changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.healing_length,
                    HEALING_MIN..=HEALING_MAX,
                    0.02,
                ))
                .changed();
            changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.dipole_separation,
                    0.8..=6.0,
                    0.1,
                ))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Cursor (x, y):");
            changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.cursor_x,
                    -CURSOR_BOUND..=CURSOR_BOUND,
                    0.1,
                ))
                .changed();
            changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.cursor_y,
                    -CURSOR_BOUND..=CURSOR_BOUND,
                    0.1,
                ))
                .changed();
        });
        if changed {
            self.state_rev = self.state_rev.wrapping_add(1);
        }
    }

    fn draw_view_and_metric_cards(&mut self, ui: &mut egui::Ui) {
        let mut vis_changed = false;
        egui::ComboBox::from_id_salt("superfluid_view_mode")
            .selected_text(self.state.view_mode.label())
            .show_ui(ui, |ui| {
                for vm in [
                    CondensateViewMode::SuperfluidDensity,
                    CondensateViewMode::CyclicPhase,
                    CondensateViewMode::WcagHighContrastDensity,
                ] {
                    vis_changed |= ui
                        .selectable_value(&mut self.state.view_mode, vm, vm.label())
                        .changed();
                }
            });
        vis_changed |= ui
            .checkbox(
                &mut self.state.show_vortex_markers,
                "Show Plaquette Vortex Markers (+2π / -2π)",
            )
            .changed();
        if vis_changed {
            self.state_rev = self.state_rev.wrapping_add(1);
        }

        ui.separator();
        self.render_superfluid_metrics(ui);
    }

    fn render_superfluid_metrics(&self, ui: &mut egui::Ui) {
        let norm_err = (self.current_invariants.norm - self.initial_invariants.norm).abs();
        render_copyable_metric(
            ui,
            "Norm M (|ΔM|)",
            &format!("{:.6} ({norm_err:.2e})", self.current_invariants.norm),
        );
        render_copyable_metric(
            ui,
            "Momentum |P|",
            &format!("{:.4}", self.current_invariants.momentum_magnitude()),
        );
        render_copyable_metric(
            ui,
            "GPE Energy H / μ",
            &format!(
                "{:.3} / {:.3}",
                self.current_invariants.energy, self.current_invariants.chemical_potential
            ),
        );
        render_copyable_metric(
            ui,
            "Vortex Cores (+1 / -1)",
            &format!(
                "{} / {}",
                self.charge_summary.positive_count, self.charge_summary.negative_count
            ),
        );
        render_copyable_metric(
            ui,
            "Net Winding Q (∮∇θ·dl)",
            &format!(
                "{:+} ({:+.3})",
                self.charge_summary.net_charge, self.charge_summary.net_circulation
            ),
        );
        render_copyable_metric(ui, "Step Budget", &format!("{:.2} ms", self.last_step_ms));
    }
}
