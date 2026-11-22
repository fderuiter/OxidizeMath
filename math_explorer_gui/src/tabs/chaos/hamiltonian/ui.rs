//! Side-panel controls and synchronized Runge-Kutta 4 comparator plot for [`HamiltonianChaosLab`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    ComparatorView, HamiltonianChaosLab, HamiltonianModelKind, SymplecticSolverChoice, MAX_ORBITS,
};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{PaneSidePanel, PlaybackAction, PlaybackTransport, scoped_panel_id};
use eframe::egui;
use egui_plot::{Corner, Legend, Line, LineStyle, Plot, PlotPoints, Points};
use math_explorer::physics::astrophysics::nbody::FigureEightThreeBody;

impl HamiltonianChaosLab {
    pub(super) fn render_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(parent_ui, "hamiltonian_chaos_controls");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Hamiltonian Chaos Lab");
            ui.separator();

            self.render_transport_controls(ui);
            ui.separator();

            self.render_model_and_solver_selectors(ui);
            ui.separator();

            self.render_parameter_sliders(ui);
            ui.separator();

            self.render_ensemble_actions(ui);
            ui.separator();

            self.render_conservation_readout(ui);
        });
    }

    fn render_transport_controls(&mut self, ui: &mut egui::Ui) {
        let action = PlaybackTransport::in_thread(
            "hamiltonian_chaos_transport",
            "Hamiltonian Chaos Lab",
            self.is_playing,
        )
        .show(ui);

        match action {
            Some(PlaybackAction::Play) => self.is_playing = true,
            Some(PlaybackAction::Pause) => self.is_playing = false,
            Some(PlaybackAction::Step) => self.step_batch(self.steps_per_frame.clamp(1, 128)),
            Some(PlaybackAction::Reset) => {
                self.is_playing = false;
                self.seed_default_showcase();
            }
            None => {}
        }
    }

    fn render_model_and_solver_selectors(&mut self, ui: &mut egui::Ui) {
        ui.label("Conservative System:");
        let prev_model = self.model;
        egui::ComboBox::from_id_salt("hamiltonian_model_selector")
            .selected_text(self.model.label())
            .show_ui(ui, |ui| {
                for candidate in [
                    HamiltonianModelKind::HenonHeiles,
                    HamiltonianModelKind::DoublePendulum,
                    HamiltonianModelKind::FigureEightThreeBody,
                ] {
                    ui.selectable_value(&mut self.model, candidate, candidate.label());
                }
            });
        if self.model != prev_model {
            let chosen = self.model;
            self.switch_model(chosen);
        }

        ui.add_space(4.0);
        ui.label("Symplectic Integrator:");
        let prev_solver = self.solver;
        egui::ComboBox::from_id_salt("hamiltonian_solver_selector")
            .selected_text(self.solver.label())
            .show_ui(ui, |ui| {
                for candidate in [
                    SymplecticSolverChoice::StormerVerlet,
                    SymplecticSolverChoice::Yoshida4,
                    SymplecticSolverChoice::GaussLegendre4,
                ] {
                    let enabled = self.model != HamiltonianModelKind::DoublePendulum
                        || candidate == SymplecticSolverChoice::GaussLegendre4;
                    ui.add_enabled_ui(enabled, |ui| {
                        ui.selectable_value(&mut self.solver, candidate, candidate.label());
                    });
                }
            });
        if self.solver != prev_solver {
            self.state_rev = self.state_rev.wrapping_add(1);
        }
        if self.model == HamiltonianModelKind::DoublePendulum {
            ui.small("Non-separable H(q,p): implicit Gauss-Legendre 4th order active.");
        }
    }

    fn render_parameter_sliders(&mut self, ui: &mut egui::Ui) {
        if self.model != HamiltonianModelKind::FigureEightThreeBody {
            let (e_min, e_max, _) = self.energy_bounds();
            if ui
                .add(
                    egui::Slider::new(&mut self.energy, e_min..=e_max)
                        .text("Energy Shell E")
                        .max_decimals(4),
                )
                .accessible_hover_text("Constant Hamiltonian energy shell H(q, p) = E")
                .changed()
            {
                self.rebuild_existing_seeds();
            }
        }

        if ui
            .add(
                egui::Slider::new(&mut self.dt, 0.001..=0.05)
                    .text("Step Δt")
                    .max_decimals(4),
            )
            .accessible_hover_text("Fixed symplectic and RK4 integration time step")
            .changed()
        {
            self.state_rev = self.state_rev.wrapping_add(1);
        }

        if ui
            .add(egui::Slider::new(&mut self.steps_per_frame, 1..=128).text("Steps / Frame"))
            .accessible_hover_text("Symplectic sub-steps integrated per 60 FPS frame")
            .changed()
        {
            self.state_rev = self.state_rev.wrapping_add(1);
        }

        if ui
            .checkbox(&mut self.show_rk4_ghost, "Show Explicit RK4 Drift Twin")
            .accessible_hover_text(
                "Overlay synchronized non-symplectic RungeKutta4 trajectory and energy drift",
            )
            .changed()
        {
            self.state_rev = self.state_rev.wrapping_add(1);
        }
    }

    fn render_ensemble_actions(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("⚡ Burst +1,000 Steps")
                .accessible_hover_text(
                    "Advance all seeded orbits by 1,000 zero-allocation steps immediately",
                )
                .clicked()
            {
                self.step_batch(1_000);
            }
            if ui
                .button("🌌 Seed Showcase")
                .accessible_hover_text("Reset and seed curated KAM islands and chaotic sea orbits")
                .clicked()
            {
                self.seed_default_showcase();
            }
            if ui
                .button("🗑 Clear Orbits")
                .accessible_hover_text("Remove all seeded Poincaré trajectories")
                .clicked()
            {
                self.clear_orbits();
            }
        });
        if let Some(note) = &self.last_status_note {
            ui.small(note);
        }
    }

    fn render_conservation_readout(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let total_crossings: usize = self.orbits.iter().map(|s| s.section_points.len()).sum();
        ui.label(format!(
            "Active Orbits: {} / {}  ({} crossings)",
            self.orbits.len(),
            MAX_ORBITS,
            total_crossings
        ));
        ui.label(format!("Sim Time: t = {:.2}  ({:.2} ms/frame)", self.sim_time, self.last_frame_ms));
        ui.colored_label(
            palette.success,
            format!("Symplectic |ΔH/H₀|: {:.6}%", self.ghost.symplectic_err_pct),
        );
        if self.show_rk4_ghost {
            ui.colored_label(
                palette.error,
                format!("Explicit RK4 |ΔH/H₀|: {:.4}%", self.ghost.rk4_err_pct),
            );
        }
    }

    pub(super) fn render_central_workspace(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let avail = ui.available_size();
            let wide = avail.x >= 680.0;
            if wide {
                let left_w = (avail.x * 0.54).max(260.0);
                let right_w = (avail.x - left_w - 10.0).max(200.0);
                ui.horizontal(|ui| {
                    self.render_poincare_canvas(ui, egui::vec2(left_w, avail.y));
                    ui.vertical(|ui| {
                        ui.set_width(right_w);
                        self.render_comparator_pane(ui, egui::vec2(right_w, avail.y));
                    });
                });
            } else {
                let top_h = (avail.y * 0.54).max(200.0);
                let bot_h = (avail.y - top_h - 10.0).max(160.0);
                self.render_poincare_canvas(ui, egui::vec2(avail.x, top_h));
                ui.add_space(6.0);
                self.render_comparator_pane(ui, egui::vec2(avail.x, bot_h));
            }
        });
    }

    fn render_comparator_pane(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        ui.horizontal_wrapped(|ui| {
            for mode in [
                ComparatorView::EnergyDrift,
                ComparatorView::ConfigurationSpace,
            ] {
                if ui
                    .selectable_label(self.comparator_view == mode, mode.label())
                    .clicked()
                {
                    self.comparator_view = mode;
                    self.state_rev = self.state_rev.wrapping_add(1);
                }
            }
        });
        let plot_h = (size.y - 30.0).max(120.0);
        let palette = SemanticPalette::from_visuals(ui.visuals());
        match self.comparator_view {
            ComparatorView::EnergyDrift => self.render_energy_drift_plot(ui, plot_h, &palette),
            ComparatorView::ConfigurationSpace => {
                self.render_configuration_plot(ui, plot_h, &palette);
            }
        }
    }

    fn render_energy_drift_plot(
        &self,
        ui: &mut egui::Ui,
        height: f32,
        palette: &SemanticPalette,
    ) {
        Plot::new("hamiltonian_energy_drift_plot")
            .height(height)
            .legend(Legend::default().position(Corner::LeftTop))
            .x_axis_label("Simulation Time t")
            .y_axis_label("Relative Energy Error |ΔH / H₀| (%)")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(
                        format!("{} |ΔH/H₀| (%)", self.solver.label()),
                        PlotPoints::from(self.ghost.symplectic_drift.clone()),
                    )
                    .color(palette.success)
                    .width(2.0_f32),
                );
                if self.show_rk4_ghost {
                    plot_ui.line(
                        Line::new(
                            "Explicit RungeKutta4 |ΔH/H₀| (%)",
                            PlotPoints::from(self.ghost.rk4_drift.clone()),
                        )
                        .color(palette.error)
                        .style(LineStyle::dashed_dense())
                        .width(2.0_f32),
                    );
                }
            });
    }

    fn render_configuration_plot(
        &self,
        ui: &mut egui::Ui,
        height: f32,
        palette: &SemanticPalette,
    ) {
        Plot::new("hamiltonian_configuration_plot")
            .height(height)
            .data_aspect(1.0)
            .legend(Legend::default().position(Corner::LeftTop))
            .x_axis_label("x")
            .y_axis_label("y")
            .show(ui, |plot_ui| {
                for (idx, slot) in self.orbits.iter().enumerate() {
                    let color = palette.series_color(slot.color_index);
                    plot_ui.line(
                        Line::new(format!("Orbit {}", idx + 1), PlotPoints::from(slot.trail.clone()))
                            .color(color)
                            .width(1.8_f32),
                    );
                }
                if self.model == HamiltonianModelKind::FigureEightThreeBody {
                    if let Some(lead) = self.orbits.last() {
                        let bodies = FigureEightThreeBody::body_positions(&lead.state12).to_vec();
                        plot_ui.points(
                            Points::new("3-Body Masses", PlotPoints::from(bodies))
                                .color(palette.accent)
                                .radius(5.0_f32),
                        );
                    }
                }
                if self.show_rk4_ghost && !self.ghost.trail.is_empty() {
                    plot_ui.line(
                        Line::new("RK4 Ghost Twin", PlotPoints::from(self.ghost.trail.clone()))
                            .color(palette.error)
                            .style(LineStyle::dashed_loose())
                            .width(1.6_f32),
                    );
                }
            });
    }
}
