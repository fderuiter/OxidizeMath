//! Side-panel controls, 2x2 payoff matrix table, and replicator trajectory/phase plots
//! for [`super::EvolutionaryBiologyTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::EvolutionPlotView;
use super::EvolutionaryBiologyTool;
use crate::app::SemanticPalette;
use crate::framework::recovery::FiniteStateGuard;
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use egui_plot::{HLine, Legend, Line, Plot, PlotPoints, Points, VLine};

impl EvolutionaryBiologyTool {
    pub(super) fn render_studio(&mut self, ui: &mut egui::Ui) {
        self.draw_side_panel(ui);
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.draw_ess_banner(ui);
            ui.add_space(4.0);
            self.draw_plots(ui);
        });
    }

    fn draw_side_panel(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "evolutionary_biology_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Evolutionary Replicator Lab");
                ui.separator();
                self.draw_recovery_card(ui);
                self.draw_view_selector(ui);
                ui.separator();
                self.draw_payoff_sliders(ui);
                ui.separator();
                self.draw_payoff_matrix_card(ui);
            });
        });
    }

    fn draw_recovery_card(&mut self, ui: &mut egui::Ui) {
        if !FiniteStateGuard::diverged(self) {
            return;
        }
        let palette = SemanticPalette::from_visuals(ui.visuals());
        egui::Frame::group(ui.style())
            .stroke(egui::Stroke::new(1.5_f32, palette.error))
            .show(ui, |ui| {
                ui.colored_label(palette.error, "⚠ Replicator Parameter Guard Triggered");
                if let Some(diag) = &self.diagnostic {
                    ui.small(&diag.message);
                }
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Clamp Parameters").clicked() {
                        FiniteStateGuard::clamp_parameters(self);
                    }
                    if ui.button("Reset Defaults").clicked() {
                        FiniteStateGuard::reset_parameters(self);
                    }
                });
            });
        ui.separator();
    }

    fn draw_view_selector(&mut self, ui: &mut egui::Ui) {
        ui.label("Plot View:");
        for view in EvolutionPlotView::ALL {
            if ui
                .selectable_label(self.state.view == view, view.label())
                .clicked()
            {
                self.state.view = view;
            }
        }
    }

    fn draw_payoff_sliders(&mut self, ui: &mut egui::Ui) {
        ui.label("Hawk-Dove Payoff & Integration:");
        let before = self.state.clone();
        ui.add(
            AnimatedSlider::new(&mut self.state.resource_value, 0.1..=20.0)
                .text("Resource Value V")
                .id_salt("evo_v"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.state.injury_cost, 0.1..=30.0)
                .text("Injury Cost C")
                .id_salt("evo_c"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.state.initial_hawk_freq, 0.01..=0.99)
                .text("Initial Hawk p₀")
                .id_salt("evo_p0"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.state.horizon, 2.0..=80.0)
                .text("Time Horizon T")
                .id_salt("evo_horizon"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.state.dt, 0.005..=0.25)
                .text("Step Δt")
                .id_salt("evo_dt"),
        );
        ui.checkbox(
            &mut self.state.show_ensemble,
            "Show Boundary Ensemble (p₀ = 0.08, 0.92)",
        );

        if self.state != before {
            self.recompute();
        }
    }

    fn draw_payoff_matrix_card(&self, ui: &mut egui::Ui) {
        let pop = self.population();
        let [[e_hh, e_hd], [e_dh, e_dd]] = pop.payoff_matrix();
        ui.label("2×2 Payoff Matrix E(row, col):");
        egui::Grid::new("hawk_dove_payoff_grid")
            .striped(true)
            .show(ui, |ui| {
                ui.strong("Strategy");
                ui.strong("vs Hawk");
                ui.strong("vs Dove");
                ui.end_row();

                ui.label("Hawk");
                ui.monospace(format!("{e_hh:+.2}"));
                ui.monospace(format!("{e_hd:+.2}"));
                ui.end_row();

                ui.label("Dove");
                ui.monospace(format!("{e_dh:+.2}"));
                ui.monospace(format!("{e_dd:+.2}"));
                ui.end_row();
            });
    }

    fn draw_ess_banner(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let pop = self.population();
        let ess = pop.ess_hawk_frequency();
        let (w_h, w_d) = pop.expected_fitness(self.state.initial_hawk_freq);
        ui.horizontal_wrapped(|ui| {
            if self.state.injury_cost > self.state.resource_value {
                ui.colored_label(
                    palette.success,
                    format!("● Mixed Polymorphic ESS p* = V/C = {ess:.3}"),
                );
            } else {
                ui.colored_label(
                    palette.warning,
                    "● Pure Hawk Dominance ESS p* = 1.000 (V ≥ C)",
                );
            }
            ui.separator();
            ui.label(format!(
                "At p₀ = {:.2}: W_Hawk = {w_h:.2}, W_Dove = {w_d:.2}",
                self.state.initial_hawk_freq
            ));
        });
    }

    fn draw_plots(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let avail_h = (ui.available_height() - 8.0).max(240.0);
        match self.state.view {
            EvolutionPlotView::TrajectoryAndPhase => {
                let half_h = (avail_h * 0.48).max(150.0);
                self.draw_trajectory_plot(ui, half_h, &palette);
                ui.add_space(6.0);
                self.draw_phase_plot(ui, half_h, &palette);
            }
            EvolutionPlotView::TrajectoryOnly => {
                self.draw_trajectory_plot(ui, avail_h, &palette);
            }
            EvolutionPlotView::PhaseLineOnly => {
                self.draw_phase_plot(ui, avail_h, &palette);
            }
        }
    }

    fn draw_trajectory_plot(&self, ui: &mut egui::Ui, height: f32, palette: &SemanticPalette) {
        let ess = self.population().ess_hawk_frequency();
        Plot::new(crate::framework::scoped_panel_id(ui, "evo_traj_plot"))
            .height(height)
            .legend(Legend::default())
            .x_axis_label("Evolutionary Time t")
            .y_axis_label("Strategy Frequency")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(
                        format!("Hawk p_H(t) (p₀={:.2})", self.state.initial_hawk_freq),
                        PlotPoints::from(self.primary_hawk_traj.clone()),
                    )
                    .color(palette.accent)
                    .width(2.4_f32),
                );
                let dove_pts: Vec<[f64; 2]> = self
                    .primary_hawk_traj
                    .iter()
                    .map(|&[t, p]| [t, 1.0 - p])
                    .collect();
                plot_ui.line(
                    Line::new("Dove p_D(t) = 1 - p_H(t)", PlotPoints::from(dove_pts))
                        .color(palette.info)
                        .width(1.8_f32),
                );
                if self.state.show_ensemble {
                    plot_ui.line(
                        Line::new(
                            "Ensemble p₀ = 0.08",
                            PlotPoints::from(self.low_traj.clone()),
                        )
                        .color(palette.muted_text)
                        .width(1.3_f32),
                    );
                    plot_ui.line(
                        Line::new(
                            "Ensemble p₀ = 0.92",
                            PlotPoints::from(self.high_traj.clone()),
                        )
                        .color(palette.muted_text)
                        .width(1.3_f32),
                    );
                }
                plot_ui.hline(
                    HLine::new(format!("ESS Hawk Equilibrium p* = {ess:.3}"), ess)
                        .color(palette.success)
                        .width(1.6_f32),
                );
            });
    }

    fn draw_phase_plot(&self, ui: &mut egui::Ui, height: f32, palette: &SemanticPalette) {
        let ess = self.population().ess_hawk_frequency();
        Plot::new(crate::framework::scoped_panel_id(ui, "evo_phase_plot"))
            .height(height)
            .legend(Legend::default())
            .x_axis_label("Hawk Frequency p_H")
            .y_axis_label("Replicator Velocity dp_H / dt")
            .show(ui, |plot_ui| {
                plot_ui.hline(HLine::new("Zero Velocity", 0.0).color(palette.muted_text));
                plot_ui.line(
                    Line::new(
                        "dp_H/dt = ½ p_H(1 - p_H)(V - p_H C)",
                        PlotPoints::from(self.phase_curve.clone()),
                    )
                    .color(palette.warning)
                    .width(2.3_f32),
                );
                plot_ui.vline(
                    VLine::new(format!("Stable ESS p* = {ess:.3}"), ess)
                        .color(palette.success)
                        .width(1.5_f32),
                );
                plot_ui.points(
                    Points::new("Stable ESS Attractor", vec![[ess, 0.0]])
                        .color(palette.success)
                        .radius(5.5_f32),
                );
            });
    }
}
