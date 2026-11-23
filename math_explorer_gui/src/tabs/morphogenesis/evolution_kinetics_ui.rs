#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::evolution_kinetics_tool::EvolutionKineticsTool;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints, Points};

impl EvolutionKineticsTool {
    fn draw_replicator_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("Hawk-Dove Replicator Game");
        changed |= ui
            .add(egui::Slider::new(&mut self.state.resource_value, 0.1..=20.0).text("Reward V"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.cost_of_injury, 0.1..=20.0).text("Injury Cost C"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.initial_hawk_freq, 0.01..=0.99)
                    .text("Initial p_H(0)"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.time_horizon, 2.0..=100.0).text("Horizon T"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.dt, 0.005..=0.25).text("Step dt"))
            .changed();
        changed
    }

    fn draw_kinetics_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("Michaelis-Menten & Hill Kinetics");
        changed |= ui
            .add(egui::Slider::new(&mut self.state.v_max, 0.5..=50.0).text("V_max"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.k_m, 0.1..=20.0).text("K_m"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.hill_n, 0.2..=8.0).text("Hill n"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.substrate_probe, 0.0..=30.0).text("Probe [S]"))
            .changed();
        changed
    }

    pub(crate) fn draw_controls(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "evolution_kinetics_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("Evolution & Kinetics");
            ui.separator();

            let mut changed = self.draw_replicator_sliders(ui);
            ui.separator();
            changed |= self.draw_kinetics_sliders(ui);
            if changed {
                self.recompute();
            }

            ui.separator();
            let ess = self.state.analytical_ess();
            let final_ph = self.hawk_curve.last().map(|p| p[1]).unwrap_or(ess);
            let (v_mm, v_hill) = self.probe_velocities();
            ui.label(format!("Analytical ESS p*: {ess:.3}"));
            ui.label(format!("Terminal p_H(T): {final_ph:.3}"));
            ui.label(format!("Michaelis-Menten v([S]): {v_mm:.3}"));
            ui.label(format!("Hill Cooperative v([S]): {v_hill:.3}"));
        });
    }

    fn draw_replicator_plot(
        &self,
        ui: &mut egui::Ui,
        palette: &crate::app::SemanticPalette,
        height: f32,
    ) -> egui::Rect {
        Plot::new("hawk_dove_replicator_plot")
            .height(height)
            .legend(Legend::default())
            .x_axis_label("Time t")
            .y_axis_label("Strategy Frequency")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(
                        "Hawk Frequency p_H(t)",
                        PlotPoints::from(self.hawk_curve.clone()),
                    )
                    .color(palette.series[0])
                    .width(2.2_f32),
                );
                plot_ui.line(
                    Line::new(
                        "Dove Frequency p_D(t)",
                        PlotPoints::from(self.dove_curve.clone()),
                    )
                    .color(palette.series[1])
                    .width(1.8_f32),
                );
                plot_ui.line(
                    Line::new("Analytical ESS p*", PlotPoints::from(self.ess_line.clone()))
                        .color(palette.accent)
                        .style(LineStyle::dashed_dense()),
                );
            })
            .response
            .rect
    }

    fn draw_kinetics_plot(
        &self,
        ui: &mut egui::Ui,
        palette: &crate::app::SemanticPalette,
        height: f32,
        probe: (f64, f64, f64),
    ) -> egui::Rect {
        let (s_probe, v_mm, v_hill) = probe;
        Plot::new("enzyme_saturation_plot")
            .height(height)
            .legend(Legend::default())
            .x_axis_label("Substrate Concentration [S]")
            .y_axis_label("Reaction Velocity v([S])")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(
                        "Michaelis-Menten (n = 1)",
                        PlotPoints::from(self.mm_curve.clone()),
                    )
                    .color(palette.series[0])
                    .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new(
                        format!("Hill Kinetics (n = {:.2})", self.state.hill_n),
                        PlotPoints::from(self.hill_curve.clone()),
                    )
                    .color(palette.accent)
                    .width(2.2_f32),
                );
                plot_ui.points(
                    Points::new(
                        "Probe Velocity",
                        PlotPoints::from(vec![[s_probe, v_mm], [s_probe, v_hill]]),
                    )
                    .radius(4.5_f32)
                    .color(palette.series[1]),
                );
            })
            .response
            .rect
    }

    pub(crate) fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let avail_h = (ui.available_height() * 0.47).max(160.0);
            let top_rect = self.draw_replicator_plot(ui, &palette, avail_h);
            ui.separator();
            let (v_mm, v_hill) = self.probe_velocities();
            let bot_rect = self.draw_kinetics_plot(
                ui,
                &palette,
                avail_h,
                (self.state.substrate_probe, v_mm, v_hill),
            );

            let focus_resp = ui.interact(
                top_rect.union(bot_rect),
                ui.id().with("evolution_kinetics_a11y"),
                egui::Sense::focusable_noninteractive(),
            );
            let summary = format!(
                "Hawk-Dove ESS {:.3}, Michaelis-Menten velocity {:.2}, Hill velocity {:.2}",
                self.state.analytical_ess(),
                v_mm,
                v_hill
            );
            crate::accessibility::register_canvas_accessibility(
                ui,
                &focus_resp,
                egui::accesskit::Role::GraphicsDocument,
                "Evolution Replicator and Enzyme Kinetics Plot",
                Some(&summary),
            );
            crate::accessibility::paint_canvas_focus_ring(ui, &focus_resp);
        });
    }
}
