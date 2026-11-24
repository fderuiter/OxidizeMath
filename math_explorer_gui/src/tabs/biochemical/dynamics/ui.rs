//! Side-panel controls and paired `egui_plot` views for [`super::BiochemicalDynamicsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::KineticsPlotView;
use super::BiochemicalDynamicsTool;
use crate::app::SemanticPalette;
use crate::async_sim::SimWorkerState;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{PlaybackAction, PlaybackTransport};
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, VLine};

impl BiochemicalDynamicsTool {
    pub(super) fn render_studio(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        if self.state.playing && !FiniteStateGuard::diverged(self) {
            self.step_cooperative_frame();
            if self.state.playing {
                ctx.request_repaint();
            }
        }
        self.draw_side_panel(ui);
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.draw_summary_strip(ui);
            ui.add_space(4.0);
            self.draw_plots(ui);
        });
    }

    fn draw_side_panel(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "biochemical_dynamics_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Enzyme Kinetics & RD");
                ui.separator();
                self.draw_recovery_card(ui);
                self.draw_transport_and_view(ui);
                ui.separator();
                self.draw_kinetics_sliders(ui);
                ui.separator();
                self.draw_spatial_rd_sliders(ui);
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
                ui.colored_label(palette.error, "⚠ Biochemical Stability Guard Triggered");
                if let Some(diag) = &self.diagnostic {
                    ui.small(&diag.message);
                }
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Auto-Scale dt").clicked() {
                        FiniteStateGuard::auto_scale_step_size(self);
                    }
                    if ui.button("Clamp Params").clicked() {
                        FiniteStateGuard::clamp_parameters(self);
                    }
                    if ui.button("Reset").clicked() {
                        FiniteStateGuard::reset_parameters(self);
                    }
                });
            });
        ui.separator();
    }

    fn draw_transport_and_view(&mut self, ui: &mut egui::Ui) {
        let worker_state = if FiniteStateGuard::diverged(self) {
            SimWorkerState::Faulted
        } else if self.state.playing {
            SimWorkerState::Running
        } else {
            SimWorkerState::Ready
        };
        if let Some(action) = PlaybackTransport::from_worker_state(
            "biochemical_rd_transport",
            "Enzyme Reaction-Diffusion",
            worker_state,
        )
        .can_reset(true)
        .show(ui)
        {
            match action {
                PlaybackAction::Play => self.state.playing = true,
                PlaybackAction::Pause => self.state.playing = false,
                PlaybackAction::Step => {
                    self.state.playing = false;
                    self.step_once();
                }
                PlaybackAction::Reset => self.rebuild_reaction_diffusion(),
                PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
            }
        }
        ui.add_space(4.0);
        ui.label("Plot Layout:");
        for view in KineticsPlotView::ALL {
            if ui
                .selectable_label(self.state.view == view, view.label())
                .clicked()
            {
                self.state.view = view;
            }
        }
    }

    fn draw_kinetics_sliders(&mut self, ui: &mut egui::Ui) {
        ui.label("Enzyme Saturation & Cooperativity:");
        let prev = (
            self.state.v_max,
            self.state.k_m,
            self.state.hill_n,
            self.state.max_substrate,
        );
        ui.add(
            AnimatedSlider::new(&mut self.state.v_max, 0.1..=20.0)
                .text("Max Velocity V_max")
                .id_salt("bio_vmax"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.state.k_m, 0.05..=10.0)
                .text("Half-Saturation K_m")
                .id_salt("bio_km"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.state.hill_n, 0.5..=8.0)
                .text("Hill Coefficient n")
                .id_salt("bio_hill_n"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.state.max_substrate, 1.0..=40.0)
                .text("Substrate Span [S]_max")
                .id_salt("bio_s_max"),
        );
        let curr = (
            self.state.v_max,
            self.state.k_m,
            self.state.hill_n,
            self.state.max_substrate,
        );
        if prev != curr {
            self.rebuild_reaction_diffusion();
        }
    }

    fn rd_param_snapshot(&self) -> (f64, f64, f64, f64, f64, usize, f64) {
        (
            self.state.substrate_reservoir,
            self.state.feed_rate,
            self.state.degradation_rate,
            self.state.diffusion_activator,
            self.state.diffusion_substrate,
            self.state.grid_points,
            self.state.dt,
        )
    }

    fn draw_spatial_rd_sliders(&mut self, ui: &mut egui::Ui) {
        ui.label("1D Reaction-Diffusion Chamber:");
        let prev = self.rd_param_snapshot();
        ui.add(
            AnimatedSlider::new(&mut self.state.substrate_reservoir, 0.2..=10.0)
                .text("Reservoir s₀")
                .id_salt("bio_s0"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.state.feed_rate, 0.01..=2.0)
                .text("Feed Rate k_feed")
                .id_salt("bio_k_feed"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.state.degradation_rate, 0.01..=2.0)
                .text("Clearance k_deg")
                .id_salt("bio_k_deg"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.state.diffusion_activator, 0.005..=2.0)
                .text("Activator D_u")
                .id_salt("bio_du"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.state.diffusion_substrate, 0.005..=2.0)
                .text("Substrate D_s")
                .id_salt("bio_ds"),
        );
        let mut gp_f64 = self.state.grid_points as f64;
        if ui
            .add(
                AnimatedSlider::new(&mut gp_f64, 16.0..=256.0)
                    .text("Grid Nodes N")
                    .id_salt("bio_grid_n"),
            )
            .changed()
        {
            self.state.grid_points = (gp_f64.round() as usize).clamp(16, 256);
        }
        ui.add(
            AnimatedSlider::new(&mut self.state.dt, 0.001..=0.1)
                .text("Step Δt")
                .id_salt("bio_dt"),
        );
        if prev != self.rd_param_snapshot() {
            self.rebuild_reaction_diffusion();
        }
    }

    fn draw_summary_strip(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let coop_label = if self.state.hill_n > 1.05 {
            ("Positive Cooperativity (n > 1)", palette.success)
        } else if self.state.hill_n < 0.95 {
            ("Negative Cooperativity (n < 1)", palette.warning)
        } else {
            ("Hyperbolic Non-Cooperative (n ≈ 1)", palette.info)
        };
        let peak_u = self
            .rd_system
            .state
            .species(0)
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(coop_label.1, format!("● {}", coop_label.0));
            ui.separator();
            ui.label(format!(
                "Half-Max V_max/2 = {:.2} at K_m = {:.2}",
                self.state.v_max * 0.5,
                self.state.k_m
            ));
            ui.separator();
            ui.label(format!(
                "RD Time t = {:.2} ({} steps, {:.2} ms/step) | Peak Activator u = {:.3}",
                self.sim_time, self.step_count, self.last_step_ms, peak_u
            ));
        });
    }

    fn draw_plots(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let avail_h = (ui.available_height() - 8.0).max(240.0);
        match self.state.view {
            KineticsPlotView::CoupledSplit => {
                let half_h = (avail_h * 0.48).max(150.0);
                self.draw_saturation_plot(ui, half_h, &palette);
                ui.add_space(6.0);
                self.draw_spatial_rd_plot(ui, half_h, &palette);
            }
            KineticsPlotView::SaturationCurve => {
                self.draw_saturation_plot(ui, avail_h, &palette);
            }
            KineticsPlotView::SpatialReactionDiffusion => {
                self.draw_spatial_rd_plot(ui, avail_h, &palette);
            }
        }
    }

    fn draw_saturation_plot(&self, ui: &mut egui::Ui, height: f32, palette: &SemanticPalette) {
        let (mm_pts, hill_pts) = self.sample_kinetics_curves();
        Plot::new(crate::framework::scoped_panel_id(ui, "bio_kinetics_plot"))
            .height(height)
            .legend(Legend::default())
            .x_axis_label("Substrate Concentration [S]")
            .y_axis_label("Reaction Velocity v([S])")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Michaelis-Menten (n = 1)", PlotPoints::from(mm_pts))
                        .color(palette.info)
                        .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new(
                        format!("Hill Cooperativity (n = {:.2})", self.state.hill_n),
                        PlotPoints::from(hill_pts),
                    )
                    .color(palette.accent)
                    .width(2.4_f32),
                );
                plot_ui.vline(
                    VLine::new("Half-Saturation K_m", self.state.k_m)
                        .color(palette.warning)
                        .width(1.2_f32),
                );
            });
    }

    fn draw_spatial_rd_plot(&self, ui: &mut egui::Ui, height: f32, palette: &SemanticPalette) {
        let (u_pts, s_pts) = self.sample_spatial_profiles();
        Plot::new(crate::framework::scoped_panel_id(ui, "bio_spatial_rd_plot"))
            .height(height)
            .legend(Legend::default())
            .x_axis_label("1D Spatial Position x")
            .y_axis_label("Concentration")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Activator u(x, t)", PlotPoints::from(u_pts))
                        .color(palette.success)
                        .width(2.3_f32),
                );
                plot_ui.line(
                    Line::new("Substrate s(x, t)", PlotPoints::from(s_pts))
                        .color(palette.warning)
                        .width(1.9_f32),
                );
            });
    }
}
