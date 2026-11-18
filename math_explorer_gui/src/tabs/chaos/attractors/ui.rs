#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AttractorPlotter, SolverExecutionState};
use crate::accessibility::AccessibleTheoryHover;
use crate::async_sim::{SimCommand, SimWorkerState};
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui::{self, Color32, CornerRadius, Stroke, Vec2};
use egui_plot::{Callout, Line, Plot, PlotPoints};
use nalgebra::Vector3;

impl AttractorPlotter {
    fn draw_simulation_section(&mut self, ui: &mut egui::Ui) {
        let transport_state = if self.diverged() {
            SimWorkerState::Faulted
        } else if self.is_paused() && self.worker.state().is_running() {
            SimWorkerState::Ready
        } else {
            self.worker.state()
        };

        if let Some(action) = PlaybackTransport::from_worker_state(
            "attractor_transport",
            "Attractor Plotter",
            transport_state,
        )
        .can_reset(true)
        .show(ui)
        {
            match action {
                PlaybackAction::Play => {
                    let _ = self.try_resume();
                }
                PlaybackAction::Pause => {
                    self.execution_state = SolverExecutionState::Paused;
                    let _ = self.worker.send_command(SimCommand::Pause);
                }
                PlaybackAction::Step => {
                    self.execution_state = SolverExecutionState::Paused;
                    let _ = self.worker.send_command(SimCommand::Step(1));
                }
                PlaybackAction::Reset => {
                    self.reset();
                }
            }
        }

        if ui
            .add(
                egui::Slider::new(&mut self.simulation_speed, 1..=50).text("Speed (steps/frame)"),
            )
            .changed()
        {
            self.on_slider_changed();
        }
        if ui
            .add(egui::Slider::new(&mut self.dt, 0.001..=0.05).text("Time Step (dt)"))
            .changed()
        {
            self.on_slider_changed();
        }
        if ui
            .add(egui::Slider::new(&mut self.max_points, 100..=10000).text("Max Points"))
            .changed()
        {
            self.on_slider_changed();
        }
    }

    pub(super) fn draw_controls(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("attractor_controls").show(ctx, |ui| {
            ui.heading("Lorenz Attractor");
            if ui.button("🎨 Starter Gallery").clicked() {
                let cur = ctx.data(|d| {
                    d.get_temp::<bool>(egui::Id::new("SHOW_STARTER_GALLERY"))
                        .unwrap_or(false)
                });
                ctx.data_mut(|d| d.insert_temp(egui::Id::new("SHOW_STARTER_GALLERY"), !cur));
            }
            ui.separator();

            ui.collapsing("Parameters", |ui| {
                if ui
                    .add(egui::Slider::new(&mut self.system.sigma, 0.0..=50.0).text("Prandtl Number (σ)"))
                    .changed()
                {
                    self.on_slider_changed();
                }
                if ui
                    .add(egui::Slider::new(&mut self.system.rho, 0.0..=100.0).text("Rayleigh Number (ρ)"))
                    .changed()
                {
                    self.on_slider_changed();
                }
                if ui
                    .add(egui::Slider::new(&mut self.system.beta, 0.0..=10.0).text("Geometric Factor (β)"))
                    .changed()
                {
                    self.on_slider_changed();
                }
            });

            egui::CollapsingHeader::new("Simulation")
                .default_open(true)
                .show(ui, |ui| {
                    self.draw_simulation_section(ui);
                });

            ui.collapsing("View", |ui| {
                self.camera.ui(ui);
            });

            ui.separator();
            ui.checkbox(&mut self.show_annotations, "Show Fixed Points");
            ui.separator();
            let state = self.system.state.vec;
            ui.label(format!("X: {:.2}", state.x));
            ui.label(format!("Y: {:.2}", state.y));
            ui.label(format!("Z: {:.2}", state.z));
        });
    }

    fn draw_fixed_point_callouts(&self, plot_ui: &mut egui_plot::PlotUi<'_>) {
        if !self.show_annotations
            || self.system.rho <= 1.0
            || self.system.beta <= 0.0
            || !self.system.rho.is_finite()
            || !self.system.beta.is_finite()
        {
            return;
        }
        let x = (self.system.beta * (self.system.rho - 1.0)).sqrt();
        let z = self.system.rho - 1.0;
        if !x.is_finite() || !z.is_finite() {
            return;
        }

        let c_pos = self.project(Vector3::new(x, x, z));
        let c_neg = self.project(Vector3::new(-x, -x, z));
        let visuals = plot_ui.ctx().style().visuals.clone();
        let bg_fill = Color32::from_rgba_unmultiplied(
            visuals.window_fill().r(),
            visuals.window_fill().g(),
            visuals.window_fill().b(),
            230,
        );
        let frame_stroke = visuals.widgets.noninteractive.bg_stroke;

        plot_ui.callout(
            Callout::new(c_pos, format!("Fixed Point C+ ({x:.1}, {x:.1}, {z:.1})"))
                .offset(Vec2::new(35.0, -25.0))
                .leader_stroke(Stroke::new(1.5_f32, Color32::from_rgb(231, 76, 60)))
                .frame(bg_fill, frame_stroke)
                .corner_radius(CornerRadius::same(4))
                .text_color(visuals.text_color()),
        );
        plot_ui.callout(
            Callout::new(
                c_neg,
                format!("Fixed Point C- ({:.1}, {:.1}, {z:.1})", -x, -x),
            )
            .offset(Vec2::new(-35.0, 25.0))
            .leader_stroke(Stroke::new(1.5_f32, Color32::from_rgb(155, 89, 182)))
            .frame(bg_fill, frame_stroke)
            .corner_radius(CornerRadius::same(4))
            .text_color(visuals.text_color()),
        );
    }

    pub(super) fn draw_central_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if let SolverExecutionState::Diverged {
                invalid_vec,
                diverged_at_step,
            } = self.execution_state
            {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "⚠️ Solver Divergence Warning: Non-finite state detected at step {} (coordinates: [{:.2}, {:.2}, {:.2}]). Simulation halted.",
                                diverged_at_step, invalid_vec.x, invalid_vec.y, invalid_vec.z
                            ))
                            .color(egui::Color32::RED)
                            .strong(),
                        );
                        if ui.button("↻ Reset Parameters").clicked() {
                            self.reset_parameters();
                        }
                        if ui.button("↻ Reset State").clicked() {
                            self.reset();
                        }
                    });
                });
            }

            let points: Vec<[f64; 2]> = self.history.iter().map(|p| self.project(*p)).collect();
            let response = Plot::new("attractor_plot")
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new("Trajectory", PlotPoints::new(points))
                            .color(egui::Color32::from_rgb(100, 200, 255)),
                    );
                    self.draw_fixed_point_callouts(plot_ui);
                })
                .response;

            self.camera.handle_interaction(&response, ui);
            response.accessible_theory_hover(&self.system);
            ui.label("Drag 'Yaw' and 'Pitch' in the side panel to rotate the view.");
        });
    }
}
