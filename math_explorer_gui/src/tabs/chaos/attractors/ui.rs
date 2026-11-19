#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AttractorPlotter, SolverExecutionState};
use crate::accessibility::AccessibleTheoryHover;
use crate::async_sim::ui::{
    render_custom_fault_banner, render_empty_plot_guidance, render_worker_fault_banner,
    WorkerFaultAction,
};
use crate::async_sim::{SimCommand, SimWorkerState};
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui::{self, Color32, CornerRadius, Stroke, Vec2};
use egui_plot::{Callout, Line, Plot, PlotPoints, Points};
use nalgebra::Vector3;

impl AttractorPlotter {
    pub(super) fn draw_side_panel(&mut self, ctx: &egui::Context) {
        let worker_faulted = self.worker.state() == SimWorkerState::Faulted;
        egui::SidePanel::left("attractor_controls").show(ctx, |ui| {
            ui.heading("Lorenz Attractor");
            ui.separator();

            if worker_faulted && !self.diverged() {
                if let Some(action) = render_worker_fault_banner(
                    ui,
                    self.worker.state(),
                    self.worker.diagnostic_report(),
                    false,
                ) {
                    self.apply_fault_action(action, ctx);
                    ui.separator();
                }
            }

            self.draw_parameter_sliders(ui);
            self.draw_simulation_controls(ui, ctx, worker_faulted);

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

    fn draw_parameter_sliders(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Parameters", |ui| {
            if ui
                .add(
                    egui::Slider::new(&mut self.system.sigma, 0.0..=50.0)
                        .text("Prandtl Number (σ)"),
                )
                .changed()
            {
                self.on_slider_changed();
            }
            if ui
                .add(
                    egui::Slider::new(&mut self.system.rho, 0.0..=100.0)
                        .text("Rayleigh Number (ρ)"),
                )
                .changed()
            {
                self.on_slider_changed();
            }
            if ui
                .add(
                    egui::Slider::new(&mut self.system.beta, 0.0..=10.0)
                        .text("Geometric Factor (β)"),
                )
                .changed()
            {
                self.on_slider_changed();
            }
        });
    }

    fn draw_simulation_controls(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        worker_faulted: bool,
    ) {
        egui::CollapsingHeader::new("Simulation")
            .default_open(true)
            .show(ui, |ui| {
                let transport_state = if self.diverged() || worker_faulted {
                    SimWorkerState::Faulted
                } else if self.worker.state().is_stepping() {
                    SimWorkerState::Stepping
                } else if self.is_paused() {
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
                            ctx.request_repaint();
                        }
                    }
                }

                if ui
                    .add(
                        egui::Slider::new(&mut self.simulation_speed, 1..=super::MAX_SIMULATION_SPEED)
                            .text("Speed (steps/frame)"),
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
            });
    }

    fn apply_fault_action(&mut self, action: WorkerFaultAction, ctx: &egui::Context) {
        use crate::framework::recovery::RecoveryAction;
        match action {
            WorkerFaultAction::Reset => self.reset(),
            WorkerFaultAction::Recovery(RecoveryAction::AutoScaleStepSize) => {
                self.dt = if self.dt.is_finite() {
                    (self.dt * 0.5).max(0.001)
                } else {
                    0.005
                };
                self.reset();
            }
            WorkerFaultAction::Recovery(RecoveryAction::ClampParameters) => {
                self.system.sigma = if self.system.sigma.is_finite() {
                    self.system.sigma.clamp(0.0, 50.0)
                } else {
                    10.0
                };
                self.system.rho = if self.system.rho.is_finite() {
                    self.system.rho.clamp(0.0, 100.0)
                } else {
                    28.0
                };
                self.system.beta = if self.system.beta.is_finite() {
                    self.system.beta.clamp(0.0, 10.0)
                } else {
                    8.0 / 3.0
                };
                self.dt = if self.dt.is_finite() {
                    self.dt.clamp(0.001, 0.05)
                } else {
                    0.01
                };
                self.reset();
            }
            WorkerFaultAction::ResetParameters
            | WorkerFaultAction::Recovery(RecoveryAction::ResetParameters) => {
                self.reset_parameters();
            }
        }
        ctx.request_repaint();
    }

    fn draw_central_fault_banner(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if let SolverExecutionState::Diverged {
            invalid_vec,
            diverged_at_step,
        } = self.execution_state
        {
            let detail = format!(
                "Step {} | Coordinates: [{:.2}, {:.2}, {:.2}]",
                diverged_at_step, invalid_vec.x, invalid_vec.y, invalid_vec.z
            );
            if let Some(action) = render_custom_fault_banner(
                ui,
                "⚠ Solver Divergence Warning: Non-finite state detected. Simulation halted.",
                Some(&detail),
                true,
            ) {
                self.apply_fault_action(action, ctx);
            }
        } else if self.worker.state() == SimWorkerState::Faulted {
            if let Some(action) = render_worker_fault_banner(
                ui,
                self.worker.state(),
                self.worker.diagnostic_report(),
                false,
            ) {
                self.apply_fault_action(action, ctx);
            }
        }
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
        let palette = crate::app::SemanticPalette::from_visuals(&visuals);
        let bg_fill = Color32::from_rgba_unmultiplied(
            visuals.window_fill().r(),
            visuals.window_fill().g(),
            visuals.window_fill().b(),
            230,
        );
        let frame_stroke = visuals.widgets.noninteractive.bg_stroke;

        let callout_pos = Callout::new(
            c_pos,
            format!("Fixed Point C+ ({x:.1}, {x:.1}, {z:.1})"),
        )
        .offset(Vec2::new(35.0, -25.0))
        .leader_stroke(Stroke::new(1.5_f32, palette.series_color(3)))
        .frame(bg_fill, frame_stroke)
        .corner_radius(CornerRadius::same(4))
        .text_color(visuals.text_color());

        let callout_neg = Callout::new(
            c_neg,
            format!("Fixed Point C- ({:.1}, {:.1}, {z:.1})", -x, -x),
        )
        .offset(Vec2::new(-35.0, 25.0))
        .leader_stroke(Stroke::new(1.5_f32, palette.series_color(4)))
        .frame(bg_fill, frame_stroke)
        .corner_radius(CornerRadius::same(4))
        .text_color(visuals.text_color());

        plot_ui.callout(callout_pos);
        plot_ui.callout(callout_neg);
    }

    pub(super) fn draw_central_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_central_fault_banner(ui, ctx);

            if self.history.is_empty() && !self.diverged() {
                render_empty_plot_guidance(
                    ui,
                    "Initial Lorenz state (1, 1, 1) ready — click ▶ Play or ⏭ Step to trace the strange attractor.",
                );
            }

            let points: Vec<[f64; 2]> = self.history.iter().map(|p| self.project(*p)).collect();
            let initial_pt = (points.is_empty() && self.is_finite_state())
                .then(|| self.project(self.system.state.vec));

            let series0 = crate::app::SemanticPalette::from_visuals(ui.visuals()).series_color(0);
            let response = Plot::new("attractor_plot")
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Trajectory", PlotPoints::new(points)).color(series0));
                    if let Some(pt) = initial_pt {
                        plot_ui.points(
                            Points::new("Initial State", vec![pt])
                                .radius(4.0_f32)
                                .color(series0),
                        );
                    }
                    self.draw_fixed_point_callouts(plot_ui);
                })
                .response;

            self.camera.handle_interaction(&response, ui);
            response.accessible_theory_hover(&self.system);
            ui.label("Drag 'Yaw' and 'Pitch' in the side panel to rotate the view.");
        });
    }
}
