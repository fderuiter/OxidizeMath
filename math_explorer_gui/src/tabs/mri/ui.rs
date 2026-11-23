#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::pulse::MriPulseMode;
use super::runner::MriSnapshot;
use super::tool::MriTool;
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::{SimCommand, SimWorkerState};
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui;
use egui_plot::{Legend, Line, Plot, Points};

impl MriTool {
    fn draw_relaxation_controls(&mut self, ui: &mut egui::Ui) {
        let locked = self.active_challenge.is_some();
        ui.collapsing("Relaxation Parameters", |ui| {
            ui.add_enabled_ui(!locked, |ui| {
                if ui
                    .add(
                        egui::Slider::new(&mut self.simulator.t1, 0.1..=5.0)
                            .logarithmic(true)
                            .text("Longitudinal (T1) [s]"),
                    )
                    .changed()
                {
                    self.params.write().unwrap_or_else(|e| e.into_inner()).t1 = self.simulator.t1;
                    self.recompute_pulse_sequence();
                }
                if ui
                    .add(
                        egui::Slider::new(&mut self.simulator.t2, 0.01..=2.0)
                            .logarithmic(true)
                            .text("Transverse (T2) [s]"),
                    )
                    .changed()
                {
                    self.params.write().unwrap_or_else(|e| e.into_inner()).t2 = self.simulator.t2;
                    self.recompute_pulse_sequence();
                }
            });
            if self.simulator.t2 > self.simulator.t1 {
                let err = crate::app::SemanticPalette::from_visuals(ui.visuals()).error;
                ui.colored_label(err, "Warning: T2 > T1 is unphysical!");
            }
        });
    }

    fn draw_rf_pulse_controls(&mut self, ui: &mut egui::Ui) {
        let challenge_locked = self.active_challenge.is_some();
        ui.collapsing("RF Pulse Sequence (Rotating Frame)", |ui| {
            let mut changed = false;
            ui.add_enabled_ui(!challenge_locked, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Sequence:");
                    egui::ComboBox::from_id_salt("mri_pulse_mode")
                        .selected_text(match self.pulse_config.mode {
                            MriPulseMode::ContinuousField => "Continuous FID",
                            MriPulseMode::InversionPulse => "RF Inversion Pulse",
                            MriPulseMode::HahnSpinEcho => "Hahn Spin Echo",
                        })
                        .show_ui(ui, |ui| {
                            changed |= ui
                                .selectable_value(
                                    &mut self.pulse_config.mode,
                                    MriPulseMode::ContinuousField,
                                    "Continuous FID",
                                )
                                .changed();
                            changed |= ui
                                .selectable_value(
                                    &mut self.pulse_config.mode,
                                    MriPulseMode::InversionPulse,
                                    "RF Inversion Pulse",
                                )
                                .changed();
                            changed |= ui
                                .selectable_value(
                                    &mut self.pulse_config.mode,
                                    MriPulseMode::HahnSpinEcho,
                                    "Hahn Spin Echo",
                                )
                                .changed();
                        });
                });
            });
            if !self.pulse_config.mode.is_continuous() {
                changed |= self.draw_rf_pulse_sliders(ui, challenge_locked);
            }
            if changed {
                if !self.pulse_config.mode.is_continuous() {
                    let _ = self.worker.send_command(SimCommand::Pause);
                    self.recompute_pulse_sequence();
                } else {
                    self.reset();
                }
            }
        });
    }

    fn draw_rf_pulse_sliders(&mut self, ui: &mut egui::Ui, challenge_locked: bool) -> bool {
        let mut changed = false;
        let mut b1_ut = self.pulse_config.b1_amp * 1.0e6;
        if ui
            .add(egui::Slider::new(&mut b1_ut, 0.0..=50.0).text("B1 Amplitude [uT]"))
            .changed()
        {
            self.pulse_config.b1_amp = b1_ut * 1.0e-6;
            changed = true;
        }
        let mut tau_ms = self.pulse_config.duration_tau * 1.0e3;
        if ui
            .add(egui::Slider::new(&mut tau_ms, 0.5..=20.0).text("Pulse Duration tau [ms]"))
            .changed()
        {
            self.pulse_config.duration_tau = tau_ms * 1.0e-3;
            changed = true;
        }
        changed |= ui
            .add(
                egui::Slider::new(&mut self.pulse_config.phase_deg, -180.0..=180.0)
                    .text("RF Phase [deg]"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.pulse_config.detuning_hz, -2000.0..=2000.0)
                    .text("Detuning Offset [Hz]"),
            )
            .changed();
        if self.pulse_config.mode == MriPulseMode::HahnSpinEcho {
            let mut te_ms = self.pulse_config.echo_time_te * 1.0e3;
            ui.add_enabled_ui(!challenge_locked, |ui| {
                if ui
                    .add(egui::Slider::new(&mut te_ms, 10.0..=120.0).text("Echo Time TE [ms]"))
                    .changed()
                {
                    self.pulse_config.echo_time_te = te_ms * 1.0e-3;
                    changed = true;
                }
            });
        }
        ui.label(format!(
            "Nominal Flip Angle: {:.1}°",
            self.pulse_config.nominal_flip_deg()
        ));
        changed
    }

    fn draw_magnetic_field_controls(&mut self, ui: &mut egui::Ui) {
        let locked = self.active_challenge.is_some();
        ui.collapsing("Magnetic Fields", |ui| {
            ui.add_enabled_ui(!locked, |ui| {
                ui.label("Main Field (B0) [T]");
                ui.horizontal(|ui| {
                    ui.label("Bz");
                    if ui
                        .add(egui::DragValue::new(&mut self.simulator.b_field.z).speed(0.01))
                        .changed()
                    {
                        self.params
                            .write()
                            .unwrap_or_else(|e| e.into_inner())
                            .b_field
                            .z = self.simulator.b_field.z;
                        self.recompute_pulse_sequence();
                    }
                });
                ui.label("Gradients / RF (B1) [T]");
                ui.horizontal(|ui| {
                    ui.label("Bx");
                    if ui
                        .add(egui::DragValue::new(&mut self.simulator.b_field.x).speed(0.001))
                        .changed()
                    {
                        self.params
                            .write()
                            .unwrap_or_else(|e| e.into_inner())
                            .b_field
                            .x = self.simulator.b_field.x;
                    }
                    ui.label("By");
                    if ui
                        .add(egui::DragValue::new(&mut self.simulator.b_field.y).speed(0.001))
                        .changed()
                    {
                        self.params
                            .write()
                            .unwrap_or_else(|e| e.into_inner())
                            .b_field
                            .y = self.simulator.b_field.y;
                    }
                });
            });
        });
    }

    fn draw_playback_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Simulation");
        if let Some(action) = PlaybackTransport::from_worker_state(
            "mri_transport",
            "MRI Physics",
            self.worker.state(),
        )
        .reset_label("↻ Reset (90° Pulse)")
        .can_reset(true)
        .with_sonification(&mut self.audio_settings)
        .show(ui)
        {
            match action {
                PlaybackAction::Play => {
                    let _ = self.worker.send_command(SimCommand::Start);
                }
                PlaybackAction::Pause => {
                    let _ = self.worker.send_command(SimCommand::Pause);
                }
                PlaybackAction::Step => {
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
                egui::Slider::new(&mut self.time_scale, 1.0..=100.0)
                    .logarithmic(true)
                    .text("Time Scale (Speed)"),
            )
            .changed()
        {
            self.params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .time_scale = self.time_scale;
        }
    }

    fn draw_side_panel(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let panel_id = crate::framework::scoped_panel_id(ui, "controls_panel");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("MRI Physics Control");
            ui.separator();

            if let Some(WorkerFaultAction::Reset | WorkerFaultAction::ResetParameters) =
                render_worker_fault_banner(
                    ui,
                    self.worker.state(),
                    self.worker.diagnostic_report(),
                    false,
                )
            {
                self.reset();
                ctx.request_repaint();
                ui.separator();
            }

            self.draw_relaxation_controls(ui);
            self.draw_magnetic_field_controls(ui);
            self.draw_rf_pulse_controls(ui);
            let _ = self.callout_manager.render_ui(ui);
            ui.separator();
            if self.pulse_config.mode.is_continuous() {
                self.draw_playback_controls(ui, &ctx);
                ui.separator();
            }
            ui.label(format!("Time: {:.3} s", self.current_time));
            ui.label(format!("|M|: {:.3}", self.simulator.magnetization.norm()));
            ui.label(format!("Mx: {:.3}", self.simulator.magnetization.x));
            ui.label(format!("My: {:.3}", self.simulator.magnetization.y));
            ui.label(format!("Mz: {:.3}", self.simulator.magnetization.z));
        });
    }

    fn draw_central_panel(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Magnetization Vector Evolution");

            if self.mx_points.len() <= 1 {
                render_empty_plot_guidance(
                    ui,
                    "Initial 90° tipped magnetization M = (1, 0, 0) ready — click ▶ Play or ⏭ Step to simulate Bloch relaxation.",
                );
            }

            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            Plot::new("bloch_plot")
                .legend(Legend::default())
                .x_axis_label("Time (s)")
                .y_axis_label("Magnetization (M/M0)")
                .view_aspect(2.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new("Mx (Transverse)", &self.mx_points[..]).color(palette.axis_x),
                    );
                    plot_ui.line(
                        Line::new("My (Transverse)", &self.my_points[..]).color(palette.axis_y),
                    );
                    plot_ui.line(
                        Line::new("Mz (Longitudinal)", &self.mz_points[..]).color(palette.axis_z),
                    );
                    if self.mx_points.len() == 1 {
                        let mx0 = [self.mx_points[0].x, self.mx_points[0].y];
                        let my0 = [self.my_points[0].x, self.my_points[0].y];
                        let mz0 = [self.mz_points[0].x, self.mz_points[0].y];
                        plot_ui.points(
                            Points::new("Mx (t=0)", vec![mx0])
                                .radius(4.5_f32)
                                .color(palette.axis_x),
                        );
                        plot_ui.points(
                            Points::new("My (t=0)", vec![my0])
                                .radius(4.5_f32)
                                .color(palette.axis_y),
                        );
                        plot_ui.points(
                            Points::new("Mz (t=0)", vec![mz0])
                                .radius(4.5_f32)
                                .color(palette.axis_z),
                        );
                    }
                    self.callout_manager.render_on_plot(plot_ui);
                });

            ui.label(
                "Observe how Mz recovers to M0 (T1 relaxation) and Mx/My decay to 0 (T2 relaxation).",
            );
            ui.label("Adjust B-fields to see precession.");
        });
    }

    #[cfg(test)]
    pub(crate) fn show_ui(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| self.show_ui_inside(ui));
    }

    pub(crate) fn show_ui_inside(&mut self, ui: &mut egui::Ui) {
        if self.pulse_config.mode.is_continuous() {
            if let Some(snapshot) = self.worker.update() {
                if let Some(data) = snapshot
                    .structured_data
                    .as_ref()
                    .and_then(|d| d.downcast_ref::<MriSnapshot>())
                {
                    self.current_time = data.current_time;
                    self.simulator.magnetization = data.magnetization;
                    self.history = data.history.clone();
                    self.update_plot_buffers();
                }
            }
        }

        if matches!(
            self.worker.state(),
            SimWorkerState::Running | SimWorkerState::Stepping
        ) {
            ui.ctx().request_repaint();
        }

        self.draw_side_panel(ui);
        self.draw_central_panel(ui);
    }
}
