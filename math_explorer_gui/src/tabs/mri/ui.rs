#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::MriSnapshot;
use super::tool::MriTool;
use crate::accessibility::{announce_status, AccessibleHoverText};
use crate::app::SemanticPalette;
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::{SimCommand, SimWorkerState};
use crate::framework::{
    branch_a_ghost_line, branch_b_solid_line, fork_marker_callout, fork_marker_vline,
    render_divergence_strip, CounterfactualTransport, PlaybackAction, PlaybackTransport,
};
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoint, Points};

impl MriTool {
    fn draw_relaxation_controls(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Relaxation Parameters", |ui| {
            if ui
                .add(
                    egui::Slider::new(&mut self.simulator.t1, 0.1..=5.0)
                        .logarithmic(true)
                        .text("Longitudinal (T1) [s]"),
                )
                .changed()
            {
                self.params.write().unwrap_or_else(|e| e.into_inner()).t1 = self.simulator.t1;
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
            }
            if self.simulator.t2 > self.simulator.t1 {
                let err = crate::app::SemanticPalette::from_visuals(ui.visuals()).error;
                ui.colored_label(err, "Warning: T2 > T1 is unphysical!");
            }
        });
    }

    fn draw_magnetic_field_controls(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Magnetic Fields", |ui| {
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

        let can_fork = self.current_step_index() > 0 && self.worker.state() != SimWorkerState::Faulted;
        let is_forked = self.twin.is_some();
        if let Some(cf_action) =
            CounterfactualTransport::new(&mut self.branch_engine, can_fork, is_forked).show(ui)
        {
            let _ = self.apply_mri_branch_action(cf_action);
            ctx.request_repaint();
        }
        self.render_refocus_pulse_button(ui, ctx);
    }

    fn render_refocus_pulse_button(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let can_pulse = self.current_step_index() > 0 || self.twin.is_some();
        let resp = ui
            .add_enabled(can_pulse, egui::Button::new("🧲 180° Refocusing Pulse"))
            .accessible_hover_text(
                "Apply a 180° RF spin-echo refocusing pulse to Branch B at the current decay step",
            )
            .on_disabled_hover_text(
                "Step or Play past t = 0 to apply a mid-decay 180° refocusing pulse",
            );
        if resp.clicked() && self.apply_refocus_pulse_180() {
            announce_status("Applied 180 degree spin-echo refocusing pulse on Branch B");
            ctx.request_repaint();
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
            ui.separator();
            self.draw_playback_controls(ui, &ctx);

            ui.separator();
            ui.label(format!("Time: {:.3} s", self.current_time));
            ui.label(format!("|M|: {:.3}", self.simulator.magnetization.norm()));
            ui.label(format!("Mx: {:.3}", self.simulator.magnetization.x));
            ui.label(format!("My: {:.3}", self.simulator.magnetization.y));
            ui.label(format!("Mz: {:.3}", self.simulator.magnetization.z));
        });
    }

    fn draw_central_panel(&self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Magnetization Vector Evolution");
            if let Some(ref snap) = self.branch_engine.active_snapshot {
                render_divergence_strip(ui, "MRI Bloch Relaxation", snap);
            }

            if self.mx_points.len() <= 1 {
                render_empty_plot_guidance(
                    ui,
                    "Initial 90° tipped magnetization M = (1, 0, 0) ready — click ▶ Play or ⏭ Step to simulate Bloch relaxation.",
                );
            }

            let palette = SemanticPalette::from_visuals(ui.visuals());
            Plot::new("bloch_plot")
                .legend(Legend::default())
                .x_axis_label("Time (s)")
                .y_axis_label("Magnetization (M/M0)")
                .view_aspect(2.0)
                .show(ui, |plot_ui| {
                    self.render_mri_plot_curves(plot_ui, &palette);
                    if self.mx_points.len() == 1 {
                        let mx0 = [self.mx_points[0].x, self.mx_points[0].y];
                        let my0 = [self.my_points[0].x, self.my_points[0].y];
                        let mz0 = [self.mz_points[0].x, self.mz_points[0].y];
                        plot_ui.points(Points::new("Mx (t=0)", vec![mx0]).radius(4.5_f32).color(palette.axis_x));
                        plot_ui.points(Points::new("My (t=0)", vec![my0]).radius(4.5_f32).color(palette.axis_y));
                        plot_ui.points(Points::new("Mz (t=0)", vec![mz0]).radius(4.5_f32).color(palette.axis_z));
                    }
                });

            ui.label(
                "Observe how Mz recovers to M0 (T1 relaxation) and Mx/My decay to 0 (T2 relaxation).",
            );
            ui.label("Adjust B-fields to see precession.");
        });
    }

    fn render_mri_plot_curves<'a>(
        &'a self,
        plot_ui: &mut egui_plot::PlotUi<'a>,
        palette: &SemanticPalette,
    ) {
        let to_vec = |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
        if let Some(ref twin) = self.twin {
            let mxa: Vec<[f64; 2]> = twin.history_a.iter().map(|(t, m)| [*t, m.x]).collect();
            let mya: Vec<[f64; 2]> = twin.history_a.iter().map(|(t, m)| [*t, m.y]).collect();
            let mza: Vec<[f64; 2]> = twin.history_a.iter().map(|(t, m)| [*t, m.z]).collect();
            plot_ui.line(branch_a_ghost_line("Mx (Branch A)", mxa, palette.axis_x));
            plot_ui.line(branch_a_ghost_line("My (Branch A)", mya, palette.axis_y));
            plot_ui.line(branch_a_ghost_line("Mz (Branch A)", mza, palette.axis_z));
            plot_ui.line(branch_b_solid_line("Mx (Branch B)", to_vec(&self.mx_points), palette.axis_x));
            plot_ui.line(branch_b_solid_line("My (Branch B)", to_vec(&self.my_points), palette.axis_y));
            plot_ui.line(branch_b_solid_line("Mz (Branch B)", to_vec(&self.mz_points), palette.axis_z));
            let fork_t = twin.tracker.fork_time();
            plot_ui.vline(fork_marker_vline(palette, fork_t));
            plot_ui.callout(fork_marker_callout([fork_t, twin.fork_m.x], fork_t, palette));
        } else {
            plot_ui.line(Line::new("Mx (Transverse)", &self.mx_points[..]).color(palette.axis_x));
            plot_ui.line(Line::new("My (Transverse)", &self.my_points[..]).color(palette.axis_y));
            plot_ui.line(Line::new("Mz (Longitudinal)", &self.mz_points[..]).color(palette.axis_z));
        }
    }

    #[cfg(test)]
    pub(crate) fn show_ui(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| self.show_ui_inside(ui));
    }

    pub(crate) fn show_ui_inside(&mut self, ui: &mut egui::Ui) {
        if let Some(snapshot) = self.worker.update() {
            if let Some(data) = snapshot
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<MriSnapshot>())
            {
                self.current_time = data.current_time;
                self.simulator.magnetization = data.magnetization;
                self.history = data.history.clone();
                self.twin = data.twin.clone();
                self.update_plot_buffers();
                self.sync_mri_branch_engine();
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
