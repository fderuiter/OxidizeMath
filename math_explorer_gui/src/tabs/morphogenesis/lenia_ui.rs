#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::lenia_tool::{LeniaPresetChoice, LeniaSnapshotPayload, LeniaTool};
use super::MorphogenesisTool;
use crate::async_sim::ui::{render_worker_fault_banner, WorkerFaultAction};
use crate::async_sim::SimCommand;
use crate::framework::recovery::{FiniteStateGuard, RecoveryAction};
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui;
use eframe::egui::ColorImage;

impl LeniaTool {
    fn sync_texture_from_snapshot(&mut self, ctx: &egui::Context) {
        if !self.dirty && self.texture.is_some() {
            return;
        }
        let Some(snapshot) = self.worker.latest_snapshot() else {
            return;
        };
        let num_pixels = snapshot.width * snapshot.height;
        if num_pixels == 0 {
            return;
        }
        if self.pixels.len() != num_pixels {
            self.pixels.resize(num_pixels, egui::Color32::BLACK);
        }
        let hc = self.last_high_contrast.unwrap_or_else(|| {
            crate::app::ThemeMode::from_visuals(&ctx.style().visuals)
                == crate::app::ThemeMode::HighContrast
        });

        if let Some(payload) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<LeniaSnapshotPayload>())
            .filter(|p| p.grid.len() == num_pixels)
        {
            self.telemetry = payload.telemetry;
            let recolored = MorphogenesisTool::plot_concentration_with_contrast(
                &payload.grid,
                snapshot.width,
                snapshot.height,
                hc,
            );
            self.pixels.copy_from_slice(&recolored.pixels);
        } else if let Ok(px) = snapshot.pixels.try_read() {
            if px.len() == num_pixels {
                self.pixels.copy_from_slice(&px);
            }
        }

        let image = ColorImage {
            size: [snapshot.width, snapshot.height],
            pixels: self.pixels.clone(),
            ..Default::default()
        };
        if let Some(tex) = &mut self.texture {
            tex.set(image, egui::TextureOptions::NEAREST);
        } else {
            self.texture = Some(ctx.load_texture("lenia_tex", image, egui::TextureOptions::NEAREST));
        }
        self.dirty = false;
    }

    fn draw_fault_banner(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if let Some(action) = render_worker_fault_banner(
            ui,
            self.worker.state(),
            self.worker.diagnostic_report(),
            true,
        ) {
            match action {
                WorkerFaultAction::Reset => self.reset(),
                WorkerFaultAction::ResetParameters => FiniteStateGuard::reset_parameters(self),
                WorkerFaultAction::Recovery(RecoveryAction::AutoScaleStepSize) => {
                    FiniteStateGuard::auto_scale_step_size(self);
                }
                WorkerFaultAction::Recovery(RecoveryAction::ClampParameters) => {
                    FiniteStateGuard::clamp_parameters(self);
                }
                WorkerFaultAction::Recovery(RecoveryAction::ResetParameters) => {
                    FiniteStateGuard::reset_parameters(self);
                }
            }
            ctx.request_repaint();
            ui.separator();
        }
    }

    pub(crate) fn draw_controls(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let panel_id = crate::framework::scoped_panel_id(ui, "lenia_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("Spectral Lenia");
            ui.separator();
            self.draw_fault_banner(ui, &ctx);

            ui.label("Lifeform Presets:");
            ui.horizontal(|ui| {
                if ui.button("Orbium Glider").clicked() {
                    self.apply_preset_choice(LeniaPresetChoice::Orbium);
                }
                if ui.button("Hydrogeminium").clicked() {
                    self.apply_preset_choice(LeniaPresetChoice::Hydrogeminium);
                }
            });
            ui.separator();

            {
                let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
                ui.add(egui::Slider::new(&mut p.mu, 0.05..=0.50).text("Growth μ"));
                ui.add(egui::Slider::new(&mut p.sigma, 0.005..=0.15).text("Width σ"));
                ui.add(egui::Slider::new(&mut p.dt, 0.01..=0.50).text("Step dt"));
                let max_r = (self.width.min(self.height) / 2).saturating_sub(1).min(32);
                ui.add(egui::Slider::new(&mut p.kernel_radius, 4..=max_r).text("Radius R"));
            }
            ui.separator();

            if let Some(action) = PlaybackTransport::from_worker_state(
                "lenia_transport",
                "Continuous Lenia",
                self.worker.state(),
            )
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

            self.draw_telemetry_readout(ui);
        });
    }

    fn draw_telemetry_readout(&self, ui: &mut egui::Ui) {
        ui.separator();
        ui.strong("Live Toroidal Telemetry");
        let tel = self.telemetry;
        ui.label(format!("Total Mass: {:.2}", tel.total_mass));
        ui.label(format!(
            "Center of Mass: ({:.1}, {:.1})",
            tel.center_of_mass[0], tel.center_of_mass[1]
        ));
        ui.label(format!(
            "Velocity: ({:+.2}, {:+.2}) cells/t",
            tel.velocity[0], tel.velocity[1]
        ));
        ui.label(format!("Speed: {:.3} cells/t", tel.speed));
        ui.label(format!(
            "Rotational Symmetry: {:.1}% ({}-fold)",
            tel.rotational_symmetry * 100.0,
            tel.dominant_symmetry_fold
        ));
    }

    pub(crate) fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.sync_texture_from_snapshot(&ctx);
            if let Some(texture) = &self.texture {
                let avail = ui.available_size();
                let side = avail.x.min(avail.y).max(96.0);
                let alt_label = format!(
                    "Continuous Lenia Spectral Field ({}x{} torus, mass {:.1}, speed {:.2})",
                    self.width, self.height, self.telemetry.total_mass, self.telemetry.speed
                );
                let response = ui.add(
                    egui::Image::new((texture.id(), egui::vec2(side, side)))
                        .alt_text(alt_label)
                        .sense(egui::Sense::focusable_noninteractive()),
                );
                crate::accessibility::paint_canvas_focus_ring(ui, &response);
            } else {
                crate::async_sim::ui::render_empty_plot_guidance(
                    ui,
                    "Spectral Lenia torus initialized — click ▶ Play or ⏭ Step to evolve continuous solitons.",
                );
            }
        });
    }
}
