#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ising_spin_color, IsingBrushMode, IsingModelTool, DEFAULT_ISING_SIZE};
use crate::accessibility::AccessibleTheoryHover;
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::SimCommand;
use crate::framework::{
    draw_brush_palette_controls, PlaybackAction, PlaybackTransport, RasterBrushCanvas,
};
use eframe::egui;

impl IsingModelTool {
    pub(super) fn request_metric_refresh(&mut self, ctx: &egui::Context) {
        if self.worker.state().is_ready()
            && self
                .worker
                .send_command(SimCommand::Custom("recompute_metrics".to_string()))
                .is_ok()
        {
            ctx.request_repaint();
        }
    }

    fn draw_parameter_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if ui
            .add(
                egui::Slider::new(&mut self.temperature, 0.1..=10.0)
                    .text("Temperature (T) - K (relative)"),
            )
            .changed()
        {
            self.params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .temperature = self.temperature;
        }
        ui.small("Critical Temp Tc ≈ 2.269 (for J=1)");

        if ui
            .add(
                egui::Slider::new(&mut self.j_coupling, -2.0..=2.0).text("Coupling Constant (J) - J"),
            )
            .changed()
        {
            self.params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .j_coupling = self.j_coupling;
            self.request_metric_refresh(ctx);
        }
        ui.small("Positive: Ferromagnetic\nNegative: Antiferromagnetic");

        if ui
            .add(egui::Slider::new(&mut self.h_field, -2.0..=2.0).text("External Field (H) - H"))
            .changed()
        {
            self.params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .h_field = self.h_field;
            self.request_metric_refresh(ctx);
        }

        ui.separator();
        if ui
            .add(
                egui::Slider::new(&mut self.steps_per_frame, 100..=50000)
                    .text("Simulation Speed (Steps/Frame)")
                    .logarithmic(true),
            )
            .changed()
        {
            self.params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .steps_per_frame = self.steps_per_frame;
        }
    }

    fn draw_brush_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.separator();
        ui.label("Spin & Boundary Brush");
        egui::ComboBox::from_id_salt("ising_brush_mode")
            .selected_text(self.brush_mode.label())
            .show_ui(ui, |ui| {
                for mode in IsingBrushMode::ALL {
                    ui.selectable_value(&mut self.brush_mode, mode, mode.label());
                }
            });
        draw_brush_palette_controls(ui, "ising", &mut self.brush_state, 10);
        if ui.button("Clear Pinned Sites").clicked() {
            self.clear_pins(Some(ctx));
        }
    }

    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "ising_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .resizable(false)
            .default_width(220.0)
            .show_inside(parent_ui, |ui| {
                ui.heading("Ising Model Controls");
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

                self.draw_parameter_controls(ui, ctx);
                ui.separator();

                if let Some(action) = PlaybackTransport::from_worker_state(
                    "ising_transport",
                    "Ising Model",
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

                self.draw_brush_controls(ui, ctx);

                ui.separator();
                ui.label(format!("Magnetization: {:.3}", self.magnetization));
                ui.label(format!("Energy: {:.3}", self.energy));
            });
    }

    fn sync_lattice_texture(&mut self, ctx: &egui::Context) {
        if !self.dirty && self.texture.is_some() {
            return;
        }
        let Some(snapshot) = self.worker.latest_snapshot() else {
            return;
        };
        let Ok(snap_pixels) = snapshot.pixels.try_read() else {
            return;
        };
        if snap_pixels.is_empty() {
            return;
        }
        let num_pixels = snapshot.width * snapshot.height;
        if self.pixels.len() != num_pixels {
            self.pixels.resize(num_pixels, egui::Color32::BLACK);
        }
        let palette = crate::app::SemanticPalette::from_visuals(&ctx.style().visuals);
        if let Some(spins) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<Vec<i8>>())
            .filter(|s| s.len() == num_pixels)
        {
            for (dst, &s) in self.pixels.iter_mut().zip(spins.iter()) {
                *dst = ising_spin_color(s, &palette);
            }
        } else if snap_pixels.len() == num_pixels {
            self.pixels.copy_from_slice(&snap_pixels);
        }

        let image = egui::ColorImage {
            size: [snapshot.width, snapshot.height],
            pixels: self.pixels.clone(),
            ..Default::default()
        };
        let texture_opts = egui::TextureOptions::NEAREST;
        if let Some(texture) = &mut self.texture {
            texture.set(image, texture_opts);
        } else {
            self.texture = Some(ctx.load_texture("ising_lattice", image, texture_opts));
        }
        self.dirty = false;
    }

    pub(super) fn draw_central_panel(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.sync_lattice_texture(ctx);

            if let Some(texture) = &self.texture {
                let tex = texture.clone();
                let alt_label = format!(
                    "2D Ising Model spin lattice visualization (T = {:.2}, M = {:.3}, E = {:.3})",
                    self.temperature, self.magnetization, self.energy
                );
                let pinned_count = self.pin_cells.iter().filter(|&&c| c != 0).count();
                let summary = format!(
                    "M = {:.3}, E = {:.1}, {pinned_count} pinned sites",
                    self.magnetization, self.energy
                );
                let (response, stroke) = RasterBrushCanvas::new(
                    "ising_lattice_canvas",
                    DEFAULT_ISING_SIZE,
                    DEFAULT_ISING_SIZE,
                    &alt_label,
                )
                .value_summary(&summary)
                .show_texture(ui, &tex, &mut self.brush_state);
                let _ = response.accessible_theory_hover(self);
                if let Some(stroke) = stroke {
                    self.apply_brush_stroke(stroke, Some(ctx));
                }
            } else {
                render_empty_plot_guidance(
                    ui,
                    "Spin lattice ready — click ▶ Play or ⏭ Step to run Metropolis spin flips.",
                );
            }
        });
    }
}
