#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::{render_wavefield_image, OpticsFrameData, WaveRenderInput};
use super::tool::OpticsWaveTool;
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility, AccessibleTheoryHover,
};
use crate::app::ThemeMode;
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::SimCommand;
use crate::framework::recovery::{FiniteStateGuard, RecoveryAction};
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui::{self, vec2, ColorImage};
use math_explorer::physics::optics::refraction::OpticsPreset;

impl OpticsWaveTool {
    fn sync_texture(&mut self, ctx: &egui::Context) {
        let high_contrast =
            ThemeMode::from_visuals(&ctx.style().visuals) == ThemeMode::HighContrast;
        if let Some(snapshot) = self.worker.latest_snapshot() {
            if self.dirty {
                if let Some(data) = snapshot
                    .structured_data
                    .as_ref()
                    .and_then(|d| d.downcast_ref::<OpticsFrameData>())
                {
                    self.latest_frame = Some(data.clone());
                } else if let Ok(px) = snapshot.pixels.try_read() {
                    if px.len() == snapshot.width * snapshot.height {
                        self.pixels = px.clone();
                    }
                }
            }
        }
        if !self.dirty && self.texture.is_some() {
            return;
        }
        let show_irradiance = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .show_irradiance;
        if let Some(ref frame) = self.latest_frame {
            let n = self.grid_size;
            if frame.field.len() == n * n {
                let recolored = render_wavefield_image(WaveRenderInput {
                    field: &frame.field,
                    irradiance: &frame.irradiance,
                    refractive_index: &frame.refractive_index,
                    obstacle: &frame.obstacle,
                    width: n,
                    height: n,
                    show_irradiance,
                    high_contrast,
                });
                self.pixels = recolored.pixels;
            }
        }
        if self.pixels.len() != self.grid_size * self.grid_size {
            return;
        }
        let image = ColorImage {
            size: [self.grid_size, self.grid_size],
            pixels: self.pixels.clone(),
            ..Default::default()
        };
        let opts = egui::TextureOptions::LINEAR;
        if let Some(tex) = &mut self.texture {
            tex.set(image, opts);
        } else {
            self.texture = Some(ctx.load_texture("optics_wave_tex", image, opts));
        }
        self.dirty = false;
    }

    fn draw_fault_and_recovery(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
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

    fn draw_medium_controls(&mut self, ui: &mut egui::Ui) {
        let mut grid_sz = self.grid_size;
        let preset_changed = {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            let prev_preset = p.preset;
            egui::ComboBox::from_label("Optical Preset")
                .selected_text(p.preset.label())
                .show_ui(ui, |ui| {
                    for candidate in OpticsPreset::ALL {
                        ui.selectable_value(&mut p.preset, candidate, candidate.label());
                    }
                });
            ui.add(egui::Slider::new(&mut p.frequency, 4.0..=24.0).text("Frequency f"));
            if p.preset == OpticsPreset::DoubleSlit {
                ui.add(
                    egui::Slider::new(&mut p.slit_separation, 0.08..=0.40)
                        .text("Slit Separation d"),
                );
                ui.add(egui::Slider::new(&mut p.aperture_width, 0.015..=0.14).text("Slit Width a"));
            } else if p.preset != OpticsPreset::VeselagoLens {
                ui.add(egui::Slider::new(&mut p.lens_radius, 0.10..=0.34).text("Lens Radius R"));
            }
            ui.add(egui::Slider::new(&mut p.cfl_scale, 0.10..=1.00).text("CFL Step Scale"));
            p.preset != prev_preset
        };
        if ui
            .add(egui::Slider::new(&mut grid_sz, 64..=256).text("FDTD Grid N"))
            .changed()
        {
            self.rebuild_for_grid_size(grid_sz);
        } else if preset_changed {
            self.dirty = true;
        }
    }

    fn draw_overlay_and_detector_controls(&mut self, ui: &mut egui::Ui) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        if ui
            .checkbox(&mut p.show_irradiance, "Show Smoothed Irradiance ⟨u²⟩")
            .changed()
        {
            self.dirty = true;
        }
        ui.checkbox(&mut p.show_rays, "Overlay Hamiltonian Rays & Caustics");
        if p.show_rays {
            ui.add(egui::Slider::new(&mut p.ray_count, 3..=31).text("Ray Count"));
        }
        ui.separator();
        ui.label("Virtual Detector Plane:");
        ui.add(egui::Slider::new(&mut p.detector_x, 0.45..=0.90).text("Detector x"));
        ui.add(egui::Slider::new(&mut p.adc_bits, 4..=16).text("Sensor ADC Bits"));
    }

    fn draw_sensor_readouts(&self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("Calibrated Sensor Readouts");
        let readout = self
            .latest_frame
            .as_ref()
            .map(|f| f.readout)
            .unwrap_or_default();
        egui::Grid::new("optics_sensor_readout_grid")
            .num_columns(2)
            .spacing([12.0, 4.0])
            .show(ui, |ui| {
                ui.label("Axial Range r₀:");
                ui.monospace(format!("{:.3} m", readout.axial_range));
                ui.end_row();
                ui.label("Peak Irradiance I_max:");
                ui.monospace(format!("{:.4e}", readout.peak_irradiance));
                ui.end_row();
                ui.label("Mean Irradiance Ī:");
                ui.monospace(format!("{:.4e}", readout.mean_irradiance));
                ui.end_row();
                ui.label("Fringe Visibility V:");
                ui.monospace(format!("{:.3}", readout.fringe_visibility));
                ui.end_row();
                ui.label("Lambertian Intensity:");
                ui.monospace(format!("{:.4} W/m²", readout.lambertian_intensity));
                ui.end_row();
                ui.label("LWS Range Resolution:");
                ui.monospace(format!("{:.3e} m", readout.lws_range_resolution));
                ui.end_row();
            });
    }

    pub(crate) fn draw_controls(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let panel_id = crate::framework::scoped_panel_id(ui, "optics_waves_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Optics & Waves Controls");
                ui.separator();
                self.draw_fault_and_recovery(ui, &ctx);
                self.draw_medium_controls(ui);
                ui.separator();
                self.draw_overlay_and_detector_controls(ui);
                ui.separator();

                if let Some(action) = PlaybackTransport::from_worker_state(
                    "optics_waves_transport",
                    "OpticsWaves",
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
                self.draw_sensor_readouts(ui);
            });
        });
    }

    pub(crate) fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.sync_texture(&ctx);
            egui_plot::commands::CommandRegistryData::register(
                &ctx,
                "Nudge Virtual Detector",
                "Nudge virtual optical detector plane along x-axis (when canvas focused)",
                egui_plot::commands::CommandTrigger::Key(egui::Key::ArrowRight),
                false,
                "Optics & Waves",
            );

            let Some(texture) = &self.texture else {
                render_empty_plot_guidance(
                    ui,
                    "2D FDTD wave grid ready — click ▶ Play or ⏭ Step to propagate wavefronts.",
                );
                return;
            };

            let (preset, freq, det_x) = {
                let p = self.params.read().unwrap_or_else(|e| e.into_inner());
                (p.preset, p.frequency, p.detector_x)
            };
            let peak_irr = self
                .latest_frame
                .as_ref()
                .map_or(0.0, |f| f.readout.peak_irradiance);
            let alt_label = format!(
                "2D Wave & Caustics Playground ({}, f = {:.1}, detector x = {:.2})",
                preset.label(),
                freq,
                det_x
            );
            let value_summary = format!("peak irradiance {:.3e}", peak_irr);

            let avail = ui.available_size();
            let side = avail.x.min(avail.y).max(120.0);
            let response = ui
                .add(
                    egui::Image::new(texture)
                        .fit_to_exact_size(vec2(side, side))
                        .alt_text(alt_label.clone())
                        .sense(egui::Sense::click_and_drag()),
                )
                .accessible_theory_hover(&self.theory);

            register_canvas_accessibility(
                ui,
                &response,
                egui::accesskit::Role::Image,
                &alt_label,
                Some(&value_summary),
            );
            self.canvas_id = response.has_focus().then_some(response.id);
            self.handle_canvas_interaction(ui, &response);
            self.paint_overlays(ui, response.rect);
            paint_canvas_focus_ring(ui, &response);
        });
    }

    fn handle_canvas_interaction(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if response.dragged() {
            if let Some(pos) = response.interact_pointer_pos() {
                let rel_x =
                    ((pos.x - response.rect.left()) / response.rect.width().max(1.0)) as f64;
                let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
                p.detector_x = rel_x.clamp(0.45, 0.90);
            }
        }
        lock_canvas_arrow_keys(ui, response);
        if response.has_focus() {
            let delta = ui.input(|i| {
                let mut d = 0.0_f64;
                if i.key_pressed(egui::Key::ArrowLeft) {
                    d -= 0.02;
                }
                if i.key_pressed(egui::Key::ArrowRight) {
                    d += 0.02;
                }
                d
            });
            if delta != 0.0 {
                let new_x = {
                    let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
                    p.detector_x = (p.detector_x + delta).clamp(0.45, 0.90);
                    p.detector_x
                };
                announce_live_status(ui.ctx(), format!("Virtual detector x moved to {new_x:.2}"));
            }
        }
    }
}
