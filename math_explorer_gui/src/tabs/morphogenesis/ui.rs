#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MorphogenesisTool, SimpleRng};
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::SimCommand;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard, RecoveryAction};
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui;
use eframe::egui::ColorImage;
use egui_plot::{Plot, PlotImage, PlotPoint};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};

impl MorphogenesisTool {
    fn sync_texture_from_snapshot(&mut self, ctx: &egui::Context) {
        if !self.dirty && self.texture.is_some() {
            return;
        }
        let Some(snapshot) = self.worker.latest_snapshot() else {
            return;
        };
        let Ok(snapshot_pixels) = snapshot.pixels.try_read() else {
            return;
        };
        if snapshot_pixels.is_empty() {
            return;
        }

        let num_pixels = snapshot.width * snapshot.height;
        if self.pixels.len() != num_pixels {
            self.pixels.resize(num_pixels, egui::Color32::BLACK);
        }
        let high_contrast = self.last_high_contrast.unwrap_or_else(|| {
            crate::app::ThemeMode::from_visuals(&ctx.style().visuals)
                == crate::app::ThemeMode::HighContrast
        });
        if let Some(u_data) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<Vec<f64>>())
            .filter(|u| u.len() == num_pixels)
        {
            let recolored = Self::plot_concentration_with_contrast(
                u_data,
                snapshot.width,
                snapshot.height,
                high_contrast,
            );
            self.pixels.copy_from_slice(&recolored.pixels);
        } else if snapshot_pixels.len() == num_pixels {
            self.pixels.copy_from_slice(&snapshot_pixels);
        }

        let image = ColorImage {
            size: [snapshot.width, snapshot.height],
            pixels: self.pixels.clone(),
            ..Default::default()
        };
        let texture_opts = egui::TextureOptions::NEAREST;
        if let Some(texture) = &mut self.texture {
            texture.set(image, texture_opts);
        } else {
            self.texture = Some(ctx.load_texture("morphogenesis_tex", image, texture_opts));
        }
        self.dirty = false;
    }

    fn stabilize_cfl_step(&mut self, ctx: &egui::Context, min_dt: f64, max_dt: f64) {
        let safe_dt = {
            let mut params = self.params.write().unwrap_or_else(|e| e.into_inner());
            let dt = (0.2485 / params.d_u.max(params.d_v).max(1e-6)).clamp(min_dt, max_dt);
            params.dt = dt;
            dt
        };
        self.reset_system();
        crate::accessibility::announce_live_status(
            ctx,
            format!("Time step stabilized to CFL bound dt = {safe_dt:.4}"),
        );
    }

    fn draw_parameter_sliders(&mut self, ui: &mut egui::Ui) {
        let constraints = self.theory.theory_parameters();
        let bound = |key: &str, min: f64, max: f64, step: f64| {
            constraints
                .get(key)
                .cloned()
                .unwrap_or(ParameterConstraint { min, max, step })
        };
        let ca = bound("a", 0.0, 1.0, 0.01);
        let cb = bound("b", 0.0, 2.0, 0.01);
        let cdu = bound("d_u", 0.1, 5.0, 0.05);
        let cdv = bound("d_v", 10.0, 200.0, 1.0);
        let cdt = bound("dt", 1e-4, 0.1, 5e-4);

        let mut changed = false;
        let cfl_exceeded = {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            changed |= ui
                .add(
                    egui::Slider::new(&mut p.a, ca.min..=ca.max)
                        .step_by(ca.step)
                        .text("a"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut p.b, cb.min..=cb.max)
                        .step_by(cb.step)
                        .text("b"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut p.d_u, cdu.min..=cdu.max)
                        .step_by(cdu.step)
                        .text("D_u"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut p.d_v, cdv.min..=cdv.max)
                        .step_by(cdv.step)
                        .text("D_v"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut p.dt, cdt.min..=cdt.max)
                        .logarithmic(true)
                        .text("dt"),
                )
                .changed();
            4.0 * p.d_u.max(p.d_v) * p.dt >= 1.0
        };
        if changed {
            let _ = self.reseed_if_washed_out_entering_pattern();
        }
        if cfl_exceeded && ui.button("⚡ Stabilize Step (CFL)").clicked() {
            self.stabilize_cfl_step(ui.ctx(), cdt.min, cdt.max);
        }
    }

    fn draw_presets(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Presets:");
        if ui.button("Spots").clicked() {
            self.apply_named_preset(0.1, 0.9, 1.0, 100.0, 0.05);
        }
        if ui.button("Stripes").clicked() {
            self.apply_named_preset(0.14, 0.86, 1.0, 50.0, 0.02);
        }
        if let Some(snapshot) = self.worker.latest_snapshot() {
            if let Some(u_data) = snapshot
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<Vec<f64>>())
            {
                let splat_count =
                    crate::volumetric_scene::presets::build_scene_from_2d_morphogenesis_slice(
                        u_data,
                        snapshot.width,
                        snapshot.height,
                    )
                    .gaussians
                    .len();
                ui.small(format!("Extruded 3D Gaussian splats: {splat_count}"));
            }
        }
        ui.separator();
        let _ = self.atlas.show_toggle_button(ui);
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

    pub(crate) fn draw_controls(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let panel_id = crate::framework::scoped_panel_id(ui, "morphogenesis_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("Morphogenesis Controls");
            ui.separator();

            self.draw_fault_and_recovery(ui, &ctx);
            self.draw_parameter_sliders(ui);
            ui.separator();

            if let Some(action) = PlaybackTransport::from_worker_state(
                "morphogenesis_transport",
                "Morphogenesis",
                self.worker.state(),
            )
            .timeline(self.worker.timeline_metadata())
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
                    PlaybackAction::StepBack => {
                        let _ = self.worker.send_command(SimCommand::StepBack(1));
                    }
                    PlaybackAction::ScrubTo(step) => {
                        let _ = self.worker.send_command(SimCommand::SeekStep(step));
                    }
                    PlaybackAction::Reset => {
                        self.reset();
                        ctx.request_repaint();
                    }
                }
            }

            self.draw_presets(ui);
        });
    }

    fn draw_turing_field_plot(&self, ui: &mut egui::Ui) {
        if let Some(texture) = &self.texture {
            let w = self.width as f32;
            let h = self.height as f32;
            Plot::new("morphogenesis_plot")
                .data_aspect(1.0)
                .view_aspect(1.0)
                .show_grid(false)
                .show_axes([false, false])
                .allow_drag(true)
                .allow_zoom(true)
                .show(ui, |plot_ui| {
                    plot_ui.image(PlotImage::new(
                        "morphogenesis_image",
                        texture.id(),
                        PlotPoint::new(f64::from(w) / 2.0, f64::from(h) / 2.0),
                        [w, h],
                    ));
                });
        } else {
            render_empty_plot_guidance(
                ui,
                "Reaction-diffusion grid ready — click ▶ Play or ⏭ Step to evolve Turing patterns.",
            );
        }
    }

    pub(crate) fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.sync_texture_from_snapshot(&ctx);
            if !self.atlas.is_visible() {
                self.draw_turing_field_plot(ui);
                return;
            }
            let (ax, ay) = self.active_axis_values();
            let sig = self.non_axis_context_sig();
            let cartographer_params = {
                let p = self.params.read().unwrap_or_else(|e| e.into_inner());
                (p.a, p.b, p.d_u, p.d_v, p.dt)
            };
            let theory_params = self.theory.theory_parameters();
            let mut teleport = None;
            if ui.available_width() >= 620.0 {
                ui.columns(2, |cols| {
                    self.draw_turing_field_plot(&mut cols[0]);
                    teleport =
                        self.atlas
                            .show_panel(&mut cols[1], "morphogenesis", ax, ay, sig, |keys| {
                                std::sync::Arc::new(super::atlas::MorphogenesisCartographer::new(
                                    &theory_params,
                                    keys,
                                    cartographer_params.0,
                                    cartographer_params.1,
                                    cartographer_params.2,
                                    cartographer_params.3,
                                    cartographer_params.4,
                                ))
                            });
                });
            } else {
                teleport = self
                    .atlas
                    .show_panel(ui, "morphogenesis", ax, ay, sig, |keys| {
                        std::sync::Arc::new(super::atlas::MorphogenesisCartographer::new(
                            &theory_params,
                            keys,
                            cartographer_params.0,
                            cartographer_params.1,
                            cartographer_params.2,
                            cartographer_params.3,
                            cartographer_params.4,
                        ))
                    });
                ui.separator();
                self.draw_turing_field_plot(ui);
            }
            if let Some(ev) = teleport {
                self.apply_atlas_teleport(ev);
                ctx.request_repaint();
            }
        });
    }

    pub(crate) fn initialize_system(
        system: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
        width: usize,
        height: usize,
    ) {
        let n = width * height;
        let mut rng = SimpleRng::new(12345);

        let a = system.kinetics.a;
        let b = system.kinetics.b;
        let u_eq = a + b;
        let v_eq = if u_eq.abs() > 1e-9 {
            b / (u_eq * u_eq)
        } else {
            0.0
        };

        for i in 0..n {
            system.u_mut()[i] = u_eq + rng.range(-0.1, 0.1);
            system.v_mut()[i] = v_eq + rng.range(-0.1, 0.1);
        }
    }

    #[cfg(test)]
    pub(crate) fn plot_concentration(data: &[f64], width: usize, height: usize) -> ColorImage {
        Self::plot_concentration_with_contrast(data, width, height, false)
    }

    pub(crate) fn plot_concentration_with_contrast(
        data: &[f64],
        width: usize,
        height: usize,
        high_contrast: bool,
    ) -> ColorImage {
        let num_pixels = width * height;
        let mut pixels = Vec::with_capacity(num_pixels);

        let (min, max) =
            data.iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
                    if val.is_finite() {
                        (min.min(val), max.max(val))
                    } else {
                        (min, max)
                    }
                });

        let (min, max) = if min.is_finite() && max.is_finite() && max >= min {
            (min, max)
        } else {
            (0.0, 1.0)
        };

        let range = max - min;
        let inv_range = if range > 1e-6 { 1.0 / range } else { 1.0 };

        for &val in data {
            let safe_val = if val.is_finite() { val } else { min };
            let norm = (safe_val - min) * inv_range;
            let safe_norm = if norm.is_finite() {
                norm.clamp(0.0, 1.0)
            } else {
                0.0
            };
            let rgb = if high_contrast {
                heatmap_color_high_contrast(safe_norm)
            } else {
                heatmap_color(safe_norm)
            };
            pixels.push(egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
        }

        ColorImage {
            size: [width, height],
            pixels,
            ..Default::default()
        }
    }
}

impl FiniteStateGuard for MorphogenesisTool {
    fn is_finite_state(&self) -> bool {
        self.worker.state() != crate::async_sim::SimWorkerState::Faulted
            && self.worker.diagnostic_report().is_none()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.worker.diagnostic_report()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.dt = (p.dt * 0.5).max(1e-4);
        drop(p);
        self.reset_system();
    }

    fn clamp_parameters(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = p.a.clamp(0.0, 1.0);
        p.b = p.b.clamp(0.0, 2.0);
        p.d_u = p.d_u.clamp(0.1, 5.0);
        p.d_v = p.d_v.clamp(10.0, 200.0);
        p.dt = p.dt.clamp(0.001, 0.1);
        drop(p);
        self.reset_system();
    }

    fn reset_parameters(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = 0.1;
        p.b = 0.9;
        p.d_u = 1.0;
        p.d_v = 100.0;
        p.dt = 0.05;
        drop(p);
        self.reset_system();
    }
}
