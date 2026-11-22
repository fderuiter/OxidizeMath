#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::MorphogenesisKineticsAdapter;
use super::{MorphogenesisTool, SimpleRng};
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::SimCommand;
use crate::framework::recovery::{FiniteStateGuard, RecoveryAction};
use crate::framework::{PlaybackAction, PlaybackTransport};
use crate::presets::MorphogenesisKineticsMode;
use eframe::egui;
use eframe::egui::ColorImage;
use egui_plot::{Plot, PlotImage, PlotPoint};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{GrayScottKinetics, TuringSystem};
#[cfg(test)]
use math_explorer::biology::morphogenesis::SchnakenbergKinetics;
use math_explorer::rand::Rng;
use math_explorer::rng::OxidizeRng;

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

    fn draw_kinetics_selector(&mut self, ui: &mut egui::Ui) {
        let current_mode = self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .mode;
        ui.label("Kinetics Model:");
        ui.horizontal_wrapped(|ui| {
            if ui
                .selectable_label(
                    current_mode == MorphogenesisKineticsMode::Schnakenberg,
                    "Schnakenberg",
                )
                .clicked()
                && current_mode != MorphogenesisKineticsMode::Schnakenberg
            {
                self.apply_named_preset(0.1, 0.9, 1.0, 100.0, 0.05);
            }
            if ui
                .selectable_label(
                    current_mode == MorphogenesisKineticsMode::GrayScott,
                    "Gray-Scott",
                )
                .clicked()
                && current_mode != MorphogenesisKineticsMode::GrayScott
            {
                let mitosis = GrayScottKinetics::mitosis();
                self.apply_gray_scott_preset(mitosis.f, mitosis.k);
            }
        });
    }

    fn draw_parameter_sliders(&self, ui: &mut egui::Ui) {
        let mut ks = self.kinetics_state.write().unwrap_or_else(|e| e.into_inner());
        let mut params = self.params.write().unwrap_or_else(|e| e.into_inner());
        match ks.mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                ui.add(egui::Slider::new(&mut params.a, 0.0..=1.0).text("a"));
                ui.add(egui::Slider::new(&mut params.b, 0.0..=2.0).text("b"));
                ui.add(egui::Slider::new(&mut params.d_u, 0.1..=5.0).text("D_u"));
                ui.add(egui::Slider::new(&mut params.d_v, 10.0..=200.0).text("D_v"));
            }
            MorphogenesisKineticsMode::GrayScott => {
                ui.add(egui::Slider::new(&mut ks.f, 0.01..=0.09).text("f (feed)"));
                ui.add(egui::Slider::new(&mut ks.k, 0.03..=0.08).text("k (kill)"));
                ui.add(egui::Slider::new(&mut params.d_u, 0.05..=0.40).text("D_u"));
                ui.add(egui::Slider::new(&mut params.d_v, 0.02..=0.25).text("D_v"));
                ui.add(egui::Slider::new(&mut params.dt, 0.1..=1.5).text("dt"));
            }
        }
    }

    fn draw_presets(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Schnakenberg Presets:");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Spots").clicked() {
                self.apply_named_preset(0.1, 0.9, 1.0, 100.0, 0.05);
            }
            if ui.button("Stripes").clicked() {
                self.apply_named_preset(0.14, 0.86, 1.0, 50.0, 0.02);
            }
        });
        ui.label("Gray-Scott Presets:");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Mitosis").clicked() {
                let m = GrayScottKinetics::mitosis();
                self.apply_gray_scott_preset(m.f, m.k);
            }
            if ui.button("Coral").clicked() {
                let c = GrayScottKinetics::coral();
                self.apply_gray_scott_preset(c.f, c.k);
            }
        });
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
            self.draw_kinetics_selector(ui);
            ui.separator();
            self.draw_parameter_sliders(ui);
            ui.separator();

            if let Some(action) = PlaybackTransport::from_worker_state(
                "morphogenesis_transport",
                "Morphogenesis",
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

            self.draw_presets(ui);
        });
    }

    pub(crate) fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.sync_texture_from_snapshot(&ctx);

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
                    "Reaction-diffusion grid ready — click ▶ Play or ⏭ Step to evolve patterns.",
                );
            }
        });
    }

    #[cfg(test)]
    pub(crate) fn initialize_schnakenberg(
        system: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
        width: usize,
        height: usize,
    ) {
        let n = width * height;
        let mut rng = SimpleRng::new(12345);
        let u_eq = system.kinetics.a + system.kinetics.b;
        let v_eq = system.kinetics.b / (u_eq * u_eq);
        for i in 0..n {
            system.u_mut()[i] = u_eq + rng.range(-0.1, 0.1);
            system.v_mut()[i] = v_eq + rng.range(-0.1, 0.1);
        }
    }

    #[cfg(test)]
    pub(crate) fn initialize_system(
        system: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
        width: usize,
        height: usize,
    ) {
        Self::initialize_schnakenberg(system, width, height);
    }

    pub(crate) fn initialize_adapter_system(
        system: &mut TuringSystem<2, MorphogenesisKineticsAdapter, FiniteDifference2D>,
        width: usize,
        height: usize,
    ) {
        match system.kinetics.mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                let n = width * height;
                let mut rng = SimpleRng::new(12345);
                let u_eq = system.kinetics.a + system.kinetics.b;
                let v_eq = system.kinetics.b / (u_eq * u_eq);
                for i in 0..n {
                    system.u_mut()[i] = u_eq + rng.range(-0.1, 0.1);
                    system.v_mut()[i] = v_eq + rng.range(-0.1, 0.1);
                }
            }
            MorphogenesisKineticsMode::GrayScott => {
                Self::seed_gray_scott_patches(system, width, height);
            }
        }
    }

    fn seed_gray_scott_patches(
        system: &mut TuringSystem<2, MorphogenesisKineticsAdapter, FiniteDifference2D>,
        width: usize,
        height: usize,
    ) {
        system.u_mut().fill(1.0);
        system.v_mut().fill(0.0);
        if width == 0 || height == 0 {
            return;
        }
        let mut rng = OxidizeRng::new(12345);
        let cx = width / 2;
        let cy = height / 2;
        let seeds = [
            (cx, cy, 5usize),
            (cx.saturating_sub(width / 5), cy.saturating_sub(height / 6), 3),
            ((cx + width / 5).min(width - 1), (cy + height / 6).min(height - 1), 3),
        ];
        for (sx, sy, half) in seeds {
            let x0 = sx.saturating_sub(half);
            let x1 = (sx + half).min(width - 1);
            let y0 = sy.saturating_sub(half);
            let y1 = (sy + half).min(height - 1);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let idx = flatten_2d_index(x, y, width);
                    let u_val = 0.50_f64 + rng.gen_range(-0.02_f64..=0.02_f64);
                    let v_val = 0.25_f64 + rng.gen_range(-0.02_f64..=0.02_f64);
                    system.u_mut()[idx] = u_val.clamp(0.0, 1.0);
                    system.v_mut()[idx] = v_val.clamp(0.0, 1.0);
                }
            }
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
        let (min, max) = data
            .iter()
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
