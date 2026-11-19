#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MorphogenesisTab, SimpleRng};
use crate::async_sim::ui::render_empty_plot_guidance;
use eframe::egui;
use eframe::egui::ColorImage;
use egui_plot::{Plot, PlotImage, PlotPoint};
use math_commons::math_kernel::colormap::heatmap_color;
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};

impl MorphogenesisTab {
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
        if snapshot_pixels.len() == num_pixels {
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

    pub(super) fn draw_canvas(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.sync_texture_from_snapshot(ctx);

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
        let v_eq = b / (u_eq * u_eq);

        for i in 0..n {
            system.u_mut()[i] = u_eq + rng.range(-0.1, 0.1);
            system.v_mut()[i] = v_eq + rng.range(-0.1, 0.1);
        }
    }

    pub(crate) fn plot_concentration(data: &[f64], width: usize, height: usize) -> ColorImage {
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
            let rgb = heatmap_color(safe_norm);
            pixels.push(egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
        }

        ColorImage {
            size: [width, height],
            pixels,
            ..Default::default()
        }
    }
}
