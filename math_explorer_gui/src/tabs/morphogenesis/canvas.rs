#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::overlay::render_morphogenesis_overlay;
use super::types::{
    MorphogenesisBrushMode, MorphogenesisKinetics, MorphogenesisKineticsMode,
    MorphogenesisSnapshotData,
};
use super::{MorphogenesisTool, SimpleRng};
use crate::app::ThemeMode;
use crate::async_sim::ui::render_empty_plot_guidance;
use crate::async_sim::SimCommand;
use eframe::egui;
use eframe::egui::ColorImage;
use egui_plot::{Plot, PlotImage, PlotPoint};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::biology::diffusion::AdvectionDiffusion2D;
use math_explorer::biology::morphogenesis::TuringSystem;
use scientific_metadata::theory::TheoryDescribable;

impl MorphogenesisTool {
    pub(crate) fn sync_texture_from_snapshot(&mut self, ctx: &egui::Context) {
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
            self.pixels.resize(num_pixels, egui::Color32::TRANSPARENT);
        }
        let theme_mode = self
            .last_theme_mode
            .unwrap_or_else(|| ThemeMode::from_visuals(&ctx.style().visuals));
        let overlay_mode = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .overlay_mode;

        if let Some(data) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<MorphogenesisSnapshotData>())
            .filter(|d| d.concentration.len() == num_pixels)
        {
            self.live_peclet = data.peclet;
            self.live_damkohler = data.damkohler;
            let recolored = render_morphogenesis_overlay(
                &data.concentration,
                &data.vorticity,
                &data.obstacles,
                snapshot.width,
                snapshot.height,
                overlay_mode,
                theme_mode,
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
        if let Some(texture) = &mut self.texture {
            texture.set(image, egui::TextureOptions::NEAREST);
        } else {
            self.texture =
                Some(ctx.load_texture("morphogenesis_tex", image, egui::TextureOptions::NEAREST));
        }
        self.dirty = false;
    }

    pub(crate) fn draw_turing_field_plot(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(texture) = &self.texture else {
            render_empty_plot_guidance(
                ui,
                "Reaction-diffusion grid ready — click ▶ Play or ⏭ Step to evolve Turing patterns.",
            );
            return;
        };
        let w = self.width as f32;
        let h = self.height as f32;
        let (brush_mode, brush_radius) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.brush_mode, p.brush_radius)
        };
        let allow_pan = brush_mode == MorphogenesisBrushMode::PanZoom;
        let tex_id = texture.id();
        let mut brush_cmd = None;

        Plot::new("morphogenesis_plot")
            .data_aspect(1.0)
            .view_aspect(1.0)
            .show_grid(false)
            .show_axes([false, false])
            .allow_drag(allow_pan)
            .allow_zoom(true)
            .show(ui, |plot_ui| {
                plot_ui.image(PlotImage::new(
                    "morphogenesis_image",
                    tex_id,
                    PlotPoint::new(f64::from(w) / 2.0, f64::from(h) / 2.0),
                    [w, h],
                ));
                let (primary, secondary) =
                    ctx.input(|i| (i.pointer.primary_down(), i.pointer.secondary_down()));
                if !allow_pan && plot_ui.response().hovered() && (primary || secondary) {
                    if let Some(pos) = plot_ui.pointer_coordinate() {
                        let gx = pos.x.round() as i32;
                        let gy = ((f64::from(h) - 1.0) - pos.y).round() as i32;
                        brush_cmd = Some(SimCommand::ApplyBrush {
                            cx: gx,
                            cy: gy,
                            r: brush_radius,
                            is_obstacle: primary,
                        });
                    }
                }
            });

        if let Some(cmd) = brush_cmd {
            if self.worker.send_command(cmd).is_ok() {
                self.dirty = true;
                ctx.request_repaint();
            }
        }
    }

    pub(crate) fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.sync_texture_from_snapshot(&ctx);
            if !self.atlas.is_visible() {
                self.draw_turing_field_plot(ui, &ctx);
                return;
            }
            let (ax, ay) = self.active_axis_values();
            let sig = self.non_axis_context_sig();
            let cart_p = {
                let p = self.params.read().unwrap_or_else(|e| e.into_inner());
                (p.a, p.b, p.d_u, p.d_v, p.dt)
            };
            let theory_params = self.theory.theory_parameters();
            let build_cart = |keys: (&'static str, &'static str)| -> std::sync::Arc<
                dyn crate::async_sim::atlas::RegimeCartographer,
            > {
                std::sync::Arc::new(super::atlas::MorphogenesisCartographer::new(
                    &theory_params,
                    keys,
                    cart_p.0,
                    cart_p.1,
                    cart_p.2,
                    cart_p.3,
                    cart_p.4,
                ))
            };
            let mut teleport = None;
            if ui.available_width() >= 620.0 {
                ui.columns(2, |cols| {
                    self.draw_turing_field_plot(&mut cols[0], &ctx);
                    teleport =
                        self.atlas
                            .show_panel(&mut cols[1], "morphogenesis", ax, ay, sig, build_cart);
                });
            } else {
                teleport = self
                    .atlas
                    .show_panel(ui, "morphogenesis", ax, ay, sig, build_cart);
                ui.separator();
                self.draw_turing_field_plot(ui, &ctx);
            }
            if let Some(ev) = teleport {
                self.apply_atlas_teleport(ev);
                ctx.request_repaint();
            }
        });
    }

    pub(crate) fn initialize_coupled_system(
        system: &mut TuringSystem<2, MorphogenesisKinetics, AdvectionDiffusion2D>,
        width: usize,
        height: usize,
    ) {
        let n = width * height;
        let mut rng = SimpleRng::new(12345);
        match system.kinetics.mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                let u_eq = (system.kinetics.a + system.kinetics.b).max(0.05);
                let v_eq = system.kinetics.b / (u_eq * u_eq);
                for i in 0..n {
                    system.u_mut()[i] = u_eq + rng.range(-0.1, 0.1);
                    system.v_mut()[i] = v_eq + rng.range(-0.1, 0.1);
                }
            }
            MorphogenesisKineticsMode::GrayScott => {
                system.u_mut().fill(1.0);
                system.v_mut().fill(0.0);
                let patches = [
                    (width / 2, height / 2),
                    (width / 3, height / 3),
                    ((2 * width) / 3, (2 * height) / 3),
                ];
                let patch_r = (width.min(height) / 10).clamp(2, 6) as isize;
                for (px, py) in patches {
                    for y in 0..height {
                        for x in 0..width {
                            let dx = x as isize - px as isize;
                            let dy = y as isize - py as isize;
                            if dx * dx + dy * dy <= patch_r * patch_r {
                                let idx = flatten_2d_index(x, y, width);
                                system.u_mut()[idx] = 0.50 + rng.range(-0.04, 0.04);
                                system.v_mut()[idx] = 0.25 + rng.range(-0.04, 0.04);
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn plot_concentration(data: &[f64], width: usize, height: usize) -> ColorImage {
        Self::plot_concentration_with_contrast(data, width, height, false)
    }

    #[cfg(test)]
    pub(crate) fn plot_concentration_with_contrast(
        data: &[f64],
        width: usize,
        height: usize,
        high_contrast: bool,
    ) -> ColorImage {
        let theme = if high_contrast {
            ThemeMode::HighContrast
        } else {
            ThemeMode::Dark
        };
        render_morphogenesis_overlay(
            data,
            &[],
            &[],
            width,
            height,
            super::types::MorphogenesisOverlayMode::Concentration,
            theme,
        )
    }
}
