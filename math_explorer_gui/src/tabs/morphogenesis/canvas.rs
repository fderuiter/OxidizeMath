//! Snapshot texture synchronization, split parameter-atlas layout, and brushable plot canvas
//! for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::counterfactual::{render_morphogenesis_heatmap, MorphogenesisHeatmapMode};
use super::runner::MorphoSnapshotData;
use super::sculpt::overlay_cell_mask_colors;
#[cfg(test)]
use super::state::{initialize_schnakenberg_system, plot_concentration_with_contrast};
use super::MorphogenesisTool;
use crate::app::ThemeMode;
use crate::async_sim::ui::render_empty_plot_guidance;
use crate::framework::{
    fork_marker_callout, render_divergence_strip, BrushMode, InteractiveTool, PlotBrushController,
    PlotBrushFrame,
};
use eframe::egui::{self, ColorImage};
use egui_plot::{Plot, PlotImage, PlotPoint};
#[cfg(test)]
use math_explorer::biology::diffusion::SpatialDiffusion;
#[cfg(test)]
use math_explorer::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};

impl MorphogenesisTool {
    fn render_snapshot_image(
        &self,
        theme: ThemeMode,
        mode: MorphogenesisHeatmapMode,
    ) -> Option<ColorImage> {
        if let Some(snapshot) = self.worker.latest_snapshot() {
            let num_pixels = snapshot.width * snapshot.height;
            if let Some(data) = snapshot
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<MorphoSnapshotData>())
                .filter(|d| {
                    num_pixels > 0
                        && d.u_data.len() == num_pixels
                        && (!self.is_forked() || d.step_count >= self.local_runner.step_count)
                })
            {
                let mut img = render_morphogenesis_heatmap(
                    &data.u_data,
                    data.u_a.as_deref(),
                    snapshot.width,
                    snapshot.height,
                    mode,
                    theme,
                );
                if mode == MorphogenesisHeatmapMode::ActiveUb && !data.mask_roles.is_empty() {
                    overlay_cell_mask_colors(
                        &mut img,
                        &data.mask_roles,
                        theme == ThemeMode::HighContrast,
                    );
                }
                return Some(img);
            }
            if !self.is_forked() {
                if let Ok(px) = snapshot.pixels.try_read() {
                    if num_pixels > 0 && px.len() == num_pixels {
                        return Some(ColorImage {
                            size: [snapshot.width, snapshot.height],
                            pixels: px.clone(),
                            ..Default::default()
                        });
                    }
                }
            }
        }
        self.is_forked()
            .then(|| self.render_current_heatmap_image())
    }

    pub(crate) fn sync_texture_from_snapshot(&mut self, ctx: &egui::Context) {
        if !self.dirty && self.texture.is_some() {
            return;
        }
        let mut theme = self
            .last_theme_mode
            .unwrap_or_else(|| ThemeMode::from_visuals(&ctx.style().visuals));
        let (mode, high_contrast) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.heatmap_mode, p.high_contrast)
        };
        if high_contrast {
            theme = ThemeMode::HighContrast;
        }
        let Some(image) = self.render_snapshot_image(theme, mode) else {
            return;
        };
        let num_pixels = image.size[0] * image.size[1];
        if self.pixels.len() != num_pixels {
            self.pixels.resize(num_pixels, egui::Color32::BLACK);
        }
        self.pixels.copy_from_slice(&image.pixels);

        let texture_opts = egui::TextureOptions::NEAREST;
        if let Some(texture) = &mut self.texture {
            texture.set(image, texture_opts);
        } else {
            self.texture = Some(ctx.load_texture("morphogenesis_tex", image, texture_opts));
        }
        self.dirty = false;
    }

    fn draw_field_plot(&mut self, ui: &mut egui::Ui) {
        let Some(texture) = self.texture.clone() else {
            render_empty_plot_guidance(
                ui,
                "Reaction-diffusion grid ready — click ▶ Play or ⏭ Step to evolve patterns.",
            );
            return;
        };

        let w = self.width as f32;
        let h = self.height as f32;
        let forked = self.local_runner.twin.as_ref().map(|t| t.fork_step);
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let allow_pan = !self.brush_kit.is_sculpting();
        let mut brush_frame: Option<PlotBrushFrame> = None;
        let plot_resp = Plot::new("morphogenesis_plot")
            .data_aspect(1.0)
            .view_aspect(1.0)
            .show_grid(false)
            .show_axes([false, false])
            .allow_drag(allow_pan)
            .allow_zoom(allow_pan)
            .allow_scroll(allow_pan)
            .show(ui, |plot_ui| {
                plot_ui.image(PlotImage::new(
                    "morphogenesis_image",
                    texture.id(),
                    PlotPoint::new(f64::from(w) / 2.0, f64::from(h) / 2.0),
                    [w, h],
                ));
                if let Some(step) = forked {
                    let dt = self.params.read().unwrap_or_else(|e| e.into_inner()).dt;
                    plot_ui.callout(fork_marker_callout(
                        [f64::from(w) * 0.5, f64::from(h) * 0.94],
                        (step as f64) * dt,
                        &palette,
                    ));
                }
                brush_frame = Some(PlotBrushController::capture_plot(
                    &mut self.brush_kit,
                    plot_ui,
                    [0.5, 0.5],
                    [f64::from(w) - 0.5, f64::from(h) - 0.5],
                    Some(BrushMode::Erase),
                ));
            });
        if let Some(frame) = brush_frame {
            frame.dispatch(ui, &plot_resp.response, self);
        }
    }

    pub(crate) fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::CentralPanel::default().show_inside(ui, |ui| {
            if let Some(snap) = self.counterfactual_snapshot() {
                render_divergence_strip(ui, self.name(), &snap);
                ui.add_space(4.0);
            }
            self.sync_texture_from_snapshot(&ctx);
            if !self.atlas.is_visible() {
                self.draw_field_plot(ui);
                return;
            }
            let (current_mode, _, _) = self.effective_kinetics_triplet();
            self.sync_atlas_options_for_mode(current_mode);
            let (ax, ay) = self.active_axis_values();
            let sig = self.non_axis_context_sig();
            let cart = self.build_active_morpho_cartographer(self.atlas.axis_keys());
            let mut teleport = None;
            if ui.available_width() >= 620.0 {
                ui.columns(2, |cols| {
                    self.draw_field_plot(&mut cols[0]);
                    teleport =
                        self.atlas
                            .show_panel(&mut cols[1], "morphogenesis", ax, ay, sig, |_| {
                                std::sync::Arc::clone(&cart)
                            });
                });
            } else {
                teleport = self
                    .atlas
                    .show_panel(ui, "morphogenesis", ax, ay, sig, |_| {
                        std::sync::Arc::clone(&cart)
                    });
                ui.separator();
                self.draw_field_plot(ui);
            }
            if let Some(ev) = teleport {
                self.apply_atlas_teleport(ev);
                ctx.request_repaint();
            }
        });
    }

    #[cfg(test)]
    pub(crate) fn initialize_system<D: SpatialDiffusion<2>>(
        system: &mut TuringSystem<2, SchnakenbergKinetics, D>,
        width: usize,
        height: usize,
    ) {
        initialize_schnakenberg_system(system, width, height);
    }

    #[cfg(test)]
    pub(crate) fn plot_concentration(data: &[f64], width: usize, height: usize) -> ColorImage {
        plot_concentration_with_contrast(data, width, height, false)
    }

    /// Maps a 2D concentration slice onto a normalized heatmap [`ColorImage`] with optional high-contrast tokens.
    #[cfg(test)]
    #[must_use]
    pub fn plot_concentration_with_contrast(
        data: &[f64],
        width: usize,
        height: usize,
        high_contrast: bool,
    ) -> ColorImage {
        plot_concentration_with_contrast(data, width, height, high_contrast)
    }
}
