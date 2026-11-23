//! Snapshot texture synchronization, `SpatialBrushKit` canvas plot, counterfactual divergence
//! strip, and side-by-side parameter atlas view.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::counterfactual::{render_morphogenesis_heatmap, MorphogenesisHeatmapMode};
use super::overlay::render_morphogenesis_overlay;
use super::runner::MorphoSnapshotData;
use super::sculpt::overlay_cell_mask_colors;
use super::types::{MorphogenesisFlowMode, MorphogenesisOverlayMode};
use super::MorphogenesisTool;
use crate::app::{SemanticPalette, ThemeMode};
use crate::async_sim::ui::render_empty_plot_guidance;
use crate::framework::{
    fork_marker_callout, render_divergence_strip, BrushMode, InteractionContext, InteractiveTool,
};
use eframe::egui;
use eframe::egui::ColorImage;
use egui_plot::{Plot, PlotImage, PlotPoint};

impl MorphogenesisTool {
    fn recolor_snapshot_data(
        mdata: &MorphoSnapshotData,
        width: usize,
        height: usize,
        flow: super::types::MorphogenesisFlowState,
        heatmap_mode: MorphogenesisHeatmapMode,
        theme_mode: ThemeMode,
    ) -> ColorImage {
        let high_contrast = theme_mode == ThemeMode::HighContrast;
        let use_twin_or_base_heatmap = (mdata.u_a.is_some()
            && heatmap_mode != MorphogenesisHeatmapMode::ActiveUb)
            || (flow.flow_mode == MorphogenesisFlowMode::None
                && flow.overlay_mode == MorphogenesisOverlayMode::Concentration);
        let mut recolored = if use_twin_or_base_heatmap {
            render_morphogenesis_heatmap(
                &mdata.u_data,
                mdata.u_a.as_deref(),
                width,
                height,
                heatmap_mode,
                theme_mode,
            )
        } else {
            render_morphogenesis_overlay(
                &mdata.u_data,
                &mdata.vorticity,
                &mdata.obstacles,
                width,
                height,
                flow.overlay_mode,
                theme_mode,
            )
        };
        if heatmap_mode == MorphogenesisHeatmapMode::ActiveUb && !mdata.mask_roles.is_empty() {
            overlay_cell_mask_colors(&mut recolored, &mdata.mask_roles, high_contrast);
        }
        recolored
    }

    fn resolve_next_canvas_image(&mut self, theme_mode: ThemeMode) -> Option<ColorImage> {
        let heatmap_mode = self.heatmap_mode();
        let flow = *self.flow_state.read().unwrap_or_else(|e| e.into_inner());
        if let Some(snapshot) = self.worker.latest_snapshot() {
            let num_pixels = snapshot.width * snapshot.height;
            if let Some(mdata) = snapshot
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<MorphoSnapshotData>())
                .filter(|m| {
                    m.u_data.len() == num_pixels
                        && (!self.is_forked() || m.step_count >= self.local_runner.step_count)
                })
            {
                self.live_peclet = mdata.peclet;
                self.live_damkohler = mdata.damkohler;
                return Some(Self::recolor_snapshot_data(
                    mdata,
                    snapshot.width,
                    snapshot.height,
                    flow,
                    heatmap_mode,
                    theme_mode,
                ));
            }
            if let Ok(px) = snapshot.pixels.try_read() {
                if px.len() == num_pixels && !px.is_empty() && !self.is_forked() {
                    return Some(ColorImage {
                        size: [snapshot.width, snapshot.height],
                        pixels: px.clone(),
                        ..Default::default()
                    });
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
        let theme_mode = self
            .last_theme_mode
            .unwrap_or_else(|| ThemeMode::from_visuals(&ctx.style().visuals));
        let Some(image) = self.resolve_next_canvas_image(theme_mode) else {
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

    pub(crate) fn draw_turing_field_plot(&mut self, ui: &mut egui::Ui) {
        if let Some(texture) = self.texture.clone() {
            let w = self.width as f32;
            let h = self.height as f32;
            let forked = self.local_runner.twin.as_ref().map(|t| t.fork_step);
            let palette = SemanticPalette::from_visuals(ui.visuals());
            let allow_pan = !self.brush_kit.is_sculpting();
            let mut strokes = Vec::new();
            let plot_resp = Plot::new("morphogenesis_plot")
                .data_aspect(1.0)
                .view_aspect(1.0)
                .show_grid(false)
                .show_axes([false, false])
                .allow_drag(allow_pan)
                .allow_zoom(allow_pan)
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
                    strokes = self.brush_kit.sample_plot(
                        plot_ui,
                        [0.5, 0.5],
                        [f64::from(w) - 0.5, f64::from(h) - 0.5],
                        Some(BrushMode::Erase),
                    );
                });
            for stroke in strokes {
                let ic = InteractionContext::from_brush_stroke(&plot_resp.response, stroke);
                self.on_brush(&ic);
            }
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
            if let Some(snap) = self.counterfactual_snapshot() {
                render_divergence_strip(ui, self.name(), &snap);
                ui.add_space(4.0);
            }
            self.sync_texture_from_snapshot(&ctx);
            if !self.atlas.is_visible() {
                self.draw_turing_field_plot(ui);
                return;
            }
            let current_mode = self
                .kinetics_state
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .mode;
            self.sync_atlas_options_for_mode(current_mode);
            let (ax, ay) = self.active_axis_values();
            let sig = self.non_axis_context_sig();
            let cart = self.build_active_morpho_cartographer(self.atlas.axis_keys());
            let mut teleport = None;
            if ui.available_width() >= 620.0 {
                ui.columns(2, |cols| {
                    self.draw_turing_field_plot(&mut cols[0]);
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
                self.draw_turing_field_plot(ui);
            }
            if let Some(ev) = teleport {
                self.apply_atlas_teleport(ev);
                ctx.request_repaint();
            }
        });
    }
}
