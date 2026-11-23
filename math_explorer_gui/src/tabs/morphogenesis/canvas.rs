//! Snapshot texture synchronization, `SpatialBrushKit` canvas plot, and side-by-side parameter atlas view.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::overlay::render_morphogenesis_overlay;
use super::runner::MorphoSnapshotData;
use super::sculpt::overlay_cell_mask_colors;
use super::types::{MorphogenesisFlowMode, MorphogenesisOverlayMode};
use super::MorphogenesisTool;
use crate::app::ThemeMode;
use crate::async_sim::ui::render_empty_plot_guidance;
use crate::framework::{BrushMode, InteractionContext, InteractiveTool};
use eframe::egui;
use eframe::egui::ColorImage;
use egui_plot::{Plot, PlotImage, PlotPoint};
use scientific_metadata::theory::TheoryDescribable;

impl MorphogenesisTool {
    fn recolor_snapshot_data(
        mdata: &MorphoSnapshotData,
        width: usize,
        height: usize,
        flow: super::types::MorphogenesisFlowState,
        theme_mode: ThemeMode,
    ) -> ColorImage {
        let high_contrast = theme_mode == ThemeMode::HighContrast;
        let mut recolored = if flow.flow_mode == MorphogenesisFlowMode::None
            && flow.overlay_mode == MorphogenesisOverlayMode::Concentration
        {
            Self::plot_concentration_with_contrast(&mdata.u_data, width, height, high_contrast)
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
        if !mdata.mask_roles.is_empty() {
            overlay_cell_mask_colors(&mut recolored, &mdata.mask_roles, high_contrast);
        }
        recolored
    }

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
            self.pixels.resize(num_pixels, egui::Color32::BLACK);
        }
        let theme_mode = self
            .last_theme_mode
            .unwrap_or_else(|| ThemeMode::from_visuals(&ctx.style().visuals));
        let flow = *self.flow_state.read().unwrap_or_else(|e| e.into_inner());

        if let Some(mdata) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<MorphoSnapshotData>())
            .filter(|m| m.u_data.len() == num_pixels)
        {
            self.live_peclet = mdata.peclet;
            self.live_damkohler = mdata.damkohler;
            let recolored = Self::recolor_snapshot_data(
                mdata,
                snapshot.width,
                snapshot.height,
                flow,
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
            self.sync_texture_from_snapshot(&ctx);
            if !self.atlas.is_visible() {
                self.draw_turing_field_plot(ui);
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
                    self.draw_turing_field_plot(&mut cols[0]);
                    teleport =
                        self.atlas
                            .show_panel(&mut cols[1], "morphogenesis", ax, ay, sig, build_cart);
                });
            } else {
                teleport = self
                    .atlas
                    .show_panel(ui, "morphogenesis", ax, ay, sig, build_cart);
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
