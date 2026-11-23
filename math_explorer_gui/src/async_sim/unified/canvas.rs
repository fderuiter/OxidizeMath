#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{UnifiedModel, UnifiedSimTool};
use crate::accessibility::AccessibleTheoryHover;
use crate::async_sim::ui::render_empty_plot_guidance;
use crate::async_sim::StateSnapshot;
use crate::framework::RasterBrushCanvas;
use eframe::egui::{self, ColorImage, TextureOptions};
use egui_plot::{HLine, Line, LineStyle, Plot, PlotPoints, Points};

impl<M: UnifiedModel> UnifiedSimTool<M> {
    fn draw_image_plot(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        texture: &egui::TextureHandle,
    ) {
        let (grid_w, grid_h, flip_y) = M::raster_brush_grid().unwrap_or_else(|| {
            let sz = texture.size();
            (sz[0].max(1), sz[1].max(1), false)
        });
        let id_salt = format!("{}_raster_canvas", M::name());
        let label = format!("{} raster simulation canvas", M::name());
        let (response, stroke) = RasterBrushCanvas::new(&id_salt, grid_w, grid_h, &label)
            .flip_y(flip_y)
            .show_texture(ui, texture, &mut self.brush_state);
        let _ = response.accessible_theory_hover(self);
        if let Some(stroke) = stroke {
            self.apply_brush_stroke(stroke, ctx);
        }
    }

    fn draw_line_plot(&self, ui: &mut egui::Ui, snapshot: &StateSnapshot) {
        if snapshot.custom_data.is_empty() {
            render_empty_plot_guidance(
                ui,
                "No trajectory samples yet — click ▶ Run in the side panel to begin.",
            );
            return;
        }
        let mut points = Vec::new();
        for i in (0..snapshot.custom_data.len()).step_by(2) {
            if i + 1 < snapshot.custom_data.len() {
                points.push([snapshot.custom_data[i], snapshot.custom_data[i + 1]]);
            }
        }
        if points.len() <= 1 {
            render_empty_plot_guidance(
                ui,
                "Initial resting state ready — click ▶ Run to evolve the trajectory over time.",
            );
        }

        let color = crate::app::SemanticPalette::from_visuals(ui.visuals()).series_color(0);
        let initial_pt = (points.len() == 1).then(|| points[0]);
        let line = Line::new(M::name(), PlotPoints::from_iter(points)).color(color);

        Plot::new(format!("{}_plot_line", M::name()))
            .view_aspect(2.0)
            .show(ui, |plot_ui| {
                plot_ui.line(line);
                if let Some(pt) = initial_pt {
                    plot_ui.points(
                        Points::new(format!("{} (t=0)", M::name()), vec![pt])
                            .radius(4.5_f32)
                            .color(color),
                    );
                    plot_ui.hline(
                        HLine::new("Initial Baseline", pt[1])
                            .style(LineStyle::Dashed { length: 6.0_f32 })
                            .color(color),
                    );
                }
            });
    }

    fn refresh_texture_if_dirty(&mut self, ctx: &egui::Context) {
        if !self.dirty && self.texture.is_some() {
            return;
        }
        let Some(snapshot) = self.controller.latest_snapshot() else {
            return;
        };
        let Ok(guard) = snapshot.pixels.try_read() else {
            return;
        };
        if guard.is_empty() {
            return;
        }
        let image = ColorImage::new([snapshot.width, snapshot.height], guard.clone());
        if let Some(texture) = &mut self.texture {
            texture.set(image, TextureOptions::NEAREST);
        } else {
            self.texture = Some(ctx.load_texture(M::name(), image, TextureOptions::NEAREST));
        }
        self.dirty = false;
    }

    fn render_primary_viewport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        self.refresh_texture_if_dirty(ctx);
        if let Some(texture) = self.texture.clone() {
            self.draw_image_plot(ui, ctx, &texture);
        } else if let Some(snapshot) = self.controller.latest_snapshot() {
            self.draw_line_plot(ui, snapshot);
        } else {
            render_empty_plot_guidance(
                ui,
                "Simulation ready — click ▶ Run in the side panel to start.",
            );
        }
    }

    pub(super) fn draw_central_panel(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::CentralPanel::default().show_inside(ui, |ui| {
            if M::custom_central_panel(ui, self.controller.latest_snapshot()) {
                return;
            }
            if !self.is_atlas_visible() {
                self.render_primary_viewport(ui, &ctx);
                return;
            }
            let mut teleport = None;
            if ui.available_width() >= 680.0 {
                ui.columns(2, |cols| {
                    self.render_primary_viewport(&mut cols[0], &ctx);
                    teleport = self.draw_atlas_panel_ui(&mut cols[1]);
                });
            } else {
                teleport = self.draw_atlas_panel_ui(ui);
                ui.separator();
                self.render_primary_viewport(ui, &ctx);
            }
            if let Some(ev) = teleport {
                self.apply_atlas_teleport(ev, Some(&ctx));
            }
        });
    }
}
