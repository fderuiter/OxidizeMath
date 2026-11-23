#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::ui::render_empty_plot_guidance;
use crate::async_sim::SimCommand;
use crate::framework::{BrushMode, InteractionContext, InteractiveTool};
use eframe::egui::{self, ColorImage, TextureOptions};
use egui_plot::{HLine, Line, LineStyle, Plot, PlotImage, PlotPoint, PlotPoints, Points};

impl<M: UnifiedModel> UnifiedSimTool<M> {
    fn sample_legacy_image_brush(&mut self, ctx: &egui::Context, plot_ui: &egui_plot::PlotUi) {
        if !plot_ui.response().hovered()
            || !ctx.input(|i| i.pointer.primary_down() || i.pointer.secondary_down())
        {
            return;
        }
        let Some(pos) = plot_ui.pointer_coordinate() else {
            return;
        };
        let grid_x = pos.x.round() as i32;
        let grid_y = pos.y.round() as i32;
        let is_obs = ctx.input(|i| i.pointer.primary_down());
        let allowed = self
            .ui_extension
            .as_mut()
            .is_none_or(|ext| ext.on_brush(grid_x, grid_y, 2, is_obs));
        if allowed
            && self
                .controller
                .send_command(SimCommand::ApplyBrush {
                    cx: grid_x,
                    cy: grid_y,
                    r: 2,
                    is_obstacle: is_obs,
                })
                .is_ok()
        {
            ctx.request_repaint();
        }
    }

    fn draw_image_plot(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        texture: &egui::TextureHandle,
    ) {
        let width = texture.size()[0] as f32;
        let height = texture.size()[1] as f32;
        let allow_pan = !M::supports_spatial_brush() || !self.brush_kit.is_sculpting();
        let mut strokes = Vec::new();
        let visuals = ui.visuals().clone();
        let mut ext_cmds = Vec::new();

        let plot_resp = Plot::new(format!("{}_plot_img", M::name()))
            .data_aspect(1.0)
            .view_aspect(1.0)
            .show_grid(false)
            .show_axes([false, false])
            .allow_drag(allow_pan)
            .allow_zoom(allow_pan)
            .allow_scroll(allow_pan)
            .show(ui, |plot_ui| {
                plot_ui.image(PlotImage::new(
                    format!("{}_image", M::name()),
                    texture.id(),
                    PlotPoint::new(f64::from(width) / 2.0, f64::from(height) / 2.0),
                    [width, height],
                ));
                if let Some(ext) = &mut self.ui_extension {
                    ext_cmds = ext.handle_canvas_overlay(ctx, &visuals, plot_ui);
                }
                if M::supports_spatial_brush() {
                    strokes = self.brush_kit.sample_plot(
                        plot_ui,
                        [0.5, 0.5],
                        [f64::from(width) - 0.5, f64::from(height) - 0.5],
                        Some(BrushMode::Erase),
                    );
                } else {
                    self.sample_legacy_image_brush(ctx, plot_ui);
                }
                self.callout_manager.render_on_plot(plot_ui);
            });

        for stroke in strokes {
            let ic = InteractionContext::from_brush_stroke(&plot_resp.response, stroke);
            self.on_brush(&ic);
        }
        self.dispatch_extension_commands(ext_cmds, Some(ctx));
    }

    fn draw_line_plot(&mut self, ui: &mut egui::Ui, custom_data: &[f64]) {
        if custom_data.is_empty() {
            render_empty_plot_guidance(
                ui,
                "No trajectory samples yet — click ▶ Run in the side panel to begin.",
            );
            return;
        }
        let mut points = Vec::new();
        for i in (0..custom_data.len()).step_by(2) {
            if i + 1 < custom_data.len() {
                points.push([custom_data[i], custom_data[i + 1]]);
            }
        }
        if points.len() <= 1 {
            render_empty_plot_guidance(
                ui,
                "Initial resting state ready — click ▶ Run to evolve the trajectory over time.",
            );
        }
        if let Some(&last_pt) = points.last() {
            self.callout_manager.set_auto_callout(
                "unified_latest",
                last_pt,
                format!("({:.2}, {:.2})", last_pt[0], last_pt[1]),
                [22.0_f32, -18.0_f32],
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
                self.callout_manager.render_on_plot(plot_ui);
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
        if let Some(ext) = &self.ui_extension {
            ext.sync_callouts(self.controller.latest_snapshot(), &mut self.callout_manager);
        }
        if let Some(texture) = self.texture.clone() {
            self.draw_image_plot(ui, ctx, &texture);
        } else if let Some(custom_data) = self
            .controller
            .latest_snapshot()
            .map(|s| s.custom_data.clone())
        {
            self.draw_line_plot(ui, &custom_data);
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
            let (handled, strokes) = M::custom_central_panel_with_brush(
                ui,
                self.controller.latest_snapshot(),
                &mut self.brush_kit,
            );
            if handled {
                if !strokes.is_empty() {
                    let resp = ui.interact(
                        ui.min_rect(),
                        ui.id().with("unified_custom_brush"),
                        egui::Sense::hover(),
                    );
                    for stroke in strokes {
                        let ic = InteractionContext::from_brush_stroke(&resp, stroke);
                        self.on_brush(&ic);
                    }
                }
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
