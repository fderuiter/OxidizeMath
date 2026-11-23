#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::ui::render_empty_plot_guidance;
use crate::async_sim::{SimCommand, SimulationController, StateSnapshot};
use eframe::egui::{self, ColorImage, TextureOptions};
use egui_plot::{HLine, Line, LineStyle, Plot, PlotImage, PlotPoint, PlotPoints, Points};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Interactive central-panel context granting `UnifiedModel` implementations access to the
/// simulation worker controller, active snapshot, shared parameters, and custom series state.
pub struct UnifiedCanvasContext<'a> {
    /// Handle to the background simulation controller for dispatching commands.
    pub controller: &'a mut SimulationController,
    /// Optional deterministic export snapshot overriding live worker snapshots.
    pub stepped_snapshot: Option<&'a StateSnapshot>,
    /// Shared scalar simulation parameters synchronized with the worker.
    pub params: &'a Arc<RwLock<HashMap<String, f64>>>,
    /// Shared custom series vectors (such as `"custom_potential"`) synchronized with the worker.
    pub custom_series: &'a Arc<RwLock<HashMap<String, Vec<f64>>>>,
}

impl UnifiedCanvasContext<'_> {
    /// Returns the active frame snapshot, preferring any stepped export snapshot.
    #[must_use]
    pub fn snapshot(&self) -> Option<&StateSnapshot> {
        self.stepped_snapshot
            .or_else(|| self.controller.latest_snapshot())
    }
}

impl<M: UnifiedModel> UnifiedSimTool<M> {
    fn draw_image_plot(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        texture: &egui::TextureHandle,
    ) {
        let width = texture.size()[0] as f32;
        let height = texture.size()[1] as f32;

        let visuals = ui.visuals().clone();
        let mut ext_cmds = Vec::new();
        Plot::new(format!("{}_plot_img", M::name()))
            .data_aspect(1.0)
            .view_aspect(1.0)
            .show_grid(false)
            .show_axes([false, false])
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .show(ui, |plot_ui| {
                plot_ui.image(PlotImage::new(
                    format!("{}_image", M::name()),
                    texture.id(),
                    PlotPoint::new(width as f64 / 2.0, height as f64 / 2.0),
                    [width, height],
                ));

                if let Some(ext) = &mut self.ui_extension {
                    ext_cmds = ext.handle_canvas_overlay(ctx, &visuals, plot_ui);
                }

                if plot_ui.response().hovered()
                    && ctx.input(|i| i.pointer.primary_down() || i.pointer.secondary_down())
                {
                    if let Some(pos) = plot_ui.pointer_coordinate() {
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
                }
                self.callout_manager.render_on_plot(plot_ui);
            });
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
            let mut canvas_ctx = super::UnifiedCanvasContext {
                controller: self.controller.controller_mut(),
                stepped_snapshot: self.stepped_snapshot.as_ref(),
                params: &self.params,
                custom_series: &self.custom_series,
            };
            if M::custom_central_panel_interactive(ui, &mut canvas_ctx) {
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
