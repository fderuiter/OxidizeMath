#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! `Plot3DViewport` consolidated 3D wireframe rendering, dirty-cached 2D raster heatmap + contour view,
//! and canvas gestures for [`super::LossLandscapeTool`].

use std::sync::Arc;

use super::surfaces::ARENA_3D_HALF_EXTENT;
use super::{CanvasInteractionMode, LossLandscapeTool};
use crate::camera::{Plot3DViewport, CAMERA_3D_HELPER_TEXT};
use crate::framework::{InputMode, InteractionContext, InteractiveTool};
use eframe::egui;
use egui_plot::{Line, Plot, PlotImage, PlotPoint};

impl LossLandscapeTool {
    pub(super) fn handle_viewport_gesture(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if self.interaction_mode == CanvasInteractionMode::Orbit {
            (
                self.slingshot_anchor,
                self.slingshot_current,
                self.last_brush_wb,
            ) = (None, None, None);
            return;
        }
        let pointer_pos = response.interact_pointer_pos();
        if self.interaction_mode == CanvasInteractionMode::SculptBrush {
            if response.dragged_by(egui::PointerButton::Primary) || response.clicked() {
                let ctx = InteractionContext {
                    pointer_pos,
                    delta: response.drag_delta(),
                    is_dragging: response.dragged(),
                    is_clicked: response.clicked(),
                    response,
                    input_mode: InputMode::Mouse,
                    multi_touch: ui.input(|i| i.multi_touch()),
                    keys_down: ui.input(|i| i.keys_down.clone()),
                    modifiers: ui.input(|i| i.modifiers),
                };
                self.on_brush(&ctx);
            } else {
                self.last_brush_wb = None;
            }
            return;
        }
        if response.drag_started_by(egui::PointerButton::Primary) {
            if let Some(pos) = pointer_pos {
                let wb = self.pointer_to_wb(response.rect, pos);
                (self.slingshot_anchor, self.slingshot_current) = (Some(wb), Some(wb));
            }
        } else if response.dragged_by(egui::PointerButton::Primary) {
            if let Some(pos) = pointer_pos {
                self.slingshot_current = Some(self.pointer_to_wb(response.rect, pos));
            }
        } else if response.drag_stopped_by(egui::PointerButton::Primary) {
            if let (Some(a), Some(c)) =
                (self.slingshot_anchor.take(), self.slingshot_current.take())
            {
                let vel = [(a[0] - c[0]) * 2.5, (a[1] - c[1]) * 2.5];
                let bounds = self.domain_bounds();
                let emp = Self::build_empirical(
                    self.surface,
                    self.predictor_mode,
                    self.target_mode,
                    self.loss_metric,
                    self.huber_delta,
                    &self.cached_targets,
                    &self.math_input,
                );
                let eval =
                    super::surfaces::SurfaceEvaluator::new(self.surface, &self.sculpt_bumps, emp);
                self.arena
                    .launch_slingshot(&mut self.arena_config, a, vel, eval, bounds);
            }
        }
    }

    pub(super) fn show_3d_arena(&mut self, ui: &mut egui::Ui, grid_key: u64) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let dark_mode = ui.visuals().dark_mode;
        let suppress_orbit = self.interaction_mode != CanvasInteractionMode::Orbit;
        let helper = match self.interaction_mode {
            CanvasInteractionMode::Orbit => CAMERA_3D_HELPER_TEXT,
            CanvasInteractionMode::Slingshot => {
                "Slingshot: drag on surface to pull back and launch active particles (orbit locked)."
            }
            CanvasInteractionMode::SculptBrush => {
                "Sculpt: drag to raise barriers (+A) or Alt-drag to carve wells (-A) (orbit locked)."
            }
        };

        let tabular = self.ensure_tabular_series(grid_key, &palette);
        self.ensure_3d_scene(grid_key, &palette, dark_mode);

        let mut cam = self.camera;
        let plot_resp = Plot3DViewport::new("loss_landscape_3d_plot", ARENA_3D_HALF_EXTENT)
            .helper_text(helper)
            .show(ui, &mut cam, |plot_ui| {
                let pid = plot_ui.plot_id();
                plot_ui.ctx().data_mut(|d| {
                    d.insert_temp(pid.with("dragging_callout"), suppress_orbit);
                });
                plot_ui.set_tabular_series(Arc::clone(&tabular));
                self.cache.scene_builder.finish_without_sidecar(plot_ui);
            });
        if !suppress_orbit {
            self.camera = cam;
        }
        self.handle_viewport_gesture(ui, &plot_resp.response);
    }

    pub(super) fn show_2d_contours(&mut self, ui: &mut egui::Ui, grid_key: u64) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let dark_mode = ui.visuals().dark_mode;
        let ctx = ui.ctx().clone();
        let (tex_id, backing_image) = self.ensure_heatmap_texture(&ctx, grid_key, &palette);
        let tabular = self.ensure_tabular_series(grid_key, &palette);
        let (overlay_segs, overlay_bounds) = self.ensure_2d_overlay(grid_key, &palette, dark_mode);
        let (w_min, w_max, b_min, b_max) = self.domain_bounds();
        let center = PlotPoint::new(0.5 * (w_min + w_max), 0.5 * (b_min + b_max));
        let size = egui::vec2((w_max - w_min) as f32, (b_max - b_min) as f32);
        let allow_pan = self.interaction_mode == CanvasInteractionMode::Orbit;

        let plot_resp = Plot::new("loss_landscape_plot")
            .x_axis_label("Weight (w)")
            .y_axis_label("Bias (b)")
            .data_aspect(1.0)
            .allow_drag(allow_pan)
            .show(ui, |plot_ui| {
                plot_ui.set_tabular_series(tabular);
                plot_ui.image(
                    PlotImage::new("Loss Heatmap", tex_id, center, size)
                        .with_backing_image(backing_image),
                );
                if !overlay_segs.is_empty() {
                    plot_ui.line(
                        Line::from_colored_segments("Contours & Trajectories", overlay_segs)
                            .with_bounds(overlay_bounds),
                    );
                }
            });
        self.handle_viewport_gesture(ui, &plot_resp.response);
    }
}
