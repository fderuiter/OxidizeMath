#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! `Plot3DViewport` batched 3D scene rendering, cached 2D raster + batched contour view,
//! live inspector telemetry bar, racer overlays, and canvas gestures.

use std::sync::Arc;

use super::surfaces::{lerp_color, ARENA_3D_HALF_EXTENT};
use super::{CanvasInteractionMode, LossLandscapeTool};
use crate::app::SemanticPalette;
use crate::camera::{Plot3DViewport, CAMERA_3D_HELPER_TEXT};
use crate::framework::{InputMode, InteractionContext, InteractiveTool};
use eframe::egui::{self, RichText, Stroke};
use egui_plot::{
    CoordinatesFormatter, Line, MarkerShape, Plot, PlotImage, PlotPoint, PlotUi, Points,
    TabularSeries,
};

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
            self.handle_sculpt_gesture(ui, response, pointer_pos);
            return;
        }
        self.handle_slingshot_gesture(response, pointer_pos);
    }

    fn handle_sculpt_gesture(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        pointer_pos: Option<egui::Pos2>,
    ) {
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
                brush_stroke: None,
            };
            self.on_brush(&ctx);
        } else {
            self.last_brush_wb = None;
        }
    }

    fn handle_slingshot_gesture(
        &mut self,
        response: &egui::Response,
        pointer_pos: Option<egui::Pos2>,
    ) {
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

    fn active_inspector_wb(&self) -> [f64; 2] {
        if let Some(wb) = self.last_hover_wb {
            return wb;
        }
        for (idx, racer) in self.arena.racers.iter().enumerate() {
            if self.arena_config.enabled_racers[idx] && racer.kind.is_available_in_build() {
                return [racer.phase.q.x, racer.phase.q.y];
            }
        }
        self.arena_config.spawn
    }

    fn render_inspector_bar(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let [w, b] = self.active_inspector_wb();
        let eval = self.surface_evaluator();
        let loss = eval.evaluate(w, b);
        let grad = eval.gradient(w, b);
        egui::Frame::NONE
            .fill(palette.surface_fill)
            .inner_margin(egui::Margin::symmetric(8, 4))
            .corner_radius(egui::CornerRadius::same(4))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        RichText::new("INSPECTOR:")
                            .monospace()
                            .strong()
                            .color(palette.info),
                    );
                    ui.label(
                        RichText::new(format!("w = {w:.3}, b = {b:.3}"))
                            .monospace()
                            .color(palette.accent),
                    );
                    ui.label(RichText::new("|").monospace().color(palette.muted_text));
                    ui.label(
                        RichText::new(format!("f(w,b) = {loss:.4}"))
                            .monospace()
                            .strong()
                            .color(palette.success),
                    );
                    ui.label(RichText::new("|").monospace().color(palette.muted_text));
                    ui.label(
                        RichText::new(format!("∇f = [{:.2}, {:.2}]", grad.x, grad.y))
                            .monospace()
                            .color(palette.series_color(4)),
                    );
                });
            });
    }

    pub(super) fn show_3d_arena(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        self.render_inspector_bar(ui, &palette);
        let suppress_orbit = self.interaction_mode != CanvasInteractionMode::Orbit;
        let saved_camera = self.camera;
        self.ensure_3d_scene(saved_camera, &palette);
        let helper = match self.interaction_mode {
            CanvasInteractionMode::Orbit => CAMERA_3D_HELPER_TEXT,
            CanvasInteractionMode::Slingshot => {
                "Slingshot: drag on surface to pull back and launch active particles (orbit locked)."
            }
            CanvasInteractionMode::SculptBrush => {
                "Sculpt: drag to raise barriers (+A) or Alt-drag to carve wells (-A) (orbit locked)."
            }
        };

        let mut cam = self.camera;
        let mut item_count = 0;
        let plot_resp = Plot3DViewport::new("loss_landscape_3d_plot", ARENA_3D_HALF_EXTENT)
            .helper_text(helper)
            .show(ui, &mut cam, |plot_ui| {
                let pid = plot_ui.plot_id();
                plot_ui.ctx().data_mut(|d| {
                    d.insert_temp(pid.with("dragging_callout"), suppress_orbit);
                });
                self.cache.scene_builder.finish(plot_ui);
                item_count = plot_ui.item_count();
            });
        self.cache.last_3d_plot_item_count = item_count;
        if !suppress_orbit {
            self.camera = cam;
        }
        self.handle_viewport_gesture(ui, &plot_resp.response);
    }

    pub(super) fn show_2d_contours(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        self.render_inspector_bar(ui, &palette);
        let heat_low = lerp_color(palette.surface_fill, palette.axis_z, 0.45);
        let heat_high = lerp_color(palette.surface_fill, palette.axis_x, 0.45);
        let raster = self.ensure_2d_raster(ui.ctx(), heat_low, heat_high);
        self.cache.last_2d_has_plot_image = raster.is_some();
        self.cache.last_2d_contour_item_count = self.cache.contours.len();
        let allow_pan = self.interaction_mode == CanvasInteractionMode::Orbit;
        let (w_min, w_max, b_min, b_max) = self.domain_bounds();
        let mut hovered_wb: Option<[f64; 2]> = None;

        let plot_resp = {
            let eval = self.surface_evaluator();
            Plot::new("loss_landscape_plot")
                .x_axis_label("Weight (w)")
                .y_axis_label("Bias (b)")
                .data_aspect(1.0)
                .allow_drag(allow_pan)
                .coordinates_formatter(
                    egui_plot::Corner::LeftBottom,
                    CoordinatesFormatter::new(move |pt, _| {
                        let l = eval.evaluate(pt.x, pt.y);
                        let g = eval.gradient(pt.x, pt.y);
                        format!(
                            "w = {:.3}, b = {:.3}\nf(w,b) = {:.4}\n∇f = [{:.2}, {:.2}]",
                            pt.x, pt.y, l, g.x, g.y
                        )
                    }),
                )
                .show(ui, |plot_ui| {
                    if plot_ui.response().hovered() {
                        if let Some(pt) = plot_ui.pointer_coordinate() {
                            if pt.x.is_finite() && pt.y.is_finite() {
                                hovered_wb = Some([pt.x, pt.y]);
                            }
                        }
                    }
                    self.submit_2d_surface_layers(
                        plot_ui,
                        &palette,
                        raster,
                        [w_min, w_max, b_min, b_max],
                    );
                    self.draw_2d_minima_and_racers(plot_ui, &palette);
                    self.submit_2d_tabular_slices(plot_ui, &palette);
                })
        };
        if let Some(wb) = hovered_wb {
            self.last_hover_wb = Some(wb);
        }
        self.handle_viewport_gesture(ui, &plot_resp.response);
    }

    fn submit_2d_surface_layers(
        &self,
        plot_ui: &mut PlotUi<'_>,
        palette: &SemanticPalette,
        raster: Option<(egui::TextureId, Arc<egui::ColorImage>)>,
        bounds: [f64; 4],
    ) {
        let [w_min, w_max, b_min, b_max] = bounds;
        if let Some((tid, img)) = raster {
            let center = PlotPoint::new(0.5 * (w_min + w_max), 0.5 * (b_min + b_max));
            let size = [(w_max - w_min) as f32, (b_max - b_min) as f32];
            plot_ui.image(
                PlotImage::new("", tid, center, size)
                    .allow_hover(false)
                    .companion_visual_only(true)
                    .with_export_image(img, bounds),
            );
        }
        for contour in &self.cache.contours {
            let color = lerp_color(palette.success, palette.node_stroke, contour.iso);
            plot_ui.add(
                Line::from_segments("Iso-Contour", Arc::clone(&contour.segments))
                    .with_bounds(contour.bounds)
                    .stroke(Stroke::new(1.5_f32, color))
                    .allow_hover(false)
                    .companion_visual_only(true),
            );
        }
    }

    fn draw_2d_minima_and_racers(&self, plot_ui: &mut PlotUi<'_>, palette: &SemanticPalette) {
        let minima: Vec<[f64; 2]> = self.surface_evaluator().global_minima().to_vec();
        if !minima.is_empty() {
            plot_ui.points(
                Points::new("Global Minima (True Solution)", minima)
                    .color(palette.success)
                    .radius(6.0_f32)
                    .shape(MarkerShape::Diamond)
                    .companion_visual_only(true),
            );
        }
        for (idx, racer) in self.arena.racers.iter().enumerate() {
            if !self.arena_config.enabled_racers[idx] || !racer.kind.is_available_in_build() {
                continue;
            }
            let color = racer.kind.color(palette);
            if racer.trail.len() >= 2 {
                let trail: Vec<[f64; 2]> = racer.trail.iter().collect();
                plot_ui.line(
                    Line::new(racer.kind.label(), trail)
                        .color(color)
                        .width(2.2_f32)
                        .companion_visual_only(true),
                );
            }
            plot_ui.points(
                Points::new(racer.kind.label(), vec![[racer.phase.q.x, racer.phase.q.y]])
                    .color(color)
                    .radius(5.0_f32)
                    .companion_visual_only(true),
            );
        }
    }

    fn submit_2d_tabular_slices(&self, plot_ui: &mut PlotUi<'_>, palette: &SemanticPalette) {
        for slice in &self.cache.tabular_slices {
            let color = lerp_color(palette.axis_z, palette.axis_x, slice.t);
            plot_ui.tabular_series(TabularSeries::new(
                slice.name.clone(),
                slice.points.clone(),
                color,
            ));
        }
    }
}
