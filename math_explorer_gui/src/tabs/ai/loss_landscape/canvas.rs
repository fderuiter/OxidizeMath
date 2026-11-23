#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! `Plot3DViewport` 3D wireframe rendering, 2D contour view, racer overlays, and canvas gestures.

use super::surfaces::{lerp_color, SurfaceGrid, ARENA_3D_HALF_EXTENT};
use super::{CanvasInteractionMode, LossLandscapeTool};
use crate::app::SemanticPalette;
use crate::camera::{Camera3D, Plot3DViewport, CAMERA_3D_HELPER_TEXT};
use crate::framework::{InputMode, InteractionContext, InteractiveTool};
use eframe::egui;
use egui_plot::{Line, MarkerShape, Plot, PlotUi, Points};

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

    pub(super) fn show_3d_arena(&mut self, ui: &mut egui::Ui, grid: &SurfaceGrid) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let suppress_orbit = self.interaction_mode != CanvasInteractionMode::Orbit;
        let saved_camera = self.camera;
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
        let plot_resp = Plot3DViewport::new("loss_landscape_3d_plot", ARENA_3D_HALF_EXTENT)
            .helper_text(helper)
            .show(ui, &mut cam, |plot_ui| {
                let pid = plot_ui.plot_id();
                plot_ui.ctx().data_mut(|d| {
                    d.insert_temp(pid.with("dragging_callout"), suppress_orbit);
                });
                let eval = self.surface_evaluator();
                for (t, line_pts) in grid.wireframe_polylines(&saved_camera) {
                    let color = lerp_color(palette.axis_z, palette.axis_x, t);
                    plot_ui.line(
                        Line::new("Surface Mesh", line_pts)
                            .color(color)
                            .width(1.1_f32),
                    );
                }
                for &[mw, mb] in eval.global_minima() {
                    let mloss = eval.evaluate(mw, mb);
                    let p2 = saved_camera.project(&grid.world_point(mw, mb, mloss, 0.03));
                    plot_ui.points(
                        Points::new("Global Minimum", vec![p2])
                            .color(palette.success)
                            .radius(5.5_f32)
                            .shape(MarkerShape::Diamond),
                    );
                }
                self.draw_3d_racers(plot_ui, grid, &saved_camera, &palette);
                if let (Some(a), Some(c)) = (self.slingshot_anchor, self.slingshot_current) {
                    let la = eval.evaluate(a[0], a[1]);
                    let lc = eval.evaluate(c[0], c[1]);
                    let pa = saved_camera.project(&grid.world_point(a[0], a[1], la, 0.05));
                    let pc = saved_camera.project(&grid.world_point(c[0], c[1], lc, 0.05));
                    plot_ui.line(
                        Line::new("Slingshot Vector", vec![pa, pc])
                            .color(palette.warning)
                            .width(2.5_f32),
                    );
                }
            });
        if !suppress_orbit {
            self.camera = cam;
        }
        self.handle_viewport_gesture(ui, &plot_resp.response);
    }

    pub(super) fn show_2d_contours(&mut self, ui: &mut egui::Ui, grid: &SurfaceGrid) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let points = grid.heatmap_points(palette.axis_z, palette.axis_x);
        let contours = grid.contour_segments(8);
        let allow_pan = self.interaction_mode == CanvasInteractionMode::Orbit;

        let plot_resp = Plot::new("loss_landscape_plot")
            .x_axis_label("Weight (w)")
            .y_axis_label("Bias (b)")
            .data_aspect(1.0)
            .allow_drag(allow_pan)
            .show(ui, |plot_ui| {
                for point in points {
                    plot_ui.points(point);
                }
                for (iso, segs) in contours {
                    let c = lerp_color(palette.axis_z, palette.axis_x, iso);
                    for seg in segs {
                        plot_ui.line(
                            Line::new("Iso-Contour", seg.to_vec())
                                .color(c)
                                .width(1.2_f32),
                        );
                    }
                }
                for &[mw, mb] in self.surface_evaluator().global_minima() {
                    plot_ui.points(
                        Points::new("Global Minima (True Solution)", vec![[mw, mb]])
                            .color(palette.success)
                            .radius(6.0_f32)
                            .shape(MarkerShape::Circle),
                    );
                }
                self.draw_2d_racers(plot_ui, &palette);
            });
        self.handle_viewport_gesture(ui, &plot_resp.response);
    }

    fn draw_3d_racers(
        &self,
        plot_ui: &mut PlotUi<'_>,
        grid: &SurfaceGrid,
        cam: &Camera3D,
        palette: &SemanticPalette,
    ) {
        let eval = self.surface_evaluator();
        for (idx, racer) in self.arena.racers.iter().enumerate() {
            if !self.arena_config.enabled_racers[idx] || !racer.kind.is_available_in_build() {
                continue;
            }
            let color = racer.kind.color(palette);
            if racer.trail.len() >= 2 {
                let pts: Vec<[f64; 2]> = racer
                    .trail
                    .iter()
                    .map(|[w, b]| {
                        let l = eval.evaluate(w, b);
                        cam.project(&grid.world_point(w, b, l, 0.035))
                    })
                    .collect();
                plot_ui.line(
                    Line::new(racer.kind.label(), pts)
                        .color(color)
                        .width(2.2_f32),
                );
            }
            let head = cam.project(&grid.world_point(
                racer.phase.q.x,
                racer.phase.q.y,
                racer.potential_energy,
                0.045,
            ));
            plot_ui.points(
                Points::new(racer.kind.label(), vec![head])
                    .color(color)
                    .radius(5.0_f32),
            );
        }
    }

    fn draw_2d_racers(&self, plot_ui: &mut PlotUi<'_>, palette: &SemanticPalette) {
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
                        .width(2.2_f32),
                );
            }
            plot_ui.points(
                Points::new(racer.kind.label(), vec![[racer.phase.q.x, racer.phase.q.y]])
                    .color(color)
                    .radius(5.0_f32),
            );
        }
    }
}
