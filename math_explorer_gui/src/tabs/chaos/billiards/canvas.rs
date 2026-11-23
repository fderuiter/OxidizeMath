//! Accessible click-to-launch and keyboard-navigable Euclidean billiard spatial canvas.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BilliardGeometryKind, DynamicalBilliardsTool, MAX_RAYS};
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::SemanticPalette;
use eframe::egui::{self, FontId, Pos2, Rect, Stroke, StrokeKind, Vec2};

struct SpatialViewport {
    rect: Rect,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
}

impl SpatialViewport {
    fn new(rect: Rect, bounds: [f64; 4]) -> Self {
        let side = rect.width().min(rect.height());
        let pad = (side * 0.06).clamp(14.0, 28.0);
        let square_side = (side - 2.0 * pad).max(24.0);
        let square = Rect::from_center_size(rect.center(), Vec2::splat(square_side));
        Self {
            rect: square,
            x_min: bounds[0],
            x_max: bounds[1],
            y_min: bounds[2],
            y_max: bounds[3],
        }
    }

    fn to_screen(&self, xy: [f64; 2]) -> Pos2 {
        let tx = ((xy[0] - self.x_min) / (self.x_max - self.x_min)) as f32;
        let ty = ((xy[1] - self.y_min) / (self.y_max - self.y_min)) as f32;
        Pos2::new(
            self.rect.left() + tx * self.rect.width(),
            self.rect.bottom() - ty * self.rect.height(),
        )
    }

    fn to_world(&self, pos: Pos2) -> [f64; 2] {
        let tx = (pos.x - self.rect.left()) / self.rect.width().max(1.0);
        let ty = (self.rect.bottom() - pos.y) / self.rect.height().max(1.0);
        [
            self.x_min + f64::from(tx) * (self.x_max - self.x_min),
            self.y_min + f64::from(ty) * (self.y_max - self.y_min),
        ]
    }

    fn world_radius_to_pixels(&self, r: f64) -> f32 {
        ((r / (self.x_max - self.x_min)) as f32) * self.rect.width()
    }
}

impl DynamicalBilliardsTool {
    pub(super) fn render_billiard_canvas(&mut self, ui: &mut egui::Ui, desired_size: Vec2) {
        let (response, painter) = ui.allocate_painter(desired_size, egui::Sense::click_and_drag());
        let _ = response.clone().accessible_hover_text(
            "Euclidean Billiard Table: click inside table or use Arrow keys + Enter to launch rays",
        );

        let vp = SpatialViewport::new(response.rect, self.world_bounds());
        self.handle_canvas_click(ui.ctx(), &response, &vp);

        if response.has_focus() {
            lock_canvas_arrow_keys(ui, &response);
            self.handle_canvas_keyboard(ui);
        }
        self.focused_canvas = response.has_focus().then_some(response.id);

        let total_bounces: usize = self.rays.iter().map(|r| r.bounce_count).sum();
        let summary = format!(
            "{} — {}/{} rays, {} bounces, launch ({:.3}, {:.3})",
            self.geometry.label(),
            self.rays.len(),
            MAX_RAYS,
            total_bounces,
            self.launch_pos[0],
            self.launch_pos[1]
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "Euclidean Billiard Table",
            Some(&summary),
        );

        let palette = SemanticPalette::from_visuals(ui.visuals());
        self.paint_table_geometry(&painter, response.rect, &vp, &palette);
        self.paint_ray_trails(&painter, &vp, &palette);
        self.paint_reticle_and_hud(
            &painter,
            response.rect,
            &vp,
            &palette,
            ui.visuals(),
            response.has_focus(),
        );
        paint_canvas_focus_ring(ui, &response);
    }

    fn handle_canvas_click(
        &mut self,
        ctx: &egui::Context,
        response: &egui::Response,
        vp: &SpatialViewport,
    ) {
        if !response.clicked() {
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        let xy = vp.to_world(pos);
        response.request_focus();
        self.launch_and_announce(ctx, xy);
    }

    fn handle_canvas_keyboard(&mut self, ui: &egui::Ui) {
        let ctx = ui.ctx();
        let [x_min, x_max, y_min, y_max] = self.world_bounds();
        let dx = (x_max - x_min) * 0.04;
        let dy = (y_max - y_min) * 0.04;

        let delta = ctx.input_mut(|i| {
            for (mods, scale) in [(egui::Modifiers::SHIFT, 0.25), (egui::Modifiers::NONE, 1.0)] {
                if i.consume_key(mods, egui::Key::ArrowLeft) {
                    return Some([-dx * scale, 0.0]);
                }
                if i.consume_key(mods, egui::Key::ArrowRight) {
                    return Some([dx * scale, 0.0]);
                }
                if i.consume_key(mods, egui::Key::ArrowUp) {
                    return Some([0.0, dy * scale]);
                }
                if i.consume_key(mods, egui::Key::ArrowDown) {
                    return Some([0.0, -dy * scale]);
                }
            }
            None
        });

        if let Some([ddx, ddy]) = delta {
            ui.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
            let candidate = [
                (self.launch_pos[0] + ddx).clamp(x_min, x_max),
                (self.launch_pos[1] + ddy).clamp(y_min, y_max),
            ];
            if self.contains_interior_point(candidate) {
                self.launch_pos = candidate;
                self.state_rev = self.state_rev.wrapping_add(1);
                announce_live_status(
                    ctx,
                    format!(
                        "Billiard launch cursor moved to ({:.3}, {:.3})",
                        candidate[0], candidate[1]
                    ),
                );
            }
        }

        let fire_pressed =
            ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter));
        if fire_pressed {
            let pos = self.launch_pos;
            self.launch_and_announce(ctx, pos);
        }
    }

    pub(super) fn launch_and_announce(&mut self, ctx: &egui::Context, xy: [f64; 2]) {
        match self.seed_single_ray_at(xy) {
            Ok(idx) => {
                let msg = format!("Launched ray {} from ({:.3}, {:.3})", idx + 1, xy[0], xy[1]);
                self.last_status_note = Some(msg.clone());
                announce_live_status(ctx, msg);
            }
            Err(reason) => {
                let msg = format!("Cannot launch at ({:.3}, {:.3}): {reason}", xy[0], xy[1]);
                self.last_status_note = Some(msg.clone());
                announce_live_status(ctx, msg);
            }
        }
    }

    fn paint_table_geometry(
        &self,
        painter: &egui::Painter,
        outer: Rect,
        vp: &SpatialViewport,
        palette: &SemanticPalette,
    ) {
        painter.rect_filled(outer, 6.0, palette.surface_fill);
        painter.rect_stroke(
            outer,
            6.0,
            Stroke::new(1.0_f32, palette.graph_edge),
            StrokeKind::Inside,
        );

        let boundary_world = self.outer_boundary_polyline();
        if boundary_world.len() >= 3 {
            let mut pts: Vec<Pos2> = boundary_world.iter().map(|&xy| vp.to_screen(xy)).collect();
            if let Some(&first) = pts.first() {
                pts.push(first);
            }
            painter.add(egui::Shape::line(pts, Stroke::new(2.2_f32, palette.info)));
        }

        if self.geometry == BilliardGeometryKind::Sinai {
            let center = vp.to_screen([0.0, 0.0]);
            let rad_px = vp.world_radius_to_pixels(self.sinai.scatterer_radius);
            painter.circle_filled(center, rad_px, palette.warning.gamma_multiply(0.18));
            painter.circle_stroke(center, rad_px, Stroke::new(2.0_f32, palette.warning));
        } else if self.geometry == BilliardGeometryKind::Ellipse {
            let c = self.ellipse.focal_distance();
            if c > 1e-4 {
                for fx in [-c, c] {
                    let fpos = vp.to_screen([fx, 0.0]);
                    painter.circle_filled(fpos, 3.8, palette.accent);
                }
            }
        }
    }

    fn paint_ray_trails(
        &self,
        painter: &egui::Painter,
        vp: &SpatialViewport,
        palette: &SemanticPalette,
    ) {
        let clip = painter.with_clip_rect(vp.rect.expand(4.0));
        for slot in &self.rays {
            let color = palette.series_color(slot.color_index);
            if slot.vertices.len() >= 2 {
                let pts: Vec<Pos2> = slot.vertices.iter().map(|&xy| vp.to_screen(xy)).collect();
                clip.add(egui::Shape::line(
                    pts,
                    Stroke::new(1.4_f32, color.gamma_multiply(0.78)),
                ));
            }
            let head = vp.to_screen(slot.ray.pos);
            clip.circle_filled(head, 3.2, color);
        }
    }

    fn paint_reticle_and_hud(
        &self,
        painter: &egui::Painter,
        outer: Rect,
        vp: &SpatialViewport,
        palette: &SemanticPalette,
        visuals: &egui::Visuals,
        focused: bool,
    ) {
        let lpos = vp.to_screen(self.launch_pos);
        let reticle_color = if focused {
            palette.accent
        } else {
            visuals.text_color().gamma_multiply(0.8)
        };
        let arm = if focused { 8.5 } else { 6.0 };
        painter.line_segment(
            [lpos - Vec2::new(arm, 0.0), lpos + Vec2::new(arm, 0.0)],
            Stroke::new(1.6_f32, reticle_color),
        );
        painter.line_segment(
            [lpos - Vec2::new(0.0, arm), lpos + Vec2::new(0.0, arm)],
            Stroke::new(1.6_f32, reticle_color),
        );
        let dir_vec = Vec2::new(
            self.launch_angle.cos() as f32,
            -(self.launch_angle.sin() as f32),
        ) * 16.0;
        painter.line_segment([lpos, lpos + dir_vec], Stroke::new(1.8_f32, palette.accent));

        painter.text(
            Pos2::new(outer.left() + 10.0, outer.top() + 8.0),
            egui::Align2::LEFT_TOP,
            format!(
                "Spatial Table: {}  |  Rays: {}/{}",
                self.geometry.label(),
                self.rays.len(),
                MAX_RAYS
            ),
            FontId::proportional(12.0),
            visuals.text_color(),
        );
        painter.text(
            Pos2::new(outer.center().x, outer.bottom() - 4.0),
            egui::Align2::CENTER_BOTTOM,
            "Click interior or Arrows+Enter to launch ray",
            FontId::proportional(11.5),
            visuals.weak_text_color(),
        );
        if let Some(note) = &self.last_status_note {
            if note.starts_with("Cannot") {
                painter.text(
                    Pos2::new(outer.right() - 10.0, outer.top() + 8.0),
                    egui::Align2::RIGHT_TOP,
                    note,
                    FontId::proportional(11.0),
                    palette.warning,
                );
            }
        }
    }
}
