//! Custom `egui::Rect` canvas adapter (`SpatialBrushKit::sample_rect`) for discrete grid tools.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BrushMode, BrushStroke, SpatialBrushKit};
use eframe::egui::{self, Key, Pos2, Rect, Shape};
use egui_plot::{
    commands::{CommandRegistryData, CommandTrigger},
    focus_ring_stroke,
};

impl SpatialBrushKit {
    pub(super) fn register_shortcuts(ctx: &egui::Context) {
        CommandRegistryData::register(
            ctx,
            "Arrow Keys",
            "Move spatial brush cursor on focused canvas",
            CommandTrigger::Key(Key::ArrowRight),
            false,
            "Spatial Brush",
        );
        CommandRegistryData::register(
            ctx,
            "[ / ]",
            "Decrease or increase spatial brush radius",
            CommandTrigger::Key(Key::OpenBracket),
            false,
            "Spatial Brush",
        );
        CommandRegistryData::register(
            ctx,
            "Space / Enter",
            "Stamp spatial brush at keyboard cursor",
            CommandTrigger::Key(Key::Space),
            false,
            "Spatial Brush",
        );
    }

    /// Samples pointer drags and WCAG 2.2 AA keyboard sculpting inside a custom `egui::Rect`
    /// canvas allocated with `egui::Sense::click_and_drag()`.
    ///
    /// Coordinates are normalized to `[domain_min, domain_max]` with `y` increasing upward so
    /// [`Self::plot_to_grid_y_down`] maps visual top-to-bottom rows to `0..height`.
    pub fn sample_rect(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        domain_min: [f64; 2],
        domain_max: [f64; 2],
        secondary_mode: Option<BrushMode>,
    ) -> Vec<BrushStroke> {
        self.ensure_cursor_in_bounds(domain_min, domain_max);
        let ctx = ui.ctx().clone();
        Self::register_shortcuts(&ctx);

        if response.has_focus() {
            self.last_focused_id = Some(response.id);
        } else if self.last_focused_id == Some(response.id) {
            self.last_focused_id = None;
        }

        let mut strokes = Vec::new();
        if self.is_sculpting() {
            if response.has_focus() {
                self.sample_keyboard(&ctx, response.id, domain_min, domain_max, &mut strokes);
            }
            self.sample_rect_pointer(
                &ctx,
                response,
                domain_min,
                domain_max,
                secondary_mode,
                &mut strokes,
            );
        } else {
            self.last_drag_pos = None;
        }

        self.draw_rect_focus_and_reticle(&ctx, response, domain_min, domain_max);
        strokes
    }

    fn rect_pos_to_domain(
        rect: Rect,
        pos: Pos2,
        domain_min: [f64; 2],
        domain_max: [f64; 2],
    ) -> [f64; 2] {
        let lo_x = domain_min[0].min(domain_max[0]);
        let hi_x = domain_min[0].max(domain_max[0]);
        let lo_y = domain_min[1].min(domain_max[1]);
        let hi_y = domain_min[1].max(domain_max[1]);
        let tx = f64::from(((pos.x - rect.min.x) / rect.width().max(1.0)).clamp(0.0, 1.0));
        let ty = f64::from(((rect.max.y - pos.y) / rect.height().max(1.0)).clamp(0.0, 1.0));
        [lo_x + tx * (hi_x - lo_x), lo_y + ty * (hi_y - lo_y)]
    }

    fn domain_to_rect_pos(
        rect: Rect,
        pt: [f64; 2],
        domain_min: [f64; 2],
        domain_max: [f64; 2],
    ) -> Pos2 {
        let lo_x = domain_min[0].min(domain_max[0]);
        let hi_x = domain_min[0].max(domain_max[0]);
        let lo_y = domain_min[1].min(domain_max[1]);
        let hi_y = domain_min[1].max(domain_max[1]);
        let ux = ((pt[0] - lo_x) / (hi_x - lo_x).max(1e-6)).clamp(0.0, 1.0) as f32;
        let uy = ((pt[1] - lo_y) / (hi_y - lo_y).max(1e-6)).clamp(0.0, 1.0) as f32;
        Pos2::new(
            rect.min.x + ux * rect.width(),
            rect.max.y - uy * rect.height(),
        )
    }

    fn sample_rect_pointer(
        &mut self,
        ctx: &egui::Context,
        response: &egui::Response,
        domain_min: [f64; 2],
        domain_max: [f64; 2],
        secondary_mode: Option<BrushMode>,
        strokes: &mut Vec<BrushStroke>,
    ) {
        let (primary, secondary, pointer_pos) = ctx.input(|i| {
            (
                i.pointer.primary_down(),
                i.pointer.secondary_down(),
                i.pointer.interact_pos().or_else(|| i.pointer.hover_pos()),
            )
        });
        let active_mode = if primary {
            Some(self.mode)
        } else if secondary {
            secondary_mode
        } else {
            None
        };

        let Some(mode) = active_mode else {
            self.last_drag_pos = None;
            return;
        };

        let pointer_active = response.hovered() || response.dragged() || response.clicked();
        if !pointer_active {
            self.last_drag_pos = None;
            return;
        }

        if let Some(pos) = response.interact_pointer_pos().or(pointer_pos) {
            if !response.rect.contains(pos) && !response.dragged() {
                self.last_drag_pos = None;
                return;
            }
            let curr = Self::rect_pos_to_domain(response.rect, pos, domain_min, domain_max);
            let prev = self.last_drag_pos.unwrap_or(curr);
            let drag_vector = [curr[0] - prev[0], curr[1] - prev[1]];
            self.cursor_pos = curr;
            self.cursor_initialized = true;
            self.last_drag_pos = Some(curr);
            strokes.push(BrushStroke {
                start: prev,
                end: curr,
                radius: self.radius,
                mode,
                drag_vector,
            });
        }
    }

    fn draw_rect_focus_and_reticle(
        &self,
        ctx: &egui::Context,
        response: &egui::Response,
        domain_min: [f64; 2],
        domain_max: [f64; 2],
    ) {
        let has_focus = response.has_focus();
        let show_reticle = self.is_sculpting() && (has_focus || response.hovered());
        if !has_focus && !show_reticle {
            return;
        }

        let visuals = ctx.style().visuals.clone();
        let stroke = focus_ring_stroke(&visuals, visuals.extreme_bg_color);
        let frame = response.rect;
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            response.id.with("spatial_brush_rect_overlay"),
        ));

        if has_focus {
            painter.rect_stroke(frame.shrink(1.0), 2.0, stroke, egui::StrokeKind::Inside);
        }

        if show_reticle {
            let center_px =
                Self::domain_to_rect_pos(frame, self.cursor_pos, domain_min, domain_max);
            let span_x = (domain_max[0] - domain_min[0]).abs().max(1.0) as f32;
            let px_radius = ((self.radius as f32) * (frame.width() / span_x)).max(4.0);
            let clipped = painter.with_clip_rect(frame);
            clipped.add(Shape::circle_stroke(center_px, px_radius, stroke));
            let arm = 4.0_f32;
            clipped.line_segment(
                [
                    Pos2::new(center_px.x - arm, center_px.y),
                    Pos2::new(center_px.x + arm, center_px.y),
                ],
                stroke,
            );
            clipped.line_segment(
                [
                    Pos2::new(center_px.x, center_px.y - arm),
                    Pos2::new(center_px.x, center_px.y + arm),
                ],
                stroke,
            );
        }
    }
}
