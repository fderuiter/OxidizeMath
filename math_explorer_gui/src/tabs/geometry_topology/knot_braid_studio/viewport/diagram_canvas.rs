#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Planar link diagram world-to-screen mapping, under-crossing gap cues, and crossing badges.

use super::{DiagramThemeColors, UNDER_CROSSING_GAP_PX};
use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};
use math_explorer::pure_math::knot_theory::{Crossing, CrossingSign, PlanarDiagram};

pub(super) struct DiagramMapper {
    center_world: [f64; 2],
    scale: f32,
    center_screen: Pos2,
}

impl DiagramMapper {
    pub(super) fn from_diagram(diag: &PlanarDiagram, rect: Rect) -> Self {
        let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
        for cr in &diag.crossings {
            min_x = min_x.min(cr.position[0]);
            max_x = max_x.max(cr.position[0]);
            min_y = min_y.min(cr.position[1]);
            max_y = max_y.max(cr.position[1]);
        }
        for seg in &diag.segments {
            for wp in &seg.waypoints {
                min_x = min_x.min(wp[0]);
                max_x = max_x.max(wp[0]);
                min_y = min_y.min(wp[1]);
                max_y = max_y.max(wp[1]);
            }
        }
        let span_x = (max_x - min_x).max(1.2) as f32;
        let span_y = (max_y - min_y).max(1.2) as f32;
        let usable = (rect.size() - Vec2::splat(56.0)).max(Vec2::splat(64.0));
        let scale = (usable.x / span_x).min(usable.y / span_y);
        Self {
            center_world: [0.5 * (min_x + max_x), 0.5 * (min_y + max_y)],
            scale,
            center_screen: rect.center(),
        }
    }

    pub(super) fn to_screen(&self, world: [f64; 2]) -> Pos2 {
        let dx = (world[0] - self.center_world[0]) as f32 * self.scale;
        let dy = (world[1] - self.center_world[1]) as f32 * self.scale;
        Pos2::new(self.center_screen.x + dx, self.center_screen.y - dy)
    }
}

pub(super) fn build_segment_screen_polyline(
    diag: &PlanarDiagram,
    seg_id: usize,
    mapper: &DiagramMapper,
) -> Vec<Pos2> {
    let seg = &diag.segments[seg_id];
    let p_start = mapper.to_screen(diag.crossings[seg.tail.crossing].position);
    let p_end = mapper.to_screen(diag.crossings[seg.head.crossing].position);
    let mut pts = Vec::with_capacity(seg.waypoints.len() + 2);
    pts.push(p_start);
    for &wp in &seg.waypoints {
        pts.push(mapper.to_screen(wp));
    }
    if pts.len() == 1 && p_start.distance(p_end) < 2.0 {
        pts.push(p_start + Vec2::new(20.0, -18.0));
        pts.push(p_start + Vec2::new(20.0, 18.0));
    }
    pts.push(p_end);

    // Pull back under-strand endpoints (`slot == 0` or `slot == 2`) to leave a clear visual break.
    if seg.tail.slot == 2 && pts.len() >= 2 {
        let dir = (pts[1] - pts[0]).normalized();
        if dir.length_sq() > 0.1 {
            pts[0] += dir * UNDER_CROSSING_GAP_PX;
        }
    }
    if seg.head.slot == 0 && pts.len() >= 2 {
        let last = pts.len() - 1;
        let dir = (pts[last - 1] - pts[last]).normalized();
        if dir.length_sq() > 0.1 {
            pts[last] += dir * UNDER_CROSSING_GAP_PX;
        }
    }
    pts
}

pub(super) fn paint_midpoint_arrow(painter: &egui::Painter, pts: &[Pos2], color: Color32) {
    if pts.len() < 2 {
        return;
    }
    let mid_idx = pts.len() / 2;
    let a = pts[mid_idx - 1];
    let b = pts[mid_idx];
    let delta = b - a;
    if delta.length() < 12.0 {
        return;
    }
    let dir = delta.normalized();
    let perp = Vec2::new(-dir.y, dir.x);
    let tip = a.lerp(b, 0.55);
    painter.line_segment(
        [tip, tip - dir * 7.0 + perp * 4.2],
        Stroke::new(2.0_f32, color),
    );
    painter.line_segment(
        [tip, tip - dir * 7.0 - perp * 4.2],
        Stroke::new(2.0_f32, color),
    );
}

pub(super) fn paint_crossing_bridge_and_badge(
    painter: &egui::Painter,
    c_idx: usize,
    cr: &Crossing,
    selected: bool,
    comp_of: &[usize],
    mapper: &DiagramMapper,
    colors: &DiagramThemeColors,
) {
    let center = mapper.to_screen(cr.position);
    let over_color = colors.component_colors[comp_of[cr.over_in()] % colors.component_colors.len()];
    painter.circle_filled(center, UNDER_CROSSING_GAP_PX * 0.65, colors.canvas_bg);
    painter.circle_filled(center, 3.2, over_color);

    if selected {
        painter.circle_stroke(center, 15.0, Stroke::new(2.6_f32, colors.focus_ring));
    }
    let sign_char = match cr.sign {
        CrossingSign::Positive => "+",
        CrossingSign::Negative => "-",
    };
    let sign_color = match cr.sign {
        CrossingSign::Positive => colors.positive_text,
        CrossingSign::Negative => colors.negative_text,
    };
    let badge_pos = center + Vec2::new(13.0, -13.0);
    let badge_rect = Rect::from_center_size(badge_pos, Vec2::new(34.0, 15.0));
    painter.rect_filled(badge_rect, 3.0, colors.badge_bg);
    painter.text(
        badge_pos,
        egui::Align2::CENTER_CENTER,
        format!("c{c_idx}{sign_char}"),
        egui::FontId::monospace(10.5),
        sign_color,
    );
}

pub(super) fn paint_free_loops(
    painter: &egui::Painter,
    rect: Rect,
    loops: usize,
    colors: &DiagramThemeColors,
) {
    let radius = (rect.width().min(rect.height()) * 0.24).clamp(24.0, 72.0);
    for idx in 0..loops {
        let offset_x = (idx as f32 - (loops.saturating_sub(1) as f32) * 0.5) * (radius * 1.15);
        let center = rect.center() + Vec2::new(offset_x, 0.0);
        let color = colors.component_colors[idx % colors.component_colors.len()];
        painter.circle_stroke(center, radius, Stroke::new(3.0_f32, color));
    }
    painter.text(
        rect.center_bottom() - Vec2::new(0.0, 18.0),
        egui::Align2::CENTER_CENTER,
        format!("Unlink ({loops} crossing-free component(s))"),
        egui::FontId::monospace(12.0),
        colors.label_text,
    );
}
