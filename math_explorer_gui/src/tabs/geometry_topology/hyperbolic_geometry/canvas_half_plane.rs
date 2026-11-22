#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::canvas::CANVAS_ARC_PTS;
use super::export::export_tile_colors;
use super::{HyperbolicGeometryTool, LabMode};
use crate::accessibility::{
    lock_canvas_arrow_keys, paint_canvas_focus_ring, register_canvas_accessibility,
};
use crate::app::SemanticPalette;
use eframe::egui;
use math_explorer::pure_math::hyperbolic::{
    GeodesicSegment, HalfPlanePoint, HorocycleDisk, HorocycleHalfPlane,
    cayley_complex_to_half_plane_clamped, sample_disk_geodesic_into,
};
use num_complex::Complex64;
use std::f64::consts::TAU;

impl HyperbolicGeometryTool {
    pub(super) fn render_half_plane_viewport(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        if response.clicked() || response.drag_started() {
            response.request_focus();
        }
        lock_canvas_arrow_keys(ui, &response);
        if response.has_focus() {
            self.focused_canvas = Some(response.id);
            self.handle_half_plane_keyboard_pan(ui);
        }
        self.handle_half_plane_pointer_drag(&response, rect);

        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::Canvas,
            "Upper Half-Plane Cayley Viewport (PSL(2,R) Drag)",
            Some("Cayley isometric image w = i(1+z)/(1-z)"),
        );

        let painter = ui.painter_at(rect);
        let palette = SemanticPalette::from_visuals(ui.visuals());
        painter.rect_filled(rect, 6.0_f32, ui.visuals().extreme_bg_color);
        self.paint_half_plane_contents(&painter, rect, &palette);
        paint_canvas_focus_ring(ui, &response);
    }

    fn handle_half_plane_keyboard_pan(&mut self, ui: &egui::Ui) {
        let mut du = 0.0_f64;
        let mut scale = 1.0_f64;
        ui.input(|i| {
            if i.key_pressed(egui::Key::ArrowRight) {
                du += 0.15;
            }
            if i.key_pressed(egui::Key::ArrowLeft) {
                du -= 0.15;
            }
            if i.key_pressed(egui::Key::ArrowUp) {
                scale = 1.12;
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                scale = 0.89;
            }
        });
        if du != 0.0 || (scale - 1.0).abs() > 1.0e-6 {
            let from = HalfPlanePoint::base_i();
            let to = HalfPlanePoint::new_clamped(du, scale);
            self.apply_half_plane_drag(from, to);
        }
    }

    fn handle_half_plane_pointer_drag(&mut self, response: &egui::Response, rect: egui::Rect) {
        let span = self.state.half_plane_span;
        if response.drag_started() {
            if let Some(pos) = response.interact_pointer_pos() {
                self.drag_hp_last = Some(screen_to_half_plane(pos, rect, span));
            }
        }
        if response.dragged() {
            if let Some(pos) = response.interact_pointer_pos() {
                let curr = screen_to_half_plane(pos, rect, span);
                if let Some(prev) = self.drag_hp_last {
                    self.apply_half_plane_drag(prev, curr);
                }
                self.drag_hp_last = Some(curr);
            }
        }
        if response.drag_stopped() {
            self.drag_hp_last = None;
        }
    }

    fn paint_half_plane_contents(
        &mut self,
        painter: &egui::Painter,
        rect: egui::Rect,
        palette: &SemanticPalette,
    ) {
        let span = self.state.half_plane_span;
        let axis_y = rect.bottom() - 18.0;
        painter.line_segment(
            [
                egui::pos2(rect.left(), axis_y),
                egui::pos2(rect.right(), axis_y),
            ],
            egui::Stroke::new(2.0_f32, palette.accent),
        );

        match self.state.mode {
            LabMode::TilingExplorer => self.paint_half_plane_tiling(painter, rect, span, palette),
            LabMode::ChaoticBilliards => {
                self.paint_half_plane_billiards(painter, rect, span, palette);
            }
        }

        painter.text(
            rect.left_top() + egui::vec2(10.0, 8.0),
            egui::Align2::LEFT_TOP,
            "Upper Half-Plane H  (PSL(2,R) Cayley Chart)",
            egui::FontId::proportional(13.0),
            palette.node_stroke,
        );
    }

    fn paint_half_plane_tiling(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        span: f64,
        palette: &SemanticPalette,
    ) {
        let count = self.transformed_tile_count.min(self.transformed_tiles.len());
        let mut arc = [[0.0_f64; 2]; CANVAS_ARC_PTS];
        let mut pts = Vec::with_capacity(16 * CANVAS_ARC_PTS);
        for (idx, tile) in self.transformed_tiles[..count].iter().enumerate() {
            if tile.vertex_count < 3 {
                continue;
            }
            pts.clear();
            for i in 0..tile.vertex_count {
                let z0 = tile.vertices[i];
                let z1 = tile.vertices[(i + 1) % tile.vertex_count];
                sample_disk_geodesic_into(z0, z1, &mut arc);
                for &[x, y] in &arc[..CANVAS_ARC_PTS - 1] {
                    let w = cayley_complex_to_half_plane_clamped(Complex64::new(x, y));
                    pts.push(half_plane_to_screen(w, rect, span));
                }
            }
            pts.push(pts[0]);
            let (_, stroke) = export_tile_colors(tile.depth, idx, palette);
            painter.add(egui::Shape::line(
                pts.clone(),
                egui::Stroke::new(1.15_f32, stroke),
            ));
        }

        if self.state.show_horocycles {
            let xi = self.state.disk_isometry.apply_complex(Complex64::new(0.0, 1.0));
            if let Ok(horo) = HorocycleDisk::from_radius(xi, self.state.horocycle_radius) {
                paint_half_plane_horocycle(
                    painter,
                    horo.to_half_plane(),
                    rect,
                    span,
                    palette.warning,
                );
            }
        }
    }

    fn paint_half_plane_billiards(
        &mut self,
        painter: &egui::Painter,
        rect: egui::Rect,
        span: f64,
        palette: &SemanticPalette,
    ) {
        let iso = self.state.disk_isometry;
        let verts = self.billiards.table().vertices();
        let mut arc = [[0.0_f64; 2]; CANVAS_ARC_PTS];
        let mut table_pts = Vec::with_capacity(verts.len() * CANVAS_ARC_PTS);
        for i in 0..verts.len() {
            let z0 = iso.apply_complex(verts[i]);
            let z1 = iso.apply_complex(verts[(i + 1) % verts.len()]);
            sample_disk_geodesic_into(z0, z1, &mut arc);
            for &[x, y] in &arc[..CANVAS_ARC_PTS - 1] {
                let w = cayley_complex_to_half_plane_clamped(Complex64::new(x, y));
                table_pts.push(half_plane_to_screen(w, rect, span));
            }
        }
        table_pts.push(table_pts[0]);
        painter.add(egui::Shape::line(
            table_pts,
            egui::Stroke::new(2.0_f32, palette.info),
        ));

        let n_p = self
            .billiards
            .copy_primary_segments_into(&mut self.primary_seg_scratch);
        paint_ray_segments_hp(
            painter,
            &self.primary_seg_scratch[n_p.saturating_sub(140)..n_p],
            iso,
            rect,
            span,
            palette.series_color(2),
        );
        let n_t = self
            .billiards
            .copy_twin_segments_into(&mut self.twin_seg_scratch);
        paint_ray_segments_hp(
            painter,
            &self.twin_seg_scratch[n_t.saturating_sub(140)..n_t],
            iso,
            rect,
            span,
            palette.series_color(1),
        );
    }

    pub(super) fn render_lyapunov_strip(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::hover());
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::Image,
            "Twin-Ray Logarithmic Separation ln(delta_t / delta_0)",
            None,
        );
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 6.0_f32, palette.surface_fill);
        let n = self
            .billiards
            .copy_separation_series_into(&mut self.sep_series_scratch);
        if n >= 2 {
            let slice = &self.sep_series_scratch[..n];
            let s_min = slice[0][0];
            let s_max = slice[n - 1][0].max(s_min + 1.0e-3);
            let mut y_min = 0.0_f64;
            let mut y_max = 1.0_f64;
            for &[_, val] in slice {
                y_min = y_min.min(val);
                y_max = y_max.max(val);
            }
            let pad_rect = rect.shrink2(egui::vec2(12.0, 18.0));
            let line_pts: Vec<egui::Pos2> = slice
                .iter()
                .step_by((n / 256).max(1))
                .map(|&[s, val]| {
                    let tx = ((s - s_min) / (s_max - s_min)) as f32;
                    let ty = ((val - y_min) / (y_max - y_min).max(1.0e-6)) as f32;
                    egui::pos2(
                        pad_rect.left() + tx * pad_rect.width(),
                        pad_rect.bottom() - ty * pad_rect.height(),
                    )
                })
                .collect();
            painter.add(egui::Shape::line(
                line_pts,
                egui::Stroke::new(1.8_f32, palette.warning),
            ));
        }
        painter.text(
            rect.left_top() + egui::vec2(10.0, 4.0),
            egui::Align2::LEFT_TOP,
            format!(
                "Lyapunov Divergence  ln(δ_t / δ_0) = {:.3}   (bounces: {})",
                self.billiards.current_log_separation(),
                self.billiards.total_bounces()
            ),
            egui::FontId::proportional(12.0),
            palette.node_stroke,
        );
    }
}

fn half_plane_to_screen(w: Complex64, rect: egui::Rect, span: f64) -> egui::Pos2 {
    let axis_y = rect.bottom() - 18.0;
    let scale = (rect.width() * 0.5) / (span as f32);
    let x = (rect.center().x + (w.re as f32) * scale).clamp(rect.left() - 50.0, rect.right() + 50.0);
    let y = (axis_y - (w.im as f32) * scale).clamp(rect.top() - 50.0, axis_y);
    egui::pos2(x, y)
}

fn screen_to_half_plane(pos: egui::Pos2, rect: egui::Rect, span: f64) -> HalfPlanePoint {
    let axis_y = rect.bottom() - 18.0;
    let scale = ((rect.width() * 0.5) / (span as f32)).max(1.0);
    let u = f64::from((pos.x - rect.center().x) / scale);
    let v = f64::from((axis_y - pos.y).max(2.0) / scale);
    HalfPlanePoint::new_clamped(u, v)
}

fn paint_ray_segments_hp(
    painter: &egui::Painter,
    segs: &[GeodesicSegment],
    iso: math_explorer::pure_math::hyperbolic::Psu11,
    rect: egui::Rect,
    span: f64,
    color: egui::Color32,
) {
    let mut arc = [[0.0_f64; 2]; CANVAS_ARC_PTS];
    let mut line = Vec::with_capacity(CANVAS_ARC_PTS);
    for seg in segs {
        line.clear();
        sample_disk_geodesic_into(
            iso.apply(seg.start).as_complex(),
            iso.apply(seg.end).as_complex(),
            &mut arc,
        );
        for &[x, y] in &arc {
            let w = cayley_complex_to_half_plane_clamped(Complex64::new(x, y));
            line.push(half_plane_to_screen(w, rect, span));
        }
        painter.add(egui::Shape::line(
            line.clone(),
            egui::Stroke::new(1.35_f32, color),
        ));
    }
}

fn paint_half_plane_horocycle(
    painter: &egui::Painter,
    horo_hp: HorocycleHalfPlane,
    rect: egui::Rect,
    span: f64,
    color: egui::Color32,
) {
    match horo_hp {
        HorocycleHalfPlane::HorizontalLine { height } => {
            let p_left = half_plane_to_screen(Complex64::new(-span, height), rect, span);
            let p_right = half_plane_to_screen(Complex64::new(span, height), rect, span);
            painter.line_segment([p_left, p_right], egui::Stroke::new(1.8_f32, color));
        }
        HorocycleHalfPlane::TangentCircle { base_x, radius } => {
            let pts: Vec<egui::Pos2> = (0..=48)
                .map(|i| {
                    let ang = TAU * (i as f64) / 48.0;
                    let w =
                        Complex64::new(base_x + radius * ang.cos(), radius + radius * ang.sin());
                    half_plane_to_screen(w, rect, span)
                })
                .collect();
            painter.add(egui::Shape::line(pts, egui::Stroke::new(1.8_f32, color)));
        }
    }
}
