#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::export::export_tile_colors;
use super::{ChartLayout, HyperbolicGeometryTool, LabMode};
use crate::accessibility::{
    lock_canvas_arrow_keys, paint_canvas_focus_ring, register_canvas_accessibility,
};
use crate::app::SemanticPalette;
use eframe::egui;
use math_explorer::pure_math::hyperbolic::{
    DiskPoint, GeodesicSegment, HorocycleDisk, TiledPolygon, sample_disk_geodesic_into,
};
use num_complex::Complex64;

pub(super) const CANVAS_ARC_PTS: usize = 9;

impl HyperbolicGeometryTool {
    pub(super) fn render_canvases(&mut self, ui: &mut egui::Ui) {
        let avail = ui.available_size();
        let show_sep_strip = self.state.mode == LabMode::ChaoticBilliards && avail.y > 260.0;
        let main_h = if show_sep_strip {
            (avail.y - 115.0).max(160.0)
        } else {
            avail.y.max(160.0)
        };

        match self.state.chart_layout {
            ChartLayout::DualSynchronized => {
                let half_w = ((avail.x - 10.0) * 0.5).max(120.0);
                ui.horizontal(|ui| {
                    self.render_disk_viewport(ui, egui::vec2(half_w, main_h));
                    self.render_half_plane_viewport(ui, egui::vec2(half_w, main_h));
                });
            }
            ChartLayout::PoincareDiskOnly => {
                self.render_disk_viewport(ui, egui::vec2(avail.x.max(160.0), main_h));
            }
            ChartLayout::UpperHalfPlaneOnly => {
                self.render_half_plane_viewport(ui, egui::vec2(avail.x.max(160.0), main_h));
            }
        }

        if show_sep_strip {
            ui.add_space(6.0);
            self.render_lyapunov_strip(ui, egui::vec2(avail.x.max(160.0), 100.0));
        }
    }

    fn render_disk_viewport(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        if response.clicked() || response.drag_started() {
            response.request_focus();
        }
        lock_canvas_arrow_keys(ui, &response);
        if response.has_focus() {
            self.focused_canvas = Some(response.id);
            self.handle_disk_keyboard_pan(ui);
        }
        self.handle_disk_pointer_drag(&response, rect);

        let a11y_val = format!(
            "mode={:?}, tiles={}, bounces={}, ln_sep={:.2}",
            self.state.mode,
            self.transformed_tile_count,
            self.billiards.total_bounces(),
            self.billiards.current_log_separation()
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::Canvas,
            "Poincaré Disk Conformal Viewport (PSU(1,1) Drag)",
            Some(&a11y_val),
        );

        let painter = ui.painter_at(rect);
        let palette = SemanticPalette::from_visuals(ui.visuals());
        painter.rect_filled(rect, 6.0_f32, ui.visuals().extreme_bg_color);
        self.paint_disk_contents(&painter, rect, &palette);
        paint_canvas_focus_ring(ui, &response);
    }

    fn handle_disk_keyboard_pan(&mut self, ui: &egui::Ui) {
        let mut dx = 0.0_f64;
        let mut dy = 0.0_f64;
        ui.input(|i| {
            if i.key_pressed(egui::Key::ArrowRight) {
                dx += 0.06;
            }
            if i.key_pressed(egui::Key::ArrowLeft) {
                dx -= 0.06;
            }
            if i.key_pressed(egui::Key::ArrowUp) {
                dy += 0.06;
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                dy -= 0.06;
            }
        });
        if dx != 0.0 || dy != 0.0 {
            self.apply_disk_drag(DiskPoint::origin(), DiskPoint::new_clamped(dx, dy));
        }
    }

    fn handle_disk_pointer_drag(&mut self, response: &egui::Response, rect: egui::Rect) {
        let radius = 0.44 * rect.width().min(rect.height()).max(20.0);
        let center = rect.center();
        if response.drag_started() {
            if let Some(pos) = response.interact_pointer_pos() {
                self.drag_disk_last = Some(screen_to_disk(pos, center, radius));
            }
        }
        if response.dragged() {
            if let Some(pos) = response.interact_pointer_pos() {
                let curr = screen_to_disk(pos, center, radius);
                if let Some(prev) = self.drag_disk_last {
                    self.apply_disk_drag(prev, curr);
                }
                self.drag_disk_last = Some(curr);
            }
        }
        if response.drag_stopped() {
            self.drag_disk_last = None;
        }
    }

    fn paint_disk_contents(
        &mut self,
        painter: &egui::Painter,
        rect: egui::Rect,
        palette: &SemanticPalette,
    ) {
        let center = rect.center();
        let radius = 0.44 * rect.width().min(rect.height()).max(20.0);
        painter.circle_filled(center, radius, palette.surface_fill);
        painter.circle_stroke(center, radius, egui::Stroke::new(2.0_f32, palette.accent));

        match self.state.mode {
            LabMode::TilingExplorer => self.paint_disk_tiling(painter, center, radius, palette),
            LabMode::ChaoticBilliards => {
                self.paint_disk_billiards(painter, center, radius, palette);
            }
        }

        painter.text(
            rect.left_top() + egui::vec2(10.0, 8.0),
            egui::Align2::LEFT_TOP,
            "Poincaré Disk D  (PSU(1,1))",
            egui::FontId::proportional(13.0),
            palette.node_stroke,
        );
    }

    fn paint_disk_tiling(
        &self,
        painter: &egui::Painter,
        center: egui::Pos2,
        radius: f32,
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
            fill_disk_tile_screen_pts(tile, center, radius, &mut arc, &mut pts);
            let (fill, stroke) = export_tile_colors(tile.depth, idx, palette);
            painter.add(egui::Shape::convex_polygon(
                pts.clone(),
                fill,
                egui::Stroke::NONE,
            ));
            pts.push(pts[0]);
            painter.add(egui::Shape::line(
                pts.clone(),
                egui::Stroke::new(1.0_f32, stroke),
            ));
        }

        if self.state.show_horocycles {
            let xi = self.state.disk_isometry.apply_complex(Complex64::new(0.0, 1.0));
            if let Ok(horo) = HorocycleDisk::from_radius(xi, self.state.horocycle_radius) {
                let hc = disk_to_screen(horo.euclidean_center, center, radius);
                let hr = (horo.euclidean_radius as f32) * radius;
                painter.circle_stroke(hc, hr, egui::Stroke::new(1.8_f32, palette.warning));
            }
        }
    }

    fn paint_disk_billiards(
        &mut self,
        painter: &egui::Painter,
        center: egui::Pos2,
        radius: f32,
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
                table_pts.push(disk_to_screen(Complex64::new(x, y), center, radius));
            }
        }
        table_pts.push(table_pts[0]);
        painter.add(egui::Shape::line(
            table_pts,
            egui::Stroke::new(2.2_f32, palette.info),
        ));

        let primary_color = palette.series_color(2);
        let twin_color = palette.series_color(1);
        let n_p = self
            .billiards
            .copy_primary_segments_into(&mut self.primary_seg_scratch);
        paint_ray_segments_disk(
            painter,
            &self.primary_seg_scratch[n_p.saturating_sub(180)..n_p],
            iso,
            center,
            radius,
            primary_color,
        );
        let n_t = self
            .billiards
            .copy_twin_segments_into(&mut self.twin_seg_scratch);
        paint_ray_segments_disk(
            painter,
            &self.twin_seg_scratch[n_t.saturating_sub(180)..n_t],
            iso,
            center,
            radius,
            twin_color,
        );

        let p_head = disk_to_screen(
            iso.apply(self.billiards.primary_ray().position).as_complex(),
            center,
            radius,
        );
        let t_head = disk_to_screen(
            iso.apply(self.billiards.twin_ray().position).as_complex(),
            center,
            radius,
        );
        painter.circle_filled(p_head, 4.2_f32, primary_color);
        painter.circle_filled(t_head, 4.2_f32, twin_color);
    }
}

fn disk_to_screen(z: Complex64, center: egui::Pos2, radius: f32) -> egui::Pos2 {
    egui::pos2(
        center.x + (z.re as f32) * radius,
        center.y - (z.im as f32) * radius,
    )
}

fn screen_to_disk(pos: egui::Pos2, center: egui::Pos2, radius: f32) -> DiskPoint {
    let re = f64::from((pos.x - center.x) / radius);
    let im = f64::from((center.y - pos.y) / radius);
    DiskPoint::new_clamped(re, im)
}

fn fill_disk_tile_screen_pts(
    tile: &TiledPolygon,
    center: egui::Pos2,
    radius: f32,
    arc: &mut [[f64; 2]; CANVAS_ARC_PTS],
    out: &mut Vec<egui::Pos2>,
) {
    for i in 0..tile.vertex_count {
        let z0 = tile.vertices[i];
        let z1 = tile.vertices[(i + 1) % tile.vertex_count];
        sample_disk_geodesic_into(z0, z1, arc);
        for &[x, y] in &arc[..CANVAS_ARC_PTS - 1] {
            out.push(disk_to_screen(Complex64::new(x, y), center, radius));
        }
    }
}

fn paint_ray_segments_disk(
    painter: &egui::Painter,
    segs: &[GeodesicSegment],
    iso: math_explorer::pure_math::hyperbolic::Psu11,
    center: egui::Pos2,
    radius: f32,
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
            line.push(disk_to_screen(Complex64::new(x, y), center, radius));
        }
        painter.add(egui::Shape::line(
            line.clone(),
            egui::Stroke::new(1.5_f32, color),
        ));
    }
}
