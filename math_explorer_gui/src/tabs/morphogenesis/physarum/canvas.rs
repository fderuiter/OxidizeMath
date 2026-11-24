//! Interactive chemotactic trail canvas, skeleton & MST vector overlays, Bresenham-interpolated
//! mouse/keyboard city/wall/sever brushes, AccessKit registration, and WCAG 2.2 AA colormaps.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::MAX_SAVED_OBSTACLE_STROKES;
use super::{PhysarumBrushMode, PhysarumTool};
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility,
};
use crate::framework::{bresenham_line, compute_aspect_locked_rect, map_pointer_to_cell};
use eframe::egui;
use eframe::egui::ColorImage;
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;

impl PhysarumTool {
    pub(crate) fn refresh_pixels_in_place(&mut self) {
        let w = self.engine.width();
        let h = self.engine.height();
        let total = w * h;
        if self.pixels.len() != total {
            self.pixels.resize(total, egui::Color32::BLACK);
        }
        let obs = self.engine.obstacles();
        let tr = self.engine.trail();
        let [or, og, ob] = self.obstacle_rgb;
        let wall_col = egui::Color32::from_rgb(or, og, ob);

        for idx in 0..total {
            if obs[idx] != 0 {
                self.pixels[idx] = wall_col;
                continue;
            }
            let raw = tr[idx];
            let norm = if raw.is_finite() && raw > 0.0 {
                (raw / (raw + 2.2)).sqrt().clamp(0.0, 1.0)
            } else {
                0.0
            };
            let rgb = if self.high_contrast {
                heatmap_color_high_contrast(norm)
            } else {
                heatmap_color(norm)
            };
            self.pixels[idx] = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
        }
        self.stamp_terminal_pixels(w, h);
        self.texture_dirty = true;
    }

    fn stamp_terminal_pixels(&mut self, w: usize, h: usize) {
        let [br, bg, bb] = self.beacon_rgb;
        let beacon_col = egui::Color32::from_rgb(br, bg, bb);
        for term in self.engine.terminals() {
            let cx = (term.x.round() as isize).clamp(1, (w - 2) as isize);
            let cy = (term.y.round() as isize).clamp(1, (h - 2) as isize);
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let nx = (cx + dx) as usize;
                    let ny = (cy + dy) as usize;
                    self.pixels[flatten_2d_index(nx, ny, w)] = beacon_col;
                }
            }
        }
    }

    fn sync_gpu_texture(&mut self, ctx: &egui::Context) {
        if !self.texture_dirty && self.texture.is_some() {
            return;
        }
        let image = ColorImage {
            size: [self.engine.width(), self.engine.height()],
            pixels: self.pixels.clone(),
            ..Default::default()
        };
        let opts = egui::TextureOptions::NEAREST;
        if let Some(tex) = &mut self.texture {
            tex.set(image, opts);
        } else {
            self.texture = Some(ctx.load_texture("physarum_trail_grid", image, opts));
        }
        self.texture_dirty = false;
    }

    pub(crate) fn draw_interactive_canvas(&mut self, ui: &mut egui::Ui, target_side: f32) {
        let ctx = ui.ctx().clone();
        self.sync_gpu_texture(&ctx);

        let side = target_side.clamp(140.0, 560.0);
        let sense = egui::Sense::click_and_drag().union(egui::Sense::focusable_noninteractive());
        let (response, painter) = ui.allocate_painter(egui::vec2(side, side), sense);
        let canvas_rect =
            compute_aspect_locked_rect(response.rect, self.engine.width(), self.engine.height());

        if let Some(tex) = &self.texture {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            painter.image(tex.id(), canvas_rect, uv, egui::Color32::WHITE);
        }

        self.handle_pointer_interaction(ui, &response, canvas_rect);
        self.handle_keyboard_interaction(ui, &response);
        if self.show_skeleton_overlay {
            self.paint_skeleton_and_terminals(ui, canvas_rect, &painter);
        }
        self.paint_cursor_reticle(ui, &response, canvas_rect, &painter);

        let b = &self.latest_benchmark;
        let summary = format!(
            "{} on {}x{} grid ({} agents, {} cities): wire/MST {:.2}, reduction {:+.1}%, stretch {:.2}, cursor ({}, {})",
            self.preset.label(),
            self.engine.width(),
            self.engine.height(),
            self.engine.agent_count(),
            self.engine.terminal_count(),
            b.cost_ratio,
            b.length_reduction_pct,
            b.mean_route_stretch,
            self.cursor[0],
            self.cursor[1],
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "Physarum Network Lab Chemotactic Canvas",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &response);
        lock_canvas_arrow_keys(ui, &response);
        self.focused_canvas = response.has_focus().then_some(response.id);
    }

    fn paint_skeleton_and_terminals(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        painter: &egui::Painter,
    ) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let w = self.engine.width().max(1) as f32;
        let h = self.engine.height().max(1) as f32;
        let map_pt = |p: [f64; 2]| -> egui::Pos2 {
            egui::pos2(
                rect.min.x + ((p[0] as f32 + 0.5) / w) * rect.width(),
                rect.min.y + ((p[1] as f32 + 0.5) / h) * rect.height(),
            )
        };

        for &(p0, p1) in &self.latest_benchmark.mst_segments {
            painter.line_segment(
                [map_pt(p0), map_pt(p1)],
                egui::Stroke::new(1.1_f32, palette.graph_edge),
            );
        }
        for edge in &self.latest_benchmark.edges {
            painter.line_segment(
                [map_pt(edge.p0), map_pt(edge.p1)],
                egui::Stroke::new(2.1_f32, palette.success),
            );
        }
        for &junc in &self.latest_benchmark.steiner_junctions {
            painter.circle_stroke(
                map_pt(junc),
                4.5_f32,
                egui::Stroke::new(1.8_f32, palette.warning),
            );
        }
        for term in self.engine.terminals() {
            let center = map_pt([term.x, term.y]);
            painter.circle_filled(center, 4.2_f32, palette.accent);
            painter.circle_stroke(
                center,
                5.2_f32,
                egui::Stroke::new(1.4_f32, palette.node_stroke),
            );
        }
    }

    fn handle_pointer_interaction(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        canvas_rect: egui::Rect,
    ) {
        if response.clicked() || response.dragged() {
            response.request_focus();
        }
        let any_down = ui.input(|i| {
            i.pointer.button_down(egui::PointerButton::Primary)
                || i.pointer.button_down(egui::PointerButton::Secondary)
        });
        let primary_click = response.clicked_by(egui::PointerButton::Primary);
        let primary_drag = response.dragged_by(egui::PointerButton::Primary);
        let secondary_drag = response.dragged_by(egui::PointerButton::Secondary);
        let secondary = response.clicked_by(egui::PointerButton::Secondary) || secondary_drag;
        if !primary_click && !primary_drag && !secondary {
            if !any_down {
                self.drag_prev = None;
            }
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        let Some((gx, gy)) = map_pointer_to_cell(
            canvas_rect,
            pos,
            self.engine.width(),
            self.engine.height(),
            false,
        ) else {
            if !any_down {
                self.drag_prev = None;
            }
            return;
        };
        self.cursor = [gx, gy];
        let start = if primary_drag || secondary_drag {
            self.drag_prev.unwrap_or((gx, gy))
        } else {
            (gx, gy)
        };
        self.dispatch_pointer_stroke(ui, start, (gx, gy), primary_click, secondary);
        self.drag_prev = any_down.then_some((gx, gy));
    }

    fn dispatch_pointer_stroke(
        &mut self,
        ui: &egui::Ui,
        start: (usize, usize),
        end: (usize, usize),
        primary_click: bool,
        secondary: bool,
    ) {
        let shift = ui.input(|i| i.modifiers.shift);
        if secondary || shift {
            if self.brush_mode == PhysarumBrushMode::PaintWall {
                self.apply_wall_drag_segment(start, end, false);
            } else {
                self.apply_sever_drag_segment(start, end);
            }
            return;
        }
        match self.brush_mode {
            PhysarumBrushMode::PlaceCity if primary_click => self.toggle_city_at(end.0, end.1),
            PhysarumBrushMode::PlaceCity => {}
            PhysarumBrushMode::PaintWall => self.apply_wall_drag_segment(start, end, true),
            PhysarumBrushMode::EraseWall => self.apply_wall_drag_segment(start, end, false),
            PhysarumBrushMode::SeverTube => self.apply_sever_drag_segment(start, end),
        }
    }

    pub(crate) fn toggle_city_at(&mut self, gx: usize, gy: usize) {
        if self.engine.toggle_terminal_at(gx as f64, gy as f64) {
            let wf = self.engine.width() as f64;
            let hf = self.engine.height() as f64;
            self.terminals = self
                .engine
                .terminals()
                .iter()
                .map(|t| {
                    [
                        ((t.x / wf) * 1e6).round() / 1e6,
                        ((t.y / hf) * 1e6).round() / 1e6,
                    ]
                })
                .collect();
            if self.obstacle_mask_rle.is_some() {
                self.sync_obstacle_mask_rle();
            } else {
                self.preset_obstacle_baseline = self.engine.obstacles().to_vec();
            }
            self.refresh_benchmark_and_record();
            self.refresh_pixels_in_place();
        }
    }

    /// Sweeps a continuous Bresenham obstacle brush segment from `start` to `end` without skipping cells.
    pub fn apply_wall_drag_segment(
        &mut self,
        start: (usize, usize),
        end: (usize, usize),
        blocked: bool,
    ) {
        let w = self.engine.width();
        let h = self.engine.height();
        let r = self.brush_radius;
        for (ix, iy) in bresenham_line(start.0 as i32, start.1 as i32, end.0 as i32, end.1 as i32) {
            let gx = (ix.max(0) as usize).min(w - 1);
            let gy = (iy.max(0) as usize).min(h - 1);
            self.engine.paint_obstacle_disc(gx, gy, r, blocked);
            self.record_obstacle_stroke_point(gx, gy, blocked);
        }
        self.sync_obstacle_mask_rle();
        self.refresh_benchmark_and_record();
        self.refresh_pixels_in_place();
    }

    fn record_obstacle_stroke_point(&mut self, gx: usize, gy: usize, blocked: bool) {
        let u = (((gx as f64) / (self.engine.width() as f64)) * 1e4).round() / 1e4;
        let v = (((gy as f64) / (self.engine.height() as f64)) * 1e4).round() / 1e4;
        let entry = [u, v, self.brush_radius, if blocked { 1.0 } else { 0.0 }];
        let is_dup = self
            .obstacle_strokes
            .last()
            .is_some_and(|last| (last[0] - u).hypot(last[1] - v) < 0.025 && last[3] == entry[3]);
        if !is_dup {
            if self.obstacle_strokes.len() >= MAX_SAVED_OBSTACLE_STROKES {
                self.obstacle_strokes.remove(0);
            }
            self.obstacle_strokes.push(entry);
        }
    }

    pub(crate) fn apply_wall_stroke(&mut self, gx: usize, gy: usize, blocked: bool) {
        self.apply_wall_drag_segment((gx, gy), (gx, gy), blocked);
    }

    /// Sweeps a continuous Bresenham artery-severing brush segment from `start` to `end`.
    pub fn apply_sever_drag_segment(&mut self, start: (usize, usize), end: (usize, usize)) {
        let w = self.engine.width();
        let h = self.engine.height();
        let r = self.brush_radius;
        for (ix, iy) in bresenham_line(start.0 as i32, start.1 as i32, end.0 as i32, end.1 as i32) {
            let gx = (ix.max(0) as usize).min(w - 1);
            let gy = (iy.max(0) as usize).min(h - 1);
            self.engine.sever_trail_disc(gx, gy, r);
        }
        self.refresh_benchmark_and_record();
        self.refresh_pixels_in_place();
    }

    pub(crate) fn apply_sever_brush(&mut self, gx: usize, gy: usize) {
        self.apply_sever_drag_segment((gx, gy), (gx, gy));
    }

    fn handle_keyboard_interaction(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if !response.has_focus() || !crate::framework::is_pane_shortcut_allowed(ui) {
            return;
        }
        let w = self.engine.width();
        let h = self.engine.height();
        let step = if ui.input(|i| i.modifiers.shift) {
            4
        } else {
            1
        };
        let mut moved = false;
        ui.input(|i| {
            if i.key_pressed(egui::Key::ArrowLeft) {
                self.cursor[0] = self.cursor[0].saturating_sub(step).max(1);
                moved = true;
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                self.cursor[0] = (self.cursor[0] + step).min(w.saturating_sub(2));
                moved = true;
            }
            if i.key_pressed(egui::Key::ArrowUp) {
                self.cursor[1] = self.cursor[1].saturating_sub(step).max(1);
                moved = true;
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                self.cursor[1] = (self.cursor[1] + step).min(h.saturating_sub(2));
                moved = true;
            }
        });
        self.dispatch_keyboard_brush(ui, moved);
    }

    fn dispatch_keyboard_brush(&mut self, ui: &egui::Ui, moved: bool) {
        let (activate, city, wall, erase, sever) = ui.input(|i| {
            let activate = i.key_pressed(egui::Key::Space) || i.key_pressed(egui::Key::Enter);
            let city = i.key_pressed(egui::Key::C);
            let wall = i.key_pressed(egui::Key::W);
            let erase = i.key_pressed(egui::Key::E);
            let sever = i.key_pressed(egui::Key::Delete)
                || i.key_pressed(egui::Key::Backspace)
                || i.key_pressed(egui::Key::Minus);
            (activate, city, wall, erase, sever)
        });
        let [cx, cy] = self.cursor;
        let action_label = if sever {
            self.apply_sever_brush(cx, cy);
            Some("Severed Physarum trail")
        } else if city {
            self.toggle_city_at(cx, cy);
            Some("Toggled Physarum city terminal")
        } else if wall {
            self.apply_wall_stroke(cx, cy, true);
            Some("Painted Physarum obstacle wall")
        } else if erase {
            self.apply_wall_stroke(cx, cy, false);
            Some("Erased Physarum obstacle wall")
        } else if activate {
            Some(self.apply_active_mode_at(cx, cy))
        } else {
            None
        };
        if let Some(prefix) = action_label {
            announce_live_status(ui.ctx(), format!("{prefix} at ({cx}, {cy})."));
        } else if moved {
            ui.ctx().request_repaint();
        }
    }

    fn apply_active_mode_at(&mut self, cx: usize, cy: usize) -> &'static str {
        match self.brush_mode {
            PhysarumBrushMode::PlaceCity => {
                self.toggle_city_at(cx, cy);
                "Toggled Physarum city terminal"
            }
            PhysarumBrushMode::PaintWall => {
                self.apply_wall_stroke(cx, cy, true);
                "Painted Physarum obstacle wall"
            }
            PhysarumBrushMode::EraseWall => {
                self.apply_wall_stroke(cx, cy, false);
                "Erased Physarum obstacle wall"
            }
            PhysarumBrushMode::SeverTube => {
                self.apply_sever_brush(cx, cy);
                "Severed Physarum trail"
            }
        }
    }

    fn paint_cursor_reticle(
        &self,
        ui: &egui::Ui,
        response: &egui::Response,
        rect: egui::Rect,
        painter: &egui::Painter,
    ) {
        if !response.has_focus() && !response.hovered() {
            return;
        }
        let w = self.engine.width().max(1) as f32;
        let h = self.engine.height().max(1) as f32;
        let cx = rect.min.x + ((self.cursor[0] as f32 + 0.5) / w) * rect.width();
        let cy = rect.min.y + ((self.cursor[1] as f32 + 0.5) / h) * rect.height();
        let rad_px = ((self.brush_radius as f32) / w) * rect.width();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let stroke_col = match self.brush_mode {
            PhysarumBrushMode::PlaceCity => palette.accent,
            PhysarumBrushMode::PaintWall => palette.info,
            PhysarumBrushMode::EraseWall => palette.success,
            PhysarumBrushMode::SeverTube => palette.warning,
        };
        painter.circle_stroke(
            egui::pos2(cx, cy),
            rad_px.max(4.0),
            egui::Stroke::new(1.5_f32, stroke_col),
        );
    }
}
