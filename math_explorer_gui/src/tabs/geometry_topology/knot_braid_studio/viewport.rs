#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! 3D tubular `Plot3DViewport` and interactive WCAG-accessible 2D planar diagram canvas.

mod diagram_canvas;

use self::diagram_canvas::{
    DiagramMapper, build_segment_screen_polyline, paint_crossing_bridge_and_badge,
    paint_free_loops, paint_midpoint_arrow,
};
use super::model::DiagramEditOp;
use super::KnotBraidStudioTool;
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility,
};
use crate::app::SemanticPalette;
use crate::framework::Plot3DViewport;
use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};
use egui_plot::LineStyle;

/// Under-crossing visual gap radius in pixels for unmistakable over/under strand cues.
pub(super) const UNDER_CROSSING_GAP_PX: f32 = 11.0;

/// Resolved WCAG 2.2 AA theme colors for the 2D planar link diagram canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DiagramThemeColors {
    pub canvas_bg: Color32,
    pub border_stroke: Color32,
    pub focus_ring: Color32,
    pub badge_bg: Color32,
    pub positive_text: Color32,
    pub negative_text: Color32,
    pub label_text: Color32,
    pub component_colors: [Color32; 6],
}

/// Resolves theme-aware planar diagram colors guaranteeing WCAG 2.2 AA contrast across
/// `ThemeMode::Dark`, `ThemeMode::Light`, and `ThemeMode::HighContrast`.
#[must_use]
pub(crate) fn diagram_theme_colors(visuals: &egui::Visuals) -> DiagramThemeColors {
    let palette = SemanticPalette::from_visuals(visuals);
    DiagramThemeColors {
        canvas_bg: visuals.extreme_bg_color,
        border_stroke: palette.graph_edge,
        focus_ring: SemanticPalette::sash_focus_outline(visuals),
        badge_bg: visuals.panel_fill,
        positive_text: palette.success,
        negative_text: palette.error,
        label_text: palette.node_stroke,
        component_colors: palette.series,
    }
}

impl KnotBraidStudioTool {
    /// Populates `self.scene_builder` with depth-sorted 3D component curves, Bishop rails, and frames.
    pub fn populate_scene(&mut self, visuals: &egui::Visuals) {
        let palette = SemanticPalette::from_visuals(visuals);
        self.scene_builder.begin(self.camera);
        self.scene_builder.add_axes(2.4, visuals.dark_mode);

        for (comp_idx, comp) in self.model.link_3d.components.iter().enumerate() {
            if comp.is_empty() {
                continue;
            }
            let mut closed = Vec::with_capacity(comp.len() + 1);
            closed.extend_from_slice(comp);
            closed.push(comp[0]);
            let color = palette.series_color(comp_idx);
            self.scene_builder.add_polyline(
                format!("Component {}", comp_idx + 1),
                &closed,
                Stroke::new(3.0_f32, color),
                LineStyle::Solid,
                true,
            );
        }

        for (comp_idx, rail) in &self.model.tube_rails {
            let color = palette.series_color(*comp_idx);
            self.scene_builder.add_polyline(
                "",
                rail,
                Stroke::new(1.15_f32, color),
                LineStyle::Solid,
                false,
            );
        }

        if self.show_tube_rings {
            for (comp_idx, ring) in &self.model.tube_rings {
                let color = palette.series_color(*comp_idx);
                self.scene_builder.add_polyline(
                    "",
                    ring,
                    Stroke::new(1.0_f32, color),
                    LineStyle::Solid,
                    false,
                );
            }
        }

        if self.show_bishop_frames {
            self.append_bishop_triads(&palette);
        }
    }

    fn append_bishop_triads(&mut self, palette: &SemanticPalette) {
        let scale = (self.tube_radius * 2.1).max(0.24);
        for (pts, frames) in self
            .model
            .link_3d
            .components
            .iter()
            .zip(self.model.bishop_frames.iter())
        {
            let step = (pts.len() / 10).max(4);
            for (pt, f) in pts.iter().zip(frames.iter()).step_by(step) {
                self.scene_builder.add_segment(
                    "Bishop U",
                    *pt,
                    pt + f.normal_u * scale,
                    Stroke::new(1.8_f32, palette.axis_x),
                    false,
                );
                self.scene_builder.add_segment(
                    "Bishop V",
                    *pt,
                    pt + f.normal_v * scale,
                    Stroke::new(1.8_f32, palette.axis_y),
                    false,
                );
            }
        }
    }

    pub(super) fn draw_viewports(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let avail = ui.available_size();
            if avail.x >= 760.0 {
                ui.columns(2, |cols| {
                    self.draw_3d_pane(&mut cols[0]);
                    self.draw_2d_diagram_pane(&mut cols[1]);
                });
            } else {
                let half_h = (avail.y * 0.52).max(220.0);
                ui.allocate_ui(Vec2::new(avail.x, half_h), |top_ui| {
                    self.draw_3d_pane(top_ui);
                });
                ui.separator();
                self.draw_2d_diagram_pane(ui);
            }
        });
    }

    fn draw_3d_pane(&mut self, ui: &mut egui::Ui) {
        let bound = (self.model.link_3d.bounding_radius() + self.tube_radius + 0.8).max(3.2);
        self.populate_scene(ui.visuals());
        let mut camera = self.camera;
        Plot3DViewport::new("knot_braid_3d", bound).show(ui, &mut camera, |plot_ui| {
            self.scene_builder.finish(plot_ui);
        });
        self.camera = camera;
    }

    fn draw_2d_diagram_pane(&mut self, ui: &mut egui::Ui) {
        let colors = diagram_theme_colors(ui.visuals());
        ui.horizontal(|ui| {
            ui.strong("2D Planar Diagram (Under-Crossing Gaps)");
            if ui.small_button("Next Crossing").clicked() {
                let _ = self.handle_diagram_key(egui::Key::ArrowRight);
            }
            if ui.small_button("Next Arc").clicked() {
                let _ = self.handle_diagram_key(egui::Key::S);
            }
        });

        let avail = ui.available_size();
        let canvas_size = Vec2::new(avail.x.max(180.0), (avail.y - 24.0).max(180.0));
        let (rect, response) = ui.allocate_exact_size(
            canvas_size,
            egui::Sense::click().union(egui::Sense::focusable_noninteractive()),
        );
        lock_canvas_arrow_keys(ui, &response);
        if response.clicked() {
            response.request_focus();
            if let Some(pos) = response.interact_pointer_pos() {
                self.select_diagram_element_at(rect, pos);
            }
        }
        if response.has_focus() {
            self.poll_diagram_keyboard(ui);
        }
        let summary = self.diagram_a11y_summary();
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "2D Planar Link Diagram Canvas",
            Some(&summary),
        );

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 6.0, colors.canvas_bg);
        painter.rect_stroke(
            rect,
            6.0,
            Stroke::new(1.25_f32, colors.border_stroke),
            egui::StrokeKind::Inside,
        );
        self.paint_planar_diagram(&painter, rect, &colors);
        paint_canvas_focus_ring(ui, &response);
        ui.small("Click or Tab to focus: Arrows select crossing, S selects arc, 1/U/2/3 apply Reidemeister moves.");
    }

    fn diagram_a11y_summary(&self) -> String {
        format!(
            "{} crossings, writhe {:+}, selected c{} and arc s{}, Jones {}",
            self.model.diagram.crossing_count(),
            self.model.writhe_diagram,
            self.selected_crossing,
            self.selected_segment,
            self.model.jones_display
        )
    }

    fn poll_diagram_keyboard(&mut self, ui: &egui::Ui) {
        let keys = [
            egui::Key::ArrowRight,
            egui::Key::ArrowLeft,
            egui::Key::ArrowDown,
            egui::Key::ArrowUp,
            egui::Key::S,
            egui::Key::Num1,
            egui::Key::U,
            egui::Key::Num2,
            egui::Key::Num3,
        ];
        for key in keys {
            if ui.input(|i| i.key_pressed(key)) && self.handle_diagram_key(key) {
                announce_live_status(ui.ctx(), self.diagram_a11y_summary());
            }
        }
    }

    /// Processes keyboard navigation and Reidemeister shortcuts for the planar diagram editor.
    pub(crate) fn handle_diagram_key(&mut self, key: egui::Key) -> bool {
        let c_len = self.model.diagram.crossings.len();
        let s_len = self.model.diagram.segments.len();
        match key {
            egui::Key::ArrowRight | egui::Key::ArrowDown if c_len > 0 => {
                self.second_crossing = self.selected_crossing;
                self.selected_crossing = (self.selected_crossing + 1) % c_len;
                true
            }
            egui::Key::ArrowLeft | egui::Key::ArrowUp if c_len > 0 => {
                self.second_crossing = self.selected_crossing;
                self.selected_crossing = (self.selected_crossing + c_len - 1) % c_len;
                true
            }
            egui::Key::S if s_len > 0 => {
                self.second_segment = self.selected_segment;
                self.selected_segment = (self.selected_segment + 1) % s_len;
                true
            }
            egui::Key::Num1 => {
                let res = self.push_reidemeister_op(DiagramEditOp::RiAdd {
                    segment: self.selected_segment,
                    positive: true,
                });
                self.record_move_result(res);
                true
            }
            egui::Key::U => {
                let res = self.push_reidemeister_op(DiagramEditOp::RiRemove {
                    crossing: self.selected_crossing,
                });
                self.record_move_result(res);
                true
            }
            egui::Key::Num2 => self.trigger_keyboard_rii(),
            egui::Key::Num3 => self.trigger_keyboard_riii(),
            _ => false,
        }
    }

    fn trigger_keyboard_rii(&mut self) -> bool {
        if self
            .model
            .diagram
            .can_apply_reidemeister_ii_remove(self.selected_crossing, self.second_crossing)
        {
            let res = self.push_reidemeister_op(DiagramEditOp::RiiRemove {
                c1: self.selected_crossing,
                c2: self.second_crossing,
            });
            self.record_move_result(res);
        } else {
            let s_len = self.model.diagram.segments.len();
            let under = if s_len >= 2 && self.second_segment == self.selected_segment {
                (self.selected_segment + 1) % s_len
            } else {
                self.second_segment
            };
            let res = self.push_reidemeister_op(DiagramEditOp::RiiAdd {
                over_seg: self.selected_segment,
                under_seg: under,
            });
            self.record_move_result(res);
        }
        true
    }

    fn trigger_keyboard_riii(&mut self) -> bool {
        if let Some(tri) = self.preferred_riii_triangle() {
            let res = self.push_reidemeister_op(DiagramEditOp::RiiiSlide {
                c_tb: tri.c_top_bot,
                c_tm: tri.c_top_mid,
                c_mb: tri.c_mid_bot,
            });
            self.record_move_result(res);
            true
        } else {
            false
        }
    }

    fn paint_planar_diagram(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        colors: &DiagramThemeColors,
    ) {
        let diag = &self.model.diagram;
        if diag.crossings.is_empty() {
            paint_free_loops(painter, rect, diag.free_loops.max(1), colors);
            return;
        }
        let mapper = DiagramMapper::from_diagram(diag, rect);
        let (_, comp_of) = diag.segment_components();

        // Pass 1: draw directed arcs with explicit pixel pullback at under-crossing slots (0 & 2).
        for (seg_id, &comp_idx) in comp_of.iter().enumerate() {
            let color = colors.component_colors[comp_idx % colors.component_colors.len()];
            let pts = build_segment_screen_polyline(diag, seg_id, &mapper);
            let width = if seg_id == self.selected_segment {
                3.8_f32
            } else {
                2.5_f32
            };
            if seg_id == self.selected_segment {
                painter.add(egui::Shape::line(
                    pts.clone(),
                    Stroke::new(6.2_f32, colors.focus_ring),
                ));
            }
            painter.add(egui::Shape::line(pts.clone(), Stroke::new(width, color)));
            paint_midpoint_arrow(painter, &pts, color);
        }

        // Pass 2: paint crossing gap masks, unbroken over-strand bridges, badges, and selection rings.
        for (c_idx, cr) in diag.crossings.iter().enumerate() {
            paint_crossing_bridge_and_badge(
                painter,
                c_idx,
                cr,
                c_idx == self.selected_crossing,
                &comp_of,
                &mapper,
                colors,
            );
        }
    }

    fn select_diagram_element_at(&mut self, rect: Rect, pos: Pos2) {
        let diag = &self.model.diagram;
        if diag.crossings.is_empty() {
            return;
        }
        let mapper = DiagramMapper::from_diagram(diag, rect);
        let mut best_c = None;
        let mut best_dist = 22.0_f32;
        for (idx, cr) in diag.crossings.iter().enumerate() {
            let d = mapper.to_screen(cr.position).distance(pos);
            if d < best_dist {
                best_dist = d;
                best_c = Some(idx);
            }
        }
        if let Some(idx) = best_c {
            if idx != self.selected_crossing {
                self.second_crossing = self.selected_crossing;
                self.selected_crossing = idx;
            }
            return;
        }
        let mut best_s = None;
        let mut best_s_dist = 18.0_f32;
        for seg_id in 0..diag.segments.len() {
            let pts = build_segment_screen_polyline(diag, seg_id, &mapper);
            for w in pts.windows(2) {
                let mid = w[0].lerp(w[1], 0.5);
                let d = mid.distance(pos);
                if d < best_s_dist {
                    best_s_dist = d;
                    best_s = Some(seg_id);
                }
            }
        }
        if let Some(seg_id) = best_s {
            if seg_id != self.selected_segment {
                self.second_segment = self.selected_segment;
                self.selected_segment = seg_id;
            }
        }
    }
}
