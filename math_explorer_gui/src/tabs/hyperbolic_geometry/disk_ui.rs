#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::disk_tool::{ARC_SAMPLE_POINTS, HyperbolicDiskTool, RulerTarget};
use crate::accessibility::{
    AccessibleHoverText, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility,
};
use crate::app::SemanticPalette;
use crate::framework::{PaneSidePanel, scoped_panel_id};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use math_explorer::num_complex::Complex64;
use math_explorer::pure_math::differential_geometry::hyperbolic::{
    GeodesicArc, HYPERBOLIC_CURVATURE, HyperbolicModel, MAX_TILING_DEPTH, gauss_bonnet_triangle,
    project_disk_point, regular_polygon_gauss_bonnet_area, unproject_disk_point,
};
use std::f64::consts::PI;

struct DiskViewport {
    center: egui::Pos2,
    radius: f32,
}

impl DiskViewport {
    fn from_rect(rect: egui::Rect) -> Self {
        let side = rect.width().min(rect.height());
        let radius = (0.45 * side).max(24.0);
        Self {
            center: rect.center(),
            radius,
        }
    }

    fn to_screen(&self, disk_pt: Complex64) -> egui::Pos2 {
        egui::pos2(
            self.center.x + (disk_pt.re as f32) * self.radius,
            self.center.y - (disk_pt.im as f32) * self.radius,
        )
    }

    fn screen_to_disk(&self, pos: egui::Pos2) -> Complex64 {
        Complex64::new(
            f64::from((pos.x - self.center.x) / self.radius),
            f64::from((self.center.y - pos.y) / self.radius),
        )
    }
}

impl HyperbolicDiskTool {
    pub(crate) fn render_disk_tool(&mut self, ui: &mut egui::Ui) {
        egui_plot::commands::CommandRegistryData::register(
            ui.ctx(),
            "Reset Möbius Pan",
            "Reset SU(1,1) camera boost and Gauss-Bonnet triangle to the disk origin",
            egui_plot::commands::CommandTrigger::Key(egui::Key::R),
            false,
            "Hyperbolic Disk Explorer",
        );
        let panel_id = scoped_panel_id(ui, "hyperbolic_disk_controls");
        PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.render_side_controls(ui);
            });
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.render_disk_canvas(ui);
        });
    }

    fn render_side_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Disk Model & Tessellation");
        ui.add_space(4.0);
        self.render_model_selector(ui);
        ui.separator();
        self.render_schlafli_controls(ui);
        ui.separator();
        self.render_ruler_controls(ui);
        ui.separator();
        self.render_gauss_bonnet_readout(ui);
    }

    fn render_model_selector(&mut self, ui: &mut egui::Ui) {
        ui.label("Projection Model:");
        ui.horizontal_wrapped(|ui| {
            let poincare_clicked = ui
                .selectable_value(&mut self.model, HyperbolicModel::Poincare, "Poincaré Disk")
                .accessible_hover_text(
                    "Conformal disk model preserving angles; geodesics are circular arcs orthogonal to |z| = 1.",
                )
                .changed();
            let klein_clicked = ui
                .selectable_value(&mut self.model, HyperbolicModel::Klein, "Beltrami-Klein")
                .accessible_hover_text(
                    "Projective disk model mapping hyperbolic geodesics to Euclidean straight chords.",
                )
                .changed();
            if poincare_clicked || klein_clicked {
                self.revision = self.revision.wrapping_add(1);
            }
        });
    }

    fn render_schlafli_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Schläfli Symbol {p, q}:");
        let mut next_p = self.p;
        let mut next_q = self.q;
        let mut next_depth = self.depth;

        let p_changed = ui
            .add(egui::Slider::new(&mut next_p, 3..=12).text("Sides p"))
            .changed();
        let q_changed = ui
            .add(egui::Slider::new(&mut next_q, 3..=12).text("Valence q"))
            .changed();
        let depth_changed = ui
            .add(egui::Slider::new(&mut next_depth, 1..=MAX_TILING_DEPTH).text("BFS Depth"))
            .changed();
        if p_changed || q_changed || depth_changed {
            let _ = self.apply_schlafli(next_p, next_q, next_depth);
        }

        ui.horizontal_wrapped(|ui| {
            for (lp, lq, label) in [
                (7, 3, "{7,3}"),
                (5, 4, "{5,4}"),
                (4, 5, "{4,5}"),
                (8, 3, "{8,3}"),
                (3, 7, "{3,7}"),
            ] {
                if ui.small_button(label).clicked() {
                    let _ = self.apply_schlafli(lp, lq, self.depth);
                }
            }
        });

        if let Some(note) = &self.status_note {
            let warn = SemanticPalette::from_visuals(ui.visuals()).warning;
            ui.colored_label(warn, note);
        }

        if ui
            .button("↺ Reset Möbius Pan")
            .accessible_hover_text("Reset SU(1,1) camera boost to the disk origin (Key: R)")
            .clicked()
        {
            self.reset_pan_and_ruler();
        }
    }

    fn render_ruler_controls(&mut self, ui: &mut egui::Ui) {
        if ui
            .checkbox(&mut self.show_ruler, "Show Gauss-Bonnet Triangle Ruler")
            .changed()
        {
            self.revision = self.revision.wrapping_add(1);
        }
        ui.label("Canvas Drag / Click Action:");
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.active_ruler_target, RulerTarget::PanOnly, "Pan Disk");
            ui.selectable_value(&mut self.active_ruler_target, RulerTarget::VertexA, "Place A");
            ui.selectable_value(&mut self.active_ruler_target, RulerTarget::VertexB, "Place B");
            ui.selectable_value(&mut self.active_ruler_target, RulerTarget::VertexC, "Place C");
        });
    }

    fn render_gauss_bonnet_readout(&self, ui: &mut egui::Ui) {
        let gb = gauss_bonnet_triangle(
            self.ruler_vertices[0],
            self.ruler_vertices[1],
            self.ruler_vertices[2],
        );
        ui.heading("Intrinsic Metrics");
        render_copyable_metric(ui, "Gaussian Curvature K", &format!("{HYPERBOLIC_CURVATURE:.1}"));
        render_copyable_metric(
            ui,
            "Active Tiles",
            &format!("{} polygons", self.tiling.polygons().len()),
        );
        if let Ok(tile_area) = regular_polygon_gauss_bonnet_area(self.p, self.q) {
            render_copyable_metric(
                ui,
                "Tile Area (Gauss-Bonnet)",
                &format!("{tile_area:.5} ({:.4} pi)", tile_area / PI),
            );
        }
        render_copyable_metric(
            ui,
            "Sides d(A,B), d(B,C), d(C,A)",
            &format!(
                "{:.4}, {:.4}, {:.4}",
                gb.side_lengths[0], gb.side_lengths[1], gb.side_lengths[2]
            ),
        );
        render_copyable_metric(
            ui,
            "Angle Sum (alpha+beta+gamma)",
            &format!("{:.4} rad ({:.2} deg)", gb.angle_sum, gb.angle_sum.to_degrees()),
        );
        render_copyable_metric(
            ui,
            "Triangle Defect / Area",
            &format!("{:.5} ({:.4} pi)", gb.area, gb.area / PI),
        );
    }

    fn render_disk_canvas(&mut self, ui: &mut egui::Ui) {
        let avail = ui.available_size();
        let canvas_size = egui::vec2(avail.x.max(180.0), avail.y.max(180.0));
        let (response, painter) = ui.allocate_painter(canvas_size, egui::Sense::click_and_drag());
        self.last_canvas_id = Some(response.id);

        let viewport = DiskViewport::from_rect(response.rect);
        self.handle_canvas_interaction(ui, &response, &viewport);

        let gb = gauss_bonnet_triangle(
            self.ruler_vertices[0],
            self.ruler_vertices[1],
            self.ruler_vertices[2],
        );
        let model_label = match self.model {
            HyperbolicModel::Poincare => "Poincaré conformal",
            HyperbolicModel::Klein => "Beltrami-Klein projective",
        };
        let summary = format!(
            "{} {{{}, {}}} tiling ({} tiles), triangle area {:.4}",
            model_label,
            self.p,
            self.q,
            self.tiling.polygons().len(),
            gb.area
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "Hyperbolic Disk Tessellation Canvas",
            Some(&summary),
        );
        lock_canvas_arrow_keys(ui, &response);

        let palette = SemanticPalette::from_visuals(ui.visuals());
        self.paint_disk_background(&painter, &viewport, &palette, ui.visuals());
        self.paint_tiling_polygons(&painter, &viewport, &palette);
        if self.show_ruler {
            self.paint_ruler_overlay(&painter, &viewport, &palette);
        }
        paint_canvas_focus_ring(ui, &response);
    }

    fn handle_canvas_interaction(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        viewport: &DiskViewport,
    ) {
        if response.clicked()
            && self.active_ruler_target == RulerTarget::PanOnly
        {
            response.request_focus();
        }
        if matches!(
            self.active_ruler_target,
            RulerTarget::VertexA | RulerTarget::VertexB | RulerTarget::VertexC
        ) && (response.clicked() || response.dragged())
        {
            if let Some(pos) = response.interact_pointer_pos() {
                response.request_focus();
                let raw = viewport.screen_to_disk(pos);
                let poincare = unproject_disk_point(raw, self.model);
                let slot = match self.active_ruler_target {
                    RulerTarget::VertexA => 0,
                    RulerTarget::VertexB => 1,
                    RulerTarget::VertexC => 2,
                    RulerTarget::PanOnly => 0,
                };
                self.ruler_vertices[slot] = poincare;
                self.revision = self.revision.wrapping_add(1);
            }
        } else if self.active_ruler_target == RulerTarget::PanOnly && response.dragged() {
            response.request_focus();
            let d = response.drag_delta();
            let scale = 2.0 / f64::from(viewport.radius.max(1.0));
            let delta = Complex64::new(f64::from(d.x) * scale, -f64::from(d.y) * scale);
            self.apply_mobius_pan(delta);
        }

        if response.has_focus() {
            let step = 0.14;
            let mut key_delta = Complex64::new(0.0, 0.0);
            ui.input(|i| {
                if i.key_pressed(egui::Key::ArrowLeft) {
                    key_delta.re -= step;
                }
                if i.key_pressed(egui::Key::ArrowRight) {
                    key_delta.re += step;
                }
                if i.key_pressed(egui::Key::ArrowUp) {
                    key_delta.im += step;
                }
                if i.key_pressed(egui::Key::ArrowDown) {
                    key_delta.im -= step;
                }
                if i.key_pressed(egui::Key::R) {
                    key_delta = Complex64::new(f64::NAN, 0.0);
                }
            });
            if key_delta.re.is_nan() {
                self.reset_pan_and_ruler();
            } else if key_delta.norm_sqr() > 0.0 {
                self.apply_mobius_pan(key_delta);
            }
        }
    }

    fn paint_disk_background(
        &self,
        painter: &egui::Painter,
        viewport: &DiskViewport,
        palette: &SemanticPalette,
        visuals: &egui::Visuals,
    ) {
        painter.circle_filled(viewport.center, viewport.radius, visuals.extreme_bg_color);
        painter.circle_stroke(
            viewport.center,
            viewport.radius,
            egui::Stroke::new(2.0_f32, palette.accent),
        );
    }

    fn paint_tiling_polygons(
        &mut self,
        painter: &egui::Painter,
        viewport: &DiskViewport,
        palette: &SemanticPalette,
    ) {
        let count = self.tiling.polygons().len();
        for idx in 0..count {
            let poly = self.tiling.polygons()[idx];
            let stroke_color = match poly.layer {
                0 => palette.accent,
                1 => palette.info,
                2 => palette.success.gamma_multiply(0.78_f32),
                _ => palette.graph_edge,
            };
            let stroke_width = if poly.layer == 0 { 2.0_f32 } else { 1.15_f32 };
            for side in 0..poly.sides {
                let a = poly.vertices[side];
                let b = poly.vertices[(side + 1) % poly.sides];
                let arc = GeodesicArc::through_points(a, b);
                arc.sample_into(self.model, &mut self.arc_scratch);
                let mut pts = Vec::with_capacity(ARC_SAMPLE_POINTS);
                for &pt in &self.arc_scratch {
                    pts.push(viewport.to_screen(pt));
                }
                painter.add(egui::Shape::line(
                    pts,
                    egui::Stroke::new(stroke_width, stroke_color),
                ));
            }
        }
    }

    fn paint_ruler_overlay(
        &mut self,
        painter: &egui::Painter,
        viewport: &DiskViewport,
        palette: &SemanticPalette,
    ) {
        let stroke = egui::Stroke::new(2.6_f32, palette.warning);
        for side in 0..3 {
            let a = self.ruler_vertices[side];
            let b = self.ruler_vertices[(side + 1) % 3];
            let arc = GeodesicArc::through_points(a, b);
            arc.sample_into(self.model, &mut self.arc_scratch);
            let pts: Vec<egui::Pos2> = self
                .arc_scratch
                .iter()
                .map(|&z| viewport.to_screen(z))
                .collect();
            painter.add(egui::Shape::line(pts, stroke));
        }

        for (idx, label) in ["A", "B", "C"].iter().enumerate() {
            let proj = project_disk_point(self.ruler_vertices[idx], self.model);
            let pos = viewport.to_screen(proj);
            painter.circle_filled(pos, 5.0, palette.error);
            painter.text(
                pos + egui::vec2(8.0, -8.0),
                egui::Align2::LEFT_BOTTOM,
                *label,
                egui::FontId::proportional(13.0),
                palette.warning,
            );
        }
    }
}

// [cite:modular_polynomials_review]
