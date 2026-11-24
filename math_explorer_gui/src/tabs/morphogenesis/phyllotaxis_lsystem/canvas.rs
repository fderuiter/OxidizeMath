//! Interactive custom 2D botanical canvas with AccessKit keyboard focus, parastichy spiral
//! overlays, and vascular L-system turtle rendering for [`super::PhyllotaxisLSystemTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::PhyllotaxisLSystemTool;
use super::presets::BotanicalStudioMode;
use crate::accessibility::{
    lock_canvas_arrow_keys, paint_canvas_focus_ring, register_canvas_accessibility,
};
use crate::app::SemanticPalette;
use eframe::egui;

impl PhyllotaxisLSystemTool {
    pub(super) fn draw_botanical_canvas(&mut self, ui: &mut egui::Ui) {
        let avail = ui.available_size();
        let canvas_size = egui::vec2(avail.x.max(260.0), (avail.y - 8.0).clamp(240.0, 680.0));
        let (response, painter) = ui.allocate_painter(canvas_size, egui::Sense::click_and_drag());
        let palette = SemanticPalette::from_visuals(ui.visuals());

        painter.rect_filled(response.rect, 6.0, ui.visuals().extreme_bg_color);
        painter.rect_stroke(
            response.rect,
            6.0_f32,
            egui::Stroke::new(1.0_f32, ui.visuals().widgets.noninteractive.bg_stroke.color),
            egui::StrokeKind::Inside,
        );

        match self.state.mode {
            BotanicalStudioMode::DualSplit => {
                let mid_x = response.rect.center().x;
                let left_rect = egui::Rect::from_min_max(
                    response.rect.min,
                    egui::pos2(mid_x - 2.0, response.rect.max.y),
                );
                let right_rect = egui::Rect::from_min_max(
                    egui::pos2(mid_x + 2.0, response.rect.min.y),
                    response.rect.max,
                );
                painter.line_segment(
                    [
                        egui::pos2(mid_x, response.rect.min.y + 8.0),
                        egui::pos2(mid_x, response.rect.max.y - 8.0),
                    ],
                    egui::Stroke::new(1.0_f32, ui.visuals().widgets.noninteractive.bg_stroke.color),
                );
                self.paint_phyllotaxis_region(&painter, left_rect, &palette);
                self.paint_lsystem_region(&painter, right_rect, &palette);
            }
            BotanicalStudioMode::PhyllotaxisOnly => {
                self.paint_phyllotaxis_region(&painter, response.rect, &palette);
            }
            BotanicalStudioMode::LSystemOnly => {
                self.paint_lsystem_region(&painter, response.rect, &palette);
            }
        }

        self.handle_canvas_interaction(ui, &response);
        self.finalize_canvas_a11y(ui, &response);
    }

    fn finalize_canvas_a11y(&mut self, ui: &mut egui::Ui, response: &egui::Response) {
        let analysis = self.vogel.analysis();
        let summary = format!(
            "{} primordia at {:.4}°: parastichies ({}, {}), Fibonacci={}, L-system {} gen {} ({} segments)",
            self.vogel.count(),
            self.state.divergence_deg,
            analysis.primary_count,
            analysis.secondary_count,
            analysis.is_fibonacci_pair,
            self.state.lsystem_preset.label(),
            self.lsystem.generation(),
            self.lsystem.segments().len()
        );
        register_canvas_accessibility(
            ui,
            response,
            egui::accesskit::Role::GraphicsDocument,
            "Botanical Phyllotaxis and L-System Morphogenesis Canvas",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, response);
        lock_canvas_arrow_keys(ui, response);
    }

    fn paint_phyllotaxis_region(
        &mut self,
        painter: &egui::Painter,
        rect: egui::Rect,
        palette: &SemanticPalette,
    ) {
        let inner = rect.shrink(16.0);
        if inner.width() <= 16.0 || inner.height() <= 16.0 {
            return;
        }
        let max_r = self
            .vogel
            .points()
            .last()
            .map_or(1.0, |p| p.radius.max(1e-6));
        let half_span = (inner.width().min(inner.height()) * 0.46) / (max_r as f32);
        let center = inner.center();

        if self.state.show_parastichies && self.vogel.count() >= 5 {
            let analysis = self.vogel.analysis();
            self.paint_spiral_family(
                painter,
                center,
                half_span,
                analysis.clockwise_step,
                egui::Stroke::new(1.35_f32, palette.accent),
            );
            if !analysis.has_radial_spokes {
                self.paint_spiral_family(
                    painter,
                    center,
                    half_span,
                    analysis.counter_clockwise_step,
                    egui::Stroke::new(1.2_f32, palette.warning),
                );
            }
        }

        let count = self.vogel.count().max(1);
        let dot_radius = (180.0 / (count as f32).sqrt()).clamp(1.6, 5.2);
        for pt in self.vogel.points() {
            let sx = center.x + (pt.x as f32) * half_span;
            let sy = center.y - (pt.y as f32) * half_span;
            let color = if pt.index == self.selected_primordium {
                palette.error
            } else if pt.index % 2 == 0 {
                palette.success
            } else {
                palette.info
            };
            let r = if pt.index == self.selected_primordium {
                dot_radius * 1.65
            } else {
                dot_radius
            };
            painter.circle_filled(egui::pos2(sx, sy), r, color);
        }
    }

    fn paint_spiral_family(
        &mut self,
        painter: &egui::Painter,
        center: egui::Pos2,
        scale_px: f32,
        step: usize,
        stroke: egui::Stroke,
    ) {
        let family_arms = step.clamp(1, 89);
        for arm in 0..family_arms {
            let count = self.vogel.count();
            let mut prev: Option<egui::Pos2> = None;
            let mut idx = arm;
            while idx < count {
                let pt = self.vogel.points()[idx];
                let pos = egui::pos2(
                    center.x + (pt.x as f32) * scale_px,
                    center.y - (pt.y as f32) * scale_px,
                );
                if let Some(p0) = prev {
                    painter.line_segment([p0, pos], stroke);
                }
                prev = Some(pos);
                idx += family_arms;
            }
        }
    }

    fn paint_lsystem_region(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        palette: &SemanticPalette,
    ) {
        let inner = rect.shrink(18.0);
        let segs = self.lsystem.segments();
        if inner.width() <= 16.0 || inner.height() <= 16.0 || segs.is_empty() {
            return;
        }
        let b = self.lsystem.bounds();
        let span_x = b.width().max(1e-6) as f32;
        let span_y = b.height().max(1e-6) as f32;
        let scale = (inner.width() / span_x).min(inner.height() / span_y) * 0.92;
        let cx_world = ((b.min_x + b.max_x) * 0.5) as f32;
        let cy_world = ((b.min_y + b.max_y) * 0.5) as f32;
        let center = inner.center();
        let max_depth = b.max_depth.max(1);

        let stride = (segs.len() / 12_000).max(1);
        for seg in segs.iter().step_by(stride) {
            let p0 = egui::pos2(
                center.x + ((seg.x0 as f32) - cx_world) * scale,
                center.y - ((seg.y0 as f32) - cy_world) * scale,
            );
            let p1 = egui::pos2(
                center.x + ((seg.x1 as f32) - cx_world) * scale,
                center.y - ((seg.y1 as f32) - cy_world) * scale,
            );
            let depth_frac = (seg.depth as f32) / (max_depth as f32);
            let color = if depth_frac < 0.35 {
                palette.warning
            } else if depth_frac < 0.70 {
                palette.success
            } else {
                palette.accent
            };
            let width = (2.2 * (1.0 - 0.55 * depth_frac)).clamp(0.85, 2.4);
            painter.line_segment([p0, p1], egui::Stroke::new(width, color));
        }
    }

    fn handle_canvas_interaction(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if response.clicked() || response.dragged() {
            response.request_focus();
        }
        if !response.has_focus() {
            return;
        }
        let mut changed = false;
        ui.input(|i| {
            if i.key_pressed(egui::Key::ArrowRight) {
                self.selected_primordium =
                    (self.selected_primordium + 1) % self.vogel.count().max(1);
            }
            if i.key_pressed(egui::Key::ArrowLeft) {
                let n = self.vogel.count().max(1);
                self.selected_primordium = (self.selected_primordium + n - 1) % n;
            }
            if i.key_pressed(egui::Key::ArrowUp) && self.state.generations < 11 {
                self.state.generations += 1;
                changed = true;
            }
            if i.key_pressed(egui::Key::ArrowDown) && self.state.generations > 1 {
                self.state.generations -= 1;
                changed = true;
            }
        });
        if changed {
            self.sync_engines();
        }
    }
}
