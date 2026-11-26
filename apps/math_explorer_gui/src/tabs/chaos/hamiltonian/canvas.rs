//! Accessible click-to-seed and keyboard-navigable Poincaré surface-of-section canvas.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{HamiltonianChaosLab, MAX_ORBITS};
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::SemanticPalette;
use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke, StrokeKind, Vec2};

struct PhaseViewport {
    rect: Rect,
    q_min: f64,
    q_max: f64,
    p_min: f64,
    p_max: f64,
}

impl PhaseViewport {
    fn new(rect: Rect, bounds: [f64; 4]) -> Self {
        let pad = (rect.width().min(rect.height()) * 0.07).clamp(16.0, 32.0);
        let inner = rect.shrink(pad);
        let safe = if inner.width() > 20.0 && inner.height() > 20.0 {
            inner
        } else {
            rect
        };
        Self {
            rect: safe,
            q_min: bounds[0],
            q_max: bounds[1],
            p_min: bounds[2],
            p_max: bounds[3],
        }
    }

    fn to_screen(&self, qp: [f64; 2]) -> Pos2 {
        let tx = ((qp[0] - self.q_min) / (self.q_max - self.q_min)) as f32;
        let ty = ((qp[1] - self.p_min) / (self.p_max - self.p_min)) as f32;
        Pos2::new(
            self.rect.left() + tx * self.rect.width(),
            self.rect.bottom() - ty * self.rect.height(),
        )
    }

    fn to_phase(&self, pos: Pos2) -> [f64; 2] {
        let tx = ((pos.x - self.rect.left()) / self.rect.width().max(1.0)).clamp(0.0, 1.0);
        let ty = ((self.rect.bottom() - pos.y) / self.rect.height().max(1.0)).clamp(0.0, 1.0);
        [
            self.q_min + f64::from(tx) * (self.q_max - self.q_min),
            self.p_min + f64::from(ty) * (self.p_max - self.p_min),
        ]
    }
}

impl HamiltonianChaosLab {
    pub(super) fn render_poincare_canvas(&mut self, ui: &mut egui::Ui, desired_size: Vec2) {
        let (response, painter) = ui.allocate_painter(desired_size, egui::Sense::click_and_drag());
        let _ = response.clone().accessible_hover_text(
            "Poincare Surface of Section: click or use Arrow keys + Enter to seed symplectic orbits",
        );

        let viewport = PhaseViewport::new(response.rect, self.poincare_bounds());
        self.handle_canvas_click(ui.ctx(), &response, &viewport);

        if response.has_focus() {
            lock_canvas_arrow_keys(ui, &response);
            self.handle_canvas_keyboard(ui);
        }
        self.focused_canvas = response.has_focus().then_some(response.id);

        let total_pts: usize = self.orbits.iter().map(|s| s.section_points.len()).sum();
        let summary = format!(
            "{} ({}) — {}/{} orbits, {} crossings, cursor ({:.3}, {:.3})",
            self.model.label(),
            self.solver.label(),
            self.orbits.len(),
            MAX_ORBITS,
            total_pts,
            self.cursor_qp[0],
            self.cursor_qp[1]
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "Poincare Surface of Section",
            Some(&summary),
        );

        let palette = SemanticPalette::from_visuals(ui.visuals());
        self.paint_canvas_backdrop(&painter, response.rect, &viewport, &palette, ui.visuals());
        self.paint_hill_shell(&painter, &viewport, &palette);
        self.paint_section_orbits(&painter, &viewport, &palette);
        self.paint_cursor_and_hud(
            &painter,
            response.rect,
            &viewport,
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
        viewport: &PhaseViewport,
    ) {
        if !response.clicked() {
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        let qp = viewport.to_phase(pos);
        self.cursor_qp = qp;
        response.request_focus();
        self.seed_and_announce(ctx, qp);
    }

    fn handle_canvas_keyboard(&mut self, ui: &egui::Ui) {
        let ctx = ui.ctx();
        let [q_min, q_max, p_min, p_max] = self.poincare_bounds();
        let dq_base = (q_max - q_min) * 0.04;
        let dp_base = (p_max - p_min) * 0.04;

        let step_delta = ctx.input_mut(|i| {
            for (mods, scale) in [(egui::Modifiers::SHIFT, 0.25), (egui::Modifiers::NONE, 1.0)] {
                if i.consume_key(mods, egui::Key::ArrowLeft) {
                    return Some([-dq_base * scale, 0.0]);
                }
                if i.consume_key(mods, egui::Key::ArrowRight) {
                    return Some([dq_base * scale, 0.0]);
                }
                if i.consume_key(mods, egui::Key::ArrowUp) {
                    return Some([0.0, dp_base * scale]);
                }
                if i.consume_key(mods, egui::Key::ArrowDown) {
                    return Some([0.0, -dp_base * scale]);
                }
            }
            None
        });

        if let Some([dq, dp]) = step_delta {
            ui.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
            self.cursor_qp[0] = (self.cursor_qp[0] + dq).clamp(q_min, q_max);
            self.cursor_qp[1] = (self.cursor_qp[1] + dp).clamp(p_min, p_max);
            self.state_rev = self.state_rev.wrapping_add(1);
            let (q_lbl, p_lbl) = self.model.section_axes();
            announce_live_status(
                ctx,
                format!(
                    "Poincare cursor moved to {q_lbl}={:.3}, {p_lbl}={:.3}",
                    self.cursor_qp[0], self.cursor_qp[1]
                ),
            );
        }

        let seed_pressed =
            ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter));
        if seed_pressed {
            self.seed_and_announce(ctx, self.cursor_qp);
        }
    }

    pub(super) fn seed_and_announce(&mut self, ctx: &egui::Context, qp: [f64; 2]) {
        let (q_lbl, p_lbl) = self.model.section_axes();
        match self.seed_orbit_at(qp) {
            Ok(idx) => {
                let msg = format!(
                    "Seeded orbit {} at ({q_lbl}={:.3}, {p_lbl}={:.3})",
                    idx + 1,
                    qp[0],
                    qp[1]
                );
                self.last_status_note = Some(msg.clone());
                announce_live_status(ctx, msg);
            }
            Err(reason) => {
                let msg = format!(
                    "Cannot seed at ({q_lbl}={:.3}, {p_lbl}={:.3}): {reason}",
                    qp[0], qp[1]
                );
                self.last_status_note = Some(msg.clone());
                announce_live_status(ctx, msg);
            }
        }
    }

    fn paint_canvas_backdrop(
        &self,
        painter: &egui::Painter,
        outer: Rect,
        vp: &PhaseViewport,
        palette: &SemanticPalette,
        visuals: &egui::Visuals,
    ) {
        painter.rect_filled(outer, 6.0, palette.surface_fill);
        painter.rect_stroke(
            outer,
            6.0,
            Stroke::new(1.0_f32, palette.graph_edge),
            StrokeKind::Inside,
        );

        let grid_stroke = Stroke::new(0.75_f32, palette.graph_edge.gamma_multiply(0.45));
        for k in 0..=4 {
            let frac = k as f64 / 4.0;
            let q = vp.q_min + frac * (vp.q_max - vp.q_min);
            let p = vp.p_min + frac * (vp.p_max - vp.p_min);
            let sx = vp.to_screen([q, vp.p_min]).x;
            let sy = vp.to_screen([vp.q_min, p]).y;
            painter.line_segment(
                [
                    Pos2::new(sx, vp.rect.top()),
                    Pos2::new(sx, vp.rect.bottom()),
                ],
                grid_stroke,
            );
            painter.line_segment(
                [
                    Pos2::new(vp.rect.left(), sy),
                    Pos2::new(vp.rect.right(), sy),
                ],
                grid_stroke,
            );
        }

        let axis_stroke = Stroke::new(1.1_f32, visuals.weak_text_color());
        if (vp.q_min..=vp.q_max).contains(&0.0) {
            let x0 = vp.to_screen([0.0, 0.0]).x;
            painter.line_segment(
                [
                    Pos2::new(x0, vp.rect.top()),
                    Pos2::new(x0, vp.rect.bottom()),
                ],
                axis_stroke,
            );
        }
        if (vp.p_min..=vp.p_max).contains(&0.0) {
            let y0 = vp.to_screen([0.0, 0.0]).y;
            painter.line_segment(
                [
                    Pos2::new(vp.rect.left(), y0),
                    Pos2::new(vp.rect.right(), y0),
                ],
                axis_stroke,
            );
        }
    }

    fn paint_hill_shell(
        &self,
        painter: &egui::Painter,
        vp: &PhaseViewport,
        palette: &SemanticPalette,
    ) {
        let curve = self.compute_hill_boundary_curve();
        if curve.len() < 3 {
            return;
        }
        let mut pts: Vec<Pos2> = curve.iter().map(|&qp| vp.to_screen(qp)).collect();
        if let Some(&first) = pts.first() {
            pts.push(first);
        }
        painter.add(egui::Shape::line(
            pts,
            Stroke::new(1.5_f32, palette.info.gamma_multiply(0.75)),
        ));
    }

    fn paint_section_orbits(
        &self,
        painter: &egui::Painter,
        vp: &PhaseViewport,
        palette: &SemanticPalette,
    ) {
        let clip = painter.with_clip_rect(vp.rect);
        for slot in &self.orbits {
            let base = palette.series_color(slot.color_index);
            let halo = base.gamma_multiply(0.22);
            for &qp in &slot.section_points {
                let pos = vp.to_screen(qp);
                clip.circle_filled(pos, 3.4, halo);
                clip.circle_filled(pos, 1.65, base);
            }
            let seed_pos = vp.to_screen(slot.seed_qp);
            clip.circle_stroke(seed_pos, 4.5, Stroke::new(1.4_f32, base));
        }

        if self.show_rk4_ghost {
            let ghost_color = palette.error.gamma_multiply(0.85);
            for &qp in &self.ghost.section_points {
                let pos = vp.to_screen(qp);
                clip.circle_stroke(pos, 2.3, Stroke::new(1.1_f32, ghost_color));
            }
        }
    }

    fn paint_cursor_and_hud(
        &self,
        painter: &egui::Painter,
        outer: Rect,
        vp: &PhaseViewport,
        palette: &SemanticPalette,
        visuals: &egui::Visuals,
        focused: bool,
    ) {
        let cpos = vp.to_screen(self.cursor_qp);
        let reticle_color = if focused {
            palette.accent
        } else {
            visuals.text_color().gamma_multiply(0.75)
        };
        let arm = if focused { 9.0 } else { 6.5 };
        painter.line_segment(
            [cpos - Vec2::new(arm, 0.0), cpos + Vec2::new(arm, 0.0)],
            Stroke::new(1.6_f32, reticle_color),
        );
        painter.line_segment(
            [cpos - Vec2::new(0.0, arm), cpos + Vec2::new(0.0, arm)],
            Stroke::new(1.6_f32, reticle_color),
        );
        painter.circle_stroke(cpos, 4.0, Stroke::new(1.4_f32, reticle_color));

        let (q_lbl, p_lbl) = self.model.section_axes();
        painter.text(
            Pos2::new(outer.center().x, outer.bottom() - 4.0),
            egui::Align2::CENTER_BOTTOM,
            format!(
                "{q_lbl} ∈ [{:.2}, {:.2}]   •   Click or Arrows+Enter to seed",
                vp.q_min, vp.q_max
            ),
            FontId::proportional(11.5),
            visuals.weak_text_color(),
        );
        painter.text(
            Pos2::new(outer.left() + 10.0, outer.top() + 8.0),
            egui::Align2::LEFT_TOP,
            format!(
                "Poincaré Section ({q_lbl}, {p_lbl})  |  Orbits: {}/{}",
                self.orbits.len(),
                MAX_ORBITS
            ),
            FontId::proportional(12.0),
            Color32::from_rgba_unmultiplied(
                visuals.text_color().r(),
                visuals.text_color().g(),
                visuals.text_color().b(),
                235,
            ),
        );
    }
}
