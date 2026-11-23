//! Torus $\mathbb{F}_p \times \mathbb{F}_p$ lattice painter and wrap-around geodesic builder
//! for [`super::elliptic_curve_playground::EllipticCurvePlayground`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::{
    lock_canvas_arrow_keys, paint_canvas_focus_ring, register_canvas_accessibility,
    AccessibleTheoryHover,
};
use crate::app::SemanticPalette;
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use math_explorer::pure_math::elliptic_curves::{AffinePointFp, ChordStepFp};
use scientific_metadata::theory::TheoryDescribable;

/// Interaction outcome returned by [`render_torus_canvas`].
#[derive(Clone, Copy, Debug, Default)]
pub struct TorusCanvasOutcome {
    /// Index in `points` clicked by pointer (including index `0` for $\mathcal{O}$).
    pub clicked_point_index: Option<usize>,
    /// Discrete keyboard step (`-1` or `+1`) along curve points when canvas holds focus.
    pub arrow_delta: i32,
    /// Whether the user pressed `Q` while focused to toggle between selecting $P$ and $Q$.
    pub toggle_selector: bool,
    /// Active canvas `Id` when this canvas currently holds keyboard focus.
    pub focused_canvas_id: Option<egui::Id>,
}

/// View state borrowed by [`render_torus_canvas`].
pub struct TorusCanvasModel<'a> {
    /// Prime modulus $p$.
    pub p: i64,
    /// Coefficient $a$.
    pub a: i64,
    /// Coefficient $b$.
    pub b: i64,
    /// Whether $4a^3 + 27b^2 \equiv 0 \pmod p$.
    pub is_singular: bool,
    /// Enumerated curve points (index `0` is $\mathcal{O}$).
    pub points: &'a [AffinePointFp],
    /// Cyclic subgroup orbit `[1]P, [2]P, ...`.
    pub orbit: &'a [AffinePointFp],
    /// First selected point $P$.
    pub point_p: AffinePointFp,
    /// Second selected point $Q$.
    pub point_q: AffinePointFp,
    /// Scalar multiple `[k]P`.
    pub scalar_kp: AffinePointFp,
    /// Active scalar $k$.
    pub scalar_k: i64,
    /// Chord/tangent step metadata when non-singular.
    pub chord: Option<ChordStepFp>,
    /// Precomputed wrap-around ray segments in torus coordinates `[0, p]^2`.
    pub ray_segments: &'a [([f32; 2], [f32; 2])],
    /// Screen-reader summary string for AccessKit canvas value.
    pub status_summary: &'a str,
}

/// Populates wrap-around secant or tangent segments on `[0, p]^2` into preallocated `out`.
pub fn populate_torus_ray_segments(
    p: i64,
    start: (i64, i64),
    step: &ChordStepFp,
    progress: f32,
    max_wraps: usize,
    out: &mut Vec<([f32; 2], [f32; 2])>,
) {
    out.clear();
    let (du, dv) = (step.geodesic_dir.0 as f64, step.geodesic_dir.1 as f64);
    let p_f = p as f64;
    let total_t = (step.geodesic_span as f64) * f64::from(progress.clamp(0.0, 1.0));
    if p <= 0 || total_t <= 1e-6 || (du.abs() < 1e-9 && dv.abs() < 1e-9) {
        return;
    }
    let mut rem_t = total_t;
    let mut cx = (start.0 as f64).rem_euclid(p_f);
    let mut cy = (start.1 as f64).rem_euclid(p_f);
    for _ in 0..max_wraps {
        if rem_t <= 1e-7 || out.len() >= out.capacity().max(max_wraps) {
            break;
        }
        let dt = rem_t
            .min(boundary_hit_time(cx, du, p_f))
            .min(boundary_hit_time(cy, dv, p_f))
            .max(1e-7);
        let nx = (cx + du * dt).clamp(0.0, p_f);
        let ny = (cy + dv * dt).clamp(0.0, p_f);
        out.push(([cx as f32, cy as f32], [nx as f32, ny as f32]));
        rem_t -= dt;
        if rem_t <= 1e-7 {
            break;
        }
        cx = wrap_boundary_coord(nx, du, p_f);
        cy = wrap_boundary_coord(ny, dv, p_f);
    }
}

fn boundary_hit_time(coord: f64, dir: f64, p_f: f64) -> f64 {
    if dir > 1e-9 {
        let gap = (p_f - coord).max(0.0);
        if gap <= 1e-9 {
            p_f / dir
        } else {
            gap / dir
        }
    } else if dir < -1e-9 {
        let gap = coord.max(0.0);
        if gap <= 1e-9 {
            p_f / (-dir)
        } else {
            gap / (-dir)
        }
    } else {
        f64::INFINITY
    }
}

fn wrap_boundary_coord(coord: f64, dir: f64, p_f: f64) -> f64 {
    if dir > 0.0 && (p_f - coord).abs() <= 1e-6 {
        0.0
    } else if dir < 0.0 && coord.abs() <= 1e-6 {
        p_f
    } else {
        coord.rem_euclid(p_f)
    }
}

struct TorusViewport {
    plot_rect: Rect,
    infinity_rect: Rect,
    p_f: f32,
}

impl TorusViewport {
    fn from_canvas(rect: Rect, p: i64) -> Self {
        let (pad_l, pad_r, pad_t, pad_b) = (30.0_f32, 14.0_f32, 34.0_f32, 26.0_f32);
        let avail_w = (rect.width() - pad_l - pad_r).max(40.0_f32);
        let avail_h = (rect.height() - pad_t - pad_b).max(40.0_f32);
        let side = avail_w.min(avail_h);
        let plot_rect = Rect::from_min_size(
            Pos2::new(
                rect.left() + pad_l + (avail_w - side) * 0.5_f32,
                rect.top() + pad_t + (avail_h - side) * 0.5_f32,
            ),
            Vec2::splat(side),
        );
        let inf_size = Vec2::new(92.0_f32, 22.0_f32);
        let infinity_rect = Rect::from_min_size(
            Pos2::new(rect.right() - inf_size.x - 10.0_f32, rect.top() + 6.0_f32),
            inf_size,
        );
        Self {
            plot_rect,
            infinity_rect,
            p_f: (p.max(5)) as f32,
        }
    }

    fn to_screen(&self, x: f32, y: f32) -> Pos2 {
        Pos2::new(
            self.plot_rect.left() + (x / self.p_f).clamp(0.0_f32, 1.0_f32) * self.plot_rect.width(),
            self.plot_rect.bottom()
                - (y / self.p_f).clamp(0.0_f32, 1.0_f32) * self.plot_rect.height(),
        )
    }
}

/// Renders the interactive $\mathbb{F}_p \times \mathbb{F}_p$ modular torus lattice canvas.
pub fn render_torus_canvas(
    ui: &mut egui::Ui,
    model: &TorusCanvasModel<'_>,
    theory: &impl TheoryDescribable,
) -> TorusCanvasOutcome {
    let desired = ui.available_size().max(Vec2::new(220.0_f32, 240.0_f32));
    let (raw_resp, painter) = ui.allocate_painter(desired, Sense::click_and_drag());
    register_canvas_accessibility(
        ui,
        &raw_resp,
        egui::accesskit::Role::Canvas,
        "Elliptic curve F_p torus lattice",
        Some(model.status_summary),
    );
    let response = raw_resp.accessible_theory_hover(theory);
    lock_canvas_arrow_keys(ui, &response);

    let palette = SemanticPalette::from_visuals(ui.visuals());
    let vp = TorusViewport::from_canvas(response.rect, model.p);
    paint_torus_backdrop(ui, &painter, &response, &vp, model, &palette);
    if model.is_singular {
        paint_singular_guardrail(&painter, &vp, model, &palette, ui.visuals().text_color());
    } else {
        paint_curve_elements(&painter, &vp, model, &palette, ui.visuals().text_color());
    }
    paint_canvas_focus_ring(ui, &response);
    collect_torus_interactions(ui, &response, &vp, model.points)
}

fn paint_torus_backdrop(
    ui: &egui::Ui,
    painter: &egui::Painter,
    response: &egui::Response,
    vp: &TorusViewport,
    model: &TorusCanvasModel<'_>,
    palette: &SemanticPalette,
) {
    painter.rect_filled(response.rect, 4.0_f32, ui.visuals().extreme_bg_color);
    painter.rect_stroke(
        vp.plot_rect,
        2.0_f32,
        Stroke::new(1.2_f32, palette.graph_edge),
        egui::StrokeKind::Inside,
    );
    let step = match model.p {
        ..=31 => 1,
        32..=101 => 5,
        102..=307 => 25,
        _ => 100,
    };
    let grid_stroke = Stroke::new(0.6_f32, palette.surface_fill);
    for idx in (step..model.p).step_by(step as usize) {
        let f = idx as f32;
        painter.line_segment(
            [vp.to_screen(f, 0.0_f32), vp.to_screen(f, vp.p_f)],
            grid_stroke,
        );
        painter.line_segment(
            [vp.to_screen(0.0_f32, f), vp.to_screen(vp.p_f, f)],
            grid_stroke,
        );
    }
    let mid_y = vp.p_f * 0.5_f32;
    let mid_stroke = Stroke::new(0.8_f32, palette.muted_text.gamma_multiply(0.35_f32));
    painter.line_segment(
        [vp.to_screen(0.0_f32, mid_y), vp.to_screen(vp.p_f, mid_y)],
        mid_stroke,
    );
    let hdr_pos = Pos2::new(
        response.rect.left() + 10.0_f32,
        response.rect.top() + 8.0_f32,
    );
    let hdr = format!(
        "Torus F_{0} x F_{0}  (y^2 = x^3 + {1}x + {2})",
        model.p, model.a, model.b
    );
    let text_col = ui.visuals().text_color();
    painter.text(
        hdr_pos,
        egui::Align2::LEFT_TOP,
        hdr,
        egui::FontId::proportional(13.0_f32),
        text_col,
    );
    paint_identity_badge(painter, vp, model, palette, text_col);
}

fn paint_identity_badge(
    painter: &egui::Painter,
    vp: &TorusViewport,
    model: &TorusCanvasModel<'_>,
    palette: &SemanticPalette,
    text_col: Color32,
) {
    let inf_stroke = if model.point_p.is_infinity() || model.point_q.is_infinity() {
        palette.accent
    } else {
        palette.graph_edge
    };
    painter.rect_filled(vp.infinity_rect, 4.0_f32, palette.surface_fill);
    painter.rect_stroke(
        vp.infinity_rect,
        4.0_f32,
        Stroke::new(1.4_f32, inf_stroke),
        egui::StrokeKind::Inside,
    );
    painter.text(
        vp.infinity_rect.center(),
        egui::Align2::CENTER_CENTER,
        "Identity O",
        egui::FontId::proportional(11.5_f32),
        text_col,
    );
}

fn paint_singular_guardrail(
    painter: &egui::Painter,
    vp: &TorusViewport,
    model: &TorusCanvasModel<'_>,
    palette: &SemanticPalette,
    text_color: Color32,
) {
    let banner = vp
        .plot_rect
        .shrink2(Vec2::new(16.0_f32, vp.plot_rect.height() * 0.26_f32));
    painter.rect_filled(banner, 6.0_f32, palette.error_banner_bg);
    painter.rect_stroke(
        banner,
        6.0_f32,
        Stroke::new(2.0_f32, palette.error_banner_stroke),
        egui::StrokeKind::Inside,
    );
    painter.text(
        Pos2::new(banner.center().x, banner.top() + 18.0_f32),
        egui::Align2::CENTER_TOP,
        format!("Singular Curve over F_{}", model.p),
        egui::FontId::proportional(15.0_f32),
        palette.error,
    );
    painter.text(
        banner.center(),
        egui::Align2::CENTER_CENTER,
        format!(
            "4a^3 + 27b^2 = 0 (mod {}) for a = {}, b = {}.\nChord/tangent slope division blocked; Hasse chart & exports stay active.",
            model.p, model.a, model.b
        ),
        egui::FontId::proportional(12.0_f32),
        text_color,
    );
}

fn paint_curve_elements(
    painter: &egui::Painter,
    vp: &TorusViewport,
    model: &TorusCanvasModel<'_>,
    palette: &SemanticPalette,
    text_color: Color32,
) {
    let r = if model.p <= 47 {
        3.8_f32
    } else if model.p <= 199 {
        2.7_f32
    } else {
        1.9_f32
    };
    for pt in model.points {
        if let Some((x, y)) = pt.coordinates() {
            painter.circle_filled(vp.to_screen(x as f32, y as f32), r, palette.info);
        }
    }
    let orbit_take = (model.scalar_k.max(0) as usize).min(model.orbit.len());
    for pt in model.orbit.iter().take(orbit_take) {
        if let Some((x, y)) = pt.coordinates() {
            painter.circle_stroke(
                vp.to_screen(x as f32, y as f32),
                r + 2.4_f32,
                Stroke::new(1.3_f32, palette.series[4]),
            );
        }
    }
    for &(a_pt, b_pt) in model.ray_segments {
        painter.line_segment(
            [
                vp.to_screen(a_pt[0], a_pt[1]),
                vp.to_screen(b_pt[0], b_pt[1]),
            ],
            Stroke::new(2.0_f32, palette.warning),
        );
    }
    if let Some(step) = model.chord {
        paint_reflection_and_sum(painter, vp, &step, palette, text_color);
    }
    for (pt, label, col, rad) in [
        (model.point_p, "P", palette.accent, 6.2_f32),
        (model.point_q, "Q", palette.series[1], 5.6_f32),
        (model.scalar_kp, "[k]P", palette.series[4], 5.2_f32),
    ] {
        paint_labeled_point(painter, vp, pt, label, col, text_color, rad);
    }
}

fn paint_reflection_and_sum(
    painter: &egui::Painter,
    vp: &TorusViewport,
    step: &ChordStepFp,
    palette: &SemanticPalette,
    text_color: Color32,
) {
    if let (Some((nx, ny)), Some((sx, sy))) = (step.neg_sum.coordinates(), step.sum.coordinates()) {
        let (p_neg, p_sum) = (
            vp.to_screen(nx as f32, ny as f32),
            vp.to_screen(sx as f32, sy as f32),
        );
        for idx in (0..9_usize).step_by(2) {
            painter.line_segment(
                [
                    p_neg.lerp(p_sum, idx as f32 / 9.0_f32),
                    p_neg.lerp(p_sum, (idx + 1) as f32 / 9.0_f32),
                ],
                Stroke::new(1.6_f32, palette.success),
            );
        }
        paint_labeled_point(
            painter,
            vp,
            step.neg_sum,
            "-R",
            palette.muted_text,
            text_color,
            4.8_f32,
        );
    }
    paint_labeled_point(
        painter,
        vp,
        step.sum,
        "P+Q",
        palette.success,
        text_color,
        6.5_f32,
    );
}

fn paint_labeled_point(
    painter: &egui::Painter,
    vp: &TorusViewport,
    point: AffinePointFp,
    label: &str,
    color: Color32,
    text_color: Color32,
    radius: f32,
) {
    let pos = match point {
        AffinePointFp::Infinity => Pos2::new(
            vp.infinity_rect.left() - 6.0_f32,
            vp.infinity_rect.center().y,
        ),
        AffinePointFp::Finite { x, y } => vp.to_screen(x as f32, y as f32),
    };
    painter.circle_filled(pos, radius, color);
    painter.circle_stroke(pos, radius + 1.2_f32, Stroke::new(1.2_f32, text_color));
    painter.text(
        Pos2::new(pos.x + radius + 3.0_f32, pos.y - radius - 2.0_f32),
        egui::Align2::LEFT_BOTTOM,
        label,
        egui::FontId::proportional(11.5_f32),
        color,
    );
}

fn collect_torus_interactions(
    ui: &egui::Ui,
    response: &egui::Response,
    vp: &TorusViewport,
    points: &[AffinePointFp],
) -> TorusCanvasOutcome {
    let mut outcome = TorusCanvasOutcome::default();
    if response.clicked() {
        response.request_focus();
        if let Some(pos) = response.interact_pointer_pos() {
            outcome.clicked_point_index = find_nearest_point(pos, vp, points);
        }
    }
    if response.has_focus() {
        outcome.focused_canvas_id = Some(response.id);
        ui.input(|i| {
            if i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::ArrowUp) {
                outcome.arrow_delta = 1;
            } else if i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::ArrowDown) {
                outcome.arrow_delta = -1;
            }
            if i.key_pressed(egui::Key::Q) {
                outcome.toggle_selector = true;
            }
        });
    }
    outcome
}

fn find_nearest_point(pos: Pos2, vp: &TorusViewport, points: &[AffinePointFp]) -> Option<usize> {
    if vp.infinity_rect.expand(4.0_f32).contains(pos) && !points.is_empty() {
        return Some(0);
    }
    let mut best: Option<(usize, f32)> = None;
    for (idx, pt) in points.iter().enumerate() {
        if let Some((x, y)) = pt.coordinates() {
            let d2 = vp.to_screen(x as f32, y as f32).distance_sq(pos);
            if d2 <= 324.0_f32 && best.is_none_or(|(_, b)| d2 < b) {
                best = Some((idx, d2));
            }
        }
    }
    best.map(|(idx, _)| idx)
}

// [cite:modular_polynomials_review]
