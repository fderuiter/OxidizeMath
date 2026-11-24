#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// [cite:hurwitz_kronecker]
// [cite:modular_polynomials_review]
//! Real and $\mathbb{F}_p$ singular locus (`Cusp`, `Crunode`, `Acnode`) canvas renderers and exports.

use super::chord_tangent_canvas::{toroidal_secant_segments, CanvasViewport, REAL_VIEW_LIMIT};
use super::chord_tangent_tool::WeierstrassPlaygroundTool;
use crate::app::SemanticPalette;
use eframe::egui;
use math_explorer::pure_math::elliptic_curves::{
    CurvePoint, FpSingularCubic, RealSingularCubic, SingularityKind,
};

/// Snaps a target $x$-coordinate onto a real singular cubic (including isolated acnode vertices).
#[must_use]
pub(crate) fn snap_real_singular_point(
    cubic: &RealSingularCubic,
    target_x: f64,
    upper: bool,
) -> CurvePoint<f64> {
    let clamped = target_x.clamp(-5.0, 5.0);
    if cubic.kind == SingularityKind::Acnode {
        let dist_node = (clamped - cubic.singular_x).abs();
        let dist_branch = (cubic.branch_min_x - clamped).max(0.0);
        if clamped < cubic.branch_min_x && dist_node <= dist_branch {
            return CurvePoint::Affine {
                x: cubic.singular_x,
                y: 0.0,
            };
        }
    }
    let x = clamped.max(cubic.branch_min_x);
    let rhs = cubic.rhs(x).max(0.0);
    let y = rhs.sqrt();
    CurvePoint::Affine {
        x,
        y: if y == 0.0 {
            0.0
        } else if upper {
            y
        } else {
            -y
        },
    }
}

fn draw_cubic_segment(
    cubic: &RealSingularCubic,
    x_start: f64,
    x_end: f64,
    sign: f64,
    painter: &egui::Painter,
    vp: CanvasViewport,
    color: egui::Color32,
) {
    if x_end <= x_start {
        return;
    }
    let span = x_end - x_start;
    let mut pts = Vec::with_capacity(161);
    for i in 0..=160 {
        let u = i as f64 / 160.0;
        let x = x_start + u * u * span;
        let rhs = cubic.rhs(x);
        if rhs >= -1e-12 {
            let y = sign * rhs.max(0.0).sqrt();
            if (vp.y_min - 2.0..=vp.y_max + 2.0).contains(&y) {
                pts.push(vp.to_screen(x, y));
            }
        }
    }
    if pts.len() >= 2 {
        painter.add(egui::Shape::line(pts, egui::Stroke::new(2.3_f32, color)));
    }
}

fn paint_singular_vertex_badge(
    painter: &egui::Painter,
    screen_pt: egui::Pos2,
    label: &str,
    palette: &SemanticPalette,
    text_color: egui::Color32,
) {
    painter.circle_filled(screen_pt, 6.2_f32, palette.error);
    painter.circle_stroke(
        screen_pt,
        9.4_f32,
        egui::Stroke::new(1.8_f32, palette.warning),
    );
    painter.text(
        screen_pt + egui::vec2(10.0_f32, -10.0_f32),
        egui::Align2::LEFT_BOTTOM,
        label,
        egui::FontId::proportional(12.5_f32),
        text_color,
    );
}

/// Renders a real singular cubic (`Cusp`, `Crunode`, or `Acnode`) and smooth-locus secant overlays.
pub(crate) fn draw_real_singular_scene(
    tool: &WeierstrassPlaygroundTool,
    cubic: &RealSingularCubic,
    painter: &egui::Painter,
    vp: CanvasViewport,
    palette: &SemanticPalette,
    visuals: &egui::Visuals,
) {
    draw_real_singular_branches(cubic, painter, vp, palette);
    if (vp.x_min..=vp.x_max).contains(&cubic.singular_x) {
        let badge = format!("{} ({:.2}, 0)", cubic.kind.label(), cubic.singular_x);
        paint_singular_vertex_badge(
            painter,
            vp.to_screen(cubic.singular_x, 0.0),
            &badge,
            palette,
            visuals.text_color(),
        );
    }
    draw_real_singular_secant_and_points(tool, cubic, painter, vp, palette, visuals);
}

fn draw_real_singular_branches(
    cubic: &RealSingularCubic,
    painter: &egui::Painter,
    vp: CanvasViewport,
    palette: &SemanticPalette,
) {
    for sign in [1.0_f64, -1.0_f64] {
        match cubic.kind {
            SingularityKind::Cusp => {
                draw_cubic_segment(
                    cubic,
                    0.0,
                    vp.x_max.max(0.1),
                    sign,
                    painter,
                    vp,
                    palette.accent,
                );
            }
            SingularityKind::Crunode => {
                let x0 = cubic.branch_min_x.max(vp.x_min);
                let x2 = vp.x_max.max(cubic.singular_x + 0.1);
                draw_cubic_segment(
                    cubic,
                    x0,
                    cubic.singular_x,
                    sign,
                    painter,
                    vp,
                    palette.accent,
                );
                draw_cubic_segment(
                    cubic,
                    cubic.singular_x,
                    x2,
                    sign,
                    painter,
                    vp,
                    palette.accent,
                );
            }
            SingularityKind::Acnode => {
                let x2 = vp.x_max.max(cubic.branch_min_x + 0.1);
                draw_cubic_segment(
                    cubic,
                    cubic.branch_min_x,
                    x2,
                    sign,
                    painter,
                    vp,
                    palette.accent,
                );
            }
        }
    }
    draw_crunode_tangent_cross(cubic, painter, vp, palette);
}

fn draw_crunode_tangent_cross(
    cubic: &RealSingularCubic,
    painter: &egui::Painter,
    vp: CanvasViewport,
    palette: &SemanticPalette,
) {
    if cubic.kind == SingularityKind::Crunode && cubic.tangent_slope_abs > 0.0 {
        let m = cubic.tangent_slope_abs;
        for sgn in [1.0_f64, -1.0_f64] {
            painter.line_segment(
                [
                    vp.to_screen(cubic.singular_x - 0.65, -sgn * m * 0.65),
                    vp.to_screen(cubic.singular_x + 0.65, sgn * m * 0.65),
                ],
                egui::Stroke::new(1.1_f32, palette.muted_text),
            );
        }
    }
}

fn draw_real_singular_secant_and_points(
    tool: &WeierstrassPlaygroundTool,
    cubic: &RealSingularCubic,
    painter: &egui::Painter,
    vp: CanvasViewport,
    palette: &SemanticPalette,
    visuals: &egui::Visuals,
) {
    let p = snap_real_singular_point(cubic, tool.state.real_px, tool.state.real_p_upper);
    let q = snap_real_singular_point(cubic, tool.state.real_qx, tool.state.real_q_upper);
    let sum = cubic.add_smooth(p, q).unwrap_or(CurvePoint::Infinity);
    let neg_sum = cubic.negate(sum);

    if tool.state.show_secant {
        if let (Some((x1, y1)), Some(m)) = (p.coords(), cubic.secant_or_tangent_slope(p, q)) {
            let ya = y1 + m * (vp.x_min - x1);
            let yb = y1 + m * (vp.x_max - x1);
            painter.line_segment(
                [vp.to_screen(vp.x_min, ya), vp.to_screen(vp.x_max, yb)],
                egui::Stroke::new(1.6_f32, palette.warning),
            );
            if let (Some((rx, ry)), Some((sx, sy))) = (neg_sum.coords(), sum.coords()) {
                painter.line_segment(
                    [vp.to_screen(rx, ry), vp.to_screen(sx, sy)],
                    egui::Stroke::new(1.4_f32, palette.success),
                );
            }
        }
    }
    for (pt, col, lbl) in [
        (p, palette.accent, "P"),
        (q, palette.warning, "Q"),
        (sum, palette.success, "P+Q"),
    ] {
        if let Some((x, y)) = pt.coords() {
            if (vp.x_min..=vp.x_max).contains(&x) && (vp.y_min..=vp.y_max).contains(&y) {
                let sc = vp.to_screen(x, y);
                painter.circle_filled(sc, 5.2_f32, col);
                painter.text(
                    sc + egui::vec2(7.0_f32, 8.0_f32),
                    egui::Align2::LEFT_TOP,
                    lbl,
                    egui::FontId::proportional(12.5_f32),
                    visuals.text_color(),
                );
            }
        }
    }
}

/// Renders a singular $\mathbb{F}_p$ curve on `WeierstrassPlaygroundTool`, highlighting $(x_s, 0)$.
pub(crate) fn draw_fp_singular_scene(
    tool: &WeierstrassPlaygroundTool,
    sing: &FpSingularCubic,
    painter: &egui::Painter,
    vp: CanvasViewport,
    palette: &SemanticPalette,
    visuals: &egui::Visuals,
) {
    let dot_radius = if sing.p <= 47 { 3.6_f32 } else { 2.0_f32 };
    for pt in &tool.lattice_points {
        if let Some((x, y)) = pt.coords() {
            painter.circle_filled(vp.to_screen(x as f64, y as f64), dot_radius, palette.info);
        }
    }

    let (p, q) = tool.current_fp_points();
    let sum = sing.add_smooth(p, q).unwrap_or(CurvePoint::Infinity);

    if tool.state.show_secant {
        if let (Some((x1, y1)), Some(m)) = (p.coords(), sing.slope_mod_p(p, q)) {
            for (a, b) in toroidal_secant_segments(sing.p, x1, y1, m) {
                painter.line_segment(
                    [vp.to_screen(a[0], a[1]), vp.to_screen(b[0], b[1])],
                    egui::Stroke::new(1.5_f32, palette.warning),
                );
            }
        }
    }
    for (pt, col, lbl) in [
        (p, palette.accent, "P"),
        (q, palette.warning, "Q"),
        (sum, palette.success, "P+Q"),
    ] {
        if let Some((x, y)) = pt.coords() {
            let sc = vp.to_screen(x as f64, y as f64);
            painter.circle_filled(sc, 5.8_f32, col);
            painter.text(
                sc + egui::vec2(7.0_f32, -6.0_f32),
                egui::Align2::LEFT_BOTTOM,
                lbl,
                egui::FontId::proportional(13.0_f32),
                visuals.text_color(),
            );
        }
    }
    let badge = format!("{} ({}, 0)", sing.kind.label(), sing.singular_x);
    paint_singular_vertex_badge(
        painter,
        vp.to_screen(sing.singular_x as f64, 0.0),
        &badge,
        palette,
        visuals.text_color(),
    );
}

/// Builds non-empty CSV/JSON datasets for real singular cubics (`Cusp`, `Crunode`, `Acnode`).
#[must_use]
pub(crate) fn build_real_singular_datasets(
    cubic: &RealSingularCubic,
) -> Vec<(String, Vec<[f64; 2]>)> {
    let mut upper = Vec::with_capacity(162);
    let mut lower = Vec::with_capacity(162);
    let x_start = cubic.branch_min_x.max(-REAL_VIEW_LIMIT);
    let x_end = REAL_VIEW_LIMIT.max(x_start + 1.0);
    for i in 0..=160 {
        let u = i as f64 / 160.0;
        let x = x_start + u * (x_end - x_start);
        let rhs = cubic.rhs(x);
        if rhs >= -1e-12 {
            let y = rhs.max(0.0).sqrt();
            upper.push([x, y]);
            lower.push([x, -y]);
        }
    }
    let mut datasets = vec![
        (format!("Real {} (+y)", cubic.kind.label()), upper),
        (format!("Real {} (-y)", cubic.kind.label()), lower),
        (
            format!("Singular {} Vertex (x_s, 0)", cubic.kind.label()),
            vec![[cubic.singular_x, 0.0]],
        ),
    ];
    if cubic.kind == math_explorer::pure_math::elliptic_curves::SingularityKind::Crunode
        && cubic.tangent_slope_abs > 0.0
    {
        let m = cubic.tangent_slope_abs;
        let xs = cubic.singular_x;
        let (x0, x1) = (
            (xs - 2.2).max(-REAL_VIEW_LIMIT),
            (xs + 2.2).min(REAL_VIEW_LIMIT),
        );
        datasets.push((
            "Crunode Tangent (+y)".to_string(),
            vec![[x0, m * (x0 - xs)], [x1, m * (x1 - xs)]],
        ));
        datasets.push((
            "Crunode Tangent (-y)".to_string(),
            vec![[x0, -m * (x0 - xs)], [x1, -m * (x1 - xs)]],
        ));
    }
    datasets
}

/// Highlights the singular vertex `(x_s, 0)` and compact header badge on the Number Theory $\mathbb{F}_p$ torus.
pub(crate) fn paint_nt_torus_singular_overlay(
    painter: &egui::Painter,
    singular_screen_pos: impl FnOnce(f32, f32) -> egui::Pos2,
    plot_rect: egui::Rect,
    curve_params: (i64, i64, i64),
    palette: &SemanticPalette,
    text_color: egui::Color32,
) {
    let (p, a, b) = curve_params;
    let Ok(Some(sing)) = FpSingularCubic::classify(a, b, p) else {
        return;
    };
    let pt = singular_screen_pos(sing.singular_x as f32, 0.0_f32);
    let label = format!("{} ({}, 0)", sing.kind.label(), sing.singular_x);
    paint_singular_vertex_badge(painter, pt, &label, palette, text_color);

    let pill = egui::Rect::from_min_size(
        egui::pos2(plot_rect.left() + 8.0_f32, plot_rect.top() + 6.0_f32),
        egui::vec2((plot_rect.width() - 16.0_f32).min(310.0_f32), 22.0_f32),
    );
    painter.rect_filled(pill, 4.0_f32, palette.error_banner_bg);
    painter.rect_stroke(
        pill,
        4.0_f32,
        egui::Stroke::new(1.2_f32, palette.error_banner_stroke),
        egui::StrokeKind::Inside,
    );
    painter.text(
        pill.center(),
        egui::Align2::CENTER_CENTER,
        format!(
            "Singular {} over F_{}: vertex ({}, 0), {} affine pts",
            sing.kind.label(),
            sing.p,
            sing.singular_x,
            sing.affine_point_count()
        ),
        egui::FontId::proportional(11.5_f32),
        text_color,
    );
}
