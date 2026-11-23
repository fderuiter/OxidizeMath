// [cite:hurwitz_kronecker]
// [cite:modular_polynomials_review]

use super::chord_tangent_canvas::{toroidal_secant_segments, CanvasViewport, REAL_VIEW_LIMIT};
use super::chord_tangent_tool::{snap_real_point, CurveFieldMode, WeierstrassPlaygroundTool};
use crate::app::SemanticPalette;
use eframe::egui;
use math_explorer::pure_math::elliptic_curves::{
    CurvePoint, PrimeWeierstrassCurve, WeierstrassCurve,
};

pub(crate) fn draw_real_scene(
    tool: &WeierstrassPlaygroundTool,
    painter: &egui::Painter,
    vp: CanvasViewport,
    palette: &SemanticPalette,
    visuals: &egui::Visuals,
) {
    let axis_stroke = egui::Stroke::new(1.0_f32, visuals.weak_text_color());
    painter.line_segment(
        [vp.to_screen(vp.x_min, 0.0), vp.to_screen(vp.x_max, 0.0)],
        axis_stroke,
    );
    painter.line_segment(
        [vp.to_screen(0.0, vp.y_min), vp.to_screen(0.0, vp.y_max)],
        axis_stroke,
    );
    let Ok(curve) = WeierstrassCurve::new(tool.state.real_a, tool.state.real_b) else {
        return;
    };

    for sign in [1.0_f64, -1.0_f64] {
        let mut run = Vec::new();
        for i in 0..=280 {
            let x = vp.x_min + (i as f64 / 280.0) * (vp.x_max - vp.x_min);
            let rhs = curve.rhs(x);
            if rhs >= 0.0 {
                let y = sign * rhs.sqrt();
                if (vp.y_min..=vp.y_max).contains(&y) {
                    run.push(vp.to_screen(x, y));
                    continue;
                }
            }
            if run.len() >= 2 {
                painter.add(egui::Shape::line(
                    std::mem::take(&mut run),
                    egui::Stroke::new(2.2_f32, palette.accent),
                ));
            } else {
                run.clear();
            }
        }
        if run.len() >= 2 {
            painter.add(egui::Shape::line(
                run,
                egui::Stroke::new(2.2_f32, palette.accent),
            ));
        }
    }
    draw_real_overlays(tool, &curve, painter, vp, palette, visuals);
}

fn draw_real_overlays(
    tool: &WeierstrassPlaygroundTool,
    curve: &WeierstrassCurve,
    painter: &egui::Painter,
    vp: CanvasViewport,
    palette: &SemanticPalette,
    visuals: &egui::Visuals,
) {
    let p = snap_real_point(curve, tool.state.real_px, tool.state.real_p_upper);
    let q = snap_real_point(curve, tool.state.real_qx, tool.state.real_q_upper);
    let sum = curve.add(p, q);
    let neg_sum = curve.negate(sum);

    if tool.state.show_secant {
        if let (Some((x1, y1)), Some(m)) = (p.coords(), curve.secant_or_tangent_slope(p, q)) {
            let ya = y1 + m * (vp.x_min - x1);
            let yb = y1 + m * (vp.x_max - x1);
            painter.line_segment(
                [vp.to_screen(vp.x_min, ya), vp.to_screen(vp.x_max, yb)],
                egui::Stroke::new(1.6_f32, palette.warning),
            );
        }
        if let (Some((rx, ry)), Some((sx, sy))) = (neg_sum.coords(), sum.coords()) {
            painter.line_segment(
                [vp.to_screen(rx, ry), vp.to_screen(sx, sy)],
                egui::Stroke::new(1.4_f32, palette.success),
            );
        }
    }
    if tool.state.show_orbit {
        let mut orbit = vec![CurvePoint::Infinity; tool.state.scalar_k.clamp(1, 64) as usize];
        let written = curve.write_orbit(p, &mut orbit);
        let pts: Vec<egui::Pos2> = orbit[..written]
            .iter()
            .filter_map(|pt| pt.coords())
            .filter(|(x, y)| x.abs() <= 12.0 && y.abs() <= 12.0)
            .map(|(x, y)| vp.to_screen(x, y))
            .collect();
        if pts.len() >= 2 {
            painter.add(egui::Shape::line(
                pts,
                egui::Stroke::new(1.4_f32, visuals.text_color()),
            ));
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
                painter.circle_filled(sc, 5.5_f32, col);
                painter.text(
                    sc + egui::vec2(7.0_f32, -7.0_f32),
                    egui::Align2::LEFT_BOTTOM,
                    lbl,
                    egui::FontId::proportional(13.0_f32),
                    visuals.text_color(),
                );
            }
        }
    }
}

pub(crate) fn draw_fp_scene(
    tool: &WeierstrassPlaygroundTool,
    painter: &egui::Painter,
    vp: CanvasViewport,
    palette: &SemanticPalette,
    visuals: &egui::Visuals,
) {
    let border = egui::Stroke::new(1.2_f32, visuals.weak_text_color());
    painter.rect_stroke(vp.rect, 0.0_f32, border, egui::StrokeKind::Inside);
    let Ok(curve) =
        PrimeWeierstrassCurve::new(tool.state.mod_a, tool.state.mod_b, tool.state.prime_p)
    else {
        return;
    };

    let dot_radius = if curve.p <= 47 { 3.6_f32 } else { 2.0_f32 };
    for pt in &tool.lattice_points {
        if let Some((x, y)) = pt.coords() {
            painter.circle_filled(
                vp.to_screen(x as f64, y as f64),
                dot_radius,
                visuals.weak_text_color(),
            );
        }
    }

    let (p, q) = tool.current_fp_points();
    let sum = curve.add(p, q);
    let neg_sum = curve.negate(sum);

    if tool.state.show_secant {
        if let Some((x1, y1)) = p.coords() {
            if let Some(m) = curve.slope_mod_p(p, q) {
                for (a, b) in toroidal_secant_segments(curve.p, x1, y1, m) {
                    painter.line_segment(
                        [vp.to_screen(a[0], a[1]), vp.to_screen(b[0], b[1])],
                        egui::Stroke::new(1.5_f32, palette.warning),
                    );
                }
            } else {
                let pf = curve.p as f64;
                painter.line_segment(
                    [vp.to_screen(x1 as f64, 0.0), vp.to_screen(x1 as f64, pf)],
                    egui::Stroke::new(1.5_f32, palette.warning),
                );
            }
            if let (Some((rx, ry)), Some((sx, sy))) = (neg_sum.coords(), sum.coords()) {
                painter.line_segment(
                    [
                        vp.to_screen(rx as f64, ry as f64),
                        vp.to_screen(sx as f64, sy as f64),
                    ],
                    egui::Stroke::new(1.5_f32, palette.success),
                );
            }
        }
    }

    draw_fp_orbit_and_markers(tool, &curve, painter, vp, palette, visuals);
}

fn draw_fp_orbit_and_markers(
    tool: &WeierstrassPlaygroundTool,
    curve: &PrimeWeierstrassCurve,
    painter: &egui::Painter,
    vp: CanvasViewport,
    palette: &SemanticPalette,
    visuals: &egui::Visuals,
) {
    let (p, q) = tool.current_fp_points();
    let sum = curve.add(p, q);
    let kp = curve.scalar_mul(tool.state.scalar_k, p);

    if tool.state.show_orbit {
        let mut orbit = vec![CurvePoint::Infinity; tool.state.scalar_k.clamp(1, 128) as usize];
        let written = curve.write_orbit(p, &mut orbit);
        let pts: Vec<egui::Pos2> = orbit[..written]
            .iter()
            .filter_map(|pt| pt.coords())
            .map(|(x, y)| vp.to_screen(x as f64, y as f64))
            .collect();
        if pts.len() >= 2 {
            painter.add(egui::Shape::line(
                pts,
                egui::Stroke::new(1.4_f32, palette.accent),
            ));
        }
    }

    for (pt, col, lbl) in [
        (p, palette.accent, "P"),
        (q, palette.warning, "Q"),
        (sum, palette.success, "P+Q"),
        (kp, palette.error, "[k]P"),
    ] {
        if let Some((x, y)) = pt.coords() {
            let sc = vp.to_screen(x as f64, y as f64);
            painter.circle_filled(sc, 6.0_f32, col);
            painter.text(
                sc + egui::vec2(7.0_f32, -6.0_f32),
                egui::Align2::LEFT_BOTTOM,
                lbl,
                egui::FontId::proportional(13.0_f32),
                visuals.text_color(),
            );
        }
    }
}

pub(crate) fn build_export_datasets(
    tool: &WeierstrassPlaygroundTool,
) -> Vec<(String, Vec<[f64; 2]>)> {
    match tool.state.field_mode {
        CurveFieldMode::Real => {
            let Ok(curve) = WeierstrassCurve::new(tool.state.real_a, tool.state.real_b) else {
                return Vec::new();
            };
            let mut branch = Vec::new();
            for i in 0..=160 {
                let x = -REAL_VIEW_LIMIT + (i as f64 / 160.0) * (2.0 * REAL_VIEW_LIMIT);
                let rhs = curve.rhs(x);
                if rhs >= 0.0 {
                    branch.push([x, rhs.sqrt()]);
                }
            }
            let p = snap_real_point(&curve, tool.state.real_px, tool.state.real_p_upper);
            let mut orbit_buf =
                vec![CurvePoint::Infinity; tool.state.scalar_k.clamp(1, 64) as usize];
            let n = curve.write_orbit(p, &mut orbit_buf);
            let orbit: Vec<[f64; 2]> = orbit_buf[..n]
                .iter()
                .filter_map(|pt| pt.coords())
                .map(|(x, y)| [x, y])
                .collect();
            vec![
                ("Real Curve Upper Branch".into(), branch),
                ("Subgroup Orbit [k]P".into(), orbit),
            ]
        }
        CurveFieldMode::FiniteField => {
            let lattice: Vec<[f64; 2]> = tool
                .lattice_points
                .iter()
                .filter_map(|pt| pt.coords())
                .map(|(x, y)| [x as f64, y as f64])
                .collect();
            let mut out = vec![("Affine Lattice E(F_p)".into(), lattice)];
            if let Ok(curve) =
                PrimeWeierstrassCurve::new(tool.state.mod_a, tool.state.mod_b, tool.state.prime_p)
            {
                let (p, q) = tool.current_fp_points();
                if let (Some((x1, y1)), Some(m)) = (p.coords(), curve.slope_mod_p(p, q)) {
                    let mut secant_pts = Vec::new();
                    for (a, b) in toroidal_secant_segments(curve.p, x1, y1, m) {
                        secant_pts.push(a);
                        secant_pts.push(b);
                    }
                    out.push(("Toroidal Secant Wrap".into(), secant_pts));
                }
                let mut orbit_buf =
                    vec![CurvePoint::Infinity; tool.state.scalar_k.clamp(1, 128) as usize];
                let n = curve.write_orbit(p, &mut orbit_buf);
                let orbit: Vec<[f64; 2]> = orbit_buf[..n]
                    .iter()
                    .filter_map(|pt| pt.coords())
                    .map(|(x, y)| [x as f64, y as f64])
                    .collect();
                out.push(("Cyclic Orbit [k]P".into(), orbit));
            }
            out
        }
    }
}
