// [cite:hurwitz_kronecker]
// [cite:modular_polynomials_review]

use super::chord_tangent_scene::{draw_fp_scene, draw_real_scene};
use super::chord_tangent_tool::{
    snap_real_point, ClickTarget, CurveFieldMode, WeierstrassPlaygroundTool,
};
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::commands::{CommandRegistryData, CommandTrigger};
use math_explorer::pure_math::elliptic_curves::{
    CurvePoint, PrimeWeierstrassCurve, WeierstrassCurve,
};

pub(crate) const REAL_VIEW_LIMIT: f64 = 4.5;

pub(crate) fn fmt_real_pt(pt: CurvePoint<f64>) -> String {
    match pt {
        CurvePoint::Infinity => "O (Infinity)".into(),
        CurvePoint::Affine { x, y } => format!("({x:.2}, {y:.2})"),
    }
}

pub(crate) fn fmt_fp_pt(pt: CurvePoint<i64>) -> String {
    match pt {
        CurvePoint::Infinity => "O (Infinity)".into(),
        CurvePoint::Affine { x, y } => format!("({x}, {y})"),
    }
}

/// Computes toroidal line segments wrapping $y \equiv m(x - x_1) + y_1 \pmod p$ across $\[0, p\]^2$.
pub(crate) fn toroidal_secant_segments(
    p: i64,
    x1: i64,
    y1: i64,
    slope: i64,
) -> Vec<([f64; 2], [f64; 2])> {
    let pf = p as f64;
    let signed_m = if slope > p / 2 { slope - p } else { slope } as f64;
    let (x1f, y1f) = (x1 as f64, y1 as f64);
    if signed_m.abs() <= 1e-12 {
        return vec![([0.0, y1f], [pf, y1f])];
    }
    let y_at_0 = y1f - signed_m * x1f;
    let y_at_p = y1f + signed_m * (pf - x1f);
    let w_min = (y_at_0.min(y_at_p) / pf).floor() as i64;
    let w_max = (y_at_0.max(y_at_p) / pf).floor() as i64;
    let (w_lo, w_hi) = if w_max - w_min > 24 {
        (-12, 12)
    } else {
        (w_min, w_max)
    };

    let mut segs = Vec::new();
    for w in w_lo..=w_hi {
        let wf = (w as f64) * pf;
        let xa = (wf - y1f) / signed_m + x1f;
        let xb = (wf + pf - y1f) / signed_m + x1f;
        let x_lo = xa.min(xb).max(0.0);
        let x_hi = xa.max(xb).min(pf);
        if x_hi - x_lo > 1e-9 {
            let y_lo = (y1f + signed_m * (x_lo - x1f) - wf).clamp(0.0, pf);
            let y_hi = (y1f + signed_m * (x_hi - x1f) - wf).clamp(0.0, pf);
            segs.push(([x_lo, y_lo], [x_hi, y_hi]));
        }
    }
    segs
}

#[derive(Clone, Copy)]
pub(crate) struct CanvasViewport {
    pub(crate) rect: egui::Rect,
    pub(crate) x_min: f64,
    pub(crate) x_max: f64,
    pub(crate) y_min: f64,
    pub(crate) y_max: f64,
}

impl CanvasViewport {
    pub(crate) fn new(outer: egui::Rect, x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        Self {
            rect: outer.shrink(22.0_f32),
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    pub(crate) fn to_screen(self, x: f64, y: f64) -> egui::Pos2 {
        let tx = ((x - self.x_min) / (self.x_max - self.x_min).max(1e-9)) as f32;
        let ty = ((y - self.y_min) / (self.y_max - self.y_min).max(1e-9)) as f32;
        egui::pos2(
            self.rect.left() + tx * self.rect.width(),
            self.rect.bottom() - ty * self.rect.height(),
        )
    }

    pub(crate) fn to_math(self, pos: egui::Pos2) -> (f64, f64) {
        let tx =
            ((pos.x - self.rect.left()) / self.rect.width().max(1.0_f32)).clamp(0.0_f32, 1.0_f32);
        let ty = ((self.rect.bottom() - pos.y) / self.rect.height().max(1.0_f32))
            .clamp(0.0_f32, 1.0_f32);
        (
            self.x_min + f64::from(tx) * (self.x_max - self.x_min),
            self.y_min + f64::from(ty) * (self.y_max - self.y_min),
        )
    }
}

pub(crate) fn render_interactive_canvas(tool: &mut WeierstrassPlaygroundTool, ui: &mut egui::Ui) {
    let avail = ui.available_size().max(egui::vec2(160.0_f32, 160.0_f32));
    let (response, painter) = ui.allocate_painter(avail, egui::Sense::click_and_drag());
    let _ = response
        .clone()
        .accessible_hover_text("Weierstrass Elliptic Curve Canvas");
    tool.focused_canvas = response.has_focus().then_some(response.id);

    register_canvas_shortcuts(ui.ctx());
    lock_canvas_arrow_keys(ui, &response);
    if response.has_focus() {
        handle_canvas_keyboard(tool, ui);
    }

    let vp = match tool.state.field_mode {
        CurveFieldMode::Real => CanvasViewport::new(
            response.rect,
            -REAL_VIEW_LIMIT,
            REAL_VIEW_LIMIT,
            -REAL_VIEW_LIMIT,
            REAL_VIEW_LIMIT,
        ),
        CurveFieldMode::FiniteField => {
            let pf = tool.state.prime_p.max(5) as f64;
            CanvasViewport::new(response.rect, 0.0, pf, 0.0, pf)
        }
    };

    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            response.request_focus();
            handle_canvas_click(tool, ui.ctx(), vp, pos, ui.input(|i| i.modifiers.shift));
        }
    }

    let summary = canvas_a11y_summary(tool);
    register_canvas_accessibility(
        ui,
        &response,
        egui::accesskit::Role::GraphicsDocument,
        "Weierstrass Elliptic Curve Canvas",
        Some(&summary),
    );

    let palette = SemanticPalette::from_visuals(ui.visuals());
    painter.rect_filled(response.rect, 4.0_f32, ui.visuals().extreme_bg_color);
    match tool.state.field_mode {
        CurveFieldMode::Real => draw_real_scene(tool, &painter, vp, &palette, ui.visuals()),
        CurveFieldMode::FiniteField => draw_fp_scene(tool, &painter, vp, &palette, ui.visuals()),
    }
    paint_canvas_focus_ring(ui, &response);
}

fn register_canvas_shortcuts(ctx: &egui::Context) {
    CommandRegistryData::register(
        ctx,
        "Step Point P Left",
        "Move curve point P backward along the curve or lattice",
        CommandTrigger::Key(egui::Key::ArrowLeft),
        true,
        "Elliptic Curve Canvas",
    );
    CommandRegistryData::register(
        ctx,
        "Step Point P Right",
        "Move curve point P forward along the curve or lattice",
        CommandTrigger::Key(egui::Key::ArrowRight),
        true,
        "Elliptic Curve Canvas",
    );
    CommandRegistryData::register(
        ctx,
        "Cycle Second Point Q",
        "Advance curve point Q along the curve or lattice",
        CommandTrigger::Key(egui::Key::ArrowUp),
        true,
        "Elliptic Curve Canvas",
    );
}

fn handle_canvas_keyboard(tool: &mut WeierstrassPlaygroundTool, ui: &mut egui::Ui) {
    let left = ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft));
    let right = ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight));
    let up = ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp));
    let down = ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown));
    if !(left || right || up || down) {
        return;
    }

    match tool.state.field_mode {
        CurveFieldMode::Real => {
            if left {
                tool.state.real_px = (tool.state.real_px - 0.2).clamp(-4.0, 4.0);
            } else if right {
                tool.state.real_px = (tool.state.real_px + 0.2).clamp(-4.0, 4.0);
            } else if up || down {
                tool.state.real_qx =
                    (tool.state.real_qx + if up { 0.2 } else { -0.2 }).clamp(-4.0, 4.0);
            }
        }
        CurveFieldMode::FiniteField if !tool.lattice_points.is_empty() => {
            let len = tool.lattice_points.len();
            if left {
                tool.state.selected_p_idx = (tool.state.selected_p_idx + len - 1) % len;
            } else if right {
                tool.state.selected_p_idx = (tool.state.selected_p_idx + 1) % len;
            } else if up {
                tool.state.selected_q_idx = (tool.state.selected_q_idx + 1) % len;
            } else if down {
                tool.state.selected_q_idx = (tool.state.selected_q_idx + len - 1) % len;
            }
        }
        CurveFieldMode::FiniteField => {}
    }
    tool.bump_revision();
    announce_live_status(ui.ctx(), canvas_a11y_summary(tool));
}

fn handle_canvas_click(
    tool: &mut WeierstrassPlaygroundTool,
    ctx: &egui::Context,
    vp: CanvasViewport,
    pos: egui::Pos2,
    shift_held: bool,
) {
    let (mx, my) = vp.to_math(pos);
    let set_q = shift_held || tool.click_target == ClickTarget::PointQ;
    match tool.state.field_mode {
        CurveFieldMode::Real => {
            if set_q {
                tool.state.real_qx = mx.clamp(-4.0, 4.0);
                tool.state.real_q_upper = my >= 0.0;
            } else {
                tool.state.real_px = mx.clamp(-4.0, 4.0);
                tool.state.real_p_upper = my >= 0.0;
            }
        }
        CurveFieldMode::FiniteField if !tool.lattice_points.is_empty() => {
            let best_idx = tool
                .lattice_points
                .iter()
                .enumerate()
                .filter_map(|(idx, pt)| pt.coords().map(|(px, py)| (idx, px as f64, py as f64)))
                .min_by(|a, b| {
                    let da = (a.1 - mx).powi(2) + (a.2 - my).powi(2);
                    let db = (b.1 - mx).powi(2) + (b.2 - my).powi(2);
                    da.total_cmp(&db)
                })
                .map_or(0, |(idx, _, _)| idx);
            if set_q {
                tool.state.selected_q_idx = best_idx;
            } else {
                tool.state.selected_p_idx = best_idx;
            }
        }
        CurveFieldMode::FiniteField => {}
    }
    tool.bump_revision();
    announce_live_status(ctx, canvas_a11y_summary(tool));
}

fn canvas_a11y_summary(tool: &WeierstrassPlaygroundTool) -> String {
    match tool.state.field_mode {
        CurveFieldMode::Real => match WeierstrassCurve::new(tool.state.real_a, tool.state.real_b) {
            Ok(curve) => {
                let p = snap_real_point(&curve, tool.state.real_px, tool.state.real_p_upper);
                let q = snap_real_point(&curve, tool.state.real_qx, tool.state.real_q_upper);
                let sum = curve.add(p, q);
                format!(
                    "Real curve y^2 = x^3 + {:.2}x + {:.2}; P = {}, Q = {}, P + Q = {}",
                    curve.a,
                    curve.b,
                    fmt_real_pt(p),
                    fmt_real_pt(q),
                    fmt_real_pt(sum)
                )
            }
            Err(err) => err.to_string(),
        },
        CurveFieldMode::FiniteField => {
            match PrimeWeierstrassCurve::new(tool.state.mod_a, tool.state.mod_b, tool.state.prime_p)
            {
                Ok(curve) => {
                    let (p, q) = tool.current_fp_points();
                    let sum = curve.add(p, q);
                    let kp = curve.scalar_mul(tool.state.scalar_k, p);
                    format!(
                        "Torus F_{}: P = {}, Q = {}, P + Q = {}, [{}]P = {}",
                        curve.p,
                        fmt_fp_pt(p),
                        fmt_fp_pt(q),
                        fmt_fp_pt(sum),
                        tool.state.scalar_k,
                        fmt_fp_pt(kp)
                    )
                }
                Err(err) => err.to_string(),
            }
        }
    }
}
