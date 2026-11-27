//! Painting and export materialization helpers for [`super::Scene3DBatchItem`].

use eframe::egui::{
    self,
    epaint::{CircleShape, PathStroke},
    pos2, Align2, Color32, Pos2, Shape, Stroke, TextStyle, TextWrapMode, Ui, WidgetText,
};
use egui_plot::{
    ExportPrimitive, LineStyle, LinearProjector, MarkerShape, PlotPoint, PlotTransform,
    SharedPoints,
};

use super::{CompiledDrawOp, CompiledScenePayload, LabelPrimitive3D};

#[inline]
fn project_point(
    pt: &PlotPoint,
    transform: &PlotTransform,
    projector: Option<&LinearProjector>,
) -> Pos2 {
    match projector {
        Some(proj) => proj.project(pt.x, pt.y),
        None => transform.position_from_point(pt),
    }
}

pub(super) fn paint_draw_op(
    op: &CompiledDrawOp,
    transform: &PlotTransform,
    projector: Option<&LinearProjector>,
    scale_mult: f32,
    shapes: &mut Vec<Shape>,
) {
    match op {
        CompiledDrawOp::Segment {
            p0,
            p1,
            stroke,
            style,
            ..
        } => {
            let s0 = project_point(p0, transform, projector);
            let s1 = project_point(p1, transform, projector);
            if !s0.is_finite() || !s1.is_finite() {
                return;
            }
            let width = stroke.width * scale_mult;
            match style {
                LineStyle::Solid => {
                    shapes.push(Shape::line_segment(
                        [s0, s1],
                        Stroke::new(width, stroke.color),
                    ));
                }
                _ => {
                    style.style_line(
                        vec![s0, s1],
                        PathStroke::new(width, stroke.color),
                        false,
                        shapes,
                    );
                }
            }
        }
        CompiledDrawOp::Point {
            pos,
            radius,
            color,
            shape,
            ..
        } => {
            let center = project_point(pos, transform, projector);
            if center.is_finite() {
                paint_marker_shape(center, *radius * scale_mult, *color, *shape, shapes);
            }
        }
    }
}

pub(super) fn paint_label_op(
    lbl: &LabelPrimitive3D,
    ui: &Ui,
    transform: &PlotTransform,
    shapes: &mut Vec<Shape>,
) {
    let pos = transform.position_from_point(&PlotPoint::new(lbl.proj[0], lbl.proj[1]));
    if pos.is_finite() {
        let galley = WidgetText::from(egui::RichText::new(&lbl.text).strong().color(lbl.color))
            .into_galley(
                ui,
                Some(TextWrapMode::Extend),
                f32::INFINITY,
                TextStyle::Body,
            );
        let rect = Align2::CENTER_CENTER.anchor_size(pos, galley.size());
        shapes.push(Shape::galley(rect.min, galley, lbl.color));
    }
}

pub(super) fn paint_marker_shape(
    center: Pos2,
    radius: f32,
    color: Color32,
    shape: MarkerShape,
    shapes: &mut Vec<Shape>,
) {
    match shape {
        MarkerShape::Circle
        | MarkerShape::Diamond
        | MarkerShape::Square
        | MarkerShape::Up
        | MarkerShape::Down
        | MarkerShape::Left
        | MarkerShape::Right => paint_filled_marker(center, radius, color, shape, shapes),
        MarkerShape::Cross | MarkerShape::Plus | MarkerShape::Asterisk => {
            paint_stroked_marker(center, radius, color, shape, shapes);
        }
    }
}

fn paint_filled_marker(
    center: Pos2,
    radius: f32,
    color: Color32,
    shape: MarkerShape,
    shapes: &mut Vec<Shape>,
) {
    let inv_sqrt_2 = std::f32::consts::FRAC_1_SQRT_2;
    let half_sqrt_3 = 0.5 * 3.0_f32.sqrt();
    let tf = |dx: f32, dy: f32| pos2(center.x + radius * dx, center.y + radius * dy);
    let poly = match shape {
        MarkerShape::Circle => {
            shapes.push(Shape::Circle(CircleShape {
                center,
                radius,
                fill: color,
                stroke: Stroke::NONE,
            }));
            return;
        }
        MarkerShape::Diamond => vec![tf(0.0, 1.0), tf(-1.0, 0.0), tf(0.0, -1.0), tf(1.0, 0.0)],
        MarkerShape::Square => vec![
            tf(-inv_sqrt_2, inv_sqrt_2),
            tf(-inv_sqrt_2, -inv_sqrt_2),
            tf(inv_sqrt_2, -inv_sqrt_2),
            tf(inv_sqrt_2, inv_sqrt_2),
        ],
        MarkerShape::Up => vec![tf(0.0, -1.0), tf(half_sqrt_3, 0.5), tf(-half_sqrt_3, 0.5)],
        MarkerShape::Down => vec![tf(0.0, 1.0), tf(-half_sqrt_3, -0.5), tf(half_sqrt_3, -0.5)],
        MarkerShape::Left => vec![tf(-1.0, 0.0), tf(0.5, -half_sqrt_3), tf(0.5, half_sqrt_3)],
        MarkerShape::Right => vec![tf(1.0, 0.0), tf(-0.5, half_sqrt_3), tf(-0.5, -half_sqrt_3)],
        _ => return,
    };
    shapes.push(Shape::convex_polygon(poly, color, Stroke::NONE));
}

fn paint_stroked_marker(
    center: Pos2,
    radius: f32,
    color: Color32,
    shape: MarkerShape,
    shapes: &mut Vec<Shape>,
) {
    let inv_sqrt_2 = std::f32::consts::FRAC_1_SQRT_2;
    let hx = 0.5 * 3.0_f32.sqrt();
    let stroke = Stroke::new((radius * 0.2).max(1.0), color);
    let tf = |dx: f32, dy: f32| pos2(center.x + radius * dx, center.y + radius * dy);
    match shape {
        MarkerShape::Cross => {
            shapes.push(Shape::line_segment(
                [tf(-inv_sqrt_2, -inv_sqrt_2), tf(inv_sqrt_2, inv_sqrt_2)],
                stroke,
            ));
            shapes.push(Shape::line_segment(
                [tf(inv_sqrt_2, -inv_sqrt_2), tf(-inv_sqrt_2, inv_sqrt_2)],
                stroke,
            ));
        }
        MarkerShape::Plus => {
            shapes.push(Shape::line_segment([tf(-1.0, 0.0), tf(1.0, 0.0)], stroke));
            shapes.push(Shape::line_segment([tf(0.0, -1.0), tf(0.0, 1.0)], stroke));
        }
        MarkerShape::Asterisk => {
            shapes.push(Shape::line_segment([tf(0.0, -1.0), tf(0.0, 1.0)], stroke));
            shapes.push(Shape::line_segment([tf(-hx, 0.5), tf(hx, -0.5)], stroke));
            shapes.push(Shape::line_segment([tf(-hx, -0.5), tf(hx, 0.5)], stroke));
        }
        _ => {}
    }
}

pub(super) fn materialize_scene_exports(
    payload: &CompiledScenePayload,
    out: &mut Vec<ExportPrimitive>,
) {
    out.reserve(payload.draw_ops.len());
    for op in &payload.draw_ops {
        match op {
            CompiledDrawOp::Segment {
                p0,
                p1,
                stroke,
                name_idx,
                ..
            } => {
                out.push(ExportPrimitive::Polyline {
                    name: payload.names[*name_idx as usize].clone(),
                    points: SharedPoints::from(vec![[p0.x, p0.y], [p1.x, p1.y]]),
                    color: stroke.color,
                    stroke_width: stroke.width,
                    closed: false,
                    fill: None,
                });
            }
            CompiledDrawOp::Point {
                pos,
                radius,
                color,
                name_idx,
                ..
            } => {
                out.push(ExportPrimitive::Points {
                    name: payload.names[*name_idx as usize].clone(),
                    points: SharedPoints::from(vec![[pos.x, pos.y]]),
                    color: *color,
                    radius: *radius,
                });
            }
        }
    }
}
