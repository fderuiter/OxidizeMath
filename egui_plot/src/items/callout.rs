use std::ops::RangeInclusive;

use egui::{
    Align2, Color32, CornerRadius, Id, Shape, Stroke, TextStyle, Ui, Vec2, WidgetText,
    epaint::{RectShape, TextShape},
};

use crate::{PlotBounds, PlotPoint, PlotTransform};

use super::{PlotGeometry, PlotItem, PlotItemBase};

/// A callout annotation inside the plot, connecting a target plot point to a text label frame using a leader line.
#[derive(Clone)]
pub struct Callout {
    base: PlotItemBase,
    pub target_point: PlotPoint,
    pub label_offset: Vec2,
    pub text: WidgetText,
    pub leader_stroke: Stroke,
    pub frame_fill: Color32,
    pub frame_stroke: Stroke,
    pub corner_radius: CornerRadius,
    pub anchor: Align2,
    pub text_color: Color32,
}

impl Callout {
    /// Create a new callout targeting a plot point with specified text label.
    pub fn new(target_point: impl Into<PlotPoint>, text: impl Into<WidgetText>) -> Self {
        Self {
            base: PlotItemBase::new(String::new()),
            target_point: target_point.into(),
            label_offset: Vec2::ZERO,
            text: text.into(),
            leader_stroke: Stroke::new(1.0_f32, Color32::TRANSPARENT),
            frame_fill: Color32::TRANSPARENT,
            frame_stroke: Stroke::new(0.0_f32, Color32::TRANSPARENT),
            corner_radius: CornerRadius::ZERO,
            anchor: Align2::CENTER_CENTER,
            text_color: Color32::TRANSPARENT,
        }
    }

    /// Offset of the callout label frame relative to target point screen position (in screen pixels).
    #[inline]
    pub fn offset(mut self, offset: impl Into<Vec2>) -> Self {
        self.label_offset = offset.into();
        self
    }

    /// Alias for [`offset`].
    #[inline]
    pub fn label_offset(mut self, offset: impl Into<Vec2>) -> Self {
        self.label_offset = offset.into();
        self
    }

    /// Leader line stroke.
    #[inline]
    pub fn leader_stroke(mut self, stroke: impl Into<Stroke>) -> Self {
        self.leader_stroke = stroke.into();
        self
    }

    /// Background frame fill color and border stroke.
    #[inline]
    pub fn frame(mut self, fill: impl Into<Color32>, stroke: impl Into<Stroke>) -> Self {
        self.frame_fill = fill.into();
        self.frame_stroke = stroke.into();
        self
    }

    /// Background frame fill color.
    #[inline]
    pub fn frame_fill(mut self, fill: impl Into<Color32>) -> Self {
        self.frame_fill = fill.into();
        self
    }

    /// Background frame border stroke.
    #[inline]
    pub fn frame_stroke(mut self, stroke: impl Into<Stroke>) -> Self {
        self.frame_stroke = stroke.into();
        self
    }

    /// Corner radius of the callout frame box.
    #[inline]
    pub fn corner_radius(mut self, corner_radius: impl Into<CornerRadius>) -> Self {
        self.corner_radius = corner_radius.into();
        self
    }

    /// Alignment anchor for the callout frame box relative to the offset target position.
    #[inline]
    pub fn anchor(mut self, anchor: Align2) -> Self {
        self.anchor = anchor;
        self
    }

    /// Text color inside the callout frame.
    #[inline]
    pub fn text_color(mut self, color: impl Into<Color32>) -> Self {
        self.text_color = color.into();
        self
    }

    builder_methods_for_base!();
}

impl PlotItem for Callout {
    fn shapes(&self, ui: &Ui, transform: &PlotTransform, shapes: &mut Vec<Shape>) {
        let text_color = if self.text_color != Color32::TRANSPARENT {
            self.text_color
        } else {
            ui.style().visuals.text_color()
        };

        let galley = self.text.clone().into_galley(
            ui,
            Some(egui::TextWrapMode::Extend),
            f32::INFINITY,
            TextStyle::Small,
        );

        let target_pos = transform.position_from_point(&self.target_point);
        let anchor_pos = target_pos + self.label_offset;

        let padding = Vec2::new(6.0, 4.0);
        let frame_size = galley.size() + 2.0 * padding;
        let frame_rect = self.anchor.anchor_size(anchor_pos, frame_size);
        let text_pos = frame_rect.min + padding;

        let connection_pos = frame_rect.clamp(target_pos);

        let leader_color = if self.leader_stroke.color != Color32::TRANSPARENT {
            self.leader_stroke.color
        } else if self.frame_stroke.color != Color32::TRANSPARENT {
            self.frame_stroke.color
        } else {
            ui.style().visuals.widgets.noninteractive.fg_stroke.color
        };

        if self.leader_stroke.width > 0.0 && target_pos.distance(connection_pos) > 0.1 {
            let stroke = Stroke::new(self.leader_stroke.width, leader_color);
            shapes.push(Shape::line_segment([target_pos, connection_pos], stroke));
        }

        let has_fill = self.frame_fill != Color32::TRANSPARENT;
        let has_stroke =
            self.frame_stroke.width > 0.0 && self.frame_stroke.color != Color32::TRANSPARENT;

        if has_fill || has_stroke {
            shapes.push(Shape::Rect(RectShape::new(
                frame_rect,
                self.corner_radius,
                self.frame_fill,
                self.frame_stroke,
                egui::StrokeKind::Inside,
            )));
        }

        shapes.push(TextShape::new(text_pos, galley, text_color).into());

        if self.base.highlight {
            shapes.push(Shape::rect_stroke(
                frame_rect.expand(1.0_f32),
                self.corner_radius,
                Stroke::new(1.0_f32, leader_color),
                egui::StrokeKind::Outside,
            ));
        }
    }

    fn initialize(&mut self, _x_range: RangeInclusive<f64>) {}

    fn color(&self) -> Color32 {
        if self.frame_stroke.color != Color32::TRANSPARENT {
            self.frame_stroke.color
        } else if self.leader_stroke.color != Color32::TRANSPARENT {
            self.leader_stroke.color
        } else {
            self.text_color
        }
    }

    fn geometry(&self) -> PlotGeometry<'_> {
        PlotGeometry::None
    }

    fn bounds(&self) -> PlotBounds {
        let mut bounds = PlotBounds::NOTHING;
        bounds.extend_with(&self.target_point);
        bounds
    }

    fn base(&self) -> &PlotItemBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut PlotItemBase {
        &mut self.base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::vec2;

    #[test]
    fn test_callout_builder() {
        let point = PlotPoint::new(1.0, 2.0);
        let callout = Callout::new(point, "Root Point")
            .offset(vec2(10.0, -15.0))
            .leader_stroke(Stroke::new(2.0_f32, Color32::RED))
            .frame(Color32::BLACK, Stroke::new(1.0_f32, Color32::WHITE))
            .corner_radius(CornerRadius::same(4))
            .anchor(Align2::LEFT_TOP)
            .name("root_callout");

        assert_eq!(callout.target_point, point);
        assert_eq!(callout.label_offset, vec2(10.0, -15.0));
        assert_eq!(callout.leader_stroke, Stroke::new(2.0_f32, Color32::RED));
        assert_eq!(callout.frame_fill, Color32::BLACK);
        assert_eq!(callout.frame_stroke, Stroke::new(1.0_f32, Color32::WHITE));
        assert_eq!(callout.corner_radius, CornerRadius::same(4));
        assert_eq!(callout.anchor, Align2::LEFT_TOP);
        assert_eq!(callout.base().name, "root_callout");
    }

    #[test]
    fn test_callout_bounds() {
        let point = PlotPoint::new(-3.5, 7.2);
        let callout = Callout::new(point, "Extreme");
        let bounds = callout.bounds();
        assert_eq!(bounds.min(), [-3.5, 7.2]);
        assert_eq!(bounds.max(), [-3.5, 7.2]);
    }

    #[test]
    fn test_callout_zero_offset() {
        let ctx = egui::Context::default();
        let _ = ctx.run(Default::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let transform = PlotTransform::new(
                    egui::Rect::from_min_size(egui::pos2(0.0, 0.0), vec2(100.0, 100.0)),
                    PlotBounds::from_min_max([0.0, 0.0], [10.0, 10.0]),
                    false,
                );

                let callout = Callout::new([5.0, 5.0], "Center Point")
                    .offset(Vec2::ZERO)
                    .leader_stroke(Stroke::new(1.0_f32, Color32::BLUE));

                let mut shapes = Vec::new();
                callout.shapes(ui, &transform, &mut shapes);

                // With offset == 0, connection_pos == target_pos, so no leader line segment is drawn.
                // Only text (and optional frame) shape is produced.
                assert!(
                    !shapes
                        .iter()
                        .any(|s| matches!(s, Shape::LineSegment { .. }))
                );
                assert!(shapes.iter().any(|s| matches!(s, Shape::Text(_))));
            });
        });
    }

    #[test]
    fn test_callout_shapes_generation() {
        let ctx = egui::Context::default();
        let _ = ctx.run(Default::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let transform = PlotTransform::new(
                    egui::Rect::from_min_size(egui::pos2(0.0, 0.0), vec2(100.0, 100.0)),
                    PlotBounds::from_min_max([0.0, 0.0], [10.0, 10.0]),
                    false,
                );

                let callout = Callout::new([2.0, 2.0], "Annotated Root")
                    .offset(vec2(20.0, -20.0))
                    .leader_stroke(Stroke::new(1.5_f32, Color32::RED))
                    .frame(Color32::DARK_GRAY, Stroke::new(1.0_f32, Color32::WHITE));

                let mut shapes = Vec::new();
                callout.shapes(ui, &transform, &mut shapes);

                // Should generate leader line, frame rectangle, and text shape
                assert!(
                    shapes
                        .iter()
                        .any(|s| matches!(s, Shape::LineSegment { .. }))
                );
                assert!(shapes.iter().any(|s| matches!(s, Shape::Rect(_))));
                assert!(shapes.iter().any(|s| matches!(s, Shape::Text(_))));
            });
        });
    }
}
