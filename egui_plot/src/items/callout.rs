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
    fn resolve_frame(&self, visuals: &egui::Visuals) -> Option<(Color32, Stroke)> {
        let has_explicit_fill =
            self.frame_fill != Color32::TRANSPARENT && self.frame_fill != Color32::PLACEHOLDER;
        let uses_theme_fill = self.frame_fill == Color32::PLACEHOLDER;
        let has_stroke =
            self.frame_stroke.width > 0.0 && self.frame_stroke.color != Color32::TRANSPARENT;

        if !(has_explicit_fill || uses_theme_fill || has_stroke) {
            return None;
        }

        let window_fill = visuals.window_fill();
        let resolved_fill = if has_explicit_fill {
            self.frame_fill
        } else {
            Color32::from_rgba_unmultiplied(window_fill.r(), window_fill.g(), window_fill.b(), 230)
        };
        let resolved_stroke = if self.frame_stroke.color == Color32::PLACEHOLDER {
            visuals.widgets.noninteractive.bg_stroke
        } else {
            self.frame_stroke
        };
        Some((resolved_fill, resolved_stroke))
    }

    fn resolve_leader_color(&self, visuals: &egui::Visuals) -> Color32 {
        if self.leader_stroke.color != Color32::TRANSPARENT
            && self.leader_stroke.color != Color32::PLACEHOLDER
        {
            self.leader_stroke.color
        } else if self.frame_stroke.color != Color32::TRANSPARENT
            && self.frame_stroke.color != Color32::PLACEHOLDER
        {
            self.frame_stroke.color
        } else {
            visuals.widgets.noninteractive.fg_stroke.color
        }
    }
}

impl PlotItem for Callout {
    fn shapes(&self, ui: &Ui, transform: &PlotTransform, shapes: &mut Vec<Shape>) {
        let visuals = &ui.style().visuals;
        let text_color =
            if self.text_color != Color32::TRANSPARENT && self.text_color != Color32::PLACEHOLDER {
                self.text_color
            } else {
                visuals.text_color()
            };

        let galley = self.text.clone().into_galley(
            ui,
            Some(egui::TextWrapMode::Extend),
            f32::INFINITY,
            TextStyle::Small,
        );

        let target_pos = transform.position_from_point(&self.target_point);
        let padding = Vec2::new(6.0, 4.0);
        let frame_rect = self.anchor.anchor_size(
            target_pos + self.label_offset,
            galley.size() + 2.0 * padding,
        );
        let connection_pos = frame_rect.clamp(target_pos);
        let leader_color = self.resolve_leader_color(visuals);

        if self.leader_stroke.width > 0.0 && target_pos.distance(connection_pos) > 0.1 {
            let stroke = Stroke::new(self.leader_stroke.width, leader_color);
            shapes.push(Shape::line_segment([target_pos, connection_pos], stroke));
        }

        if let Some((resolved_fill, resolved_stroke)) = self.resolve_frame(visuals) {
            shapes.push(Shape::Rect(RectShape::new(
                frame_rect,
                self.corner_radius,
                resolved_fill,
                resolved_stroke,
                egui::StrokeKind::Inside,
            )));
        }

        shapes.push(TextShape::new(frame_rect.min + padding, galley, text_color).into());

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
        if self.frame_stroke.color != Color32::TRANSPARENT
            && self.frame_stroke.color != Color32::PLACEHOLDER
        {
            self.frame_stroke.color
        } else if self.leader_stroke.color != Color32::TRANSPARENT
            && self.leader_stroke.color != Color32::PLACEHOLDER
        {
            self.leader_stroke.color
        } else {
            self.text_color
        }
    }

    fn geometry(&self) -> PlotGeometry<'_> {
        PlotGeometry::None
    }

    fn matches_cached_shared_points(&self, cached: &super::SharedPoints) -> bool {
        cached.is_empty()
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

/// A serializable representation of a user callout annotation stored in [`crate::PlotMemory`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct SavedCallout {
    /// Target plot coordinate [x, y].
    pub point: [f64; 2],
    /// Callout text label (limited to 128 characters).
    pub label: String,
    /// Screen pixel offset [dx, dy] for label placement and leader line.
    pub offset: [f32; 2],
}

impl SavedCallout {
    /// Create a new saved callout annotation, truncating the label to 128 characters.
    pub fn new(point: [f64; 2], label: impl Into<String>, offset: [f32; 2]) -> Self {
        let mut label = label.into();
        if label.trim().is_empty() {
            let x_str = crate::format::format_readout_value(point[0], 2, 4);
            let y_str = crate::format::format_readout_value(point[1], 2, 4);
            label = format!("({x_str}, {y_str})");
        }
        if label.chars().count() > 128 {
            label = label.chars().take(128).collect();
        }
        Self {
            point,
            label,
            offset,
        }
    }

    /// Convert this saved callout into a renderable [`Callout`] plot item.
    ///
    /// Frame fill and border stroke default to [`Color32::PLACEHOLDER`] so [`Callout::shapes`]
    /// resolves them dynamically from `ui.visuals().window_fill()` across Dark, Light, and HighContrast themes.
    pub fn to_callout(&self) -> Callout {
        let label = if self.label.trim().is_empty() {
            let x_str = crate::format::format_readout_value(self.point[0], 2, 4);
            let y_str = crate::format::format_readout_value(self.point[1], 2, 4);
            format!("({x_str}, {y_str})")
        } else if self.label.chars().count() > 128 {
            self.label.chars().take(128).collect()
        } else {
            self.label.clone()
        };

        Callout::new(self.point, label)
            .offset(Vec2::new(self.offset[0], self.offset[1]))
            .leader_stroke(Stroke::new(1.5_f32, Color32::TRANSPARENT))
            .frame(
                Color32::PLACEHOLDER,
                Stroke::new(1.0_f32, Color32::PLACEHOLDER),
            )
            .corner_radius(CornerRadius::same(4))
            .name(&self.label)
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

    #[test]
    fn test_saved_callout_creation_and_truncation() {
        let long_label = "A".repeat(200);
        let saved = SavedCallout::new([1.5, -2.5], long_label, [15.0, -25.0]);
        assert_eq!(saved.point, [1.5, -2.5]);
        assert_eq!(saved.label.len(), 128);
        assert_eq!(saved.offset, [15.0, -25.0]);

        // Empty label defaults to coordinate string
        let empty_saved = SavedCallout::new([3.0, 4.0], "", [10.0, 10.0]);
        assert_eq!(empty_saved.label, "(3.00, 4.00)");
    }

    #[test]
    fn test_saved_callout_to_callout() {
        let saved = SavedCallout::new([10.0, 20.0], "Equilibrium", [30.0, -30.0]);
        let callout = saved.to_callout();
        assert_eq!(callout.target_point, PlotPoint::new(10.0, 20.0));
        assert_eq!(callout.label_offset, vec2(30.0, -30.0));
        assert_ne!(callout.frame_fill, Color32::TRANSPARENT);
    }

    fn srgb_to_linear(channel: u8) -> f64 {
        let c = f64::from(channel) / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    fn wcag_contrast_ratio(c1: Color32, c2: Color32) -> f64 {
        let lum = |c: Color32| {
            0.2126 * srgb_to_linear(c.r())
                + 0.7152 * srgb_to_linear(c.g())
                + 0.0722 * srgb_to_linear(c.b())
        };
        let l1 = lum(c1);
        let l2 = lum(c2);
        (l1.max(l2) + 0.05) / (l1.min(l2) + 0.05)
    }

    fn blend_over(fg: Color32, bg: Color32) -> Color32 {
        let [fr, fg_c, fb, fa] = fg.to_srgba_unmultiplied();
        let a = f32::from(fa) / 255.0;
        let inv = 1.0 - a;
        let blend = |f: u8, b: u8| (f32::from(f) * a + f32::from(b) * inv).round() as u8;
        Color32::from_rgb(blend(fr, bg.r()), blend(fg_c, bg.g()), blend(fb, bg.b()))
    }

    fn high_contrast_visuals() -> egui::Visuals {
        let mut visuals = egui::Visuals::dark();
        visuals.extreme_bg_color = Color32::BLACK;
        visuals.panel_fill = Color32::BLACK;
        visuals.window_fill = Color32::BLACK;
        visuals.widgets.noninteractive.bg_fill = Color32::BLACK;
        visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, Color32::WHITE);
        visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, Color32::WHITE);
        visuals
    }

    #[test]
    fn test_saved_callout_wcag_contrast_across_themes_in_plot_show() {
        for (name, visuals) in [
            ("dark", egui::Visuals::dark()),
            ("light", egui::Visuals::light()),
            ("high_contrast", high_contrast_visuals()),
        ] {
            let ctx = egui::Context::default();
            ctx.set_visuals(visuals.clone());
            let mut captured_shapes = Vec::new();
            let output = ctx.run(Default::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let saved = SavedCallout::new([1.0, 2.0], "Bifurcation", [25.0, -25.0]);
                    crate::Plot::new(format!("callout_contrast_{name}")).show(ui, |plot_ui| {
                        plot_ui.callout(saved.to_callout());
                    });

                    let transform = PlotTransform::new(
                        egui::Rect::from_min_size(egui::pos2(0.0, 0.0), vec2(200.0, 200.0)),
                        PlotBounds::from_min_max([0.0, 0.0], [10.0, 10.0]),
                        false,
                    );
                    saved
                        .to_callout()
                        .shapes(ui, &transform, &mut captured_shapes);
                });
            });
            assert!(!output.shapes.is_empty());

            let rect_fill = captured_shapes
                .iter()
                .find_map(|s| match s {
                    Shape::Rect(r) => Some(r.fill),
                    _ => None,
                })
                .expect("SavedCallout should emit a background frame RectShape");
            let text_color = captured_shapes
                .iter()
                .find_map(|s| match s {
                    Shape::Text(t) => Some(t.override_text_color.unwrap_or(visuals.text_color())),
                    _ => None,
                })
                .expect("SavedCallout should emit a label TextShape");

            let composite_bg = blend_over(rect_fill, visuals.extreme_bg_color);
            let ratio = wcag_contrast_ratio(text_color, composite_bg);
            assert!(
                ratio >= 4.5,
                "SavedCallout in {name} mode achieved {ratio:.2}:1 contrast (< 4.5:1)"
            );
        }
    }

    #[test]
    fn test_unset_callout_frame_fill_resolves_from_window_fill() {
        let ctx = egui::Context::default();
        ctx.set_visuals(egui::Visuals::light());
        let _ = ctx.run(Default::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let transform = PlotTransform::new(
                    egui::Rect::from_min_size(egui::pos2(0.0, 0.0), vec2(100.0, 100.0)),
                    PlotBounds::from_min_max([0.0, 0.0], [10.0, 10.0]),
                    false,
                );
                let callout = Callout::new([3.0, 3.0], "Light Note")
                    .offset(vec2(15.0, -15.0))
                    .frame_stroke(Stroke::new(1.0_f32, Color32::from_gray(90)));
                let mut shapes = Vec::new();
                callout.shapes(ui, &transform, &mut shapes);
                let rect_fill = shapes
                    .iter()
                    .find_map(|s| match s {
                        Shape::Rect(r) => Some(r.fill),
                        _ => None,
                    })
                    .expect("Framed callout should resolve unset fill from window_fill");
                let wf = ui.visuals().window_fill();
                assert_eq!(
                    rect_fill.to_srgba_unmultiplied(),
                    [wf.r(), wf.g(), wf.b(), 230]
                );
            });
        });
    }
}
