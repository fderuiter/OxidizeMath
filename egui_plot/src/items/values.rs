use std::ops::{Bound, RangeBounds, RangeInclusive};

use egui::{
    Pos2, Rect, Shape, Stroke, Vec2,
    epaint::{ColorMode, PathStroke},
    lerp, pos2,
};

use crate::transform::PlotBounds;

/// A point coordinate in the plot.
///
/// Uses f64 for improved accuracy to enable plotting
/// large values (e.g. unix time on x axis).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlotPoint {
    /// This is often something monotonically increasing, such as time, but doesn't have to be.
    /// Goes from left to right.
    pub x: f64,

    /// Goes from bottom to top (inverse of everything else in egui!).
    pub y: f64,
}

impl From<[f64; 2]> for PlotPoint {
    #[inline]
    fn from([x, y]: [f64; 2]) -> Self {
        Self { x, y }
    }
}

impl PlotPoint {
    #[inline(always)]
    pub fn new(x: impl Into<f64>, y: impl Into<f64>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
        }
    }

    #[inline(always)]
    pub fn to_pos2(self) -> Pos2 {
        Pos2::new(self.x as f32, self.y as f32)
    }

    #[inline(always)]
    pub fn to_vec2(self) -> Vec2 {
        Vec2::new(self.x as f32, self.y as f32)
    }
}

// ----------------------------------------------------------------------------

/// Solid, dotted, dashed, etc.
#[derive(Debug, PartialEq, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum LineStyle {
    Solid,
    Dotted { spacing: f32 },
    Dashed { length: f32 },
}

impl LineStyle {
    pub fn dashed_loose() -> Self {
        Self::Dashed { length: 10.0 }
    }

    pub fn dashed_dense() -> Self {
        Self::Dashed { length: 5.0 }
    }

    pub fn dotted_loose() -> Self {
        Self::Dotted { spacing: 10.0 }
    }

    pub fn dotted_dense() -> Self {
        Self::Dotted { spacing: 5.0 }
    }

    pub(super) fn style_line(
        &self,
        line: Vec<Pos2>,
        mut stroke: PathStroke,
        highlight: bool,
        shapes: &mut Vec<Shape>,
    ) {
        let path_stroke_color = match &stroke.color {
            ColorMode::Solid(c) => *c,
            ColorMode::UV(callback) => {
                callback(Rect::from_min_max(pos2(0., 0.), pos2(0., 0.)), pos2(0., 0.))
            }
        };
        match line.len() {
            0 => {}
            1 => {
                let mut radius = stroke.width / 2.0;
                if highlight {
                    radius *= 2f32.sqrt();
                }
                shapes.push(Shape::circle_filled(line[0], radius, path_stroke_color));
            }
            _ => {
                match self {
                    Self::Solid => {
                        if highlight {
                            stroke.width *= 2.0;
                        }
                        shapes.push(Shape::line(line, stroke));
                    }
                    Self::Dotted { spacing } => {
                        // Take the stroke width for the radius even though it's not "correct", otherwise
                        // the dots would become too small.
                        let mut radius = stroke.width;
                        if highlight {
                            radius *= 2f32.sqrt();
                        }
                        if line.len() > 128 {
                            if let Some(mesh) =
                                build_dotted_line_mesh(&line, path_stroke_color, *spacing, radius)
                            {
                                shapes.push(Shape::Mesh(std::sync::Arc::new(mesh)));
                            }
                        } else {
                            shapes.extend(Shape::dotted_line(
                                &line,
                                path_stroke_color,
                                *spacing,
                                radius,
                            ));
                        }
                    }
                    Self::Dashed { length } => {
                        if highlight {
                            stroke.width *= 2.0;
                        }
                        let golden_ratio = (5.0_f32.sqrt() - 1.0) / 2.0; // 0.61803398875
                        let gap = length * golden_ratio;
                        if line.len() > 128 {
                            if let Some(mesh) = build_dashed_line_mesh(
                                &line,
                                stroke.width,
                                path_stroke_color,
                                *length,
                                gap,
                            ) {
                                shapes.push(Shape::Mesh(std::sync::Arc::new(mesh)));
                            }
                        } else {
                            shapes.extend(Shape::dashed_line(
                                &line,
                                Stroke::new(stroke.width, path_stroke_color),
                                *length,
                                gap,
                            ));
                        }
                    }
                }
            }
        }
    }
}

fn build_dashed_line_mesh(
    line: &[Pos2],
    width: f32,
    color: egui::Color32,
    dash_length: f32,
    gap_length: f32,
) -> Option<egui::Mesh> {
    let dash_len = dash_length.max(0.5);
    let gap_len = gap_length.max(0.5);
    let half_w = width.max(0.5) * 0.5;
    let mut mesh = egui::Mesh::default();
    let mut in_dash = true;
    let mut remaining = dash_len;

    for pair in line.windows(2) {
        let (p0, p1) = (pair[0], pair[1]);
        if !p0.is_finite() || !p1.is_finite() {
            continue;
        }
        let seg = p1 - p0;
        let seg_len = seg.length();
        if seg_len <= 1e-5 {
            continue;
        }
        let dir = seg / seg_len;
        let half_normal = Vec2::new(-dir.y, dir.x) * half_w;
        let mut cursor = 0.0_f32;

        while cursor < seg_len {
            let step = (seg_len - cursor).min(remaining);
            if in_dash && step > 1e-5 {
                let a = p0 + dir * cursor;
                let b = p0 + dir * (cursor + step);
                let base = mesh.vertices.len() as u32;
                mesh.colored_vertex(a + half_normal, color);
                mesh.colored_vertex(a - half_normal, color);
                mesh.colored_vertex(b - half_normal, color);
                mesh.colored_vertex(b + half_normal, color);
                mesh.add_triangle(base, base + 1, base + 2);
                mesh.add_triangle(base, base + 2, base + 3);
            }
            cursor += step;
            remaining -= step;
            if remaining <= 1e-5 {
                in_dash = !in_dash;
                remaining = if in_dash { dash_len } else { gap_len };
            }
        }
    }

    (!mesh.is_empty()).then_some(mesh)
}

fn build_dotted_line_mesh(
    line: &[Pos2],
    color: egui::Color32,
    spacing: f32,
    radius: f32,
) -> Option<egui::Mesh> {
    let step_len = spacing.max(1.0);
    let r = radius.max(0.5);
    let mut mesh = egui::Mesh::default();
    let mut until_next = 0.0_f32;
    const OFFSETS: [(f32, f32); 8] = [
        (1.0, 0.0),
        (
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ),
        (0.0, 1.0),
        (
            -std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ),
        (-1.0, 0.0),
        (
            -std::f32::consts::FRAC_1_SQRT_2,
            -std::f32::consts::FRAC_1_SQRT_2,
        ),
        (0.0, -1.0),
        (
            std::f32::consts::FRAC_1_SQRT_2,
            -std::f32::consts::FRAC_1_SQRT_2,
        ),
    ];

    for pair in line.windows(2) {
        let (p0, p1) = (pair[0], pair[1]);
        if !p0.is_finite() || !p1.is_finite() {
            continue;
        }
        let seg = p1 - p0;
        let seg_len = seg.length();
        if seg_len <= 1e-5 {
            continue;
        }
        let dir = seg / seg_len;
        let mut cursor = until_next;
        while cursor <= seg_len {
            let center = p0 + dir * cursor;
            let base = mesh.vertices.len() as u32;
            mesh.colored_vertex(center, color);
            for (ox, oy) in OFFSETS {
                mesh.colored_vertex(center + Vec2::new(ox * r, oy * r), color);
            }
            for i in 0..8_u32 {
                mesh.add_triangle(base, base + 1 + i, base + 1 + ((i + 1) % 8));
            }
            cursor += step_len;
        }
        until_next = cursor - seg_len;
    }

    (!mesh.is_empty()).then_some(mesh)
}

impl std::fmt::Display for LineStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Solid => write!(f, "Solid"),
            Self::Dotted { spacing } => write!(f, "Dotted({spacing} px)"),
            Self::Dashed { length } => write!(f, "Dashed({length} px)"),
        }
    }
}

// ----------------------------------------------------------------------------

/// Determines whether a plot element is vertically or horizontally oriented.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Orientation {
    Horizontal,
    #[default]
    Vertical,
}

// ----------------------------------------------------------------------------

/// Represents many [`PlotPoint`]s.
///
/// These can be an owned `Vec`, an `Arc`-backed [`super::SharedPoints`] buffer,
/// generated on-the-fly by a function, or borrowed from a slice.
pub enum PlotPoints<'a> {
    Owned(Vec<PlotPoint>),
    Shared(super::SharedPoints),
    Generator(ExplicitGenerator<'a>),
    Borrowed(&'a [PlotPoint]),
}

impl Default for PlotPoints<'_> {
    fn default() -> Self {
        Self::Owned(Vec::new())
    }
}

impl From<[f64; 2]> for PlotPoints<'_> {
    fn from(coordinate: [f64; 2]) -> Self {
        Self::new(vec![coordinate])
    }
}

impl From<Vec<[f64; 2]>> for PlotPoints<'_> {
    #[inline]
    fn from(coordinates: Vec<[f64; 2]>) -> Self {
        Self::new(coordinates)
    }
}

impl From<super::SharedPoints> for PlotPoints<'_> {
    #[inline]
    fn from(points: super::SharedPoints) -> Self {
        Self::Shared(points)
    }
}

impl From<&super::SharedPoints> for PlotPoints<'_> {
    #[inline]
    fn from(points: &super::SharedPoints) -> Self {
        Self::Shared(points.clone())
    }
}

impl<'a> From<&'a [PlotPoint]> for PlotPoints<'a> {
    #[inline]
    fn from(points: &'a [PlotPoint]) -> Self {
        Self::Borrowed(points)
    }
}

impl FromIterator<[f64; 2]> for PlotPoints<'_> {
    fn from_iter<T: IntoIterator<Item = [f64; 2]>>(iter: T) -> Self {
        Self::Owned(iter.into_iter().map(|point| point.into()).collect())
    }
}

impl<'a> PlotPoints<'a> {
    pub fn new(points: Vec<[f64; 2]>) -> Self {
        Self::from_iter(points)
    }

    /// Construct [`PlotPoints`] backed by a shared coordinate buffer without copying.
    #[inline]
    pub fn from_shared(points: impl Into<super::SharedPoints>) -> Self {
        Self::Shared(points.into())
    }

    pub fn points(&self) -> &[PlotPoint] {
        match self {
            Self::Owned(points) => points.as_slice(),
            Self::Shared(_) | Self::Generator(_) => &[],
            Self::Borrowed(points) => points,
        }
    }

    /// Returns the underlying [`super::SharedPoints`] if backed by a shared buffer.
    #[inline]
    pub fn shared(&self) -> Option<&super::SharedPoints> {
        match self {
            Self::Shared(points) => Some(points),
            Self::Owned(_) | Self::Generator(_) | Self::Borrowed(_) => None,
        }
    }

    /// Number of materialized points currently held in this series.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Owned(points) => points.len(),
            Self::Shared(points) => points.len(),
            Self::Generator(_) => 0,
            Self::Borrowed(points) => points.len(),
        }
    }

    /// Extract or reuse a [`super::SharedPoints`] buffer for this series.
    #[inline]
    pub fn to_shared_points(&self) -> super::SharedPoints {
        match self {
            Self::Shared(points) => points.clone(),
            Self::Owned(points) => super::SharedPoints::from_plot_points(points),
            Self::Borrowed(points) => super::SharedPoints::from_plot_points(points),
            Self::Generator(_) => super::SharedPoints::default(),
        }
    }

    /// Returns `true` when `cached` already holds identical coordinates for `self`.
    #[inline]
    pub(crate) fn matches_cached_shared(&self, cached: &super::SharedPoints) -> bool {
        match self {
            Self::Shared(points) => points.ptr_eq(cached),
            Self::Owned(points) => cached.matches_plot_points(points),
            Self::Borrowed(points) => cached.matches_plot_points(points),
            Self::Generator(_) => false,
        }
    }

    /// Returns the appropriate [`PlotGeometry`] view without copying coordinates.
    #[inline]
    pub(super) fn geometry(&self) -> PlotGeometry<'_> {
        match self {
            Self::Owned(points) => PlotGeometry::Points(points.as_slice()),
            Self::Shared(points) => PlotGeometry::Shared(points.as_slice()),
            Self::Generator(_) => PlotGeometry::Points(&[]),
            Self::Borrowed(points) => PlotGeometry::Points(points),
        }
    }

    /// Iterate over all materialized points as [`PlotPoint`] values without heap allocation.
    #[inline]
    pub(crate) fn iter_plot_points(&self) -> PlotPointsIter<'_> {
        match self {
            Self::Owned(points) => PlotPointsIter::Slice(points.iter()),
            Self::Shared(points) => PlotPointsIter::Shared(points.iter()),
            Self::Generator(_) => PlotPointsIter::Slice([].iter()),
            Self::Borrowed(points) => PlotPointsIter::Slice(points.iter()),
        }
    }

    /// Draw a line based on a function `y=f(x)`, a range (which can be infinite) for x and the number of points.
    pub fn from_explicit_callback(
        function: impl Fn(f64) -> f64 + 'a,
        x_range: impl RangeBounds<f64>,
        points: usize,
    ) -> Self {
        let start = match x_range.start_bound() {
            Bound::Included(x) | Bound::Excluded(x) => *x,
            Bound::Unbounded => f64::NEG_INFINITY,
        };
        let end = match x_range.end_bound() {
            Bound::Included(x) | Bound::Excluded(x) => *x,
            Bound::Unbounded => f64::INFINITY,
        };
        let x_range = start..=end;

        let generator = ExplicitGenerator {
            function: Box::new(function),
            x_range,
            points,
        };

        Self::Generator(generator)
    }

    /// Draw a line based on a function `(x,y)=f(t)`, a range for t and the number of points.
    /// The range may be specified as start..end or as start..=end.
    pub fn from_parametric_callback(
        function: impl Fn(f64) -> (f64, f64),
        t_range: impl RangeBounds<f64>,
        points: usize,
    ) -> Self {
        let start = match t_range.start_bound() {
            Bound::Included(x) => x,
            Bound::Excluded(_) => unreachable!(),
            Bound::Unbounded => panic!("The range for parametric functions must be bounded!"),
        };
        let end = match t_range.end_bound() {
            Bound::Included(x) | Bound::Excluded(x) => x,
            Bound::Unbounded => panic!("The range for parametric functions must be bounded!"),
        };
        let last_point_included = matches!(t_range.end_bound(), Bound::Included(_));
        let increment = if last_point_included {
            (end - start) / (points - 1) as f64
        } else {
            (end - start) / points as f64
        };
        (0..points)
            .map(|i| {
                let t = start + i as f64 * increment;
                function(t).into()
            })
            .collect()
    }

    /// From a series of y-values.
    /// The x-values will be the indices of these values
    pub fn from_ys_f32(ys: &[f32]) -> Self {
        ys.iter()
            .enumerate()
            .map(|(i, &y)| [i as f64, y as f64])
            .collect()
    }

    /// From a series of y-values.
    /// The x-values will be the indices of these values
    pub fn from_ys_f64(ys: &[f64]) -> Self {
        ys.iter().enumerate().map(|(i, &y)| [i as f64, y]).collect()
    }

    /// Returns true if there are no data points available and there is no function to generate any.
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Owned(points) => points.is_empty(),
            Self::Shared(points) => points.is_empty(),
            Self::Generator(_) => false,
            Self::Borrowed(points) => points.is_empty(),
        }
    }

    /// If initialized with a generator function, this will generate `n` evenly spaced points in the
    /// given range.
    pub(super) fn generate_points(&mut self, x_range: RangeInclusive<f64>) {
        if let Self::Generator(generator) = self {
            *self = Self::range_intersection(&x_range, &generator.x_range)
                .map(|intersection| {
                    let increment =
                        (intersection.end() - intersection.start()) / (generator.points - 1) as f64;
                    (0..generator.points)
                        .map(|i| {
                            let x = intersection.start() + i as f64 * increment;
                            let y = (generator.function)(x);
                            [x, y]
                        })
                        .collect()
                })
                .unwrap_or_default();
        }
    }

    /// Returns the intersection of two ranges if they intersect.
    fn range_intersection(
        range1: &RangeInclusive<f64>,
        range2: &RangeInclusive<f64>,
    ) -> Option<RangeInclusive<f64>> {
        let start = range1.start().max(*range2.start());
        let end = range1.end().min(*range2.end());
        (start < end).then_some(start..=end)
    }

    pub(super) fn bounds(&self) -> PlotBounds {
        match self {
            Self::Owned(points) => {
                let mut bounds = PlotBounds::NOTHING;
                for point in points {
                    bounds.extend_with(point);
                }
                bounds
            }
            Self::Shared(points) => {
                let mut bounds = PlotBounds::NOTHING;
                for &[x, y] in points.as_slice() {
                    bounds.extend_with(&PlotPoint::new(x, y));
                }
                bounds
            }
            Self::Generator(generator) => generator.estimate_bounds(),
            Self::Borrowed(points) => {
                let mut bounds = PlotBounds::NOTHING;
                for point in *points {
                    bounds.extend_with(point);
                }
                bounds
            }
        }
    }
}

pub(crate) enum PlotPointsIter<'a> {
    Slice(std::slice::Iter<'a, PlotPoint>),
    Shared(std::slice::Iter<'a, [f64; 2]>),
}

impl Iterator for PlotPointsIter<'_> {
    type Item = PlotPoint;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Slice(it) => it.next().copied(),
            Self::Shared(it) => it.next().map(|&[x, y]| PlotPoint::new(x, y)),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Slice(it) => it.size_hint(),
            Self::Shared(it) => it.size_hint(),
        }
    }
}

impl ExactSizeIterator for PlotPointsIter<'_> {}

// ----------------------------------------------------------------------------

/// Circle, Diamond, Square, Cross, …
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum MarkerShape {
    Circle,
    Diamond,
    Square,
    Cross,
    Plus,
    Up,
    Down,
    Left,
    Right,
    Asterisk,
}

impl MarkerShape {
    /// Get a vector containing all marker shapes.
    pub fn all() -> impl ExactSizeIterator<Item = Self> {
        [
            Self::Circle,
            Self::Diamond,
            Self::Square,
            Self::Cross,
            Self::Plus,
            Self::Up,
            Self::Down,
            Self::Left,
            Self::Right,
            Self::Asterisk,
        ]
        .iter()
        .copied()
    }
}

// ----------------------------------------------------------------------------

/// Query the points of the plot, for geometric relations like closest checks
pub enum PlotGeometry<'a> {
    /// No geometry based on single elements (examples: text, image, horizontal/vertical line)
    None,

    /// Point values (X-Y graphs)
    Points(&'a [PlotPoint]),

    /// Point values stored as shared `[f64; 2]` coordinate pairs
    Shared(&'a [[f64; 2]]),

    /// Rectangles (examples: boxes or bars)
    // Has currently no data, as it would require copying rects or iterating a list of pointers.
    // Instead, geometry-based functions are directly implemented in the respective PlotItem impl.
    Rects,
}

// ----------------------------------------------------------------------------

/// Describes a function y = f(x) with an optional range for x and a number of points.
pub struct ExplicitGenerator<'a> {
    function: Box<dyn Fn(f64) -> f64 + 'a>,
    x_range: RangeInclusive<f64>,
    points: usize,
}

impl ExplicitGenerator<'_> {
    fn estimate_bounds(&self) -> PlotBounds {
        let mut bounds = PlotBounds::NOTHING;

        let mut add_x = |x: f64| {
            // avoid infinities, as we cannot auto-bound on them!
            if x.is_finite() {
                bounds.extend_with_x(x);
            }
            let y = (self.function)(x);
            if y.is_finite() {
                bounds.extend_with_y(y);
            }
        };

        let min_x = *self.x_range.start();
        let max_x = *self.x_range.end();

        add_x(min_x);
        add_x(max_x);

        if min_x.is_finite() && max_x.is_finite() {
            // Sample some points in the interval:
            const N: u32 = 8;
            for i in 1..N {
                let t = i as f64 / (N - 1) as f64;
                let x = lerp(min_x..=max_x, t);
                add_x(x);
            }
        } else {
            // Try adding some points anyway:
            for x in [-1, 0, 1] {
                let x = x as f64;
                if min_x <= x && x <= max_x {
                    add_x(x);
                }
            }
        }

        bounds
    }
}

// ----------------------------------------------------------------------------

/// Result of [`super::PlotItem::find_closest()`] search, identifies an element inside the item for immediate use
pub struct ClosestElem {
    /// Position of hovered-over value (or bar/box-plot/…) in `PlotItem`
    pub index: usize,

    /// Squared distance from the mouse cursor (needed to compare against other `PlotItems`, which might be nearer)
    pub dist_sq: f32,
}
