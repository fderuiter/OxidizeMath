use std::ops::RangeInclusive;

use egui::{Pos2, Rect, Vec2, Vec2b, pos2, remap};

use crate::Axis;

use super::PlotPoint;

/// 2D bounding box of f64 precision.
///
/// The range of data values we show.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct PlotBounds {
    pub(crate) min: [f64; 2],
    pub(crate) max: [f64; 2],
}

impl PlotBounds {
    pub const NOTHING: Self = Self {
        min: [f64::INFINITY; 2],
        max: [-f64::INFINITY; 2],
    };

    #[inline]
    pub fn from_min_max(min: [f64; 2], max: [f64; 2]) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn min(&self) -> [f64; 2] {
        self.min
    }

    #[inline]
    pub fn max(&self) -> [f64; 2] {
        self.max
    }

    #[inline]
    pub fn new_symmetrical(half_extent: f64) -> Self {
        Self {
            min: [-half_extent; 2],
            max: [half_extent; 2],
        }
    }

    #[inline]
    pub fn is_finite(&self) -> bool {
        self.min[0].is_finite()
            && self.min[1].is_finite()
            && self.max[0].is_finite()
            && self.max[1].is_finite()
    }

    #[inline]
    pub fn is_finite_x(&self) -> bool {
        self.min[0].is_finite() && self.max[0].is_finite()
    }

    #[inline]
    pub fn is_finite_y(&self) -> bool {
        self.min[1].is_finite() && self.max[1].is_finite()
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        self.is_finite() && self.width() > 0.0 && self.height() > 0.0
    }

    #[inline]
    pub fn is_valid_x(&self) -> bool {
        self.is_finite_x() && self.width() > 0.0
    }

    #[inline]
    pub fn is_valid_y(&self) -> bool {
        self.is_finite_y() && self.height() > 0.0
    }

    #[inline]
    pub fn width(&self) -> f64 {
        self.max[0] - self.min[0]
    }

    #[inline]
    pub fn height(&self) -> f64 {
        self.max[1] - self.min[1]
    }

    #[inline]
    pub fn center(&self) -> PlotPoint {
        [
            emath::fast_midpoint(self.min[0], self.max[0]),
            emath::fast_midpoint(self.min[1], self.max[1]),
        ]
        .into()
    }

    /// Expand to include the given (x,y) value
    #[inline]
    pub fn extend_with(&mut self, value: &PlotPoint) {
        self.extend_with_x(value.x);
        self.extend_with_y(value.y);
    }

    /// Expand to include the given x coordinate
    #[inline]
    pub fn extend_with_x(&mut self, x: f64) {
        self.min[0] = self.min[0].min(x);
        self.max[0] = self.max[0].max(x);
    }

    /// Expand to include the given y coordinate
    #[inline]
    pub fn extend_with_y(&mut self, y: f64) {
        self.min[1] = self.min[1].min(y);
        self.max[1] = self.max[1].max(y);
    }

    #[inline]
    fn clamp_to_finite(&mut self) {
        for d in 0..2 {
            self.min[d] = self.min[d].clamp(f64::MIN, f64::MAX);
            if self.min[d].is_nan() {
                self.min[d] = 0.0;
            }

            self.max[d] = self.max[d].clamp(f64::MIN, f64::MAX);
            if self.max[d].is_nan() {
                self.max[d] = 0.0;
            }
        }
    }

    #[inline]
    pub fn expand_x(&mut self, pad: f64) {
        if pad.is_finite() {
            self.min[0] -= pad;
            self.max[0] += pad;
            self.clamp_to_finite();
        }
    }

    #[inline]
    pub fn expand_y(&mut self, pad: f64) {
        if pad.is_finite() {
            self.min[1] -= pad;
            self.max[1] += pad;
            self.clamp_to_finite();
        }
    }

    #[inline]
    pub fn merge_x(&mut self, other: &Self) {
        self.min[0] = self.min[0].min(other.min[0]);
        self.max[0] = self.max[0].max(other.max[0]);
    }

    #[inline]
    pub fn merge_y(&mut self, other: &Self) {
        self.min[1] = self.min[1].min(other.min[1]);
        self.max[1] = self.max[1].max(other.max[1]);
    }

    #[inline]
    pub fn set_x(&mut self, other: &Self) {
        self.min[0] = other.min[0];
        self.max[0] = other.max[0];
    }

    #[inline]
    pub fn set_x_center_width(&mut self, x: f64, width: f64) {
        self.min[0] = x - width / 2.0;
        self.max[0] = x + width / 2.0;
    }

    #[inline]
    pub fn set_y(&mut self, other: &Self) {
        self.min[1] = other.min[1];
        self.max[1] = other.max[1];
    }

    #[inline]
    pub fn set_y_center_height(&mut self, y: f64, height: f64) {
        self.min[1] = y - height / 2.0;
        self.max[1] = y + height / 2.0;
    }

    #[inline]
    pub fn merge(&mut self, other: &Self) {
        self.min[0] = self.min[0].min(other.min[0]);
        self.min[1] = self.min[1].min(other.min[1]);
        self.max[0] = self.max[0].max(other.max[0]);
        self.max[1] = self.max[1].max(other.max[1]);
    }

    #[inline]
    pub fn translate_x(&mut self, delta: f64) {
        if delta.is_finite() {
            self.min[0] += delta;
            self.max[0] += delta;
            self.clamp_to_finite();
        }
    }

    #[inline]
    pub fn translate_y(&mut self, delta: f64) {
        if delta.is_finite() {
            self.min[1] += delta;
            self.max[1] += delta;
            self.clamp_to_finite();
        }
    }

    #[inline]
    pub fn translate(&mut self, delta: (f64, f64)) {
        self.translate_x(delta.0);
        self.translate_y(delta.1);
    }

    #[inline]
    pub fn zoom(&mut self, zoom_factor: Vec2, center: PlotPoint) {
        self.min[0] = center.x + (self.min[0] - center.x) / (zoom_factor.x as f64);
        self.max[0] = center.x + (self.max[0] - center.x) / (zoom_factor.x as f64);
        self.min[1] = center.y + (self.min[1] - center.y) / (zoom_factor.y as f64);
        self.max[1] = center.y + (self.max[1] - center.y) / (zoom_factor.y as f64);
    }

    pub(crate) fn zoom_scaled(&mut self, zoom_factor: Vec2, center: PlotPoint, log_axis: Vec2b) {
        let centers = [center.x, center.y];
        for d in 0..2 {
            let zf = zoom_factor[d] as f64;
            if !zf.is_finite() || zf <= 0.0 {
                continue;
            }
            if log_axis[d] {
                let min_log = log10_clamped_bound(self.min[d]);
                let max_log = log10_clamped_max(min_log, self.max[d]);
                let c_log = if centers[d].is_finite() && centers[d] > 0.0 {
                    centers[d].log10().clamp(MIN_LOG_EXP, MAX_LOG_EXP)
                } else {
                    0.5 * (min_log + max_log)
                };
                let new_min = (c_log + (min_log - c_log) / zf).clamp(MIN_LOG_EXP, MAX_LOG_EXP);
                let new_max = (c_log + (max_log - c_log) / zf).clamp(MIN_LOG_EXP, MAX_LOG_EXP);
                if new_max > new_min {
                    let lo = exp10_clamped(new_min);
                    let hi = exp10_clamped(new_max);
                    if hi > lo {
                        self.min[d] = lo;
                        self.max[d] = hi;
                    }
                }
            } else {
                self.min[d] = centers[d] + (self.min[d] - centers[d]) / zf;
                self.max[d] = centers[d] + (self.max[d] - centers[d]) / zf;
            }
        }
    }

    #[inline]
    pub fn add_relative_margin_x(&mut self, margin_fraction: Vec2) {
        let width = self.width().max(0.0);
        self.expand_x(margin_fraction.x as f64 * width);
    }

    #[inline]
    pub fn add_relative_margin_y(&mut self, margin_fraction: Vec2) {
        let height = self.height().max(0.0);
        self.expand_y(margin_fraction.y as f64 * height);
    }

    pub(crate) fn add_log_margin_axis(&mut self, axis: usize, frac: f64) {
        sanitize_log_axis_pair(&mut self.min[axis], &mut self.max[axis]);
        if !frac.is_finite() || frac <= 0.0 {
            return;
        }
        let min_log = self.min[axis].log10();
        let max_log = self.max[axis].log10();
        let span = (max_log - min_log).max(1e-6);
        let pad = frac * span;
        let new_min_log = (min_log - pad).clamp(MIN_LOG_EXP, MAX_LOG_EXP - 1e-6);
        let new_max_log = (max_log + pad).clamp(new_min_log + 1e-6, MAX_LOG_EXP);
        let lo = exp10_clamped(new_min_log);
        let hi = exp10_clamped(new_max_log);
        if hi > lo {
            self.min[axis] = lo;
            self.max[axis] = hi;
        }
    }

    #[inline]
    pub fn range_x(&self) -> RangeInclusive<f64> {
        self.min[0]..=self.max[0]
    }

    #[inline]
    pub fn range_y(&self) -> RangeInclusive<f64> {
        self.min[1]..=self.max[1]
    }

    #[inline]
    pub fn make_x_symmetrical(&mut self) {
        let x_abs = self.min[0].abs().max(self.max[0].abs());
        self.min[0] = -x_abs;
        self.max[0] = x_abs;
    }

    #[inline]
    pub fn make_y_symmetrical(&mut self) {
        let y_abs = self.min[1].abs().max(self.max[1].abs());
        self.min[1] = -y_abs;
        self.max[1] = y_abs;
    }
}

const DEFAULT_MIN_LOG_BOUND: f64 = 1e-6;
const MIN_LOG_EXP: f64 = -300.0;
const MAX_LOG_EXP: f64 = 300.0;
const MIN_LOG_BOUND: f64 = 1e-300;
const MAX_LOG_BOUND: f64 = 1e300;

#[inline]
fn exp10_clamped(exp: f64) -> f64 {
    10.0_f64
        .powf(exp.clamp(MIN_LOG_EXP, MAX_LOG_EXP))
        .clamp(MIN_LOG_BOUND, MAX_LOG_BOUND)
}

#[inline]
fn sanitize_log_axis_pair(min: &mut f64, max: &mut f64) {
    if !min.is_finite() || *min <= 0.0 {
        *min = DEFAULT_MIN_LOG_BOUND;
    } else {
        *min = min.clamp(MIN_LOG_BOUND, 1e299);
    }
    if !max.is_finite() || *max <= *min {
        *max = (*min * 10.0).min(MAX_LOG_BOUND);
    } else {
        *max = max.min(MAX_LOG_BOUND);
    }
}

#[inline]
fn log10_clamped_bound(v: f64) -> f64 {
    if v.is_finite() && v > 0.0 {
        v.log10().clamp(MIN_LOG_EXP, MAX_LOG_EXP)
    } else {
        DEFAULT_MIN_LOG_BOUND.log10()
    }
}

#[inline]
fn log10_clamped_max(min_log: f64, max_v: f64) -> f64 {
    if max_v.is_finite() && max_v > 0.0 {
        let m = max_v.log10().clamp(MIN_LOG_EXP, MAX_LOG_EXP);
        if m > min_log {
            m
        } else {
            (min_log + 1.0).min(MAX_LOG_EXP)
        }
    } else {
        (min_log + 1.0).min(MAX_LOG_EXP)
    }
}

/// Contains the screen rectangle and the plot bounds and provides methods to transform between them.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Copy, Debug)]
pub struct PlotTransform {
    /// The screen rectangle.
    frame: Rect,

    /// The plot bounds.
    bounds: PlotBounds,

    /// Whether to always center the x-range or y-range of the bounds.
    centered: Vec2b,

    /// Whether to always invert the x and/or y axis
    inverted_axis: Vec2b,

    /// Whether to project the x and/or y axis in base-10 logarithmic space.
    #[cfg_attr(feature = "serde", serde(default))]
    log_axis: Vec2b,
}

impl PlotTransform {
    pub fn new(frame: Rect, bounds: PlotBounds, center_axis: impl Into<Vec2b>) -> Self {
        debug_assert!(
            0.0 <= frame.width() && 0.0 <= frame.height(),
            "Bad plot frame: {frame:?}"
        );
        let center_axis = center_axis.into();

        // Since the current Y bounds an affect the final X bounds and vice versa, we need to keep
        // the original version of the `bounds` before we start modifying it.
        let mut new_bounds = bounds;

        // Sanitize bounds.
        //
        // When a given bound axis is "thin" (e.g. width or height is 0) but finite, we center the
        // bounds around that value. If the other axis is "fat", we reuse its extent for the thin
        // axis, and default to +/- 1.0 otherwise.
        if !bounds.is_finite_x() {
            new_bounds.set_x(&PlotBounds::new_symmetrical(1.0));
        } else if bounds.width() <= 0.0 {
            new_bounds.set_x_center_width(
                bounds.center().x,
                if bounds.is_valid_y() {
                    bounds.height()
                } else {
                    1.0
                },
            );
        }

        if !bounds.is_finite_y() {
            new_bounds.set_y(&PlotBounds::new_symmetrical(1.0));
        } else if bounds.height() <= 0.0 {
            new_bounds.set_y_center_height(
                bounds.center().y,
                if bounds.is_valid_x() {
                    bounds.width()
                } else {
                    1.0
                },
            );
        }

        // Scale axes so that the origin is in the center.
        if center_axis.x {
            new_bounds.make_x_symmetrical();
        }
        if center_axis.y {
            new_bounds.make_y_symmetrical();
        }

        debug_assert!(
            new_bounds.is_valid(),
            "Bad final plot bounds: {new_bounds:?}"
        );

        Self {
            frame,
            bounds: new_bounds,
            centered: center_axis,
            inverted_axis: Vec2b::new(false, false),
            log_axis: Vec2b::new(false, false),
        }
    }

    pub fn new_with_invert_axis(
        frame: Rect,
        bounds: PlotBounds,
        center_axis: impl Into<Vec2b>,
        invert_axis: impl Into<Vec2b>,
    ) -> Self {
        let mut new = Self::new(frame, bounds, center_axis);
        new.inverted_axis = invert_axis.into();
        new
    }

    /// Returns whether logarithmic scaling is enabled on each axis.
    #[inline]
    pub fn log_axis(&self) -> Vec2b {
        self.log_axis
    }

    /// Configures base-10 logarithmic scaling per axis, clamping non-positive bounds to `1e-6`
    /// and disabling zero-centered symmetry on logarithmic axes.
    pub fn set_log_axis(&mut self, log_axis: impl Into<Vec2b>) {
        let log_axis = log_axis.into();
        self.log_axis = log_axis;
        if log_axis.x {
            self.centered.x = false;
            sanitize_log_axis_pair(&mut self.bounds.min[0], &mut self.bounds.max[0]);
        }
        if log_axis.y {
            self.centered.y = false;
            sanitize_log_axis_pair(&mut self.bounds.min[1], &mut self.bounds.max[1]);
        }
    }

    /// Returns `true` if `point` has finite coordinates and strictly positive values on active log axes.
    #[inline]
    pub fn is_point_valid_for_scale(&self, point: &PlotPoint) -> bool {
        point.x.is_finite()
            && point.y.is_finite()
            && (!self.log_axis.x || point.x > 0.0)
            && (!self.log_axis.y || point.y > 0.0)
    }

    /// ui-space rectangle.
    #[inline]
    pub fn frame(&self) -> &Rect {
        &self.frame
    }

    /// Plot-space bounds.
    #[inline]
    pub fn bounds(&self) -> &PlotBounds {
        &self.bounds
    }

    #[inline]
    pub fn set_bounds(&mut self, mut bounds: PlotBounds) {
        if self.log_axis.x {
            sanitize_log_axis_pair(&mut bounds.min[0], &mut bounds.max[0]);
        }
        if self.log_axis.y {
            sanitize_log_axis_pair(&mut bounds.min[1], &mut bounds.max[1]);
        }
        self.bounds = bounds;
    }

    pub fn translate_bounds(&mut self, mut delta_pos: (f64, f64)) {
        if self.centered.x && !self.log_axis.x {
            delta_pos.0 = 0.;
        }
        if self.centered.y && !self.log_axis.y {
            delta_pos.1 = 0.;
        }
        if self.log_axis.x {
            self.translate_log_axis(0, delta_pos.0);
        } else {
            self.bounds.translate_x(delta_pos.0 * self.dvalue_dpos()[0]);
        }
        if self.log_axis.y {
            self.translate_log_axis(1, delta_pos.1);
        } else {
            self.bounds.translate_y(delta_pos.1 * self.dvalue_dpos()[1]);
        }
    }

    fn translate_log_axis(&mut self, axis: usize, delta_px: f64) {
        if !delta_px.is_finite() || delta_px == 0.0 {
            return;
        }
        let span_px = if axis == 0 {
            self.frame.width() as f64
        } else {
            self.frame.height() as f64
        };
        if span_px <= 0.0 {
            return;
        }
        let flip = match (axis, self.inverted_axis[axis]) {
            (0, true) | (1, false) => -1.0,
            _ => 1.0,
        };
        let min_log = log10_clamped_bound(self.bounds.min[axis]);
        let max_log = log10_clamped_max(min_log, self.bounds.max[axis]);
        let delta_log = delta_px * flip * (max_log - min_log) / span_px;
        let lo = MIN_LOG_EXP - min_log;
        let hi = MAX_LOG_EXP - max_log;
        let clamped_delta = if lo <= hi {
            delta_log.clamp(lo, hi)
        } else {
            0.0
        };
        let lo = exp10_clamped(min_log + clamped_delta);
        let hi = exp10_clamped(max_log + clamped_delta);
        if hi > lo {
            self.bounds.min[axis] = lo;
            self.bounds.max[axis] = hi;
        }
    }

    /// Zoom by a relative factor with the given screen position as center.
    pub fn zoom(&mut self, zoom_factor: Vec2, center: Pos2) {
        let center = self.value_from_position(center);

        let mut new_bounds = self.bounds;
        new_bounds.zoom_scaled(zoom_factor, center, self.log_axis);

        if new_bounds.is_valid() {
            self.bounds = new_bounds;
        }
    }

    pub fn position_from_point_x(&self, value: f64) -> f32 {
        let screen_range = if self.inverted_axis[0] {
            (self.frame.right() as f64)..=(self.frame.left() as f64)
        } else {
            (self.frame.left() as f64)..=(self.frame.right() as f64)
        };
        if self.log_axis.x {
            let min_log = log10_clamped_bound(self.bounds.min[0]);
            let max_log = log10_clamped_max(min_log, self.bounds.max[0]);
            let val_log = if value.is_finite() && value > 0.0 {
                value.log10().clamp(MIN_LOG_EXP, MAX_LOG_EXP)
            } else {
                min_log.min(DEFAULT_MIN_LOG_BOUND.log10())
            };
            remap(val_log, min_log..=max_log, screen_range) as f32
        } else {
            remap(value, self.bounds.min[0]..=self.bounds.max[0], screen_range) as f32
        }
    }

    pub fn position_from_point_y(&self, value: f64) -> f32 {
        // negated y axis by default
        let screen_range = if self.inverted_axis[1] {
            (self.frame.top() as f64)..=(self.frame.bottom() as f64)
        } else {
            (self.frame.bottom() as f64)..=(self.frame.top() as f64)
        };
        if self.log_axis.y {
            let min_log = log10_clamped_bound(self.bounds.min[1]);
            let max_log = log10_clamped_max(min_log, self.bounds.max[1]);
            let val_log = if value.is_finite() && value > 0.0 {
                value.log10().clamp(MIN_LOG_EXP, MAX_LOG_EXP)
            } else {
                min_log.min(DEFAULT_MIN_LOG_BOUND.log10())
            };
            remap(val_log, min_log..=max_log, screen_range) as f32
        } else {
            remap(value, self.bounds.min[1]..=self.bounds.max[1], screen_range) as f32
        }
    }

    /// Screen/ui position from point on plot.
    pub fn position_from_point(&self, value: &PlotPoint) -> Pos2 {
        pos2(
            self.position_from_point_x(value.x),
            self.position_from_point_y(value.y),
        )
    }

    /// Plot point from screen/ui position.
    pub fn value_from_position(&self, pos: Pos2) -> PlotPoint {
        let x_screen = if self.inverted_axis[0] {
            (self.frame.right() as f64)..=(self.frame.left() as f64)
        } else {
            (self.frame.left() as f64)..=(self.frame.right() as f64)
        };
        let y_screen = if self.inverted_axis[1] {
            (self.frame.top() as f64)..=(self.frame.bottom() as f64)
        } else {
            (self.frame.bottom() as f64)..=(self.frame.top() as f64)
        };

        let x = if self.log_axis.x {
            let min_log = log10_clamped_bound(self.bounds.min[0]);
            let max_log = log10_clamped_max(min_log, self.bounds.max[0]);
            let log_x = remap(pos.x as f64, x_screen, min_log..=max_log);
            exp10_clamped(log_x)
        } else {
            remap(pos.x as f64, x_screen, self.bounds.range_x())
        };

        let y = if self.log_axis.y {
            let min_log = log10_clamped_bound(self.bounds.min[1]);
            let max_log = log10_clamped_max(min_log, self.bounds.max[1]);
            let log_y = remap(pos.y as f64, y_screen, min_log..=max_log);
            exp10_clamped(log_y)
        } else {
            remap(pos.y as f64, y_screen, self.bounds.range_y())
        };

        PlotPoint::new(x, y)
    }

    /// Computes screen-space spacing in ui points for a grid/axis mark on `axis`.
    pub fn step_spacing_in_points(&self, axis: Axis, step: super::GridMark) -> f32 {
        let iaxis = usize::from(axis);
        if !self.log_axis[iaxis] {
            return (self.dpos_dvalue()[iaxis] * step.step_size).abs() as f32;
        }
        let v = step.value;
        let s = step.step_size;
        if !v.is_finite() || !s.is_finite() || v <= 0.0 || s <= 0.0 {
            return 0.0;
        }
        let pos_at = |val: f64| match axis {
            Axis::X => self.position_from_point_x(val),
            Axis::Y => self.position_from_point_y(val),
        };
        let is_major_decade = (v - s).abs() <= 1e-4 * s;
        if is_major_decade {
            let neighbor = if (10.0 * v).is_finite() {
                10.0 * v
            } else {
                0.1 * v
            };
            let one_decade_px = (pos_at(neighbor) - pos_at(v)).abs();
            let exp = v.log10().round() as i32;
            let stride = if exp == 0 {
                256_u32
            } else {
                1_u32 << exp.unsigned_abs().trailing_zeros().min(8)
            };
            one_decade_px * (stride as f32)
        } else {
            let neighbor = if (v + s).is_finite() {
                v + s
            } else {
                (v - s).max(1e-300)
            };
            (pos_at(neighbor) - pos_at(v)).abs()
        }
    }

    /// Transform a rectangle of plot values to a screen-coordinate rectangle.
    ///
    /// This typically means that the rect is mirrored vertically (top becomes bottom and vice versa),
    /// since the plot's coordinate system has +Y up, while egui has +Y down.
    pub fn rect_from_values(&self, value1: &PlotPoint, value2: &PlotPoint) -> Rect {
        let pos1 = self.position_from_point(value1);
        let pos2 = self.position_from_point(value2);

        let mut rect = Rect::NOTHING;
        rect.extend_with(pos1);
        rect.extend_with(pos2);
        rect
    }

    /// delta position / delta value = how many ui points per step in the X axis in "plot space"
    pub fn dpos_dvalue_x(&self) -> f64 {
        let flip = if self.inverted_axis[0] { -1.0 } else { 1.0 };
        flip * (self.frame.width() as f64) / self.bounds.width()
    }

    /// delta position / delta value = how many ui points per step in the Y axis in "plot space"
    pub fn dpos_dvalue_y(&self) -> f64 {
        let flip = if self.inverted_axis[1] { 1.0 } else { -1.0 };
        flip * (self.frame.height() as f64) / self.bounds.height()
    }

    /// delta position / delta value = how many ui points per step in "plot space"
    pub fn dpos_dvalue(&self) -> [f64; 2] {
        [self.dpos_dvalue_x(), self.dpos_dvalue_y()]
    }

    /// delta value / delta position = how much ground do we cover in "plot space" per ui point?
    pub fn dvalue_dpos(&self) -> [f64; 2] {
        [1.0 / self.dpos_dvalue_x(), 1.0 / self.dpos_dvalue_y()]
    }

    /// scale.x/scale.y ratio.
    ///
    /// If 1.0, it means the scale factor is the same in both axes.
    fn aspect(&self) -> f64 {
        let rw = self.frame.width() as f64;
        let rh = self.frame.height() as f64;
        (self.bounds.width() / rw) / (self.bounds.height() / rh)
    }

    /// Sets the aspect ratio by expanding the x- or y-axis.
    ///
    /// This never contracts, so we don't miss out on any data.
    pub(crate) fn set_aspect_by_expanding(&mut self, aspect: f64) {
        if self.log_axis.any() {
            return;
        }
        let current_aspect = self.aspect();

        let epsilon = 1e-5;
        if (current_aspect - aspect).abs() < epsilon {
            // Don't make any changes when the aspect is already almost correct.
            return;
        }

        if current_aspect < aspect {
            self.bounds
                .expand_x((aspect / current_aspect - 1.0) * self.bounds.width() * 0.5);
        } else {
            self.bounds
                .expand_y((current_aspect / aspect - 1.0) * self.bounds.height() * 0.5);
        }
    }

    /// Sets the aspect ratio by changing either the X or Y axis (callers choice).
    pub(crate) fn set_aspect_by_changing_axis(&mut self, aspect: f64, axis: Axis) {
        if self.log_axis.any() {
            return;
        }
        let current_aspect = self.aspect();

        let epsilon = 1e-5;
        if (current_aspect - aspect).abs() < epsilon {
            // Don't make any changes when the aspect is already almost correct.
            return;
        }

        match axis {
            Axis::X => {
                self.bounds
                    .expand_x((aspect / current_aspect - 1.0) * self.bounds.width() * 0.5);
            }
            Axis::Y => {
                self.bounds
                    .expand_y((current_aspect / aspect - 1.0) * self.bounds.height() * 0.5);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp10_clamped_and_extreme_log_bounds_stay_within_domain() {
        assert_eq!(exp10_clamped(-300.0), 1e-300);
        assert_eq!(exp10_clamped(-500.0), 1e-300);
        assert_eq!(exp10_clamped(300.0), 1e300);
        assert_eq!(exp10_clamped(500.0), 1e300);

        let mut bounds = PlotBounds::from_min_max([1e-295, 1e-295], [1e295, 1e295]);
        bounds.add_log_margin_axis(0, 0.5);
        bounds.add_log_margin_axis(1, 0.5);
        assert!(bounds.min()[0] >= 1e-300 && bounds.max()[0] <= 1e300);
        assert!(bounds.min()[1] >= 1e-300 && bounds.max()[1] <= 1e300);

        let frame = Rect::from_min_size(Pos2::ZERO, emath::vec2(200.0, 200.0));
        let mut tf = PlotTransform::new(frame, bounds, Vec2b::FALSE);
        tf.set_log_axis(Vec2b::TRUE);
        let lo_pt = tf.value_from_position(Pos2::new(-1000.0, 2000.0));
        let hi_pt = tf.value_from_position(Pos2::new(2000.0, -1000.0));
        assert!(lo_pt.x >= 1e-300 && lo_pt.y >= 1e-300);
        assert!(hi_pt.x <= 1e300 && hi_pt.y <= 1e300);
    }
}
