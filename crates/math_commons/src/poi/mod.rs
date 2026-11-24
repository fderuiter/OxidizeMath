//! Sub-segment critical-point and polyline intersection solvers (`poi`).
//!
//! Provides zero-allocation monotone cubic Hermite root finding, vertical line crossings,
//! 3-point Lagrange parabolic and linear-time plateau extrema refinement, single-pass
//! critical-point scanning, and monotone/general 2D polyline intersections.
// [cite:math_commons]

mod extrema;
mod interp;
mod intersect;

#[cfg(test)]
mod tests;

pub use crate::series::{PointSeries, Sample2D};
pub use extrema::{
    LinePoiScan, detect_local_extrema, detect_local_extrema_into, parabolic_vertex, peak_extremum,
    scan_line_pois,
};
pub use interp::{
    closest_threshold_crossing, interpolate_threshold_crossings, interpolate_y_at_x,
    threshold_crossings_into, y_at_x_into,
};
pub use intersect::{
    MAX_POLYLINE_INTERSECTIONS, eval_segment_at_x, intersect_polylines, intersect_polylines_into,
    segment_intersection_2d,
};

/// Sub-segment interior local extremum (maximum or minimum) refined via parabolic interpolation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LocalExtremum {
    /// Refined `[x, y]` coordinate of the local extremum.
    pub point: [f64; 2],
    /// `true` for a local maximum, `false` for a local minimum.
    pub is_max: bool,
}
