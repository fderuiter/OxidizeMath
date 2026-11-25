//! Sub-segment critical-point and polyline intersection solvers (`poi`).
//!
//! Provides zero-allocation monotone cubic Hermite root finding, vertical line crossings,
//! 3-point Lagrange parabolic and linear-time plateau extrema refinement, single-pass
//! critical-point scanning, and monotone/general 2D polyline intersections.
// [cite:math_commons]

mod extrema;
mod interp;
mod intersect;

#[cfg(all(test, feature = "std"))]
mod tests;

pub use crate::series::{PointSeries, Sample2D};
#[cfg(feature = "alloc")]
pub use extrema::{LinePoiScan, detect_local_extrema, scan_line_pois};
pub use extrema::{detect_local_extrema_into, parabolic_vertex, peak_extremum};
pub use interp::{closest_threshold_crossing, threshold_crossings_into, y_at_x_into};
#[cfg(feature = "alloc")]
pub use interp::{interpolate_threshold_crossings, interpolate_y_at_x};
#[cfg(feature = "alloc")]
pub use intersect::intersect_polylines;
pub use intersect::{
    MAX_POLYLINE_INTERSECTIONS, eval_segment_at_x, intersect_polylines_into,
    segment_intersection_2d,
};

/// Sub-segment interior local extremum (maximum or minimum) refined via parabolic interpolation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalExtremum {
    /// Refined `[x, y]` coordinate of the local extremum.
    pub point: [f64; 2],
    /// `true` for a local maximum, `false` for a local minimum.
    pub is_max: bool,
}
