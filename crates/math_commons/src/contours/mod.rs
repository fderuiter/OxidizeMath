//! 2D marching-squares contour extraction, segment deduplication, nearest-neighbor
//! toolpath ordering, and Runge-Kutta streamline integration for vector exports.
//!
//! # Example
//!
//! ```rust
//! use math_commons::contours::{
//!     extract_contours, order_paths_nearest_neighbor, GridBounds2D,
//! };
//!
//! let bounds = GridBounds2D::new(-1.0, 1.0, -1.0, 1.0).unwrap();
//! let mut grid = Vec::with_capacity(21 * 21);
//! for iy in 0..21 {
//!     let y = -1.0 + 2.0 * (iy as f64) / 20.0;
//!     for ix in 0..21 {
//!         let x = -1.0 + 2.0 * (ix as f64) / 20.0;
//!         grid.push(x * x + y * y);
//!     }
//! }
//! let contours = extract_contours(&grid, 21, 21, &bounds, 0.25).unwrap();
//! let ordered = order_paths_nearest_neighbor(&contours);
//! assert_eq!(ordered.len(), 1);
//! assert!(ordered[0].closed);
//! ```

mod marching_squares;
mod path_order;
mod streamlines;

#[cfg(test)]
mod tests;

pub use marching_squares::{extract_contours, extract_multi_contours};
pub use path_order::{
    chain_segments, deduplicate_segments, order_paths_nearest_neighbor, pen_lift_segments,
    pen_lift_travel_distance,
};
pub use streamlines::{StreamlineConfig, trace_streamlines_rk4};

use crate::error::{MathError, MathResult};

/// Planar 2D coordinate `[x, y]`.
pub type Point2D = [f64; 2];

/// Undirected or directed 2D line segment `[p0, p1]`.
pub type Segment2D = [Point2D; 2];

/// Axis-aligned finite 2D rectangular domain bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridBounds2D {
    /// Minimum x-coordinate.
    pub min_x: f64,
    /// Maximum x-coordinate (`> min_x`).
    pub max_x: f64,
    /// Minimum y-coordinate.
    pub min_y: f64,
    /// Maximum y-coordinate (`> min_y`).
    pub max_y: f64,
}

impl GridBounds2D {
    /// Constructs validated finite 2D bounds with strictly positive width and height.
    #[verified_engine::verified]
    pub fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> MathResult<Self> {
        if !min_x.is_finite()
            || !max_x.is_finite()
            || !min_y.is_finite()
            || !max_y.is_finite()
            || max_x <= min_x
            || max_y <= min_y
        {
            return Err(MathError::NumericalError {
                reason: format!(
                    "Invalid GridBounds2D [{min_x}, {max_x}] x [{min_y}, {max_y}]: bounds must be finite and strictly ordered"
                ),
            });
        }
        Ok(Self {
            min_x,
            max_x,
            min_y,
            max_y,
        })
    }

    /// Returns `true` when `pt` has finite coordinates inside or on the boundary of `self`.
    #[must_use]
    #[verified_engine::verified]
    pub fn contains(&self, pt: Point2D) -> bool {
        pt[0].is_finite()
            && pt[1].is_finite()
            && pt[0] >= self.min_x
            && pt[0] <= self.max_x
            && pt[1] >= self.min_y
            && pt[1] <= self.max_y
    }

    /// Horizontal domain span `max_x - min_x`.
    #[must_use]
    #[verified_engine::verified]
    pub fn span_x(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// Vertical domain span `max_y - min_y`.
    #[must_use]
    #[verified_engine::verified]
    pub fn span_y(&self) -> f64 {
        self.max_y - self.min_y
    }
}

/// Chained 2D polyline or closed polygon contour.
#[derive(Debug, Clone, PartialEq)]
pub struct ContourPath {
    /// Ordered 2D vertices along the contour.
    ///
    /// When `closed` is `true`, the closing segment implicitly connects the last vertex
    /// back to `points[0]` (emit SVG `Z` or use [`Self::polyline_points`] for line plots).
    pub points: Vec<Point2D>,
    /// Whether the contour forms a closed loop.
    pub closed: bool,
}

impl ContourPath {
    /// Creates a new [`ContourPath`] from `points` and loop flag `closed`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(points: Vec<Point2D>, closed: bool) -> Self {
        Self { points, closed }
    }

    /// Number of stored vertices.
    #[must_use]
    #[verified_engine::verified]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the contour contains fewer than 2 vertices.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_empty(&self) -> bool {
        self.points.len() < 2
    }

    /// Euclidean arc length of the path (including the closing edge when `closed` is `true`).
    #[must_use]
    #[verified_engine::verified]
    pub fn arc_length(&self) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        for pair in self.points.windows(2) {
            total += euclidean_distance(pair[0], pair[1]);
        }
        if self.closed && self.points.len() >= 3 {
            total += euclidean_distance(self.points[self.points.len() - 1], self.points[0]);
        }
        total
    }

    /// Returns vertices suitable for open polyline plotters, appending `points[0]` when closed.
    #[must_use]
    #[verified_engine::verified]
    pub fn polyline_points(&self) -> Vec<Point2D> {
        let mut out = self.points.clone();
        if self.closed && out.len() >= 3 && euclidean_distance(out[0], out[out.len() - 1]) > 1e-12 {
            out.push(out[0]);
        }
        out
    }

    /// Ending pen position after tracing `self` from `points[0]`.
    #[must_use]
    #[verified_engine::verified]
    pub fn end_pen_position(&self) -> Option<Point2D> {
        if self.closed {
            self.points.first().copied()
        } else {
            self.points.last().copied()
        }
    }
}

#[verified_engine::verified]
pub(crate) fn euclidean_distance(a: Point2D, b: Point2D) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    dx.hypot(dy)
}

// [cite:math_commons]
