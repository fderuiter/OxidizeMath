//! 3D point clouds, farthest-point landmarking, and Takens delay embeddings.

use crate::error::TdaError;

/// A point in 3D Euclidean space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Z coordinate.
    pub z: f64,
}

impl Point3D {
    /// Creates a new 3D point.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::tda::Point3D;
    ///
    /// let p = Point3D::new(1.0, 2.0, 3.0);
    /// assert_eq!(p.z, 3.0);
    /// ```
    #[verified_engine::verified]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Returns `true` if all three coordinates are finite numbers.
    #[verified_engine::verified]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Computes squared Euclidean distance to another 3D point.
    #[verified_engine::verified]
    pub fn distance_squared(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    /// Computes Euclidean distance to another 3D point.
    #[verified_engine::verified]
    pub fn distance(&self, other: &Point3D) -> f64 {
        self.distance_squared(other).sqrt()
    }
}

/// A validated finite point cloud in 3D space.
#[derive(Debug, Clone, PartialEq)]
pub struct PointCloud3D {
    /// Points belonging to the 3D cloud.
    pub points: Vec<Point3D>,
}

impl PointCloud3D {
    /// Creates a validated 3D point cloud.
    ///
    /// # Errors
    ///
    /// Returns [`TdaError::EmptyPointCloud`] if `points` is empty or
    /// [`TdaError::MatrixError`] if any coordinate is non-finite.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::tda::{Point3D, PointCloud3D};
    ///
    /// let cloud = PointCloud3D::new(vec![
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 0.0, 0.0),
    /// ]).unwrap();
    /// assert_eq!(cloud.size(), 2);
    /// ```
    #[verified_engine::verified]
    pub fn new(points: Vec<Point3D>) -> Result<Self, TdaError> {
        if points.is_empty() {
            return Err(TdaError::EmptyPointCloud);
        }
        if !points.iter().all(Point3D::is_finite) {
            return Err(TdaError::MatrixError {
                reason: "PointCloud3D contains non-finite coordinates".to_string(),
            });
        }
        Ok(Self { points })
    }

    /// Returns the number of points in the cloud.
    #[verified_engine::verified]
    pub fn size(&self) -> usize {
        self.points.len()
    }

    /// Computes the symmetric pairwise Euclidean distance matrix.
    #[verified_engine::verified]
    pub fn distance_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.points.len();
        let mut dist = vec![vec![0.0; n]; n];
        let _ = self.fill_distance_matrix(&mut dist);
        dist
    }

    /// Fills a preallocated `n x n` distance matrix in-place without allocating.
    ///
    /// # Errors
    ///
    /// Returns [`TdaError::MatrixError`] if `dist` does not have dimensions `n x n`.
    #[verified_engine::verified]
    pub fn fill_distance_matrix(&self, dist: &mut [Vec<f64>]) -> Result<(), TdaError> {
        let n = self.points.len();
        if dist.len() != n || dist.iter().any(|row| row.len() != n) {
            return Err(TdaError::MatrixError {
                reason: "Distance matrix buffer dimensions do not match cloud size".to_string(),
            });
        }
        for (i, row) in dist.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = if i == j {
                    0.0
                } else {
                    self.points[i].distance(&self.points[j])
                };
            }
        }
        Ok(())
    }
}

/// Selects up to `max_landmarks` deterministic greedy farthest-point landmarks from `points`.
///
/// Starts at `points[0]` and iteratively chooses the point maximizing minimum distance to
/// previously selected landmarks, preserving global attractor geometry across lobes.
///
/// # Errors
///
/// Returns [`TdaError::EmptyPointCloud`] when `points` is empty,
/// [`TdaError::InsufficientPoints`] when `max_landmarks == 0`, or
/// [`TdaError::MatrixError`] when any coordinate is non-finite.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::tda::{Point3D, farthest_point_landmarks};
///
/// let pts = vec![
///     Point3D::new(0.0, 0.0, 0.0),
///     Point3D::new(0.1, 0.0, 0.0),
///     Point3D::new(10.0, 0.0, 0.0),
/// ];
/// let landmarks = farthest_point_landmarks(&pts, 2).unwrap();
/// assert_eq!(landmarks.len(), 2);
/// assert_eq!(landmarks[1], Point3D::new(10.0, 0.0, 0.0));
/// ```
#[verified_engine::verified]
pub fn farthest_point_landmarks(
    points: &[Point3D],
    max_landmarks: usize,
) -> Result<Vec<Point3D>, TdaError> {
    if points.is_empty() {
        return Err(TdaError::EmptyPointCloud);
    }
    if max_landmarks == 0 {
        return Err(TdaError::InsufficientPoints {
            required: 1,
            actual: 0,
        });
    }
    let target = points.len().min(max_landmarks);
    let mut min_dist_sq = vec![f64::INFINITY; points.len()];
    let mut out = vec![Point3D::new(0.0, 0.0, 0.0); target];
    let written = farthest_point_landmarks_into(points, max_landmarks, &mut min_dist_sq, &mut out)?;
    out.truncate(written);
    Ok(out)
}

/// Zero-allocation farthest-point landmarking into caller-supplied scratch and output slices.
///
/// # Errors
///
/// Returns [`TdaError`] if `points` is empty, `max_landmarks == 0`, any point is non-finite,
/// or scratch buffers are smaller than required.
#[verified_engine::verified]
pub fn farthest_point_landmarks_into(
    points: &[Point3D],
    max_landmarks: usize,
    min_dist_sq: &mut [f64],
    out: &mut [Point3D],
) -> Result<usize, TdaError> {
    if points.is_empty() {
        return Err(TdaError::EmptyPointCloud);
    }
    if max_landmarks == 0 {
        return Err(TdaError::InsufficientPoints {
            required: 1,
            actual: 0,
        });
    }
    if !points.iter().all(Point3D::is_finite) {
        return Err(TdaError::MatrixError {
            reason: "Input points contain non-finite coordinates".to_string(),
        });
    }
    let target = points.len().min(max_landmarks);
    if min_dist_sq.len() < points.len() || out.len() < target {
        return Err(TdaError::MatrixError {
            reason: "Landmark scratch buffer is too small".to_string(),
        });
    }
    if points.len() <= max_landmarks {
        out[..target].copy_from_slice(points);
        return Ok(target);
    }
    for slot in min_dist_sq.iter_mut().take(points.len()) {
        *slot = f64::INFINITY;
    }
    let mut chosen = points[0];
    out[0] = chosen;
    for out_slot in out.iter_mut().take(target).skip(1) {
        let mut best_idx = 0usize;
        let mut best_dist = -1.0_f64;
        for (i, pt) in points.iter().enumerate() {
            let d2 = pt.distance_squared(&chosen);
            if d2 < min_dist_sq[i] {
                min_dist_sq[i] = d2;
            }
            if min_dist_sq[i] > best_dist {
                best_dist = min_dist_sq[i];
                best_idx = i;
            }
        }
        chosen = points[best_idx];
        *out_slot = chosen;
    }
    Ok(target)
}

/// Constructs a 3D Takens delay-coordinate embedding `(x(t), x(t - lag), x(t - 2 * lag))`
/// from a scalar time series `x(t)`.
///
/// # Errors
///
/// Returns [`TdaError::InvalidDimension`] if `lag == 0`, [`TdaError::InsufficientPoints`]
/// if `series.len() <= 2 * lag`, or [`TdaError::MatrixError`] if `series` contains non-finite values.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::tda::takens_delay_embedding_3d;
///
/// let series = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let cloud = takens_delay_embedding_3d(&series, 1).unwrap();
/// assert_eq!(cloud.size(), 3);
/// assert_eq!(cloud.points[0].x, 3.0);
/// assert_eq!(cloud.points[0].y, 2.0);
/// assert_eq!(cloud.points[0].z, 1.0);
/// ```
#[verified_engine::verified]
pub fn takens_delay_embedding_3d(series: &[f64], lag: usize) -> Result<PointCloud3D, TdaError> {
    if lag == 0 {
        return Err(TdaError::InvalidDimension { dimension: 0 });
    }
    let required = 2usize.saturating_mul(lag).saturating_add(1);
    if series.len() < required {
        return Err(TdaError::InsufficientPoints {
            required,
            actual: series.len(),
        });
    }
    if !series.iter().all(|v| v.is_finite()) {
        return Err(TdaError::MatrixError {
            reason: "Takens time series contains non-finite observations".to_string(),
        });
    }
    let mut embedded = Vec::with_capacity(series.len() - 2 * lag);
    for t in (2 * lag)..series.len() {
        embedded.push(Point3D::new(
            series[t],
            series[t - lag],
            series[t - 2 * lag],
        ));
    }
    PointCloud3D::new(embedded)
}

#[cfg(test)]
#[path = "space3d_tests.rs"]
mod tests;
