//! Core types for Topological Data Analysis.

use crate::error::TdaError;

/// Maximum number of points supported in bounded Vietoris-Rips point clouds.
pub const MAX_TDA_POINTS: usize = 250;

/// Maximum ambient Euclidean dimension $\mathbb{R}^d$ supported for [`NdPointCloud`].
pub const MAX_TDA_DIMENSION: usize = 32;

/// A point in 2D space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    #[allow(missing_docs)]
    pub x: f64,
    #[allow(missing_docs)]
    pub y: f64,
}

impl Point2D {
    /// Creates a new 2D point.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::tda::Point2D;
    ///
    /// let p = Point2D::new(1.0, 2.0);
    /// assert_eq!(p.x, 1.0);
    /// assert_eq!(p.y, 2.0);
    /// ```
    #[verified_engine::verified]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Computes Euclidean distance to another point.
    ///
    /// Formula: d = √((x₁ - x₂)² + (y₁ - y₂)²)
    #[verified_engine::verified]
    pub fn distance(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// A collection of points in 2D space bounded to [`MAX_TDA_POINTS`].
#[derive(Debug, Clone)]
pub struct PointCloud {
    #[allow(missing_docs)]
    pub points: Vec<Point2D>,
}

impl PointCloud {
    /// Creates a new bounded 2D point cloud.
    ///
    /// # Errors
    ///
    /// Returns [`TdaError::EmptyPointCloud`] if `points` is empty, or [`TdaError::MatrixError`]
    /// if `points.len() > MAX_TDA_POINTS` or any coordinate is non-finite.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::tda::{PointCloud, Point2D};
    ///
    /// let points = vec![
    ///     Point2D::new(0.0, 0.0),
    ///     Point2D::new(1.0, 0.0),
    ///     Point2D::new(0.0, 1.0),
    /// ];
    /// let cloud = PointCloud::new(points).unwrap();
    /// assert_eq!(cloud.size(), 3);
    /// ```
    #[verified_engine::verified]
    pub fn new(points: Vec<Point2D>) -> Result<Self, TdaError> {
        if points.is_empty() {
            return Err(TdaError::EmptyPointCloud);
        }
        if points.len() > MAX_TDA_POINTS {
            return Err(TdaError::MatrixError {
                reason: format!(
                    "point cloud size {} exceeds MAX_TDA_POINTS ({MAX_TDA_POINTS})",
                    points.len()
                ),
            });
        }
        if points.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
            return Err(TdaError::MatrixError {
                reason: "point cloud coordinates must be finite".to_string(),
            });
        }
        Ok(Self { points })
    }

    /// Returns the number of points in the cloud.
    #[verified_engine::verified]
    pub fn size(&self) -> usize {
        self.points.len()
    }

    /// Computes the distance matrix between all pairs of points.
    ///
    /// Returns a symmetric matrix where element `(i, j)` is the Euclidean distance
    /// between points `i` and `j`.
    #[verified_engine::verified]
    pub fn distance_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.points.len();
        let mut dist = vec![vec![0.0; n]; n];

        for (i, row) in dist.iter_mut().enumerate().take(n) {
            for (j, val) in row.iter_mut().enumerate().take(n).skip(i + 1) {
                let d = self.points[i].distance(&self.points[j]);
                *val = d;
            }
        }

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..i {
                dist[i][j] = dist[j][i];
            }
        }

        dist
    }
}

/// A bounded collection of points in $\mathbb{R}^d$ ($1 \le d \le 32$, $1 \le n \le 250$).
#[derive(Debug, Clone, PartialEq)]
pub struct NdPointCloud {
    points: Vec<Vec<f64>>,
    dimension: usize,
}

impl NdPointCloud {
    /// Creates a validated $d$-dimensional point cloud.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`TdaError::EmptyPointCloud`] if `points` is empty.
    /// - [`TdaError::InvalidDimension`] if $d = 0$, $d > 32$, or rows have inconsistent lengths.
    /// - [`TdaError::MatrixError`] if `points.len() > 250` or any coordinate is non-finite.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::tda::NdPointCloud;
    ///
    /// let cloud = NdPointCloud::new(vec![
    ///     vec![0.0, 0.0, 1.0],
    ///     vec![3.0, 4.0, 1.0],
    /// ]).unwrap();
    /// assert_eq!(cloud.size(), 2);
    /// assert_eq!(cloud.dimension(), 3);
    /// assert!((cloud.distance_matrix()[0][1] - 5.0).abs() < 1e-12);
    /// ```
    #[verified_engine::verified]
    pub fn new(points: Vec<Vec<f64>>) -> Result<Self, TdaError> {
        if points.is_empty() {
            return Err(TdaError::EmptyPointCloud);
        }
        if points.len() > MAX_TDA_POINTS {
            return Err(TdaError::MatrixError {
                reason: format!(
                    "point cloud size {} exceeds MAX_TDA_POINTS ({MAX_TDA_POINTS})",
                    points.len()
                ),
            });
        }
        let dim = points[0].len();
        if dim == 0 || dim > MAX_TDA_DIMENSION {
            return Err(TdaError::InvalidDimension { dimension: dim });
        }
        for row in &points {
            if row.len() != dim {
                return Err(TdaError::InvalidDimension {
                    dimension: row.len(),
                });
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err(TdaError::MatrixError {
                    reason: "point cloud coordinates must be finite".to_string(),
                });
            }
        }
        Ok(Self {
            points,
            dimension: dim,
        })
    }

    /// Lifts a 2D [`PointCloud`] into a 2-dimensional [`NdPointCloud`].
    ///
    /// # Errors
    ///
    /// Propagates validation errors from [`Self::new`].
    #[verified_engine::verified]
    pub fn from_point_cloud_2d(cloud: &PointCloud) -> Result<Self, TdaError> {
        let rows = cloud.points.iter().map(|p| vec![p.x, p.y]).collect();
        Self::new(rows)
    }

    /// Returns the number of points in the cloud.
    #[verified_engine::verified]
    pub fn size(&self) -> usize {
        self.points.len()
    }

    /// Returns the ambient Euclidean dimension $d$.
    #[verified_engine::verified]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the coordinate rows of the cloud.
    #[verified_engine::verified]
    pub fn points(&self) -> &[Vec<f64>] {
        &self.points
    }

    /// Computes the symmetric pairwise Euclidean distance matrix in $\mathbb{R}^d$.
    #[verified_engine::verified]
    pub fn distance_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.points.len();
        let mut dist = vec![vec![0.0; n]; n];

        for (i, row) in dist.iter_mut().enumerate().take(n) {
            let pi = &self.points[i];
            for (j, val) in row.iter_mut().enumerate().take(n).skip(i + 1) {
                *val = euclidean_slice_distance(pi, &self.points[j]);
            }
        }

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..i {
                dist[i][j] = dist[j][i];
            }
        }

        dist
    }

    /// Projects two chosen coordinate axes `(axis_x, axis_y)` into a 2D [`PointCloud`].
    ///
    /// # Errors
    ///
    /// Returns [`TdaError::InvalidDimension`] if `axis_x >= self.dimension` or `axis_y >= self.dimension`.
    #[verified_engine::verified]
    pub fn project_2d(&self, axis_x: usize, axis_y: usize) -> Result<PointCloud, TdaError> {
        if axis_x >= self.dimension {
            return Err(TdaError::InvalidDimension { dimension: axis_x });
        }
        if axis_y >= self.dimension {
            return Err(TdaError::InvalidDimension { dimension: axis_y });
        }
        let projected = self
            .points
            .iter()
            .map(|row| Point2D::new(row[axis_x], row[axis_y]))
            .collect();
        PointCloud::new(projected)
    }
}

#[verified_engine::verified]
fn euclidean_slice_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut sum_sq = 0.0_f64;
    for (&va, &vb) in a.iter().zip(b.iter()) {
        let diff = va - vb;
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

/// A simplex is a generalization of a triangle to arbitrary dimensions.
///
/// - 0-simplex: vertex (1 point)
/// - 1-simplex: edge (2 points)
/// - 2-simplex: triangle (3 points)
/// - 3-simplex: tetrahedron (4 points)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Simplex {
    /// Indices of vertices in the point cloud.
    /// Vertices are stored in sorted order for canonical representation.
    pub vertices: Vec<usize>,
}

impl Simplex {
    /// Creates a new simplex from vertex indices.
    ///
    /// Vertices are automatically sorted for canonical representation.
    ///
    /// # Errors
    ///
    /// Returns `TdaError::InvalidSimplex` if the vector is empty or contains duplicate indices.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::tda::Simplex;
    ///
    /// let simplex = Simplex::new(vec![0, 1, 2]).unwrap();
    /// assert_eq!(simplex.dimension(), 2); // Triangle
    /// ```
    #[verified_engine::verified]
    pub fn new(mut vertices: Vec<usize>) -> Result<Self, TdaError> {
        if vertices.is_empty() {
            return Err(TdaError::InvalidSimplex {
                reason: "Empty vertex list".to_string(),
            });
        }

        // Sort for canonical representation
        vertices.sort_unstable();

        // Check for duplicate vertices
        for i in 1..vertices.len() {
            if vertices[i] == vertices[i - 1] {
                return Err(TdaError::InvalidSimplex {
                    reason: "Duplicate vertices".to_string(),
                });
            }
        }

        Ok(Self { vertices })
    }

    /// Returns the dimension of the simplex.
    ///
    /// A simplex with (n+1) vertices has dimension n.
    #[verified_engine::verified]
    pub fn dimension(&self) -> usize {
        self.vertices.len() - 1
    }

    /// Returns all faces (sub-simplices of dimension one less).
    ///
    /// For an edge `[0,1]`, the faces are vertices `[0]` and `[1]`.
    /// For a triangle `[0,1,2]`, the faces are edges `[0,1]`, `[0,2]`, `[1,2]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the created face simplex is invalid (which should not
    /// conceptually happen due to the invariants of a valid parent simplex).
    ///
    /// # Examples
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::tda::Simplex;
    ///
    /// let triangle = Simplex::new(vec![0, 1, 2]).unwrap();
    /// let faces = triangle.faces().unwrap();
    ///
    /// assert_eq!(faces.len(), 3);
    /// assert!(faces.contains(&Simplex::new(vec![0, 1]).unwrap()));
    /// assert!(faces.contains(&Simplex::new(vec![0, 2]).unwrap()));
    /// assert!(faces.contains(&Simplex::new(vec![1, 2]).unwrap()));
    /// ```
    #[verified_engine::verified]
    pub fn faces(&self) -> Result<Vec<Simplex>, TdaError> {
        if self.vertices.len() == 1 {
            return Ok(vec![]); // 0-simplex has no faces
        }

        let mut faces = Vec::new();
        for i in 0..self.vertices.len() {
            let mut face_vertices = self.vertices.clone();
            face_vertices.remove(i);
            faces.push(Simplex::new(face_vertices)?);
        }

        Ok(faces)
    }

    /// Computes the diameter of the simplex in the point cloud.
    ///
    /// The diameter is the maximum distance between any pair of vertices.
    #[verified_engine::verified]
    pub fn diameter(&self, distance_matrix: &[Vec<f64>]) -> f64 {
        let mut max_dist = 0.0;
        for i in 0..self.vertices.len() {
            for j in (i + 1)..self.vertices.len() {
                let vi = self.vertices[i];
                let vj = self.vertices[j];
                let d = distance_matrix[vi][vj];
                if d > max_dist {
                    max_dist = d;
                }
            }
        }
        max_dist
    }
}

impl PartialOrd for Simplex {
    #[verified_engine::verified]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Simplex {
    /// Lexicographic ordering on vertices.
    #[verified_engine::verified]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.vertices.cmp(&other.vertices)
    }
}

// [cite:clinical_trials_statistics]

#[cfg(test)]
#[path = "core_tests.rs"]
mod tests;
