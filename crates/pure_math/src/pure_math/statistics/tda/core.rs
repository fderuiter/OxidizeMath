//! Core types for Topological Data Analysis.

pub use super::space3d::{
    Point3D, PointCloud3D, farthest_point_landmarks, farthest_point_landmarks_into,
    takens_delay_embedding_3d,
};
use crate::error::TdaError;

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

/// A collection of points in 2D space.
#[derive(Debug, Clone)]
pub struct PointCloud {
    #[allow(missing_docs)]
    pub points: Vec<Point2D>,
}

impl PointCloud {
    /// Creates a new point cloud.
    ///
    /// # Errors
    ///
    /// Returns `TdaError::EmptyPointCloud` if the given vector is empty.
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
        Ok(Self { points })
    }

    /// Returns the number of points in the cloud.
    #[verified_engine::verified]
    pub fn size(&self) -> usize {
        self.points.len()
    }

    /// Computes the distance matrix between all pairs of points.
    ///
    /// Returns a symmetric matrix where element (i, j) is the distance
    /// between points i and j.
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

        // Mirror the matrix
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..i {
                dist[i][j] = dist[j][i];
            }
        }

        dist
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_point2d_distance() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);

        let d = p1.distance(&p2);
        assert!((d - 5.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_point_cloud_empty() {
        let result = PointCloud::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_point_cloud_distance_matrix() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
        ];
        let cloud = PointCloud::new(points).unwrap();
        let dist = cloud.distance_matrix();

        assert_eq!(dist.len(), 3);
        assert!((dist[0][1] - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
        assert!((dist[0][2] - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
        assert!((dist[1][2] - 2.0_f64.sqrt()).abs() < math_commons::registry::TOLERANCE_FAST);

        // Symmetry
        assert_eq!(dist[0][1], dist[1][0]);
    }

    #[test]
    #[verified_engine::verified]
    fn test_simplex_vertex() {
        let s = Simplex::new(vec![0]).unwrap();
        assert_eq!(s.dimension(), 0);
        assert_eq!(s.faces().unwrap().len(), 0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_simplex_edge() {
        let s = Simplex::new(vec![0, 1]).unwrap();
        assert_eq!(s.dimension(), 1);
        assert_eq!(s.faces().unwrap().len(), 2);
    }

    #[test]
    #[verified_engine::verified]
    fn test_simplex_triangle() {
        let s = Simplex::new(vec![0, 1, 2]).unwrap();
        assert_eq!(s.dimension(), 2);
        assert_eq!(s.faces().unwrap().len(), 3);

        // All faces should be edges
        for face in s.faces().unwrap() {
            assert_eq!(face.dimension(), 1);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_simplex_sorted() {
        let s1 = Simplex::new(vec![2, 0, 1]).unwrap();
        let s2 = Simplex::new(vec![0, 1, 2]).unwrap();

        // Should be equal after sorting
        assert_eq!(s1, s2);
    }

    #[test]
    #[verified_engine::verified]
    fn test_simplex_duplicate_vertices() {
        let result = Simplex::new(vec![0, 1, 1]);
        assert!(result.is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_simplex_diameter() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
        ];
        let cloud = PointCloud::new(points).unwrap();
        let dist = cloud.distance_matrix();

        let triangle = Simplex::new(vec![0, 1, 2]).unwrap();
        let diameter = triangle.diameter(&dist);

        // Maximum distance is between points 1 and 2
        assert!((diameter - 2.0_f64.sqrt()).abs() < math_commons::registry::TOLERANCE_FAST);
    }
}
