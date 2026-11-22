//! Simplicial complex construction and management.

use super::core::{PointCloud, Simplex};
use super::space3d::PointCloud3D;
use crate::error::TdaError;
use std::collections::HashSet;

/// A simplicial complex is a collection of simplices with the property
/// that every face of a simplex in the complex is also in the complex.
#[derive(Debug, Clone)]
pub struct SimplicialComplex {
    /// All simplices in the complex, organized by dimension.
    /// `simplices[d]` contains all `d`-dimensional simplices.
    pub simplices: Vec<HashSet<Simplex>>,
}

impl SimplicialComplex {
    /// Creates a new empty simplicial complex.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self {
            simplices: Vec::new(),
        }
    }

    /// Adds a simplex to the complex.
    ///
    /// Automatically adds all faces of the simplex as well to maintain
    /// the simplicial complex property.
    ///
    /// # Errors
    ///
    /// Returns a [`TdaError::InvalidSimplex`] if generating the faces of the simplex fails
    /// (e.g., due to duplicate vertices or other invalid structural states).
    #[verified_engine::verified]
    pub fn add_simplex(&mut self, simplex: Simplex) -> Result<(), TdaError> {
        let mut stack = vec![simplex];

        while let Some(current) = stack.pop() {
            let dim = current.dimension();

            // Ensure we have enough dimension levels
            while self.simplices.len() <= dim {
                self.simplices.push(HashSet::new());
            }

            // Add the simplex
            if self.simplices[dim].insert(current.clone()) {
                // Recursively add all faces (push in reverse to maintain order if it matters, though HashSet doesn't care)
                for face in current.faces()? {
                    stack.push(face);
                }
            }
        }
        Ok(())
    }

    /// Returns the number of simplices of a given dimension.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::tda::{SimplicialComplex, Simplex};
    ///
    /// let mut complex = SimplicialComplex::new();
    /// complex.add_simplex(Simplex::new(vec![0, 1]).unwrap()).unwrap();
    ///
    /// assert_eq!(complex.count_simplices(0), 2); // 2 vertices
    /// assert_eq!(complex.count_simplices(1), 1); // 1 edge
    /// ```
    #[verified_engine::verified]
    pub fn count_simplices(&self, dimension: usize) -> usize {
        if dimension >= self.simplices.len() {
            0
        } else {
            self.simplices[dimension].len()
        }
    }

    /// Returns all simplices of a given dimension.
    #[verified_engine::verified]
    pub fn get_simplices(&self, dimension: usize) -> Vec<Simplex> {
        if dimension >= self.simplices.len() {
            Vec::new()
        } else {
            self.simplices[dimension].iter().cloned().collect()
        }
    }

    /// Returns the maximum dimension of any simplex in the complex.
    #[verified_engine::verified]
    pub fn dimension(&self) -> usize {
        if self.simplices.is_empty() {
            0
        } else {
            self.simplices.len() - 1
        }
    }

    /// Checks if the complex contains a given simplex.
    #[verified_engine::verified]
    pub fn contains(&self, simplex: &Simplex) -> bool {
        let dim = simplex.dimension();
        if dim >= self.simplices.len() {
            false
        } else {
            self.simplices[dim].contains(simplex)
        }
    }
}

impl Default for SimplicialComplex {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

/// Constructs a Vietoris-Rips complex at a given radius.
///
/// The Vietoris-Rips complex at radius ε includes:
/// - All vertices (0-simplices)
/// - An edge between vertices i and j if d(i,j) ≤ ε
/// - A triangle `[i, j, k]` if all pairwise distances ≤ ε
/// - Higher-dimensional simplices similarly
///
/// This implementation constructs up to 2-dimensional simplices (triangles).
///
/// # Arguments
///
/// * `cloud` - The point cloud
/// * `radius` - The radius parameter ε
///
/// # Returns
///
/// * `Result<SimplicialComplex, TdaError>` - The Vietoris-Rips complex or error
///
/// # Errors
///
/// Returns a [`TdaError::InvalidRadius`] if `radius` is negative or non-finite.
/// Additionally, returns a [`TdaError::InvalidSimplex`] if creating any of the required
/// simplices fails during the complex construction.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::tda::{
///     PointCloud, Point2D, vietoris_rips_complex
/// };
///
/// let points = vec![
///     Point2D::new(0.0, 0.0),
///     Point2D::new(1.0, 0.0),
///     Point2D::new(0.0, 1.0),
/// ];
/// let cloud = PointCloud::new(points).unwrap();
///
/// let complex = vietoris_rips_complex(&cloud, 1.5).unwrap();
/// println!("Vertices: {}", complex.count_simplices(0));
/// println!("Edges: {}", complex.count_simplices(1));
/// ```
#[verified_engine::verified]
pub fn vietoris_rips_complex(
    cloud: &PointCloud,
    radius: f64,
) -> Result<SimplicialComplex, TdaError> {
    let dist = cloud.distance_matrix();
    vietoris_rips_from_distances(&dist, radius)
}

/// Constructs a 2-skeleton Vietoris-Rips complex from a 3D point cloud.
///
/// # Errors
///
/// Returns [`TdaError::InvalidRadius`] if `radius` is negative or non-finite.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::tda::{
///     Point3D, PointCloud3D, vietoris_rips_complex_3d
/// };
///
/// let cloud = PointCloud3D::new(vec![
///     Point3D::new(0.0, 0.0, 0.0),
///     Point3D::new(1.0, 0.0, 0.0),
///     Point3D::new(0.0, 1.0, 0.0),
/// ]).unwrap();
/// let complex = vietoris_rips_complex_3d(&cloud, 1.5).unwrap();
/// assert_eq!(complex.count_simplices(2), 1);
/// ```
#[verified_engine::verified]
pub fn vietoris_rips_complex_3d(
    cloud: &PointCloud3D,
    radius: f64,
) -> Result<SimplicialComplex, TdaError> {
    let dist = cloud.distance_matrix();
    vietoris_rips_from_distances(&dist, radius)
}

/// Builds a Vietoris-Rips simplicial complex (up to 2-simplices) from any precomputed
/// symmetric pairwise distance matrix.
///
/// # Errors
///
/// Returns [`TdaError::InvalidRadius`] if `radius` is negative or non-finite, or
/// [`TdaError::MatrixError`] if `dist` is non-square or contains negative/non-finite entries.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::tda::vietoris_rips_from_distances;
///
/// let dist = vec![
///     vec![0.0, 1.0, 1.0],
///     vec![1.0, 0.0, 1.0],
///     vec![1.0, 1.0, 0.0],
/// ];
/// let complex = vietoris_rips_from_distances(&dist, 1.0).unwrap();
/// assert_eq!(complex.count_simplices(2), 1);
/// ```
#[verified_engine::verified]
pub fn vietoris_rips_from_distances(
    dist: &[Vec<f64>],
    radius: f64,
) -> Result<SimplicialComplex, TdaError> {
    if radius < 0.0 || !radius.is_finite() {
        return Err(TdaError::InvalidRadius { value: radius });
    }
    validate_distance_matrix(dist)?;

    let n = dist.len();
    let mut complex = SimplicialComplex::new();
    for i in 0..n {
        complex.add_simplex(Simplex::new(vec![i])?)?;
    }
    for (i, row) in dist.iter().enumerate().take(n) {
        for (j, &d_ij) in row.iter().enumerate().take(n).skip(i + 1) {
            if d_ij > radius {
                continue;
            }
            complex.add_simplex(Simplex::new(vec![i, j])?)?;
            let row_j = &dist[j];
            for (k, &d_ik) in row.iter().enumerate().take(n).skip(j + 1) {
                if d_ik <= radius && row_j[k] <= radius {
                    complex.add_simplex(Simplex::new(vec![i, j, k])?)?;
                }
            }
        }
    }
    Ok(complex)
}

#[verified_engine::verified]
pub(crate) fn validate_distance_matrix(dist: &[Vec<f64>]) -> Result<(), TdaError> {
    let n = dist.len();
    for row in dist {
        if row.len() != n || row.iter().any(|&d| !d.is_finite() || d < 0.0) {
            return Err(TdaError::MatrixError {
                reason: "Distance matrix must be square with finite non-negative entries"
                    .to_string(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::statistics::tda::core::Point2D;

    #[test]
    #[verified_engine::verified]
    fn test_simplicial_complex_add_edge() {
        let mut complex = SimplicialComplex::new();
        let edge = Simplex::new(vec![0, 1]).unwrap();
        complex.add_simplex(edge).unwrap();

        // Should have 2 vertices and 1 edge
        assert_eq!(complex.count_simplices(0), 2);
        assert_eq!(complex.count_simplices(1), 1);
    }

    #[test]
    #[verified_engine::verified]
    fn test_simplicial_complex_add_triangle() {
        let mut complex = SimplicialComplex::new();
        let triangle = Simplex::new(vec![0, 1, 2]).unwrap();
        complex.add_simplex(triangle).unwrap();

        // Should have 3 vertices, 3 edges, and 1 triangle
        assert_eq!(complex.count_simplices(0), 3);
        assert_eq!(complex.count_simplices(1), 3);
        assert_eq!(complex.count_simplices(2), 1);
    }

    #[test]
    #[verified_engine::verified]
    fn test_simplicial_complex_contains() {
        let mut complex = SimplicialComplex::new();
        let edge = Simplex::new(vec![0, 1]).unwrap();
        complex.add_simplex(edge.clone()).unwrap();

        assert!(complex.contains(&edge));
        assert!(!complex.contains(&Simplex::new(vec![1, 2]).unwrap()));
    }

    #[test]
    #[verified_engine::verified]
    fn test_vietoris_rips_three_points_small_radius() {
        // Three points forming a right triangle
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
        ];
        let cloud = PointCloud::new(points).unwrap();

        // Small radius: only vertices
        let complex = vietoris_rips_complex(&cloud, 0.5).unwrap();
        assert_eq!(complex.count_simplices(0), 3); // 3 vertices
        assert_eq!(complex.count_simplices(1), 0); // 0 edges
    }

    #[test]
    #[verified_engine::verified]
    fn test_vietoris_rips_three_points_medium_radius() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
        ];
        let cloud = PointCloud::new(points).unwrap();

        // Medium radius: vertices and 2 edges
        let complex = vietoris_rips_complex(&cloud, 1.0).unwrap();
        assert_eq!(complex.count_simplices(0), 3); // 3 vertices
        assert_eq!(complex.count_simplices(1), 2); // 2 edges (sides of length 1)
        assert_eq!(complex.count_simplices(2), 0); // 0 triangles
    }

    #[test]
    #[verified_engine::verified]
    fn test_vietoris_rips_three_points_large_radius() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
        ];
        let cloud = PointCloud::new(points).unwrap();

        // Large radius: complete complex
        let complex = vietoris_rips_complex(&cloud, 2.0).unwrap();
        assert_eq!(complex.count_simplices(0), 3); // 3 vertices
        assert_eq!(complex.count_simplices(1), 3); // 3 edges
        assert_eq!(complex.count_simplices(2), 1); // 1 triangle
    }

    #[test]
    #[verified_engine::verified]
    fn test_vietoris_rips_invalid_radius() {
        let points = vec![Point2D::new(0.0, 0.0), Point2D::new(1.0, 0.0)];
        let cloud = PointCloud::new(points).unwrap();

        assert!(vietoris_rips_complex(&cloud, -1.0).is_err());
        assert!(vietoris_rips_complex(&cloud, f64::NAN).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_vietoris_rips_square() {
        // Four points forming a square
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ];
        let cloud = PointCloud::new(points).unwrap();

        // Radius = 1.0: should have edges around the square
        let complex = vietoris_rips_complex(&cloud, 1.0).unwrap();
        assert_eq!(complex.count_simplices(0), 4); // 4 vertices
        assert_eq!(complex.count_simplices(1), 4); // 4 edges (sides)
        assert_eq!(complex.count_simplices(2), 0); // 0 triangles

        // Radius = sqrt(2): should have diagonals too
        let complex = vietoris_rips_complex(&cloud, 1.5).unwrap();
        assert_eq!(complex.count_simplices(0), 4); // 4 vertices
        assert_eq!(complex.count_simplices(1), 6); // 4 sides + 2 diagonals
        assert_eq!(complex.count_simplices(2), 4); // 4 triangles
    }

    #[test]
    #[verified_engine::verified]
    fn test_vietoris_rips_from_distances() {
        let dist = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];
        let complex = vietoris_rips_from_distances(&dist, 1.0).unwrap();
        assert_eq!(complex.count_simplices(0), 3);
        assert_eq!(complex.count_simplices(1), 2);
        assert_eq!(complex.count_simplices(2), 0);
    }
}
