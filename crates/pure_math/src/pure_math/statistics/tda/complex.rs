//! Simplicial complex construction and management.

use super::core::{MAX_TDA_POINTS, NdPointCloud, PointCloud, Simplex};
use super::space3d::PointCloud3D;
use crate::error::TdaError;
use std::collections::HashSet;

/// Upper bound on 2-simplices (triangles) retained in a Vietoris-Rips filtration or complex.
pub const MAX_FILTRATION_TRIANGLES: usize = 100_000;

/// A simplicial complex is a collection of simplices with the property
/// that every face of a simplex in the complex is also in the complex.
#[derive(Debug, Clone, PartialEq)]
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

            while self.simplices.len() <= dim {
                self.simplices.push(HashSet::new());
            }

            if self.simplices[dim].insert(current.clone()) {
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
/// This implementation constructs up to 2-dimensional simplices (triangles).
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
/// assert_eq!(complex.count_simplices(0), 3);
/// assert_eq!(complex.count_simplices(2), 1);
/// ```
#[verified_engine::verified]
pub fn vietoris_rips_complex(
    cloud: &PointCloud,
    radius: f64,
) -> Result<SimplicialComplex, TdaError> {
    let dist = cloud.distance_matrix();
    vietoris_rips_from_distances(&dist, radius)
}

/// Constructs a 2-skeleton Vietoris-Rips complex from a 3D [`PointCloud3D`].
///
/// # Errors
///
/// Returns [`TdaError::InvalidRadius`] if `radius` is negative or non-finite, or
/// [`TdaError::MatrixError`] if `cloud` exceeds [`MAX_TDA_POINTS`].
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

/// Constructs a Vietoris-Rips complex up to dimension 2 from an $N$-dimensional [`NdPointCloud`],
/// capping 2-simplices at [`MAX_FILTRATION_TRIANGLES`].
///
/// # Errors
///
/// Returns [`TdaError::InvalidRadius`] if `radius` is negative or non-finite.
#[verified_engine::verified]
pub fn vietoris_rips_complex_nd(
    cloud: &NdPointCloud,
    radius: f64,
) -> Result<SimplicialComplex, TdaError> {
    let dist = cloud.distance_matrix();
    let (complex, _truncated) =
        vietoris_rips_from_distance_matrix(&dist, radius, MAX_FILTRATION_TRIANGLES)?;
    Ok(complex)
}

/// Builds a Vietoris-Rips complex from a precomputed pairwise distance matrix, capping
/// 2-simplices at `max_triangles` and returning `(complex, triangles_truncated)`.
///
/// # Errors
///
/// Returns [`TdaError::InvalidRadius`] if `radius` is negative or non-finite,
/// [`TdaError::EmptyPointCloud`] if `dist` is empty, or [`TdaError::MatrixError`] if `dist`
/// is non-square, exceeds [`MAX_TDA_POINTS`], or contains negative or non-finite distances.
#[verified_engine::verified]
pub fn vietoris_rips_from_distance_matrix(
    dist: &[Vec<f64>],
    radius: f64,
    max_triangles: usize,
) -> Result<(SimplicialComplex, bool), TdaError> {
    if radius < 0.0 || !radius.is_finite() {
        return Err(TdaError::InvalidRadius { value: radius });
    }
    validate_distance_matrix(dist)?;

    let n = dist.len();
    let mut complex = SimplicialComplex::new();
    complex.simplices = vec![HashSet::with_capacity(n), HashSet::new(), HashSet::new()];

    for i in 0..n {
        complex.simplices[0].insert(Simplex { vertices: vec![i] });
    }

    add_edges_within_radius(&mut complex, dist, radius);
    let truncated = add_triangles_bounded(&mut complex, dist, radius, max_triangles);
    if complex.simplices[2].is_empty() {
        complex.simplices.pop();
        if complex.simplices[1].is_empty() {
            complex.simplices.pop();
        }
    }

    Ok((complex, truncated))
}

/// Builds a Vietoris-Rips simplicial complex (up to 2-simplices) from any precomputed
/// symmetric pairwise distance matrix.
///
/// # Errors
///
/// Returns [`TdaError::InvalidRadius`] if `radius` is negative or non-finite, or
/// [`TdaError::MatrixError`] if `dist` is non-square, exceeds [`MAX_TDA_POINTS`], or contains
/// negative/non-finite entries.
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
    let (complex, _truncated) =
        vietoris_rips_from_distance_matrix(dist, radius, MAX_FILTRATION_TRIANGLES)?;
    Ok(complex)
}

#[verified_engine::verified]
pub(crate) fn validate_distance_matrix(dist: &[Vec<f64>]) -> Result<(), TdaError> {
    let n = dist.len();
    if n == 0 {
        return Err(TdaError::EmptyPointCloud);
    }
    if n > MAX_TDA_POINTS {
        return Err(TdaError::MatrixError {
            reason: format!("distance matrix size {n} exceeds MAX_TDA_POINTS ({MAX_TDA_POINTS})"),
        });
    }
    for row in dist {
        if row.len() != n {
            return Err(TdaError::MatrixError {
                reason: format!("distance matrix must be {n}x{n}, got row len {}", row.len()),
            });
        }
        if row.iter().any(|&d| !d.is_finite() || d < 0.0) {
            return Err(TdaError::MatrixError {
                reason: "distance matrix entries must be non-negative and finite".to_string(),
            });
        }
    }
    Ok(())
}

#[verified_engine::verified]
fn add_edges_within_radius(complex: &mut SimplicialComplex, dist: &[Vec<f64>], radius: f64) {
    let n = dist.len();
    for (i, row) in dist.iter().enumerate().take(n) {
        for (j, &d) in row.iter().enumerate().take(n).skip(i + 1) {
            if d <= radius {
                complex.simplices[1].insert(Simplex {
                    vertices: vec![i, j],
                });
            }
        }
    }
}

#[verified_engine::verified]
fn add_triangles_bounded(
    complex: &mut SimplicialComplex,
    dist: &[Vec<f64>],
    radius: f64,
    max_triangles: usize,
) -> bool {
    let n = dist.len();
    let mut added = 0_usize;
    for (i, row) in dist.iter().enumerate().take(n) {
        for (j, &d_ij) in row.iter().enumerate().take(n).skip(i + 1) {
            if d_ij > radius {
                continue;
            }
            let row_j = &dist[j];
            for (k, &d_ik) in row.iter().enumerate().take(n).skip(j + 1) {
                if d_ik <= radius && row_j[k] <= radius {
                    if added >= max_triangles {
                        return true;
                    }
                    complex.simplices[2].insert(Simplex {
                        vertices: vec![i, j, k],
                    });
                    added += 1;
                }
            }
        }
    }
    false
}

#[cfg(test)]
#[path = "complex_tests.rs"]
mod tests;

// [cite:clinical_trials_statistics]
