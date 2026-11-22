//! Persistence computation and barcode generation.

use super::complex::{validate_distance_matrix, vietoris_rips_from_distances};
use super::core::PointCloud;
use super::homology::{betti_number_0, betti_number_1};
use crate::error::TdaError;

/// A persistence interval representing the lifetime of a topological feature.
///
/// A feature "births" at radius `birth` and "dies" at radius `death`.
/// The persistence (lifetime) is `death - birth`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersistenceInterval {
    /// The radius at which the feature appears.
    pub birth: f64,
    /// The radius at which the feature disappears.
    pub death: f64,
    /// The dimension of the feature (0 for components, 1 for holes).
    pub dimension: usize,
}

impl PersistenceInterval {
    /// Creates a new persistence interval.
    ///
    /// # Errors
    ///
    /// Returns [`TdaError::InvalidRadius`] if `birth` is negative, `death < birth`,
    /// or either endpoint is non-finite.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::tda::PersistenceInterval;
    /// use math_explorer::error::TdaError;
    ///
    /// let interval = PersistenceInterval::new(0.5, 2.0, 1).unwrap();
    /// assert_eq!(interval.persistence(), 1.5);
    /// assert!(matches!(
    ///     PersistenceInterval::new(2.0, 1.0, 1),
    ///     Err(TdaError::InvalidRadius { .. })
    /// ));
    /// ```
    #[verified_engine::verified]
    pub fn new(birth: f64, death: f64, dimension: usize) -> Result<Self, TdaError> {
        if birth < 0.0 || death < birth || !birth.is_finite() || !death.is_finite() {
            return Err(TdaError::InvalidRadius { value: birth });
        }
        Ok(Self {
            birth,
            death,
            dimension,
        })
    }

    /// Returns the persistence (lifetime) of the feature (`death - birth`).
    #[verified_engine::verified]
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }

    /// Returns true if this is a significant feature (`persistence > threshold`).
    #[verified_engine::verified]
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.persistence() > threshold
    }
}

/// A persistence barcode is a collection of persistence intervals.
#[derive(Debug, Clone)]
pub struct PersistenceBarcode {
    /// All persistence intervals in the barcode.
    pub intervals: Vec<PersistenceInterval>,
}

impl PersistenceBarcode {
    /// Creates a new empty barcode.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self {
            intervals: Vec::new(),
        }
    }

    /// Adds an interval to the barcode.
    #[verified_engine::verified]
    pub fn add_interval(&mut self, interval: PersistenceInterval) {
        self.intervals.push(interval);
    }

    /// Returns the number of intervals.
    #[verified_engine::verified]
    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    /// Returns true if the barcode is empty.
    #[verified_engine::verified]
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Filters intervals by dimension.
    ///
    /// # Example
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::tda::{
    ///     PersistenceBarcode, PersistenceInterval
    /// };
    ///
    /// let mut barcode = PersistenceBarcode::new();
    /// barcode.add_interval(PersistenceInterval::new(0.0, 1.0, 0).unwrap());
    /// barcode.add_interval(PersistenceInterval::new(0.5, 2.0, 1).unwrap());
    /// assert_eq!(barcode.filter_by_dimension(1).len(), 1);
    /// ```
    #[verified_engine::verified]
    pub fn filter_by_dimension(&self, dimension: usize) -> Vec<&PersistenceInterval> {
        self.intervals
            .iter()
            .filter(|interval| interval.dimension == dimension)
            .collect()
    }

    /// Filters intervals by minimum persistence (`persistence > threshold`).
    #[verified_engine::verified]
    pub fn filter_by_persistence(&self, threshold: f64) -> Vec<&PersistenceInterval> {
        self.intervals
            .iter()
            .filter(|interval| interval.is_significant(threshold))
            .collect()
    }

    /// Returns the most persistent feature of a given dimension.
    #[verified_engine::verified]
    pub fn most_persistent(&self, dimension: usize) -> Option<&PersistenceInterval> {
        self.filter_by_dimension(dimension)
            .into_iter()
            .max_by(|a, b| {
                a.persistence()
                    .partial_cmp(&b.persistence())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

impl Default for PersistenceBarcode {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

/// Computes a persistence barcode from a 2D [`PointCloud`] across `radii`.
///
/// # Errors
///
/// Returns [`TdaError::InvalidRadius`] if `radii` is empty or contains invalid values.
///
/// # Example
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::tda::{
///     PointCloud, Point2D, compute_persistence
/// };
/// use math_explorer::error::TdaError;
///
/// let cloud = PointCloud::new(vec![
///     Point2D::new(0.0, 0.0),
///     Point2D::new(1.0, 0.0),
///     Point2D::new(0.5, 0.866),
/// ]).unwrap();
/// let radii: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
/// let barcode = compute_persistence(&cloud, &radii).unwrap();
/// assert!(!barcode.is_empty());
/// assert!(matches!(
///     compute_persistence(&cloud, &[]),
///     Err(TdaError::InvalidRadius { .. })
/// ));
/// ```
#[verified_engine::verified]
pub fn compute_persistence(
    cloud: &PointCloud,
    radii: &[f64],
) -> Result<PersistenceBarcode, TdaError> {
    let dist = cloud.distance_matrix();
    compute_persistence_from_distances(&dist, radii)
}

/// Computes a persistence barcode directly from a precomputed symmetric distance matrix.
///
/// Supports arbitrary metric spaces including 3D strange attractors and Takens delay embeddings.
///
/// # Errors
///
/// Returns [`TdaError::InvalidRadius`] if `radii` is empty or contains negative/non-finite
/// values, or [`TdaError::MatrixError`] if `dist` is non-square or non-finite.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::tda::{
///     Point3D, PointCloud3D, compute_persistence_from_distances
/// };
///
/// let cloud = PointCloud3D::new(vec![
///     Point3D::new(0.0, 0.0, 0.0),
///     Point3D::new(1.0, 0.0, 0.0),
///     Point3D::new(1.0, 1.0, 0.0),
///     Point3D::new(0.0, 1.0, 0.0),
/// ]).unwrap();
/// let radii = [0.0, 0.5, 1.1, 1.5];
/// let barcode = compute_persistence_from_distances(&cloud.distance_matrix(), &radii).unwrap();
/// assert!(!barcode.filter_by_dimension(1).is_empty());
/// ```
#[verified_engine::verified]
pub fn compute_persistence_from_distances(
    dist: &[Vec<f64>],
    radii: &[f64],
) -> Result<PersistenceBarcode, TdaError> {
    if radii.is_empty() {
        return Err(TdaError::InvalidRadius { value: 0.0 });
    }
    validate_distance_matrix(dist)?;

    let mut barcode = PersistenceBarcode::new();
    let mut prev_beta0 = 0usize;
    let mut prev_beta1 = 0usize;
    let mut component_births: Vec<f64> = Vec::new();
    let mut hole_births: Vec<f64> = Vec::new();

    for &radius in radii {
        let complex = vietoris_rips_from_distances(dist, radius)?;
        let beta0 = betti_number_0(&complex)?;
        let beta1 = betti_number_1(&complex)?;
        record_dimension_step(
            &mut component_births,
            prev_beta0,
            beta0,
            radius,
            0,
            &mut barcode,
        )?;
        record_dimension_step(&mut hole_births, prev_beta1, beta1, radius, 1, &mut barcode)?;
        prev_beta0 = beta0;
        prev_beta1 = beta1;
    }

    let final_radius = radii
        .last()
        .copied()
        .ok_or(TdaError::InvalidRadius { value: 0.0 })?;
    flush_surviving_features(&component_births, final_radius, 0, &mut barcode)?;
    flush_surviving_features(&hole_births, final_radius, 1, &mut barcode)?;
    Ok(barcode)
}

#[verified_engine::verified]
fn record_dimension_step(
    births: &mut Vec<f64>,
    prev_betti: usize,
    curr_betti: usize,
    radius: f64,
    dim: usize,
    barcode: &mut PersistenceBarcode,
) -> Result<(), TdaError> {
    if curr_betti > prev_betti {
        for _ in 0..(curr_betti - prev_betti) {
            births.push(radius);
        }
    } else if curr_betti < prev_betti {
        for _ in 0..(prev_betti - curr_betti) {
            if let Some(birth) = births.pop()
                && birth < radius
            {
                barcode.add_interval(PersistenceInterval::new(birth, radius, dim)?);
            }
        }
    }
    Ok(())
}

#[verified_engine::verified]
fn flush_surviving_features(
    births: &[f64],
    final_radius: f64,
    dim: usize,
    barcode: &mut PersistenceBarcode,
) -> Result<(), TdaError> {
    for &birth in births {
        if birth < final_radius {
            barcode.add_interval(PersistenceInterval::new(birth, final_radius, dim)?);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "persistence_tests.rs"]
mod tests;
