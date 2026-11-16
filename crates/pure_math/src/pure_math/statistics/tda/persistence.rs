//! Persistence computation and barcode generation.

use super::complex::vietoris_rips_from_distances;
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
    /// Returns [`TdaError::InvalidRadius`] if:
    /// - `birth` is negative.
    /// - `death` is strictly less than `birth`.
    /// - Either `birth` or `death` is not finite (e.g., `NaN` or `Infinity`).
    ///
    /// # Example
    ///
    /// ```
    /// use pure_math::pure_math::statistics::tda::PersistenceInterval;
    /// use pure_math::error::TdaError;
    ///
    /// let interval = PersistenceInterval::new(0.5, 2.0, 1).unwrap();
    /// assert_eq!(interval.persistence(), 1.5);
    ///
    /// // Invalid radii result in an error
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

    /// Returns the persistence (lifetime) of the feature.
    ///
    /// Formula: persistence = death - birth
    #[verified_engine::verified]
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }

    /// Returns true if this is a significant feature (persistence > threshold).
    #[verified_engine::verified]
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.persistence() > threshold
    }
}

/// A persistence barcode is a collection of persistence intervals.
///
/// The barcode visualizes the lifetime of topological features across
/// different scales (filtration parameter values).
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
    ///
    /// ```
    /// use pure_math::pure_math::statistics::tda::{
    ///     PersistenceBarcode, PersistenceInterval
    /// };
    ///
    /// let mut barcode = PersistenceBarcode::new();
    /// barcode.add_interval(PersistenceInterval::new(0.0, 1.0, 0).unwrap());
    /// barcode.add_interval(PersistenceInterval::new(0.5, 2.0, 1).unwrap());
    ///
    /// let holes = barcode.filter_by_dimension(1);
    /// assert_eq!(holes.len(), 1);
    /// ```
    #[verified_engine::verified]
    pub fn filter_by_dimension(&self, dimension: usize) -> Vec<&PersistenceInterval> {
        self.intervals
            .iter()
            .filter(|interval| interval.dimension == dimension)
            .collect()
    }

    /// Filters intervals by minimum persistence.
    ///
    /// Only returns features with persistence > threshold.
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

/// Computes persistence barcode using a simple filtration.
///
/// This performs a filtration of the Vietoris-Rips complex at multiple
/// radius values and tracks when topological features appear and disappear.
///
/// # Arguments
///
/// * `cloud` - The point cloud
/// * `radii` - Sorted sequence of radii to use for filtration
///
/// # Returns
///
/// * `Result<PersistenceBarcode, TdaError>` - The persistence barcode or error
///
/// # Errors
///
/// Returns [`TdaError::InvalidRadius`] if `radii` is empty.
/// Propagates errors from `vietoris_rips_complex` or Betti number computations if the point cloud or radii are invalid.
///
/// # Errors
///
/// Returns [`TdaError::InvalidRadius`] if `radii` is empty or if invalid radii are encountered.
/// Propagates errors from `vietoris_rips_complex`, Betti number computations, and `PersistenceInterval::new`.
///
/// # Example
///
/// ```
/// use pure_math::pure_math::statistics::tda::{
///     PointCloud, Point2D, compute_persistence
/// };
/// use pure_math::error::TdaError;
///
/// let points = vec![
///     Point2D::new(0.0, 0.0),
///     Point2D::new(1.0, 0.0),
///     Point2D::new(0.5, 0.866),
/// ];
/// let cloud = PointCloud::new(points).unwrap();
///
/// let radii: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
/// let barcode = compute_persistence(&cloud, &radii).unwrap();
///
/// println!("Found {} features", barcode.len());
/// for interval in &barcode.intervals {
///     println!("Dim {}: [{:.2}, {:.2}] (persistence: {:.2})",
///         interval.dimension, interval.birth, interval.death,
///         interval.persistence());
/// }
///
/// // Empty radii results in an error
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
    if radii.is_empty() {
        return Err(TdaError::InvalidRadius { value: 0.0 });
    }

    let dist = cloud.distance_matrix();
    let mut barcode = PersistenceBarcode::new();

    // Track previous Betti numbers
    let mut prev_beta0 = 0;
    let mut prev_beta1 = 0;

    // Track births of features
    let mut component_births: Vec<(usize, f64)> = Vec::new(); // (id, birth_radius)
    let mut hole_births: Vec<f64> = Vec::new();

    for &radius in radii {
        let complex = vietoris_rips_from_distances(&dist, radius)?;
        let beta0 = betti_number_0(&complex)?;
        let beta1 = betti_number_1(&complex)?;

        // Track connected components (β₀)
        if beta0 > prev_beta0 {
            // New components appeared
            for _ in 0..(beta0 - prev_beta0) {
                component_births.push((component_births.len(), radius));
            }
        } else if beta0 < prev_beta0 {
            // Components merged (died)
            let deaths = prev_beta0 - beta0;
            for _ in 0..deaths {
                match component_births.pop() {
                    Some((_, birth)) if birth < radius => {
                        // Only record if it has non-zero persistence
                        barcode.add_interval(PersistenceInterval::new(birth, radius, 0)?);
                    }
                    _ => {}
                }
            }
        }

        // Track holes (β₁)
        if beta1 > prev_beta1 {
            // New holes appeared
            for _ in 0..(beta1 - prev_beta1) {
                hole_births.push(radius);
            }
        } else if beta1 < prev_beta1 {
            // Holes filled (died)
            let deaths = prev_beta1 - beta1;
            for _ in 0..deaths {
                match hole_births.pop() {
                    Some(birth) if birth < radius => {
                        barcode.add_interval(PersistenceInterval::new(birth, radius, 1)?);
                    }
                    _ => {}
                }
            }
        }

        prev_beta0 = beta0;
        prev_beta1 = beta1;
    }

    // Handle features that persist to the end
    let final_radius = radii
        .last()
        .copied()
        .ok_or(TdaError::InvalidRadius { value: 0.0 })?;

    // Remaining components
    for (_, birth) in component_births {
        if birth < final_radius {
            barcode.add_interval(PersistenceInterval::new(birth, final_radius, 0)?);
        }
    }

    // Remaining holes
    for birth in hole_births {
        if birth < final_radius {
            barcode.add_interval(PersistenceInterval::new(birth, final_radius, 1)?);
        }
    }

    Ok(barcode)
}

#[cfg(test)]
mod tests {
    use super::super::complex::vietoris_rips_complex;
    use super::*;
    use crate::pure_math::statistics::tda::core::Point2D;

    #[test]
    #[verified_engine::verified]
    fn test_persistence_interval() {
        let interval = PersistenceInterval::new(0.5, 2.0, 1).unwrap();
        assert_eq!(interval.birth, 0.5);
        assert_eq!(interval.death, 2.0);
        assert_eq!(interval.dimension, 1);
        assert!((interval.persistence() - 1.5).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_persistence_interval_invalid() {
        assert!(PersistenceInterval::new(2.0, 1.0, 0).is_err()); // birth > death
        assert!(PersistenceInterval::new(-1.0, 1.0, 0).is_err()); // negative birth
    }

    #[test]
    #[verified_engine::verified]
    fn test_persistence_interval_is_significant() {
        let interval = PersistenceInterval::new(0.5, 2.0, 1).unwrap();
        assert!(interval.is_significant(1.0));
        assert!(!interval.is_significant(2.0));
    }

    #[test]
    #[verified_engine::verified]
    fn test_barcode_filter_by_dimension() {
        let mut barcode = PersistenceBarcode::new();
        barcode.add_interval(PersistenceInterval::new(0.0, 1.0, 0).unwrap());
        barcode.add_interval(PersistenceInterval::new(0.5, 2.0, 1).unwrap());
        barcode.add_interval(PersistenceInterval::new(1.0, 3.0, 1).unwrap());

        let dim0 = barcode.filter_by_dimension(0);
        assert_eq!(dim0.len(), 1);

        let dim1 = barcode.filter_by_dimension(1);
        assert_eq!(dim1.len(), 2);
    }

    #[test]
    #[verified_engine::verified]
    fn test_barcode_filter_by_persistence() {
        let mut barcode = PersistenceBarcode::new();
        barcode.add_interval(PersistenceInterval::new(0.0, 0.5, 0).unwrap()); // persistence 0.5
        barcode.add_interval(PersistenceInterval::new(0.5, 2.0, 1).unwrap()); // persistence 1.5

        let significant = barcode.filter_by_persistence(1.0);
        assert_eq!(significant.len(), 1);
        assert_eq!(significant[0].dimension, 1);
    }

    #[test]
    #[verified_engine::verified]
    fn test_barcode_most_persistent() {
        let mut barcode = PersistenceBarcode::new();
        barcode.add_interval(PersistenceInterval::new(0.0, 1.0, 1).unwrap()); // persistence 1.0
        barcode.add_interval(PersistenceInterval::new(0.5, 3.0, 1).unwrap()); // persistence 2.5
        barcode.add_interval(PersistenceInterval::new(1.0, 2.0, 1).unwrap()); // persistence 1.0

        let most_persistent = barcode.most_persistent(1).unwrap();
        assert!(
            (most_persistent.persistence() - 2.5).abs() < math_commons::registry::TOLERANCE_FAST
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_compute_persistence_line() {
        // Points on a line
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(2.0, 0.0),
        ];
        let cloud = PointCloud::new(points).unwrap();

        let radii: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
        let barcode = compute_persistence(&cloud, &radii).unwrap();

        // Should have components merging but no persistent holes
        assert!(!barcode.is_empty());

        let holes = barcode.filter_by_dimension(1);
        // A line should not have significant holes
        let significant_holes: Vec<_> = holes.iter().filter(|h| h.is_significant(0.5)).collect();
        assert_eq!(significant_holes.len(), 0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_compute_persistence_triangle() {
        // Equilateral triangle - this will form a hollow triangle at some radius
        // but might fill in at larger radii
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(1.0, 1.732),
        ];
        let cloud = PointCloud::new(points).unwrap();

        let radii: Vec<f64> = (0..30).map(|i| i as f64 * 0.15).collect();
        let barcode = compute_persistence(&cloud, &radii).unwrap();

        // The barcode should contain some features
        // Note: With only 3 points, we may not detect persistent holes
        // depending on the filtration
        assert!(!barcode.is_empty());
    }

    #[test]
    #[verified_engine::verified]
    fn test_compute_persistence_two_clusters() {
        // Two well-separated clusters
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(10.0, 0.0),
            Point2D::new(11.0, 0.0),
        ];
        let cloud = PointCloud::new(points).unwrap();

        let radii: Vec<f64> = (0..50).map(|i| i as f64 * 0.3).collect();
        let barcode = compute_persistence(&cloud, &radii).unwrap();

        // Should have components that persist for a while
        let components = barcode.filter_by_dimension(0);
        let significant_components: Vec<_> = components
            .iter()
            .filter(|c| c.is_significant(1.0))
            .collect();

        // At least one component should have significant persistence
        assert!(!significant_components.is_empty());
    }

    #[test]
    #[verified_engine::verified]
    fn test_compute_persistence_invalid_radius() {
        let points = vec![Point2D::new(0.0, 0.0), Point2D::new(1.0, 0.0)];
        let cloud = PointCloud::new(points).unwrap();

        assert!(matches!(
            compute_persistence(&cloud, &[]),
            Err(TdaError::InvalidRadius { .. })
        ));
        assert!(matches!(
            compute_persistence(&cloud, &[-1.0]),
            Err(TdaError::InvalidRadius { .. })
        ));
        assert!(matches!(
            compute_persistence(&cloud, &[f64::NAN]),
            Err(TdaError::InvalidRadius { .. })
        ));
    }

    #[test]
    #[verified_engine::verified]
    fn test_compute_persistence_barcode_parity() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(1.0, 1.0),
        ];
        let cloud = PointCloud::new(points).unwrap();
        let radii = [0.0, 0.5, 1.0, 1.5, 2.0];
        let barcode = compute_persistence(&cloud, &radii).unwrap();

        let mut b0_sequence = Vec::new();
        for &r in &radii {
            let c = vietoris_rips_complex(&cloud, r).unwrap();
            b0_sequence.push(betti_number_0(&c).unwrap());
        }
        assert_eq!(b0_sequence, vec![4, 4, 1, 1, 1]);
        assert_eq!(barcode.intervals.len(), 5);
    }
}
