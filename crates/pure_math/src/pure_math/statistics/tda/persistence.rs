//! Exact simplicial persistent homology computation and barcode generation over $\mathbb{Z}_2$.

use super::boundary::{
    FilteredEdge, FilteredTriangle, Z2BoundaryWorkspace, build_cleared_partial_1_workspace,
    build_partial_2_workspace, collect_sorted_edges, collect_sorted_triangles,
};
use super::complex::{MAX_FILTRATION_TRIANGLES, validate_distance_matrix};
use super::core::{NdPointCloud, PointCloud};
use crate::error::TdaError;
use std::cmp::Ordering;

/// A persistence interval representing the lifetime of a topological feature.
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
    /// Returns [`TdaError::InvalidRadius`] if `birth < 0.0`, `death < birth`, or either endpoint is non-finite.
    ///
    /// # Example
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::tda::PersistenceInterval;
    /// use math_explorer::error::TdaError;
    ///
    /// let interval = PersistenceInterval::new(0.5, 2.0, 1).unwrap();
    /// assert_eq!(interval.persistence(), 1.5);
    /// assert!(matches!(PersistenceInterval::new(2.0, 1.0, 1), Err(TdaError::InvalidRadius { .. })));
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

    /// Returns the persistence (`death - birth`) of the feature.
    #[verified_engine::verified]
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }

    /// Returns `true` if `self.persistence() > threshold`.
    #[verified_engine::verified]
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.persistence() > threshold
    }
}

/// Exact Vietoris-Rips persistence interval paired via $\mathbb{Z}_2$ boundary reduction.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RipsInterval {
    /// Homology dimension (`0` for $H_0$ components, `1` for $H_1$ loops).
    pub dimension: usize,
    /// Exact filtration diameter at which the feature is created.
    pub birth: f64,
    /// Exact filtration diameter at which the feature dies, or finite `max_radius` when `is_essential` is `true`.
    pub death: f64,
    /// `true` when the feature survives past the maximum filtration scale without a killing simplex.
    pub is_essential: bool,
}

impl RipsInterval {
    /// Returns the finite persistence span `death - birth`.
    #[verified_engine::verified]
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }

    /// Returns `true` if this homology class is active at filtration threshold `epsilon`.
    #[verified_engine::verified]
    pub fn is_alive_at(&self, epsilon: f64) -> bool {
        if self.is_essential {
            self.birth <= epsilon
        } else {
            self.birth <= epsilon && epsilon < self.death
        }
    }

    /// Converts this interval into a [`PersistenceInterval`].
    ///
    /// # Errors
    ///
    /// Propagates [`TdaError::InvalidRadius`] if endpoints are invalid.
    #[verified_engine::verified]
    pub fn to_persistence_interval(&self) -> Result<PersistenceInterval, TdaError> {
        PersistenceInterval::new(self.birth, self.death, self.dimension)
    }
}

/// Result of exact simplicial Vietoris-Rips persistent homology up to $H_1$.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RipsPersistenceResult {
    /// Exact $H_0$ and $H_1$ intervals ordered by `(dimension, birth, death, is_essential)`.
    pub intervals: Vec<RipsInterval>,
    /// Maximum pairwise distance across the point cloud used to cap essential intervals.
    pub max_radius: f64,
    /// `true` if candidate 2-simplices exceeded [`MAX_FILTRATION_TRIANGLES`] and were truncated.
    pub truncated_triangles: bool,
}

impl RipsPersistenceResult {
    /// Converts positive-lifetime intervals into a [`PersistenceBarcode`].
    ///
    /// # Errors
    ///
    /// Propagates [`TdaError::InvalidRadius`] if any interval has invalid bounds.
    #[verified_engine::verified]
    pub fn to_barcode(&self) -> Result<PersistenceBarcode, TdaError> {
        let mut barcode = PersistenceBarcode::new();
        for item in &self.intervals {
            if item.death > item.birth || item.is_essential {
                barcode.add_interval(item.to_persistence_interval()?);
            }
        }
        Ok(barcode)
    }

    /// Evaluates $(\beta_0(\epsilon), \beta_1(\epsilon))$ directly from active intervals at `epsilon`.
    #[verified_engine::verified]
    pub fn betti_numbers_at(&self, epsilon: f64) -> (usize, usize) {
        let mut b0 = 0_usize;
        let mut b1 = 0_usize;
        for item in &self.intervals {
            if item.is_alive_at(epsilon) {
                if item.dimension == 0 {
                    b0 += 1;
                } else if item.dimension == 1 {
                    b1 += 1;
                }
            }
        }
        (b0, b1)
    }
}

/// A persistence barcode is a collection of persistence intervals.
///
/// The barcode visualizes the lifetime of topological features across
/// different scales (filtration parameter values).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// Returns `true` if the barcode is empty.
    #[verified_engine::verified]
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Filters intervals by homology dimension.
    ///
    /// # Example
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::tda::{PersistenceBarcode, PersistenceInterval};
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
                    .unwrap_or(Ordering::Equal)
            })
    }
}

impl Default for PersistenceBarcode {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

/// Computes exact simplicial $H_0$ and $H_1$ persistence intervals on a 2D [`PointCloud`].
///
/// # Errors
///
/// Propagates [`TdaError`] if distance matrix validation or boundary workspace setup fails.
///
/// # Example
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::tda::{Point2D, PointCloud, compute_rips_persistence};
///
/// let square = PointCloud::new(vec![
///     Point2D::new(0.0, 0.0),
///     Point2D::new(1.0, 0.0),
///     Point2D::new(1.0, 1.0),
///     Point2D::new(0.0, 1.0),
/// ]).unwrap();
/// assert_eq!(compute_rips_persistence(&square).unwrap().betti_numbers_at(1.1), (1, 1));
/// ```
#[verified_engine::verified]
pub fn compute_rips_persistence(cloud: &PointCloud) -> Result<RipsPersistenceResult, TdaError> {
    compute_rips_persistence_from_distances(&cloud.distance_matrix())
}

/// Computes exact simplicial $H_0$ and $H_1$ persistence intervals on an $N$-dimensional [`NdPointCloud`].
///
/// # Errors
///
/// Propagates [`TdaError`] if distance matrix validation or boundary workspace setup fails.
#[verified_engine::verified]
pub fn compute_rips_persistence_nd(
    cloud: &NdPointCloud,
) -> Result<RipsPersistenceResult, TdaError> {
    compute_rips_persistence_from_distances(&cloud.distance_matrix())
}

/// Computes exact simplicial $H_0$ and $H_1$ persistence intervals from a precomputed distance matrix.
///
/// Sorts 0-, 1-, and 2-simplices by `(diameter, dimension, vertices)`, reduces $\partial_2$ first
/// over $\mathbb{Z}_2$, clears $\partial_2$ pivot columns in $\partial_1$, and reduces $\partial_1$.
///
/// # Errors
///
/// Returns [`TdaError`] if `dist` is empty, non-square, exceeds `MAX_TDA_POINTS`, or contains invalid distances.
#[verified_engine::verified]
pub fn compute_rips_persistence_from_distances(
    dist: &[Vec<f64>],
) -> Result<RipsPersistenceResult, TdaError> {
    compute_rips_persistence_bounded(dist, MAX_FILTRATION_TRIANGLES)
}

#[verified_engine::verified]
pub(crate) fn compute_rips_persistence_bounded(
    dist: &[Vec<f64>],
    max_triangles: usize,
) -> Result<RipsPersistenceResult, TdaError> {
    validate_distance_matrix(dist)?;
    let n = dist.len();
    let (edges, edge_lookup, max_radius) = collect_sorted_edges(dist);
    let (triangles, truncated_triangles) = collect_sorted_triangles(dist, max_triangles);

    let mut partial_2 = build_partial_2_workspace(n, edges.len(), &edge_lookup, &triangles)?;
    partial_2.reduce();

    let mut partial_1 = build_cleared_partial_1_workspace(n, &edges, &partial_2)?;
    partial_1.reduce();

    let intervals =
        assemble_rips_intervals(n, &edges, &triangles, &partial_1, &partial_2, max_radius);
    Ok(RipsPersistenceResult {
        intervals,
        max_radius,
        truncated_triangles,
    })
}

#[verified_engine::verified]
fn assemble_rips_intervals(
    n: usize,
    edges: &[FilteredEdge],
    triangles: &[FilteredTriangle],
    partial_1: &Z2BoundaryWorkspace,
    partial_2: &Z2BoundaryWorkspace,
    max_radius: f64,
) -> Vec<RipsInterval> {
    let mut intervals = Vec::with_capacity(n + edges.len().min(n * 4));

    for v in 0..n {
        if let Some(e_idx) = partial_1.pivot_col_for_row(v) {
            let death = edges[e_idx].diameter;
            if death > 0.0 {
                intervals.push(RipsInterval {
                    dimension: 0,
                    birth: 0.0,
                    death,
                    is_essential: false,
                });
            }
        } else {
            intervals.push(RipsInterval {
                dimension: 0,
                birth: 0.0,
                death: max_radius,
                is_essential: true,
            });
        }
    }

    for (e_idx, edge) in edges.iter().enumerate() {
        let birth = edge.diameter;
        if let Some(t_idx) = partial_2.pivot_col_for_row(e_idx) {
            let death = triangles[t_idx].diameter;
            if death > birth {
                intervals.push(RipsInterval {
                    dimension: 1,
                    birth,
                    death,
                    is_essential: false,
                });
            }
        } else if !partial_1.is_cleared(e_idx) && partial_1.col_pivot(e_idx).is_none() {
            intervals.push(RipsInterval {
                dimension: 1,
                birth,
                death: max_radius.max(birth),
                is_essential: true,
            });
        }
    }

    intervals.sort_by(|a, b| {
        a.dimension
            .cmp(&b.dimension)
            .then_with(|| a.birth.total_cmp(&b.birth))
            .then_with(|| a.death.total_cmp(&b.death))
            .then_with(|| a.is_essential.cmp(&b.is_essential))
    });
    intervals
}

/// Computes persistence barcode mapped onto caller filtration `radii` via exact $\mathbb{Z}_2$ pairing.
///
/// # Errors
///
/// Returns [`TdaError::InvalidRadius`] if `radii` is empty or contains negative or non-finite values.
///
/// # Example
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::tda::{PointCloud, Point2D, compute_persistence};
/// use math_explorer::error::TdaError;
///
/// let cloud = PointCloud::new(vec![
///     Point2D::new(0.0, 0.0),
///     Point2D::new(1.0, 0.0),
///     Point2D::new(0.5, 0.866),
/// ]).unwrap();
/// let radii: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
/// assert!(!compute_persistence(&cloud, &radii).unwrap().is_empty());
/// assert!(matches!(compute_persistence(&cloud, &[]), Err(TdaError::InvalidRadius { .. })));
/// ```
#[verified_engine::verified]
pub fn compute_persistence(
    cloud: &PointCloud,
    radii: &[f64],
) -> Result<PersistenceBarcode, TdaError> {
    if radii.is_empty() {
        return Err(TdaError::InvalidRadius { value: 0.0 });
    }
    let mut sorted_radii = Vec::with_capacity(radii.len());
    for &r in radii {
        if r < 0.0 || !r.is_finite() {
            return Err(TdaError::InvalidRadius { value: r });
        }
        sorted_radii.push(r);
    }
    sorted_radii.sort_by(f64::total_cmp);

    let exact = compute_rips_persistence(cloud)?;
    map_rips_intervals_to_radii(&exact, &sorted_radii)
}

#[verified_engine::verified]
fn map_rips_intervals_to_radii(
    exact: &RipsPersistenceResult,
    sorted_radii: &[f64],
) -> Result<PersistenceBarcode, TdaError> {
    let final_radius = *sorted_radii
        .last()
        .ok_or(TdaError::InvalidRadius { value: 0.0 })?;
    let mut barcode = PersistenceBarcode::new();

    for item in &exact.intervals {
        let birth_idx = sorted_radii.partition_point(|&r| r < item.birth);
        if birth_idx >= sorted_radii.len() {
            continue;
        }
        let mapped_birth = sorted_radii[birth_idx];
        let mapped_death = if item.is_essential {
            final_radius
        } else {
            let death_idx = sorted_radii.partition_point(|&r| r < item.death);
            sorted_radii.get(death_idx).copied().unwrap_or(final_radius)
        };

        if mapped_birth < mapped_death {
            barcode.add_interval(PersistenceInterval::new(
                mapped_birth,
                mapped_death,
                item.dimension,
            )?);
        }
    }
    Ok(barcode)
}

// [cite:clinical_trials_statistics]

#[cfg(test)]
#[path = "persistence_tests.rs"]
mod tests;
