use std::{ops::Deref, sync::Arc};

use super::PlotPoint;

/// Immutable, reference-counted 2D coordinate buffer (`Arc<[[f64; 2]]>`).
///
/// Cloning [`SharedPoints`] increments an atomic reference count in $O(1)$
/// without copying coordinate data, allowing plot items, export primitives,
/// thread-local plot captures, data tables, trace overlays, and framework
/// recovery snapshots to share a single allocation across frames.
#[derive(Clone, Debug, Default)]
pub struct SharedPoints(Arc<[[f64; 2]]>);

impl SharedPoints {
    /// Wrap any convertible coordinate collection into [`SharedPoints`].
    #[inline]
    pub fn new(points: impl Into<Self>) -> Self {
        points.into()
    }

    /// Copy a borrowed `[[f64; 2]]` slice into a new reference-counted buffer.
    #[inline]
    pub fn from_slice(points: &[[f64; 2]]) -> Self {
        Self(Arc::from(points))
    }

    /// Copy a borrowed [`PlotPoint`] slice into a new reference-counted buffer.
    #[inline]
    pub fn from_plot_points(points: &[PlotPoint]) -> Self {
        let coords: Vec<[f64; 2]> = points.iter().map(|p| [p.x, p.y]).collect();
        Self(Arc::from(coords))
    }

    /// Borrow the underlying coordinate slice.
    #[inline]
    pub fn as_slice(&self) -> &[[f64; 2]] {
        &self.0
    }

    /// Returns `true` when `self` and `other` reference the same allocation.
    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Returns a buffer containing at most the trailing `keep` points.
    ///
    /// Reuses the existing `Arc` handle when `keep >= self.len()`.
    #[inline]
    pub fn tail(&self, keep: usize) -> Self {
        if keep >= self.0.len() {
            self.clone()
        } else if keep == 0 {
            Self::default()
        } else {
            Self::from_slice(&self.0[self.0.len() - keep..])
        }
    }

    /// Returns `true` if `pts` has the exact same length and bitwise coordinates.
    #[inline]
    pub(crate) fn matches_plot_points(&self, pts: &[PlotPoint]) -> bool {
        self.0.len() == pts.len()
            && self
                .0
                .iter()
                .zip(pts.iter())
                .all(|(c, p)| c[0].to_bits() == p.x.to_bits() && c[1].to_bits() == p.y.to_bits())
    }
}

impl Deref for SharedPoints {
    type Target = [[f64; 2]];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<[[f64; 2]]> for SharedPoints {
    #[inline]
    fn as_ref(&self) -> &[[f64; 2]] {
        &self.0
    }
}

impl math_commons::series::PointSeries for SharedPoints {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    fn x_at(&self, index: usize) -> f64 {
        self.0[index][0]
    }

    #[inline(always)]
    fn y_at(&self, index: usize) -> f64 {
        self.0[index][1]
    }

    #[inline(always)]
    fn point_at(&self, index: usize) -> [f64; 2] {
        self.0[index]
    }
}

impl From<Vec<[f64; 2]>> for SharedPoints {
    #[inline]
    fn from(points: Vec<[f64; 2]>) -> Self {
        Self(Arc::from(points))
    }
}

impl From<&[[f64; 2]]> for SharedPoints {
    #[inline]
    fn from(points: &[[f64; 2]]) -> Self {
        Self(Arc::from(points))
    }
}

impl From<Arc<[[f64; 2]]>> for SharedPoints {
    #[inline]
    fn from(points: Arc<[[f64; 2]]>) -> Self {
        Self(points)
    }
}

impl From<&[PlotPoint]> for SharedPoints {
    #[inline]
    fn from(points: &[PlotPoint]) -> Self {
        Self::from_plot_points(points)
    }
}

impl From<Vec<PlotPoint>> for SharedPoints {
    #[inline]
    fn from(points: Vec<PlotPoint>) -> Self {
        Self::from_plot_points(&points)
    }
}

impl FromIterator<[f64; 2]> for SharedPoints {
    #[inline]
    fn from_iter<T: IntoIterator<Item = [f64; 2]>>(iter: T) -> Self {
        let vec: Vec<[f64; 2]> = iter.into_iter().collect();
        Self(Arc::from(vec))
    }
}

impl<'a> IntoIterator for &'a SharedPoints {
    type Item = &'a [f64; 2];
    type IntoIter = std::slice::Iter<'a, [f64; 2]>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl PartialEq for SharedPoints {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other) || self.0.as_ref() == other.0.as_ref()
    }
}

impl PartialEq<Vec<[f64; 2]>> for SharedPoints {
    #[inline]
    fn eq(&self, other: &Vec<[f64; 2]>) -> bool {
        self.0.as_ref() == other.as_slice()
    }
}

impl PartialEq<SharedPoints> for Vec<[f64; 2]> {
    #[inline]
    fn eq(&self, other: &SharedPoints) -> bool {
        self.as_slice() == other.0.as_ref()
    }
}

impl PartialEq<[[f64; 2]]> for SharedPoints {
    #[inline]
    fn eq(&self, other: &[[f64; 2]]) -> bool {
        self.0.as_ref() == other
    }
}

impl PartialEq<&[[f64; 2]]> for SharedPoints {
    #[inline]
    fn eq(&self, other: &&[[f64; 2]]) -> bool {
        self.0.as_ref() == *other
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for SharedPoints {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.as_ref().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for SharedPoints {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let vec = Vec::<[f64; 2]>::deserialize(deserializer)?;
        Ok(Self::from(vec))
    }
}
