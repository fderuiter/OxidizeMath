// [cite:math_commons]
use std::ops::{Deref, DerefMut};

/// A physical step size, typically used for grid spacing.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StepSize(pub f64);

impl Deref for StepSize {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for StepSize {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A grid index, representing a coordinate or an offset in a flattened array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GridIndex(pub usize);

impl Deref for GridIndex {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for GridIndex {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A dimension representing width or height.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Dimension(pub usize);

impl Deref for Dimension {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Dimension {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Converts a 2D coordinate into a 1D flat index.
///
/// # Example
/// ```
/// use math_commons::math_kernel::types::flatten_2d_index;
///
/// let index = flatten_2d_index(4, 2, 10);
/// assert_eq!(index, 24);
/// ```
#[inline(always)]
pub fn flatten_2d_index(x: usize, y: usize, width: usize) -> usize {
    y * width + x
}
