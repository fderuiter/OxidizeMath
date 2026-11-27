// [cite:math_scalars]
//! Caller-owned [`SolverWorkspace`], [`StepSize`], [`GridIndex`], [`Dimension`], and flat indexers.

use crate::error::ScalarError;
use crate::traits::RealField;
use core::fmt;
use core::ops::{Deref, DerefMut};

/// A physical or numerical grid step size, generic over [`RealField`] (defaulting to `f64`).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StepSize<T = f64>(pub T);

impl<T: RealField> StepSize<T> {
    /// Fallibly constructs a validated [`StepSize`] requiring `value > 0` and `value.is_finite()`.
    #[inline]
    pub fn try_new(value: T) -> Result<Self, ScalarError> {
        if value.is_finite() && value > T::ZERO {
            Ok(Self(value))
        } else {
            Err(ScalarError::InvalidStepSize {
                value: value.to_f64(),
            })
        }
    }

    /// Constructs a validated [`StepSize`], returning `None` if `value` is non-finite or `<= 0`.
    #[inline]
    #[must_use]
    pub fn try_new_checked(value: T) -> Option<Self> {
        (value.is_finite() && value > T::ZERO).then_some(Self(value))
    }

    /// Returns the inner scalar step value.
    #[inline]
    #[must_use]
    pub const fn value(&self) -> T {
        self.0
    }
}

impl<T> Deref for StepSize<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for StepSize<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: fmt::Display> fmt::Display for StepSize<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// A grid index representing a coordinate or an offset in a flattened array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GridIndex(pub usize);

impl Deref for GridIndex {
    type Target = usize;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for GridIndex {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl fmt::Display for GridIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// A grid or vector dimension representing width, height, depth, or state count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Dimension(pub usize);

impl Deref for Dimension {
    type Target = usize;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Dimension {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// Converts a 2D `(x, y)` coordinate into a row-major 1D flat index `y * width + x`.
#[inline(always)]
#[must_use]
pub const fn flatten_2d_index(x: usize, y: usize, width: usize) -> usize {
    y * width + x
}

/// Converts a 3D `(x, y, z)` coordinate into a 1D flat index `(z * height + y) * width + x`.
#[inline(always)]
#[must_use]
pub const fn flatten_3d_index(x: usize, y: usize, z: usize, width: usize, height: usize) -> usize {
    (z * height + y) * width + x
}

/// Zero-allocation caller-owned mutable scratch arena shared across linear algebra, ODE/SDE,
/// PDE/SPDE, and optimization solvers.
#[derive(Debug)]
pub struct SolverWorkspace<'a, T> {
    buf: &'a mut [T],
}

impl<'a, T> SolverWorkspace<'a, T> {
    /// Wraps a caller-owned mutable slice as a reusable solver workspace.
    #[inline]
    #[must_use]
    pub fn new(buf: &'a mut [T]) -> Self {
        Self { buf }
    }

    /// Creates an empty zero-capacity workspace for routines that operate strictly in-place.
    #[inline]
    #[must_use]
    pub fn empty() -> Self {
        Self { buf: &mut [] }
    }

    /// Returns the remaining number of scalar elements available in this workspace.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns `true` if the workspace holds zero remaining scalar elements.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Borrows the remaining workspace slice immutably.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.buf
    }

    /// Borrows the remaining workspace slice mutably.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.buf
    }

    /// Reborrows the remaining scratch slice for a nested solver call without consuming `self`.
    #[inline]
    pub fn reborrow(&mut self) -> SolverWorkspace<'_, T> {
        SolverWorkspace { buf: self.buf }
    }

    /// Fallibly carves off `count` leading elements from the workspace, advancing `self` to the tail.
    #[inline]
    pub fn try_take_slice(&mut self, count: usize) -> Result<&'a mut [T], ScalarError> {
        let available = self.buf.len();
        if count > available {
            return Err(ScalarError::WorkspaceTooSmall {
                required: count,
                available,
            });
        }
        let full = core::mem::take(&mut self.buf);
        let (head, tail) = full.split_at_mut(count);
        self.buf = tail;
        Ok(head)
    }

    /// Carves off `count` leading elements from the workspace, advancing `self` to the tail.
    ///
    /// # Panics
    /// Panics if `count > self.len()`; prefer [`Self::try_take_slice`].
    #[inline]
    pub fn take_slice(&mut self, count: usize) -> &'a mut [T] {
        self.try_take_slice(count)
            .expect("SolverWorkspace::take_slice capacity exceeded")
    }
}

impl<'a, T: RealField> SolverWorkspace<'a, T> {
    /// Fills all remaining workspace elements with [`RealField::ZERO`].
    #[inline]
    pub fn zero_fill(&mut self) {
        self.buf.fill(T::ZERO);
    }

    /// Splits off `count` zero-initialized scalar elements and borrows the remaining sub-workspace.
    #[inline]
    pub fn try_split(
        &mut self,
        count: usize,
    ) -> Result<(&mut [T], SolverWorkspace<'_, T>), ScalarError> {
        let available = self.buf.len();
        if available < count {
            return Err(ScalarError::WorkspaceTooSmall {
                required: count,
                available,
            });
        }
        let (head, tail) = self.buf.split_at_mut(count);
        head.fill(T::ZERO);
        Ok((head, SolverWorkspace { buf: tail }))
    }
}

/// Optional heap-backed workspace convenience wrapper enabled by the `alloc` Cargo feature.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct OwnedWorkspace<T> {
    buf: alloc::vec::Vec<T>,
}

#[cfg(feature = "alloc")]
impl<T: RealField> OwnedWorkspace<T> {
    /// Allocates a workspace of `capacity` zeroed scalars once for repeated zero-alloc solver calls.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: alloc::vec![T::ZERO; capacity],
        }
    }

    /// Borrows the owned buffer as a [`SolverWorkspace`].
    #[inline]
    pub fn as_workspace(&mut self) -> SolverWorkspace<'_, T> {
        SolverWorkspace::new(&mut self.buf)
    }
}
