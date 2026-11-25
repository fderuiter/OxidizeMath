// [cite:math_commons]
use core::fmt::Debug;
use core::ops::{Deref, DerefMut};
pub use math_scalars::RealField as Scalar;
use math_scalars::Ring;
use rand_core::RngCore;

/// A physical step size, typically used for grid spacing or integrator time steps.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StepSize<T = f64>(pub T);

impl<T> Deref for StepSize<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for StepSize<T> {
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

/// Converts a 3D coordinate `(x, y, z)` into a 1D flat index in `(z * height + y) * width + x` order.
///
/// # Example
/// ```
/// use math_commons::math_kernel::types::flatten_3d_index;
///
/// let index = flatten_3d_index(3, 2, 1, 8, 6);
/// assert_eq!(index, (6 + 2) * 8 + 3);
/// ```
#[inline(always)]
pub fn flatten_3d_index(x: usize, y: usize, z: usize, width: usize, height: usize) -> usize {
    (z * height + y) * width + x
}

/// Zero-allocation borrowed scratch workspace partitioning caller-supplied mutable slices across solver stages.
///
/// # Example
/// ```
/// use math_commons::math_kernel::types::SolverWorkspace;
///
/// let mut raw = [0.0_f64; 8];
/// let mut ws = SolverWorkspace::new(&mut raw);
/// let mut step_ws = ws.reborrow();
/// let stage_a = step_ws.take_slice(4);
/// let stage_b = step_ws.take_slice(4);
/// stage_a[0] = 1.0;
/// stage_b[0] = 2.0;
/// assert_eq!(stage_a[0] + stage_b[0], 3.0);
/// ```
#[derive(Debug)]
pub struct SolverWorkspace<'a, T = f64> {
    buf: &'a mut [T],
}

impl<'a, T> SolverWorkspace<'a, T> {
    /// Wraps a caller-provided mutable scratch slice as a zero-allocation [`SolverWorkspace`].
    #[inline]
    pub fn new(buf: &'a mut [T]) -> Self {
        Self { buf }
    }

    /// Returns the remaining number of unpartitioned scalar elements in this workspace.
    #[inline]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns `true` if no unpartitioned scalar elements remain in this workspace.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Borrows the remaining workspace buffer as an immutable slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.buf
    }

    /// Borrows the remaining workspace buffer as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.buf
    }

    /// Creates a scoped sub-workspace over the remaining slice so per-step scratch slices can be
    /// partitioned repeatedly inside loops without consuming the outer workspace.
    #[inline]
    pub fn reborrow(&mut self) -> SolverWorkspace<'_, T> {
        SolverWorkspace { buf: self.buf }
    }

    /// Fills the remaining workspace buffer with [`Ring::zero`].
    #[verified_engine::verified]
    pub fn zero_fill(&mut self)
    where
        T: Ring,
    {
        let z = T::zero();
        self.buf.fill(z.clone());
        debug_assert!(self.buf.first().is_none_or(|v| *v == z));
    }

    /// Carves off a disjoint mutable slice of length `len` from the front of the workspace,
    /// returning [`MathError::WorkspaceTooSmall`](crate::error::MathError::WorkspaceTooSmall)
    /// without heap allocation when `len > self.len()`.
    #[verified_engine::verified]
    pub fn try_take_slice(&mut self, len: usize) -> Result<&'a mut [T], crate::error::MathError> {
        let available = self.buf.len();
        if len > available {
            return Err(crate::error::MathError::WorkspaceTooSmall {
                required: len,
                available,
            });
        }
        debug_assert!(len <= available);
        let full = core::mem::take(&mut self.buf);
        let (head, tail) = full.split_at_mut(len);
        self.buf = tail;
        Ok(head)
    }

    /// Carves off a disjoint mutable slice of length `len` from the front of the workspace,
    /// advancing `self` to the remaining suffix without heap allocation.
    ///
    /// # Panics
    /// Panics if `len > self.len()`.
    #[verified_engine::verified]
    pub fn take_slice(&mut self, len: usize) -> &'a mut [T] {
        assert!(
            len <= self.buf.len(),
            "SolverWorkspace::take_slice requested {len} elements from buffer of length {}",
            self.buf.len()
        );
        let full = core::mem::take(&mut self.buf);
        let (head, tail) = full.split_at_mut(len);
        self.buf = tail;
        head
    }
}

/// Samples a deterministic uniform variate strictly in `(0, 1)` (clamped `<= 1.0` after `f32` cast)
/// via [`math_rng::sample_open`], avoiding `0.0` so `ln(u)` in Box-Muller sampling never overflows.
#[inline]
#[verified_engine::verified]
pub fn sample_open_unit<T: Scalar, R: RngCore + ?Sized>(rng: &mut R) -> T {
    let u_f64: f64 = math_rng::sample_open(rng);
    let u = T::from_f64(u_f64);
    debug_assert!(u > T::ZERO && u <= T::ONE);
    u
}

/// Samples an independent pair of standard normal $\mathcal{N}(0, 1)$ variates via the Box-Muller transform.
#[inline]
#[verified_engine::verified]
pub fn sample_standard_normal_pair<T: Scalar, R: RngCore + ?Sized>(rng: &mut R) -> (T, T) {
    let u1 = sample_open_unit::<T, R>(rng);
    let u2 = sample_open_unit::<T, R>(rng);
    let neg_two_ln = -(T::TWO * u1.ln());
    let radius = neg_two_ln.max(T::ZERO).sqrt();
    let theta = T::TAU * u2;
    let (s, c) = theta.sin_cos();
    let z0 = radius * c;
    let z1 = radius * s;
    debug_assert!(z0.is_finite() && z1.is_finite());
    (z0, z1)
}

/// Samples a single standard normal $\mathcal{N}(0, 1)$ variate from `rng` via Box-Muller.
#[inline]
#[verified_engine::verified]
pub fn sample_standard_normal<T: Scalar, R: RngCore + ?Sized>(rng: &mut R) -> T {
    let (z0, _) = sample_standard_normal_pair::<T, R>(rng);
    debug_assert!(z0.is_finite());
    z0
}

/// Fills `out` in-place with independent standard normal $\mathcal{N}(0, 1)$ samples drawn from `rng`.
#[verified_engine::verified]
pub fn fill_standard_normal<T: Scalar, R: RngCore + ?Sized>(out: &mut [T], rng: &mut R) {
    let (pairs, rem) = out.as_chunks_mut::<2>();
    for [p0, p1] in pairs {
        let (z0, z1) = sample_standard_normal_pair::<T, R>(rng);
        *p0 = z0;
        *p1 = z1;
    }
    if let Some(last) = rem.first_mut() {
        *last = sample_standard_normal::<T, R>(rng);
    }
    debug_assert!(out.iter().all(|v| v.is_finite()));
}

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
