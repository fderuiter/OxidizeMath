// [cite:math_commons]
use core::ops::{Deref, DerefMut};
use rand_core::RngCore;

pub use math_scalars::Scalar;

/// A physical grid or integration step size requiring explicit scalar extraction via [`StepSize::get`].
///
/// `StepSize` intentionally omits `Deref` and `DerefMut` so callers cannot silently coerce grid
/// spacings into bare `f64` expressions:
///
/// ```compile_fail
/// use math_commons::math_kernel::types::StepSize;
///
/// let dx = StepSize::new(0.05);
/// let _raw: f64 = *dx;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct StepSize(pub f64);

impl StepSize {
    /// Wraps a raw scalar grid or time step value.
    #[inline(always)]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Returns the underlying scalar step value by copy.
    #[inline(always)]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// Returns the underlying scalar step value.
    #[inline(always)]
    pub const fn value(&self) -> f64 {
        self.0
    }

    /// Returns `true` if the step size is neither infinite nor `NaN`.
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}

/// A grid index, representing a coordinate or an offset in a flattened array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
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
pub struct SolverWorkspace<'a, T: Scalar> {
    buf: &'a mut [T],
}

impl<'a, T: Scalar> SolverWorkspace<'a, T> {
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

    /// Fills the remaining workspace buffer with [`Scalar::ZERO`].
    #[verified_engine::verified]
    pub fn zero_fill(&mut self) {
        self.buf.fill(T::ZERO);
        debug_assert!(self.buf.first().is_none_or(|v| *v == T::ZERO));
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

/// Samples a deterministic uniform variate strictly in `(0, 1]` using 53 bits from `rng`.
///
/// Shifting `rng.next_u64() >> 11` and adding `1` yields integers in `1..=2^53`, avoiding `0.0`
/// so `ln(u)` in Box-Muller sampling never overflows or produces `-inf`.
#[inline]
#[verified_engine::verified]
pub fn sample_open_unit<T: Scalar, R: RngCore + ?Sized>(rng: &mut R) -> T {
    const INV_TWO_POW_53: f64 = 1.0 / 9_007_199_254_740_992.0;
    let mantissa_plus_one = (rng.next_u64() >> 11) + 1;
    let u = T::from_f64((mantissa_plus_one as f64) * INV_TWO_POW_53);
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
    let z0 = radius * theta.cos();
    let z1 = radius * theta.sin();
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
