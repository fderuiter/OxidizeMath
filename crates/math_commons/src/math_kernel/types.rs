// [cite:math_commons]
use core::fmt::Debug;
use core::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};
use rand_core::RngCore;

/// A physical step size, typically used for grid spacing.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
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

/// `#![no_std]` Tier 0 real floating-point scalar contract for deterministic ODE, SDE, PDE, and SPDE solvers.
///
/// Implemented for `f64` and `f32` via `libm` so transcendental evaluations are available in bare `#![no_std]`
/// builds and deterministic across native and `wasm32-unknown-unknown` targets.
pub trait Scalar:
    Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Debug
    + Default
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// Additive identity `0`.
    const ZERO: Self;
    /// Multiplicative identity `1`.
    const ONE: Self;
    /// Constant `2`.
    const TWO: Self;
    /// Constant `0.5`.
    const HALF: Self;
    /// Archimedes' constant $\pi$.
    const PI: Self;
    /// Full-turn constant $\tau = 2\pi$.
    const TAU: Self;
    /// Machine epsilon for this floating-point type.
    const EPSILON: Self;

    /// Converts an `f64` literal or parameter into `Self`.
    fn from_f64(value: f64) -> Self;
    /// Converts `Self` into an `f64` value.
    fn to_f64(self) -> f64;
    /// Converts a `usize` count or grid dimension into `Self`.
    fn from_usize(value: usize) -> Self;
    /// Returns `true` if `self` is neither infinite nor `NaN`.
    fn is_finite(self) -> bool;
    /// Returns the absolute value $|x|$.
    fn abs(self) -> Self;
    /// Returns the principal square root $\sqrt{x}$.
    fn sqrt(self) -> Self;
    /// Returns the natural logarithm $\ln(x)$.
    fn ln(self) -> Self;
    /// Returns the natural exponential $e^x$.
    fn exp(self) -> Self;
    /// Returns the sine $\sin(x)$.
    fn sin(self) -> Self;
    /// Returns the cosine $\cos(x)$.
    fn cos(self) -> Self;
    /// Returns the minimum of `self` and `other` following IEEE-754 `fmin` semantics.
    fn min(self, other: Self) -> Self;
    /// Returns the maximum of `self` and `other` following IEEE-754 `fmax` semantics.
    fn max(self, other: Self) -> Self;
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f64::consts::PI;
    const TAU: Self = core::f64::consts::TAU;
    const EPSILON: Self = f64::EPSILON;

    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        value
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        value as f64
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        libm::fabs(self)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline(always)]
    fn ln(self) -> Self {
        libm::log(self)
    }
    #[inline(always)]
    fn exp(self) -> Self {
        libm::exp(self)
    }
    #[inline(always)]
    fn sin(self) -> Self {
        libm::sin(self)
    }
    #[inline(always)]
    fn cos(self) -> Self {
        libm::cos(self)
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        libm::fmin(self, other)
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        libm::fmax(self, other)
    }
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f32::consts::PI;
    const TAU: Self = core::f32::consts::TAU;
    const EPSILON: Self = f32::EPSILON;

    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        value as f32
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        libm::fabsf(self)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline(always)]
    fn ln(self) -> Self {
        libm::logf(self)
    }
    #[inline(always)]
    fn exp(self) -> Self {
        libm::expf(self)
    }
    #[inline(always)]
    fn sin(self) -> Self {
        libm::sinf(self)
    }
    #[inline(always)]
    fn cos(self) -> Self {
        libm::cosf(self)
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        libm::fminf(self, other)
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        libm::fmaxf(self, other)
    }
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
#[derive(Debug, PartialEq)]
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
