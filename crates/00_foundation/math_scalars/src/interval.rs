// [cite:math_scalars]
//! Outward-rounded IEEE-754 interval enclosure `Interval<T>` (`\[lo, hi\]`).

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::rounding::{add_bounds, div_bounds, mul_bounds, sqrt_bounds, sub_bounds};
use crate::traits::RealScalar;

/// Closed real interval `\[lo, hi\]` with portable IEEE-754 outward-rounded operations.
///
/// Never mutates FPU rounding-mode control registers (`fesetround` is unused), guaranteeing
/// bit-identical valid enclosures on both `x86_64` and `wasm32-unknown-unknown`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Interval<T> {
    /// Lower bound of the enclosure.
    pub lo: T,
    /// Upper bound of the enclosure.
    pub hi: T,
}

impl<T: RealScalar> Default for Interval<T> {
    #[inline]
    fn default() -> Self {
        Self::point(T::ZERO)
    }
}

impl<T: RealScalar> Interval<T> {
    /// Constructs `\[lo, hi\]` when `lo <= hi`, or `(NaN, NaN)` if bounds are invalid/`NaN`.
    #[inline]
    #[must_use]
    pub fn new(lo: T, hi: T) -> Self {
        if lo.is_nan() || hi.is_nan() || lo > hi {
            Self {
                lo: T::NAN,
                hi: T::NAN,
            }
        } else {
            Self { lo, hi }
        }
    }

    /// Degenerate point interval `\[v, v\]`.
    #[inline]
    #[must_use]
    pub const fn point(v: T) -> Self {
        Self { lo: v, hi: v }
    }

    /// Entire extended real line `\[-inf, +inf\]`.
    #[inline]
    #[must_use]
    pub const fn entire() -> Self {
        Self {
            lo: T::NEG_INFINITY,
            hi: T::INFINITY,
        }
    }

    /// Invalid/empty sentinel `(NaN, NaN)`.
    #[inline]
    #[must_use]
    pub const fn nan() -> Self {
        Self {
            lo: T::NAN,
            hi: T::NAN,
        }
    }

    /// Rigorous 1-ULP enclosure of $\pi$.
    #[inline]
    #[must_use]
    pub const fn pi_enclosure() -> Self {
        Self {
            lo: T::PI_LO,
            hi: T::PI_HI,
        }
    }

    /// Rigorous enclosure of $2\pi$.
    #[inline]
    #[must_use]
    pub fn tau_enclosure() -> Self {
        Self::new(T::TAU.next_down(), T::TAU.next_up())
    }

    /// Rigorous enclosure of $\pi / 2$.
    #[inline]
    #[must_use]
    pub fn frac_pi_2_enclosure() -> Self {
        Self::new(T::FRAC_PI_2.next_down(), T::FRAC_PI_2.next_up())
    }

    /// Returns `true` if either bound is `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.lo.is_nan() || self.hi.is_nan()
    }

    /// Returns `true` if `self` is non-`NaN` and both bounds are finite.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        !self.is_nan() && self.lo.is_finite() && self.hi.is_finite()
    }

    /// Returns `true` when `x` lies inside `\[lo, hi\]`.
    #[inline]
    #[must_use]
    pub fn contains(self, x: T) -> bool {
        !self.is_nan() && !x.is_nan() && self.lo <= x && x <= self.hi
    }

    /// Returns `true` when `other` is a subset of `self` (`self.lo <= other.lo && other.hi <= self.hi`).
    #[inline]
    #[must_use]
    pub fn contains_interval(self, other: Self) -> bool {
        !self.is_nan() && !other.is_nan() && self.lo <= other.lo && other.hi <= self.hi
    }

    /// Overflow-safe midpoint `0.5 * lo + 0.5 * hi` (clamped into `\[lo, hi\]`).
    #[inline]
    #[must_use]
    pub fn midpoint(self) -> T {
        if self.is_nan() {
            return T::NAN;
        }
        if self.lo == self.hi {
            return self.lo;
        }
        let m = T::HALF * self.lo + T::HALF * self.hi;
        if m.is_finite() {
            m.max_val(self.lo).min_val(self.hi)
        } else {
            T::ZERO.max_val(self.lo).min_val(self.hi)
        }
    }

    /// Outward-rounded width `hi - lo`.
    #[inline]
    #[must_use]
    pub fn width(self) -> T {
        if self.is_nan() {
            T::NAN
        } else {
            sub_bounds(self.hi, self.lo).1
        }
    }

    /// Outward-rounded radius around [`Self::midpoint`].
    #[inline]
    #[must_use]
    pub fn radius(self) -> T {
        if self.is_nan() {
            return T::NAN;
        }
        let m = self.midpoint();
        let r_lo = sub_bounds(m, self.lo).1;
        let r_hi = sub_bounds(self.hi, m).1;
        r_lo.max_val(r_hi).max_val(T::ZERO)
    }

    /// Number of IEEE-754 ULP steps spanning `\[lo, hi\]` (`u64::MAX` if `NaN`).
    #[inline]
    #[must_use]
    pub fn width_ulps(self) -> u64 {
        if self.is_nan() {
            u64::MAX
        } else {
            self.lo.ulp_distance(self.hi)
        }
    }

    /// Convex hull enclosing both `self` and `other`.
    #[inline]
    #[must_use]
    pub fn hull(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            Self::nan()
        } else {
            Self::new(self.lo.min_val(other.lo), self.hi.max_val(other.hi))
        }
    }

    /// Set intersection `self ∩ other`, returning `None` if disjoint or `NaN`.
    #[inline]
    #[must_use]
    pub fn intersect(self, other: Self) -> Option<Self> {
        if self.is_nan() || other.is_nan() {
            return None;
        }
        let lo = self.lo.max_val(other.lo);
        let hi = self.hi.min_val(other.hi);
        if lo <= hi {
            Some(Self { lo, hi })
        } else {
            None
        }
    }

    /// Magnitude range $\{|x| : x \in \[lo, hi\]\}$.
    #[inline]
    #[must_use]
    pub fn abs_interval(self) -> Self {
        if self.is_nan() {
            return Self::nan();
        }
        if self.lo >= T::ZERO {
            self
        } else if self.hi <= T::ZERO {
            -self
        } else {
            Self::new(T::ZERO, (-self.lo).max_val(self.hi))
        }
    }

    /// Algebraic square $\{x^2 : x \in \[lo, hi\]\}$ exploiting non-negativity.
    #[inline]
    #[must_use]
    pub fn sqr_interval(self) -> Self {
        let mag = self.abs_interval();
        if mag.is_nan() {
            return Self::nan();
        }
        let lo = mul_bounds(mag.lo, mag.lo).0.max_val(T::ZERO);
        let hi = mul_bounds(mag.hi, mag.hi).1;
        Self::new(lo, hi)
    }

    /// Outward-rounded principal square root $\sqrt{\[lo, hi\]}$ (`NaN` if `lo < 0`).
    #[inline]
    #[must_use]
    pub fn sqrt_interval(self) -> Self {
        if self.is_nan() || self.lo < T::ZERO {
            return Self::nan();
        }
        let lo = sqrt_bounds(self.lo).0.max_val(T::ZERO);
        let hi = sqrt_bounds(self.hi).1;
        Self::new(lo, hi)
    }
}

impl<T: RealScalar> Neg for Interval<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        if self.is_nan() {
            Self::nan()
        } else {
            Self {
                lo: -self.hi,
                hi: -self.lo,
            }
        }
    }
}

impl<T: RealScalar> Add for Interval<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            return Self::nan();
        }
        Self::new(add_bounds(self.lo, rhs.lo).0, add_bounds(self.hi, rhs.hi).1)
    }
}

impl<T: RealScalar> Sub for Interval<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            return Self::nan();
        }
        Self::new(sub_bounds(self.lo, rhs.hi).0, sub_bounds(self.hi, rhs.lo).1)
    }
}

impl<T: RealScalar> Mul for Interval<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            return Self::nan();
        }
        let p1 = mul_bounds(self.lo, rhs.lo);
        let p2 = mul_bounds(self.lo, rhs.hi);
        let p3 = mul_bounds(self.hi, rhs.lo);
        let p4 = mul_bounds(self.hi, rhs.hi);
        if p1.0.is_nan() || p2.0.is_nan() || p3.0.is_nan() || p4.0.is_nan() {
            return Self::entire();
        }
        let lo = p1.0.min_val(p2.0).min_val(p3.0).min_val(p4.0);
        let hi = p1.1.max_val(p2.1).max_val(p3.1).max_val(p4.1);
        Self::new(lo, hi)
    }
}

impl<T: RealScalar> Div for Interval<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            return Self::nan();
        }
        if rhs.lo <= T::ZERO && T::ZERO <= rhs.hi {
            if rhs.lo == T::ZERO && rhs.hi == T::ZERO {
                return Self::nan();
            }
            return Self::entire();
        }
        let q1 = div_bounds(self.lo, rhs.lo);
        let q2 = div_bounds(self.lo, rhs.hi);
        let q3 = div_bounds(self.hi, rhs.lo);
        let q4 = div_bounds(self.hi, rhs.hi);
        if q1.0.is_nan() || q2.0.is_nan() || q3.0.is_nan() || q4.0.is_nan() {
            return Self::entire();
        }
        let lo = q1.0.min_val(q2.0).min_val(q3.0).min_val(q4.0);
        let hi = q1.1.max_val(q2.1).max_val(q3.1).max_val(q4.1);
        Self::new(lo, hi)
    }
}

impl<T: RealScalar> AddAssign for Interval<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: RealScalar> SubAssign for Interval<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: RealScalar> MulAssign for Interval<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: RealScalar> DivAssign for Interval<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
