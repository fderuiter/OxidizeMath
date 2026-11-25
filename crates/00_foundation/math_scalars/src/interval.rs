//! Outward-rounded closed real interval `[lo, hi]` (`Interval<T>`).
// [cite:math_commons]

use crate::directed::DirectedRounding;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Closed real interval $[lo, hi]$ guaranteeing outward-rounded containment across
/// native (`x86_64`, `aarch64`) and `wasm32-unknown-unknown` targets.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Interval<T> {
    /// Lower endpoint $lo \le hi$.
    pub lo: T,
    /// Upper endpoint $hi \ge lo$.
    pub hi: T,
}

impl<T: DirectedRounding> Default for Interval<T> {
    #[inline]
    fn default() -> Self {
        Self::point(T::ZERO)
    }
}

impl<T: DirectedRounding> Interval<T> {
    /// Constructs a validated interval `[min(a, b), max(a, b)]` (widening `NaN` to `[-inf, +inf]`).
    #[inline]
    #[must_use]
    pub fn new(a: T, b: T) -> Self {
        if a.is_nan() || b.is_nan() {
            return Self::entire();
        }
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        debug_assert!(lo <= hi, "Interval endpoints must satisfy lo <= hi");
        Self { lo, hi }
    }

    /// Constructs a degenerate point interval `[x, x]`.
    #[inline]
    #[must_use]
    pub fn point(x: T) -> Self {
        if x.is_nan() {
            Self::entire()
        } else {
            Self { lo: x, hi: x }
        }
    }

    /// Constructs the entire extended real line $[-\infty, +\infty]$.
    #[inline]
    #[must_use]
    pub const fn entire() -> Self {
        Self {
            lo: T::NEG_INFINITY,
            hi: T::INFINITY,
        }
    }

    /// Constructs a tight outward-rounded interval enclosing an `f64` constant.
    #[inline]
    #[must_use]
    pub fn from_f64_enclosure(v: f64) -> Self {
        if v.is_nan() {
            return Self::entire();
        }
        let approx = T::from_f64(v);
        let approx_f64 = approx.to_f64();
        if approx_f64 == v {
            Self {
                lo: approx,
                hi: approx,
            }
        } else if approx_f64 < v {
            Self {
                lo: approx,
                hi: approx.next_up(),
            }
        } else {
            Self {
                lo: approx.next_down(),
                hi: approx,
            }
        }
    }

    /// Returns `true` if $x \in [lo, hi]$.
    #[inline]
    #[must_use]
    pub fn contains(self, x: T) -> bool {
        !x.is_nan() && self.lo <= x && x <= self.hi
    }

    /// Returns `true` if $0 \in [lo, hi]$.
    #[inline]
    #[must_use]
    pub fn contains_zero(self) -> bool {
        self.lo <= T::ZERO && T::ZERO <= self.hi
    }

    /// Returns `true` if `other` is a subset of `self` ($self.lo \le other.lo$ and $other.hi \le self.hi$).
    #[inline]
    #[must_use]
    pub fn contains_subset(self, other: Self) -> bool {
        self.lo <= other.lo && other.hi <= self.hi
    }

    /// Returns `true` if `other` lies strictly inside the interior of `self` ($self.lo < other.lo$ and $other.hi < self.hi$).
    #[inline]
    #[must_use]
    pub fn interior_contains(self, other: Self) -> bool {
        self.lo < other.lo && other.hi < self.hi
    }

    /// Intersects `self` and `other`, returning `None` if disjoint.
    #[inline]
    #[must_use]
    pub fn intersect(self, other: Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi {
            Some(Self { lo, hi })
        } else {
            None
        }
    }

    /// Convex hull enclosing both `self` and `other`.
    #[inline]
    #[must_use]
    pub fn hull(self, other: Self) -> Self {
        if self.lo.is_nan() || self.hi.is_nan() || other.lo.is_nan() || other.hi.is_nan() {
            return Self::entire();
        }
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Floating-point midpoint inside `[lo, hi]` (overflow-safe via `0.5 * lo + 0.5 * hi`).
    #[inline]
    #[must_use]
    pub fn midpoint(self) -> T {
        if self.lo == self.hi {
            return self.lo;
        }
        if !self.lo.is_finite() && !self.hi.is_finite() {
            return T::ZERO;
        }
        if !self.lo.is_finite() {
            return -T::MAX_VALUE;
        }
        if !self.hi.is_finite() {
            return T::MAX_VALUE;
        }
        let mid = T::HALF * self.lo + T::HALF * self.hi;
        mid.max(self.lo).min(self.hi)
    }

    /// Outward-rounded width $hi -_{\uparrow} lo \ge 0$.
    #[inline]
    #[must_use]
    pub fn width(self) -> T {
        self.hi.sub_up(self.lo).max(T::ZERO)
    }

    /// Outward-rounded radius around `self.midpoint()`.
    #[inline]
    #[must_use]
    pub fn radius(self) -> T {
        let m = self.midpoint();
        let r_lo = m.sub_up(self.lo);
        let r_hi = self.hi.sub_up(m);
        r_lo.max(r_hi).max(T::ZERO)
    }

    /// Exact IEEE-754 ULP distance between `lo` and `hi` (`0` for point intervals).
    #[inline]
    #[must_use]
    pub fn ulp_width(self) -> u64 {
        self.lo.ulp_distance(self.hi)
    }

    /// Splits `[lo, hi]` at its midpoint into `([lo, mid], [mid, hi])`.
    #[inline]
    #[must_use]
    pub fn bisect(self) -> (Self, Self) {
        let m = self.midpoint();
        (Self { lo: self.lo, hi: m }, Self { lo: m, hi: self.hi })
    }

    /// Magnitude $\max(|lo|, |hi|)$.
    #[inline]
    #[must_use]
    pub fn mag(self) -> T {
        self.lo.abs().max(self.hi.abs())
    }

    /// Mignitude $\min_{x \in [lo, hi]} |x|$.
    #[inline]
    #[must_use]
    pub fn mig(self) -> T {
        if self.contains_zero() {
            T::ZERO
        } else {
            self.lo.abs().min(self.hi.abs())
        }
    }

    /// Tight outward-rounded square $X^2 = \{x^2 : x \in X\}$ exploiting non-negativity.
    #[inline]
    #[must_use]
    pub fn sqr(self) -> Self {
        if self.contains_zero() {
            let m = self.mag();
            Self {
                lo: T::ZERO,
                hi: m.mul_up(m),
            }
        } else {
            let a = self.lo.abs().min(self.hi.abs());
            let b = self.mag();
            Self {
                lo: a.mul_down(a).max(T::ZERO),
                hi: b.mul_up(b),
            }
        }
    }
}

impl<T: DirectedRounding> Add for Interval<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.lo.add_down(rhs.lo), self.hi.add_up(rhs.hi))
    }
}

impl<T: DirectedRounding> Add<T> for Interval<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: T) -> Self {
        self + Self::point(rhs)
    }
}

impl<T: DirectedRounding> Sub for Interval<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.lo.sub_down(rhs.hi), self.hi.sub_up(rhs.lo))
    }
}

impl<T: DirectedRounding> Sub<T> for Interval<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: T) -> Self {
        self - Self::point(rhs)
    }
}

impl<T: DirectedRounding> Mul for Interval<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let p1_lo = self.lo.mul_down(rhs.lo);
        let p2_lo = self.lo.mul_down(rhs.hi);
        let p3_lo = self.hi.mul_down(rhs.lo);
        let p4_lo = self.hi.mul_down(rhs.hi);

        let p1_hi = self.lo.mul_up(rhs.lo);
        let p2_hi = self.lo.mul_up(rhs.hi);
        let p3_hi = self.hi.mul_up(rhs.lo);
        let p4_hi = self.hi.mul_up(rhs.hi);

        if p1_lo.is_nan()
            || p2_lo.is_nan()
            || p3_lo.is_nan()
            || p4_lo.is_nan()
            || p1_hi.is_nan()
            || p2_hi.is_nan()
            || p3_hi.is_nan()
            || p4_hi.is_nan()
        {
            return Self::entire();
        }
        let lo = p1_lo.min(p2_lo).min(p3_lo.min(p4_lo));
        let hi = p1_hi.max(p2_hi).max(p3_hi.max(p4_hi));
        Self::new(lo, hi)
    }
}

impl<T: DirectedRounding> Mul<T> for Interval<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self {
        self * Self::point(rhs)
    }
}

impl<T: DirectedRounding> Div for Interval<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        if rhs.contains_zero() {
            if rhs.lo == T::ZERO && rhs.hi > T::ZERO {
                let q1 = self.lo.div_down(rhs.hi);
                let q2 = self.hi.div_up(rhs.hi);
                if self.lo >= T::ZERO {
                    return Self::new(q1.max(T::ZERO), T::INFINITY);
                }
                if self.hi <= T::ZERO {
                    return Self::new(T::NEG_INFINITY, q2.min(T::ZERO));
                }
            }
            if rhs.hi == T::ZERO && rhs.lo < T::ZERO {
                let q1 = self.hi.div_down(rhs.lo);
                let q2 = self.lo.div_up(rhs.lo);
                if self.lo >= T::ZERO {
                    return Self::new(T::NEG_INFINITY, q2.min(T::ZERO));
                }
                if self.hi <= T::ZERO {
                    return Self::new(q1.max(T::ZERO), T::INFINITY);
                }
            }
            return Self::entire();
        }

        let d1_lo = self.lo.div_down(rhs.lo);
        let d2_lo = self.lo.div_down(rhs.hi);
        let d3_lo = self.hi.div_down(rhs.lo);
        let d4_lo = self.hi.div_down(rhs.hi);

        let d1_hi = self.lo.div_up(rhs.lo);
        let d2_hi = self.lo.div_up(rhs.hi);
        let d3_hi = self.hi.div_up(rhs.lo);
        let d4_hi = self.hi.div_up(rhs.hi);

        let lo = d1_lo.min(d2_lo).min(d3_lo.min(d4_lo));
        let hi = d1_hi.max(d2_hi).max(d3_hi.max(d4_hi));
        Self::new(lo, hi)
    }
}

impl<T: DirectedRounding> Div<T> for Interval<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self {
        self / Self::point(rhs)
    }
}

impl<T: DirectedRounding> Neg for Interval<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.hi, -self.lo)
    }
}

impl<T: DirectedRounding> AddAssign for Interval<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: DirectedRounding> SubAssign for Interval<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: DirectedRounding> MulAssign for Interval<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: DirectedRounding> DivAssign for Interval<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
