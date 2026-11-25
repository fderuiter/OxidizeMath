// [cite:math_commons]
//! Outward-rounded IEEE-754 interval arithmetic enclosures for verified numerical kernels.

use crate::error::MathError;
use core::f64::consts::{FRAC_PI_2, PI, TAU};
use core::ops::{Add, Mul, Neg, Sub};

#[inline]
fn ulp_down_1(x: f64) -> f64 {
    if x == 0.0 {
        (-0.0_f64).next_down()
    } else {
        x.next_down()
    }
}

#[inline]
fn ulp_up_1(x: f64) -> f64 {
    if x == 0.0 {
        0.0_f64.next_up()
    } else {
        x.next_up()
    }
}

#[inline]
fn ulp_down_2(x: f64) -> f64 {
    ulp_down_1(ulp_down_1(x))
}

#[inline]
fn ulp_up_2(x: f64) -> f64 {
    ulp_up_1(ulp_up_1(x))
}

/// Closed real interval $[a, b]$ with directed outward IEEE-754 rounding (`next_down` / `next_up`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "[f64; 2]", into = "[f64; 2]")
)]
pub struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    /// Constructs a validated closed interval `[lo, hi]` when both bounds are finite and `lo <= hi`.
    #[verified_engine::verified]
    pub fn new(lo: f64, hi: f64) -> Result<Self, MathError> {
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Err(MathError::OutOfBounds {
                value: lo,
                min: lo.min(hi),
                max: hi,
            });
        }
        let n_lo = if lo == 0.0 { 0.0 } else { lo };
        let n_hi = if hi == 0.0 { 0.0 } else { hi };
        assert!(n_lo <= n_hi);
        Ok(Self { lo: n_lo, hi: n_hi })
    }

    /// Alias for [`Self::new`] constructing a validated closed interval `[lo, hi]`.
    #[verified_engine::verified]
    pub fn try_new(lo: f64, hi: f64) -> Result<Self, MathError> {
        Self::new(lo, hi)
    }

    /// Constructs a degenerate (point) interval `[x, x]` from a finite scalar `x`.
    #[verified_engine::verified]
    pub fn point(x: f64) -> Result<Self, MathError> {
        Self::new(x, x)
    }

    /// Constructs an outward-padded enclosure `[x.next_down(), x.next_up()]` around a finite scalar `x`.
    #[verified_engine::verified]
    pub fn from_approx(x: f64) -> Result<Self, MathError> {
        if !x.is_finite() {
            return Err(MathError::OutOfBounds {
                value: x,
                min: f64::MIN,
                max: f64::MAX,
            });
        }
        let lo = ulp_down_1(x);
        let hi = ulp_up_1(x);
        assert!(lo <= hi);
        Ok(Self { lo, hi })
    }

    /// Returns the lower endpoint $\underline{x}$ of the interval.
    #[inline]
    #[must_use]
    pub const fn lo(&self) -> f64 {
        self.lo
    }

    /// Returns the upper endpoint $\overline{x}$ of the interval.
    #[inline]
    #[must_use]
    pub const fn hi(&self) -> f64 {
        self.hi
    }

    /// Returns both endpoints `[lo, hi]` as a fixed-size array.
    #[inline]
    #[must_use]
    pub const fn bounds(&self) -> [f64; 2] {
        [self.lo, self.hi]
    }

    /// Computes an outward-rounded upper bound on the interval width $\overline{x} - \underline{x} \ge 0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn width(&self) -> f64 {
        if self.lo == self.hi {
            return 0.0;
        }
        let w = ulp_up_1(self.hi - self.lo).max(0.0);
        assert!(w >= 0.0);
        w
    }

    /// Computes an outward-rounded upper bound on the interval half-width (radius).
    #[must_use]
    #[verified_engine::verified]
    pub fn radius(&self) -> f64 {
        if self.lo == self.hi {
            return 0.0;
        }
        let r = ulp_up_1(0.5 * (self.hi - self.lo)).max(0.0);
        assert!(r >= 0.0);
        r
    }

    /// Computes the midpoint $(\underline{x} + \overline{x}) / 2$ of the interval.
    #[must_use]
    #[verified_engine::verified]
    pub fn midpoint(&self) -> f64 {
        let mid = (0.5 * self.lo + 0.5 * self.hi).clamp(self.lo, self.hi);
        assert!(mid >= self.lo && mid <= self.hi);
        mid
    }

    /// Returns `true` when scalar `x` lies inside `[self.lo, self.hi]`.
    #[inline]
    #[must_use]
    pub fn contains(&self, x: f64) -> bool {
        x.is_finite() && x >= self.lo && x <= self.hi
    }

    /// Returns `true` when `self` encloses `inner` (`self.lo <= inner.lo && inner.hi <= self.hi`).
    #[inline]
    #[must_use]
    pub fn encloses(&self, inner: Self) -> bool {
        self.lo <= inner.lo && inner.hi <= self.hi
    }

    /// Returns the smallest closed interval containing both `self` and `other`.
    #[must_use]
    #[verified_engine::verified]
    pub fn hull(self, other: Self) -> Self {
        let (lo, hi) = (self.lo.min(other.lo), self.hi.max(other.hi));
        assert!(lo <= hi);
        Self { lo, hi }
    }

    /// Returns the intersection of `self` and `other`, or `None` if disjoint.
    #[must_use]
    pub fn intersect(self, other: Self) -> Option<Self> {
        let (lo, hi) = (self.lo.max(other.lo), self.hi.min(other.hi));
        (lo <= hi).then_some(Self { lo, hi })
    }

    /// Clamps the interval into `[min_bound, max_bound]`, returning `None` if completely outside.
    #[must_use]
    pub fn clamp_domain(self, min_bound: f64, max_bound: f64) -> Option<Self> {
        let (lo, hi) = (self.lo.max(min_bound), self.hi.min(max_bound));
        (lo <= hi).then_some(Self { lo, hi })
    }

    /// Outward-rounded absolute value $\{ |x| : x \in [\text{lo}, \text{hi}] \}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn abs(self) -> Self {
        let (lo, hi) = if self.lo >= 0.0 {
            (self.lo, self.hi)
        } else if self.hi <= 0.0 {
            (-self.hi, -self.lo)
        } else {
            (0.0, (-self.lo).max(self.hi))
        };
        assert!(lo >= 0.0 && lo <= hi);
        Self { lo, hi }
    }

    /// Outward-rounded algebraic square $\{ x^2 : x \in [\text{lo}, \text{hi}] \}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn sqr(self) -> Self {
        let (a2, b2) = (self.lo * self.lo, self.hi * self.hi);
        let lo = if self.lo <= 0.0 && self.hi >= 0.0 {
            0.0
        } else {
            ulp_down_1(a2.min(b2)).max(0.0)
        };
        let hi = ulp_up_1(a2.max(b2));
        assert!(lo <= hi);
        Self { lo, hi }
    }

    /// Outward-rounded integer power $x^n$ for $n \ge 0$ computed iteratively without recursion.
    #[must_use]
    #[verified_engine::verified]
    pub fn powu(self, exp: u32) -> Self {
        if exp == 0 {
            return Self { lo: 1.0, hi: 1.0 };
        }
        if exp == 1 {
            return self;
        }
        let mut acc = if exp % 2 == 1 {
            self
        } else {
            Self { lo: 1.0, hi: 1.0 }
        };
        let mut base = self.sqr();
        let mut rem = exp / 2;
        for _ in 0..32 {
            if rem == 0 {
                break;
            }
            if rem % 2 == 1 {
                acc = acc * base;
            }
            rem /= 2;
            if rem > 0 {
                base = base * base;
            }
        }
        assert!(acc.lo <= acc.hi);
        acc
    }

    /// Outward-rounded division `self / rhs`, rejecting intervals `rhs` that contain `0.0`.
    #[verified_engine::verified]
    pub fn checked_div(self, rhs: Self) -> Result<Self, MathError> {
        if rhs.lo <= 0.0 && rhs.hi >= 0.0 {
            return Err(MathError::StrictlyOutOfBounds {
                value: 0.0,
                min: rhs.lo,
                max: rhs.hi,
            });
        }
        let (c0, c1, c2, c3) = (
            self.lo / rhs.lo,
            self.lo / rhs.hi,
            self.hi / rhs.lo,
            self.hi / rhs.hi,
        );
        let lo = ulp_down_1(c0.min(c1).min(c2).min(c3));
        let hi = ulp_up_1(c0.max(c1).max(c2).max(c3));
        assert!(lo <= hi);
        Ok(Self { lo, hi })
    }

    /// Outward-rounded principal square root $\sqrt{x}$ for non-negative intervals (`self.lo >= 0.0`).
    #[verified_engine::verified]
    pub fn sqrt(self) -> Result<Self, MathError> {
        if self.lo < 0.0 {
            return Err(MathError::OutOfBounds {
                value: self.lo,
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        let lo = if self.lo == 0.0 {
            0.0
        } else {
            ulp_down_1(libm::sqrt(self.lo)).max(0.0)
        };
        let hi = ulp_up_1(libm::sqrt(self.hi));
        assert!(lo <= hi);
        Ok(Self { lo, hi })
    }

    /// Outward-rounded natural exponential $e^x$.
    #[must_use]
    #[verified_engine::verified]
    pub fn exp(self) -> Self {
        let lo = ulp_down_2(libm::exp(self.lo)).max(0.0);
        let hi = ulp_up_2(libm::exp(self.hi));
        assert!(lo <= hi);
        Self { lo, hi }
    }

    /// Outward-rounded natural logarithm $\ln(x)$ for strictly positive intervals (`self.lo > 0.0`).
    #[verified_engine::verified]
    pub fn ln(self) -> Result<Self, MathError> {
        if self.lo <= 0.0 {
            return Err(MathError::StrictlyOutOfBounds {
                value: self.lo,
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        let (lo, hi) = (ulp_down_2(libm::log(self.lo)), ulp_up_2(libm::log(self.hi)));
        assert!(lo <= hi);
        Ok(Self { lo, hi })
    }

    /// Outward-rounded cosine enclosure $\cos([\text{lo}, \text{hi}]) \subseteq [-1, 1]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn cos(self) -> Self {
        if self.hi - self.lo >= TAU {
            return Self { lo: -1.0, hi: 1.0 };
        }
        let (c_lo, c_hi) = (libm::cos(self.lo), libm::cos(self.hi));
        let mut lo = ulp_down_2(c_lo.min(c_hi)).max(-1.0);
        let mut hi = ulp_up_2(c_lo.max(c_hi)).min(1.0);
        if libm::ceil(self.lo / TAU) * TAU <= self.hi + 1e-15 {
            hi = 1.0;
        }
        if libm::ceil((self.lo - PI) / TAU) * TAU + PI <= self.hi + 1e-15 {
            lo = -1.0;
        }
        assert!(lo <= hi && lo >= -1.0 && hi <= 1.0);
        Self { lo, hi }
    }

    /// Outward-rounded sine enclosure $\sin([\text{lo}, \text{hi}]) \subseteq [-1, 1]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn sin(self) -> Self {
        let shifted = Self {
            lo: ulp_down_1(self.lo - FRAC_PI_2),
            hi: ulp_up_1(self.hi - FRAC_PI_2),
        };
        shifted.cos()
    }
}

impl Neg for Interval {
    type Output = Self;
    #[verified_engine::verified]
    fn neg(self) -> Self::Output {
        let out = Self {
            lo: -self.hi,
            hi: -self.lo,
        };
        assert!(out.lo <= out.hi);
        out
    }
}

impl Add for Interval {
    type Output = Self;
    #[verified_engine::verified]
    fn add(self, rhs: Self) -> Self::Output {
        let (lo, hi) = (ulp_down_1(self.lo + rhs.lo), ulp_up_1(self.hi + rhs.hi));
        assert!(lo <= hi);
        Self { lo, hi }
    }
}

impl Sub for Interval {
    type Output = Self;
    #[verified_engine::verified]
    fn sub(self, rhs: Self) -> Self::Output {
        let (lo, hi) = (ulp_down_1(self.lo - rhs.hi), ulp_up_1(self.hi - rhs.lo));
        assert!(lo <= hi);
        Self { lo, hi }
    }
}

impl Mul for Interval {
    type Output = Self;
    #[verified_engine::verified]
    fn mul(self, rhs: Self) -> Self::Output {
        let (p0, p1, p2, p3) = (
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        );
        let lo = ulp_down_1(p0.min(p1).min(p2).min(p3));
        let hi = ulp_up_1(p0.max(p1).max(p2).max(p3));
        assert!(lo <= hi);
        Self { lo, hi }
    }
}

impl Interval {
    #[inline]
    pub(super) const fn from_raw_checked(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }
}

impl From<Interval> for [f64; 2] {
    #[inline]
    fn from(interval: Interval) -> Self {
        [interval.lo, interval.hi]
    }
}

impl core::convert::TryFrom<[f64; 2]> for Interval {
    type Error = MathError;

    #[verified_engine::verified]
    fn try_from(bounds: [f64; 2]) -> Result<Self, Self::Error> {
        Self::new(bounds[0], bounds[1])
    }
}
