//! Outward-rounded midpoint-radius ball arithmetic (`Ball<T>`).
// [cite:math_commons]

use crate::directed::DirectedRounding;
use crate::interval::Interval;
use crate::traits::Scalar;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Midpoint-radius real ball $(mid, rad)$ representing $\{x \in \mathbb{R} : |x - mid| \le rad\}$.
///
/// Propagates rounding errors of midpoint operations outward into `rad` via `DirectedRounding`
/// without mutating FPU state, ensuring portable enclosures on `x86_64` and `wasm32`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ball<T> {
    /// Center approximation.
    pub mid: T,
    /// Non-negative outward-rounded radius bound.
    pub rad: T,
}

impl<T: DirectedRounding> Default for Ball<T> {
    #[inline]
    fn default() -> Self {
        Self::point(T::ZERO)
    }
}

impl<T: DirectedRounding> Ball<T> {
    /// Constructs a ball `(mid, |rad|)` (widening `NaN` to `(0, +inf)`).
    #[inline]
    #[must_use]
    pub fn new(mid: T, rad: T) -> Self {
        if mid.is_nan() || rad.is_nan() {
            return Self {
                mid: T::ZERO,
                rad: T::INFINITY,
            };
        }
        Self {
            mid,
            rad: rad.abs(),
        }
    }

    /// Constructs an exact zero-radius ball `(x, 0)`.
    #[inline]
    #[must_use]
    pub fn point(x: T) -> Self {
        if x.is_nan() {
            Self {
                mid: T::ZERO,
                rad: T::INFINITY,
            }
        } else {
            Self {
                mid: x,
                rad: T::ZERO,
            }
        }
    }

    /// Converts a closed interval `[lo, hi]` into an outward-rounded enclosing ball.
    #[inline]
    #[must_use]
    pub fn from_interval(iv: Interval<T>) -> Self {
        if !iv.is_finite() {
            return Self {
                mid: T::ZERO,
                rad: T::INFINITY,
            };
        }
        let mid = iv.midpoint();
        let rad = mid.sub_up(iv.lo).max(iv.hi.sub_up(mid)).max(T::ZERO);
        Self { mid, rad }
    }

    /// Converts `self` into an outward-rounded closed interval `[mid -_down rad, mid +_up rad]`.
    #[inline]
    #[must_use]
    pub fn to_interval(self) -> Interval<T> {
        if !self.mid.is_finite() || !self.rad.is_finite() {
            return Interval::entire();
        }
        Interval::new(self.mid.sub_down(self.rad), self.mid.add_up(self.rad))
    }

    /// Returns `true` if `x` lies inside the outward-rounded interval of `self`.
    #[inline]
    #[must_use]
    pub fn contains(self, x: T) -> bool {
        self.to_interval().contains(x)
    }

    /// Returns `true` if `0` lies inside `self`.
    #[inline]
    #[must_use]
    pub fn contains_zero(self) -> bool {
        self.to_interval().contains_zero()
    }

    /// Exact IEEE-754 ULP width of `self.to_interval()`.
    #[inline]
    #[must_use]
    pub fn ulp_width(self) -> u64 {
        self.to_interval().ulp_width()
    }
}

impl<T: DirectedRounding> Add for Ball<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        if !self.mid.is_finite()
            || !rhs.mid.is_finite()
            || !self.rad.is_finite()
            || !rhs.rad.is_finite()
        {
            return Self::from_interval(self.to_interval() + rhs.to_interval());
        }
        let mid = self.mid + rhs.mid;
        let lo_m = self.mid.add_down(rhs.mid);
        let hi_m = self.mid.add_up(rhs.mid);
        let err_m = hi_m.sub_up(mid).max(mid.sub_up(lo_m));
        let rad = err_m.add_up(self.rad.add_up(rhs.rad));
        Self::new(mid, rad)
    }
}

impl<T: DirectedRounding> Add<T> for Ball<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: T) -> Self {
        self + Self::point(rhs)
    }
}

impl<T: DirectedRounding> Sub for Ball<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<T: DirectedRounding> Sub<T> for Ball<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: T) -> Self {
        self - Self::point(rhs)
    }
}

impl<T: DirectedRounding> Mul for Ball<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        if !self.mid.is_finite()
            || !rhs.mid.is_finite()
            || !self.rad.is_finite()
            || !rhs.rad.is_finite()
        {
            return Self::from_interval(self.to_interval() * rhs.to_interval());
        }
        let mid = self.mid * rhs.mid;
        let lo_m = self.mid.mul_down(rhs.mid);
        let hi_m = self.mid.mul_up(rhs.mid);
        let err_m = hi_m.sub_up(mid).max(mid.sub_up(lo_m));
        let t1 = self.mid.abs().mul_up(rhs.rad);
        let t2 = rhs.mid.abs().mul_up(self.rad);
        let t3 = self.rad.mul_up(rhs.rad);
        let rad = err_m.add_up(t1.add_up(t2.add_up(t3)));
        Self::new(mid, rad)
    }
}

impl<T: DirectedRounding> Mul<T> for Ball<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self {
        self * Self::point(rhs)
    }
}

impl<T: DirectedRounding> Div for Ball<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self::from_interval(self.to_interval() / rhs.to_interval())
    }
}

impl<T: DirectedRounding> Div<T> for Ball<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self {
        self / Self::point(rhs)
    }
}

impl<T: DirectedRounding> Neg for Ball<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            mid: -self.mid,
            rad: self.rad,
        }
    }
}

impl<T: DirectedRounding> AddAssign for Ball<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: DirectedRounding> SubAssign for Ball<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: DirectedRounding> MulAssign for Ball<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: DirectedRounding> DivAssign for Ball<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: DirectedRounding> Scalar for Ball<T> {
    const ZERO: Self = Self {
        mid: T::ZERO,
        rad: T::ZERO,
    };
    const ONE: Self = Self {
        mid: T::ONE,
        rad: T::ZERO,
    };
    const TWO: Self = Self {
        mid: T::TWO,
        rad: T::ZERO,
    };
    const HALF: Self = Self {
        mid: T::HALF,
        rad: T::ZERO,
    };
    const PI: Self = Self {
        mid: T::PI,
        rad: T::PI_RAD,
    };
    const TAU: Self = Self {
        mid: T::TAU,
        rad: T::TAU_RAD,
    };

    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::from_interval(Interval::from_f64_enclosure(v))
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.mid.is_finite() && self.rad.is_finite()
    }

    #[inline]
    fn abs(self) -> Self {
        Self::from_interval(self.to_interval().abs())
    }

    #[inline]
    fn signum(self) -> Self {
        Self::from_interval(self.to_interval().signum())
    }

    #[inline]
    fn sqrt(self) -> Self {
        Self::from_interval(self.to_interval().sqrt())
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        Self::from_interval(self.to_interval().powi(n))
    }

    #[inline]
    fn powf(self, exp: Self) -> Self {
        Self::from_interval(self.to_interval().powf(exp.to_interval()))
    }

    #[inline]
    fn exp(self) -> Self {
        Self::from_interval(self.to_interval().exp())
    }

    #[inline]
    fn ln(self) -> Self {
        Self::from_interval(self.to_interval().ln())
    }

    #[inline]
    fn sin(self) -> Self {
        Self::from_interval(self.to_interval().sin())
    }

    #[inline]
    fn cos(self) -> Self {
        Self::from_interval(self.to_interval().cos())
    }

    #[inline]
    fn tan(self) -> Self {
        Self::from_interval(self.to_interval().tan())
    }

    #[inline]
    fn asin(self) -> Self {
        Self::from_interval(self.to_interval().asin())
    }

    #[inline]
    fn acos(self) -> Self {
        Self::from_interval(self.to_interval().acos())
    }

    #[inline]
    fn atan(self) -> Self {
        Self::from_interval(self.to_interval().atan())
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        Self::from_interval(self.to_interval().atan2(other.to_interval()))
    }

    #[inline]
    fn sinh(self) -> Self {
        Self::from_interval(self.to_interval().sinh())
    }

    #[inline]
    fn cosh(self) -> Self {
        Self::from_interval(self.to_interval().cosh())
    }

    #[inline]
    fn tanh(self) -> Self {
        Self::from_interval(self.to_interval().tanh())
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        Self::from_interval(self.to_interval().hypot(other.to_interval()))
    }
}
