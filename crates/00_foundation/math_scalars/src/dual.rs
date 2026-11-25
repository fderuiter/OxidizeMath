//! First-order scalar forward-mode automatic differentiation (`Dual<T>`).
// [cite:math_commons]

use crate::traits::Scalar;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// First-order forward-mode dual number $x = \text{val} + \text{eps}\,\varepsilon$ ($\varepsilon^2 = 0$).
///
/// Propagates exact directional derivatives on the stack with zero heap allocations.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Dual<T> {
    /// Primal function value $f(x)$.
    pub val: T,
    /// First-order tangent / directional derivative $f'(x)\,\dot{x}$.
    pub eps: T,
}

impl<T: Scalar> Dual<T> {
    /// Constructs a dual number with explicit primal `val` and tangent `eps`.
    #[inline]
    #[must_use]
    pub const fn new(val: T, eps: T) -> Self {
        Self { val, eps }
    }

    /// Constructs a constant dual number `(val, 0)`.
    #[inline]
    #[must_use]
    pub const fn constant(val: T) -> Self {
        Self { val, eps: T::ZERO }
    }

    /// Constructs an active independent variable `(val, 1)` seeded with unit tangent.
    #[inline]
    #[must_use]
    pub const fn variable(val: T) -> Self {
        Self { val, eps: T::ONE }
    }
}

impl<T: Scalar> Add for Dual<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.val + rhs.val, self.eps + rhs.eps)
    }
}

impl<T: Scalar> Add<T> for Dual<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: T) -> Self {
        Self::new(self.val + rhs, self.eps)
    }
}

impl<T: Scalar> Sub for Dual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.val - rhs.val, self.eps - rhs.eps)
    }
}

impl<T: Scalar> Sub<T> for Dual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: T) -> Self {
        Self::new(self.val - rhs, self.eps)
    }
}

impl<T: Scalar> Mul for Dual<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.val * rhs.val, self.eps * rhs.val + self.val * rhs.eps)
    }
}

impl<T: Scalar> Mul<T> for Dual<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self {
        Self::new(self.val * rhs, self.eps * rhs)
    }
}

impl<T: Scalar> Div for Dual<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = T::ONE / rhs.val;
        let val = self.val * inv;
        let eps = (self.eps - val * rhs.eps) * inv;
        Self::new(val, eps)
    }
}

impl<T: Scalar> Div<T> for Dual<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self {
        let inv = T::ONE / rhs;
        Self::new(self.val * inv, self.eps * inv)
    }
}

impl<T: Scalar> Neg for Dual<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.val, -self.eps)
    }
}

impl<T: Scalar> AddAssign for Dual<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: Scalar> SubAssign for Dual<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: Scalar> MulAssign for Dual<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: Scalar> DivAssign for Dual<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: Scalar> Scalar for Dual<T> {
    const ZERO: Self = Self::constant(T::ZERO);
    const ONE: Self = Self::constant(T::ONE);
    const TWO: Self = Self::constant(T::TWO);
    const HALF: Self = Self::constant(T::HALF);
    const PI: Self = Self::constant(T::PI);
    const TAU: Self = Self::constant(T::TAU);

    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::constant(T::from_f64(v))
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.val.is_finite() && self.eps.is_finite()
    }

    #[inline]
    fn abs(self) -> Self {
        Self::new(self.val.abs(), self.eps * self.val.signum())
    }

    #[inline]
    fn signum(self) -> Self {
        Self::constant(self.val.signum())
    }

    #[inline]
    fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        Self::new(s, self.eps / (T::TWO * s))
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        if n == 0 {
            Self::ONE
        } else if n == 1 {
            self
        } else {
            let p_m1 = self.val.powi(n - 1);
            let nf = T::from_f64(f64::from(n));
            Self::new(p_m1 * self.val, nf * p_m1 * self.eps)
        }
    }

    #[inline]
    fn powf(self, exp: Self) -> Self {
        if exp.eps == T::ZERO {
            let v = self.val.powf(exp.val);
            let dv = exp.val * self.val.powf(exp.val - T::ONE) * self.eps;
            Self::new(v, dv)
        } else {
            let v = self.val.powf(exp.val);
            let dv = v * (exp.eps * self.val.ln() + exp.val * self.eps / self.val);
            Self::new(v, dv)
        }
    }

    #[inline]
    fn exp(self) -> Self {
        let e = self.val.exp();
        Self::new(e, e * self.eps)
    }

    #[inline]
    fn ln(self) -> Self {
        Self::new(self.val.ln(), self.eps / self.val)
    }

    #[inline]
    fn sin(self) -> Self {
        let (s, c) = self.val.sin_cos();
        Self::new(s, c * self.eps)
    }

    #[inline]
    fn cos(self) -> Self {
        let (s, c) = self.val.sin_cos();
        Self::new(c, -s * self.eps)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.val.sin_cos();
        (Self::new(s, c * self.eps), Self::new(c, -s * self.eps))
    }

    #[inline]
    fn tan(self) -> Self {
        let t = self.val.tan();
        let sec2 = T::ONE + t * t;
        Self::new(t, sec2 * self.eps)
    }

    #[inline]
    fn asin(self) -> Self {
        let denom = (T::ONE - self.val * self.val).sqrt();
        Self::new(self.val.asin(), self.eps / denom)
    }

    #[inline]
    fn acos(self) -> Self {
        let denom = (T::ONE - self.val * self.val).sqrt();
        Self::new(self.val.acos(), -self.eps / denom)
    }

    #[inline]
    fn atan(self) -> Self {
        let denom = T::ONE + self.val * self.val;
        Self::new(self.val.atan(), self.eps / denom)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        let denom = self.val * self.val + other.val * other.val;
        let dv = (other.val * self.eps - self.val * other.eps) / denom;
        Self::new(self.val.atan2(other.val), dv)
    }

    #[inline]
    fn sinh(self) -> Self {
        Self::new(self.val.sinh(), self.val.cosh() * self.eps)
    }

    #[inline]
    fn cosh(self) -> Self {
        Self::new(self.val.cosh(), self.val.sinh() * self.eps)
    }

    #[inline]
    fn tanh(self) -> Self {
        let th = self.val.tanh();
        Self::new(th, (T::ONE - th * th) * self.eps)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        let r = self.val.hypot(other.val);
        if r == T::ZERO {
            Self::ZERO
        } else {
            let dr = (self.val / r) * self.eps + (other.val / r) * other.eps;
            Self::new(r, dr)
        }
    }
}
