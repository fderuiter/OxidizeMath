//! Second-order forward-mode automatic differentiation (`HyperDual<T>`).
//!
//! Represents $x = \text{val} + \text{eps1}\,\varepsilon_1 + \text{eps2}\,\varepsilon_2 + \text{eps12}\,\varepsilon_1\varepsilon_2$
//! with $\varepsilon_1^2 = \varepsilon_2^2 = 0$, propagating exact first and second partial
//! derivatives without subtractive cancellation.
// [cite:math_commons]

use crate::dual::Dual;
use crate::traits::Scalar;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Second-order hyper-dual number propagating exact first and mixed/pure second derivatives.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct HyperDual<T> {
    /// Primal value $f(u, v)$.
    pub val: T,
    /// First partial along $\varepsilon_1$ ($\partial f / \partial u$).
    pub eps1: T,
    /// First partial along $\varepsilon_2$ ($\partial f / \partial v$).
    pub eps2: T,
    /// Second partial along $\varepsilon_1 \varepsilon_2$ ($\partial^2 f / \partial u \partial v$).
    pub eps12: T,
}

impl<T: Scalar> HyperDual<T> {
    /// Constructs a hyper-dual number from explicit components.
    #[inline]
    #[must_use]
    pub const fn new(val: T, eps1: T, eps2: T, eps12: T) -> Self {
        Self {
            val,
            eps1,
            eps2,
            eps12,
        }
    }

    /// Constructs a constant hyper-dual `(val, 0, 0, 0)`.
    #[inline]
    #[must_use]
    pub const fn constant(val: T) -> Self {
        Self::new(val, T::ZERO, T::ZERO, T::ZERO)
    }

    /// Seeds a univariate variable `(val, 1, 1, 0)` so `eps1` is $f'(x)$ and `eps12` is $f''(x)$.
    #[inline]
    #[must_use]
    pub const fn univariate(val: T) -> Self {
        Self::new(val, T::ONE, T::ONE, T::ZERO)
    }

    /// Seeds the first bivariate coordinate $u$ as `(val, 1, 0, 0)` for mixed partial $\partial^2 f / \partial u \partial v$.
    #[inline]
    #[must_use]
    pub const fn bivariate_u(val: T) -> Self {
        Self::new(val, T::ONE, T::ZERO, T::ZERO)
    }

    /// Seeds the second bivariate coordinate $v$ as `(val, 0, 1, 0)` for mixed partial $\partial^2 f / \partial u \partial v$.
    #[inline]
    #[must_use]
    pub const fn bivariate_v(val: T) -> Self {
        Self::new(val, T::ZERO, T::ONE, T::ZERO)
    }

    #[inline]
    fn unary_chain(self, f0: T, df: T, d2f: T) -> Self {
        Self::new(
            f0,
            df * self.eps1,
            df * self.eps2,
            df * self.eps12 + d2f * self.eps1 * self.eps2,
        )
    }

    #[inline]
    fn to_nested_dual(self) -> Dual<Dual<T>> {
        Dual::new(
            Dual::new(self.val, self.eps2),
            Dual::new(self.eps1, self.eps12),
        )
    }

    #[inline]
    fn from_nested_dual(d: Dual<Dual<T>>) -> Self {
        Self::new(d.val.val, d.eps.val, d.val.eps, d.eps.eps)
    }
}

impl<T: Scalar> Add for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(
            self.val + rhs.val,
            self.eps1 + rhs.eps1,
            self.eps2 + rhs.eps2,
            self.eps12 + rhs.eps12,
        )
    }
}

impl<T: Scalar> Add<T> for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: T) -> Self {
        Self::new(self.val + rhs, self.eps1, self.eps2, self.eps12)
    }
}

impl<T: Scalar> Sub for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(
            self.val - rhs.val,
            self.eps1 - rhs.eps1,
            self.eps2 - rhs.eps2,
            self.eps12 - rhs.eps12,
        )
    }
}

impl<T: Scalar> Sub<T> for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: T) -> Self {
        Self::new(self.val - rhs, self.eps1, self.eps2, self.eps12)
    }
}

impl<T: Scalar> Mul for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.val * rhs.val,
            self.eps1 * rhs.val + self.val * rhs.eps1,
            self.eps2 * rhs.val + self.val * rhs.eps2,
            self.eps12 * rhs.val
                + self.eps1 * rhs.eps2
                + self.eps2 * rhs.eps1
                + self.val * rhs.eps12,
        )
    }
}

impl<T: Scalar> Mul<T> for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self {
        Self::new(
            self.val * rhs,
            self.eps1 * rhs,
            self.eps2 * rhs,
            self.eps12 * rhs,
        )
    }
}

impl<T: Scalar> Div for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = T::ONE / rhs.val;
        let inv2 = inv * inv;
        let rhs_recip = rhs.unary_chain(inv, -inv2, T::TWO * inv2 * inv);
        self * rhs_recip
    }
}

impl<T: Scalar> Div<T> for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self {
        self * (T::ONE / rhs)
    }
}

impl<T: Scalar> Neg for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.val, -self.eps1, -self.eps2, -self.eps12)
    }
}

impl<T: Scalar> AddAssign for HyperDual<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: Scalar> SubAssign for HyperDual<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: Scalar> MulAssign for HyperDual<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: Scalar> DivAssign for HyperDual<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: Scalar> Scalar for HyperDual<T> {
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
        self.val.is_finite()
            && self.eps1.is_finite()
            && self.eps2.is_finite()
            && self.eps12.is_finite()
    }

    #[inline]
    fn abs(self) -> Self {
        self.unary_chain(self.val.abs(), self.val.signum(), T::ZERO)
    }

    #[inline]
    fn signum(self) -> Self {
        Self::constant(self.val.signum())
    }

    #[inline]
    fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        let df = T::HALF / s;
        let d2f = -df / (T::TWO * self.val);
        self.unary_chain(s, df, d2f)
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        match n {
            0 => Self::ONE,
            1 => self,
            2 => self * self,
            _ => {
                let p_m2 = self.val.powi(n - 2);
                let p_m1 = p_m2 * self.val;
                let nf = T::from_f64(f64::from(n));
                let nm1 = T::from_f64(f64::from(n - 1));
                self.unary_chain(p_m1 * self.val, nf * p_m1, nf * nm1 * p_m2)
            }
        }
    }

    #[inline]
    fn powf(self, exp: Self) -> Self {
        Self::from_nested_dual(self.to_nested_dual().powf(exp.to_nested_dual()))
    }

    #[inline]
    fn exp(self) -> Self {
        let e = self.val.exp();
        self.unary_chain(e, e, e)
    }

    #[inline]
    fn ln(self) -> Self {
        let inv = T::ONE / self.val;
        self.unary_chain(self.val.ln(), inv, -inv * inv)
    }

    #[inline]
    fn sin(self) -> Self {
        let (s, c) = self.val.sin_cos();
        self.unary_chain(s, c, -s)
    }

    #[inline]
    fn cos(self) -> Self {
        let (s, c) = self.val.sin_cos();
        self.unary_chain(c, -s, -c)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.val.sin_cos();
        (self.unary_chain(s, c, -s), self.unary_chain(c, -s, -c))
    }

    #[inline]
    fn tan(self) -> Self {
        let t = self.val.tan();
        let sec2 = T::ONE + t * t;
        self.unary_chain(t, sec2, T::TWO * t * sec2)
    }

    #[inline]
    fn asin(self) -> Self {
        let om = T::ONE - self.val * self.val;
        let inv_s = T::ONE / om.sqrt();
        self.unary_chain(self.val.asin(), inv_s, self.val * inv_s / om)
    }

    #[inline]
    fn acos(self) -> Self {
        let om = T::ONE - self.val * self.val;
        let inv_s = T::ONE / om.sqrt();
        self.unary_chain(self.val.acos(), -inv_s, -self.val * inv_s / om)
    }

    #[inline]
    fn atan(self) -> Self {
        let inv = T::ONE / (T::ONE + self.val * self.val);
        self.unary_chain(self.val.atan(), inv, -T::TWO * self.val * inv * inv)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        Self::from_nested_dual(self.to_nested_dual().atan2(other.to_nested_dual()))
    }

    #[inline]
    fn sinh(self) -> Self {
        let sh = self.val.sinh();
        let ch = self.val.cosh();
        self.unary_chain(sh, ch, sh)
    }

    #[inline]
    fn cosh(self) -> Self {
        let sh = self.val.sinh();
        let ch = self.val.cosh();
        self.unary_chain(ch, sh, ch)
    }

    #[inline]
    fn tanh(self) -> Self {
        let th = self.val.tanh();
        let sech2 = T::ONE - th * th;
        self.unary_chain(th, sech2, -T::TWO * th * sech2)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        Self::from_nested_dual(self.to_nested_dual().hypot(other.to_nested_dual()))
    }
}
