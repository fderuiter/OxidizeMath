// [cite:math_scalars]
//! Second-order forward-mode automatic differentiation [`HyperDual`] number.
//!
//! Represents $f_0 + f_1 \varepsilon_1 + f_2 \varepsilon_2 + f_{12} \varepsilon_1 \varepsilon_2$
//! with $\varepsilon_1^2 = \varepsilon_2^2 = 0$ and $\varepsilon_1 \varepsilon_2 \ne 0$,
//! yielding exact pure and mixed second-order derivatives without subtractive cancellation.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::traits::{Scalar, powi_generic};

/// Second-order hyper-dual number $(f, \partial_1 f, \partial_2 f, \partial_{12} f)$.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HyperDual<T> {
    /// Primal value $f(x)$.
    pub val: T,
    /// First-order derivative along direction 1.
    pub eps1: T,
    /// First-order derivative along direction 2.
    pub eps2: T,
    /// Second-order mixed/directional derivative along directions 1 and 2.
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

    /// Constructs a constant hyper-dual number `(val, 0, 0, 0)`.
    #[inline]
    #[must_use]
    pub const fn constant(val: T) -> Self {
        Self::new(val, T::ZERO, T::ZERO, T::ZERO)
    }

    /// Seeds a univariate variable `(val, 1, 1, 0)` so `eps1 == eps2 == f'(x)` and `eps12 == f''(x)`.
    #[inline]
    #[must_use]
    pub const fn second_order(val: T) -> Self {
        Self::new(val, T::ONE, T::ONE, T::ZERO)
    }

    /// Seeds direction 1 `(val, 1, 0, 0)` for mixed partial $\partial^2 f / \partial u \partial v$.
    #[inline]
    #[must_use]
    pub const fn var1(val: T) -> Self {
        Self::new(val, T::ONE, T::ZERO, T::ZERO)
    }

    /// Seeds direction 2 `(val, 0, 1, 0)` for mixed partial $\partial^2 f / \partial u \partial v$.
    #[inline]
    #[must_use]
    pub const fn var2(val: T) -> Self {
        Self::new(val, T::ZERO, T::ONE, T::ZERO)
    }

    #[inline]
    fn mul_hyper(self, rhs: Self) -> Self {
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

    #[inline]
    fn chain_unary(self, f0: T, df: T, d2f: T) -> Self {
        Self {
            val: f0,
            eps1: df * self.eps1,
            eps2: df * self.eps2,
            eps12: df * self.eps12 + d2f * (self.eps1 * self.eps2),
        }
    }
}

impl<T: Scalar> Neg for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.val, -self.eps1, -self.eps2, -self.eps12)
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

impl<T: Scalar> Mul for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_hyper(rhs)
    }
}

impl<T: Scalar> Div for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self.mul_hyper(rhs.recip())
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

    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::constant(T::from_f64(v))
    }
    #[inline]
    fn log2_e_scale() -> Self {
        Self::constant(T::log2_e_scale())
    }
    #[inline]
    fn log10_e_scale() -> Self {
        Self::constant(T::log10_e_scale())
    }
    #[inline]
    fn real_part(self) -> f64 {
        self.val.real_part()
    }
    #[inline]
    fn is_finite_val(self) -> bool {
        self.val.is_finite_val()
            && self.eps1.is_finite_val()
            && self.eps2.is_finite_val()
            && self.eps12.is_finite_val()
    }
    #[inline]
    fn abs(self) -> Self {
        if self.val == T::ZERO {
            Self::constant(T::ZERO)
        } else if self.val.real_part() < 0.0 {
            -self
        } else {
            self
        }
    }
    #[inline]
    fn sqr(self) -> Self {
        self.chain_unary(self.val.sqr(), T::TWO * self.val, T::TWO)
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        let df = T::HALF / s;
        let d2f = -(T::HALF * T::HALF) / (s * self.val);
        self.chain_unary(s, df, d2f)
    }
    #[inline]
    fn recip(self) -> Self {
        let r = self.val.recip();
        let r2 = r.sqr();
        self.chain_unary(r, -r2, T::TWO * r2 * r)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        powi_generic(self, n)
    }
    #[inline]
    fn powf(self, exp: Self) -> Self {
        if exp.eps1 == T::ZERO && exp.eps2 == T::ZERO && exp.eps12 == T::ZERO {
            let e = exp.val;
            let er = e.real_part();
            if er.is_finite() && er >= f64::from(i32::MIN) && er <= f64::from(i32::MAX) {
                let n = er as i32;
                if e == T::from_f64(f64::from(n)) {
                    return self.powi(n);
                }
            }
            let p = self.val.powf(e);
            let df = e * self.val.powf(e - T::ONE);
            let d2f = e * (e - T::ONE) * self.val.powf(e - T::TWO);
            self.chain_unary(p, df, d2f)
        } else {
            (exp * self.ln()).exp()
        }
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.val.exp();
        self.chain_unary(e, e, e)
    }
    #[inline]
    fn ln(self) -> Self {
        let inv = self.val.recip();
        self.chain_unary(self.val.ln(), inv, -inv.sqr())
    }
    #[inline]
    fn log2(self) -> Self {
        let scale = T::log2_e_scale();
        let inv = self.val.recip();
        self.chain_unary(self.val.log2(), inv * scale, -inv.sqr() * scale)
    }
    #[inline]
    fn log10(self) -> Self {
        let scale = T::log10_e_scale();
        let inv = self.val.recip();
        self.chain_unary(self.val.log10(), inv * scale, -inv.sqr() * scale)
    }
    #[inline]
    fn sin(self) -> Self {
        let (s, c) = self.val.sin_cos();
        self.chain_unary(s, c, -s)
    }
    #[inline]
    fn cos(self) -> Self {
        let (s, c) = self.val.sin_cos();
        self.chain_unary(c, -s, -c)
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.val.sin_cos();
        (self.chain_unary(s, c, -s), self.chain_unary(c, -s, -c))
    }
    #[inline]
    fn tan(self) -> Self {
        let t = self.val.tan();
        let sec2 = T::ONE + t.sqr();
        self.chain_unary(t, sec2, T::TWO * t * sec2)
    }
    #[inline]
    fn asin(self) -> Self {
        let omx2 = T::ONE - self.val.sqr();
        let inv_s = omx2.sqrt().recip();
        self.chain_unary(self.val.asin(), inv_s, self.val * inv_s / omx2)
    }
    #[inline]
    fn acos(self) -> Self {
        let omx2 = T::ONE - self.val.sqr();
        let inv_s = omx2.sqrt().recip();
        self.chain_unary(self.val.acos(), -inv_s, -(self.val * inv_s / omx2))
    }
    #[inline]
    fn atan(self) -> Self {
        let inv = (T::ONE + self.val.sqr()).recip();
        self.chain_unary(self.val.atan(), inv, -(T::TWO * self.val * inv.sqr()))
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        let (y0, x0) = (self.val, other.val);
        let inv_r2 = (x0.sqr() + y0.sqr()).recip();
        let inv_r4 = inv_r2.sqr();
        let eps1 = (x0 * self.eps1 - y0 * other.eps1) * inv_r2;
        let eps2 = (x0 * self.eps2 - y0 * other.eps2) * inv_r2;
        let sym = (y0.sqr() - x0.sqr()) * (other.eps1 * self.eps2 + other.eps2 * self.eps1)
            + T::TWO * x0 * y0 * (other.eps1 * other.eps2 - self.eps1 * self.eps2);
        let eps12 = (x0 * self.eps12 - y0 * other.eps12) * inv_r2 + sym * inv_r4;
        Self::new(y0.atan2(x0), eps1, eps2, eps12)
    }
    #[inline]
    fn sinh(self) -> Self {
        let (sh, ch) = (self.val.sinh(), self.val.cosh());
        self.chain_unary(sh, ch, sh)
    }
    #[inline]
    fn cosh(self) -> Self {
        let (sh, ch) = (self.val.sinh(), self.val.cosh());
        self.chain_unary(ch, sh, ch)
    }
    #[inline]
    fn tanh(self) -> Self {
        let th = self.val.tanh();
        let sech2 = T::ONE - th.sqr();
        self.chain_unary(th, sech2, -(T::TWO * th * sech2))
    }
    #[inline]
    fn asinh(self) -> Self {
        let s2 = T::ONE + self.val.sqr();
        let inv_s = s2.sqrt().recip();
        self.chain_unary(self.val.asinh(), inv_s, -(self.val * inv_s / s2))
    }
    #[inline]
    fn acosh(self) -> Self {
        let s2 = self.val.sqr() - T::ONE;
        let inv_s = s2.sqrt().recip();
        self.chain_unary(self.val.acosh(), inv_s, -(self.val * inv_s / s2))
    }
    #[inline]
    fn atanh(self) -> Self {
        let inv = (T::ONE - self.val.sqr()).recip();
        self.chain_unary(self.val.atanh(), inv, T::TWO * self.val * inv.sqr())
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        let r = self.val.hypot(other.val);
        if r == T::ZERO {
            return Self::constant(T::ZERO);
        }
        let inv_r = r.recip();
        let eps1 = (self.val * self.eps1 + other.val * other.eps1) * inv_r;
        let eps2 = (self.val * self.eps2 + other.val * other.eps2) * inv_r;
        let dot12 = self.eps1 * self.eps2
            + self.val * self.eps12
            + other.eps1 * other.eps2
            + other.val * other.eps12;
        let eps12 = (dot12 - eps1 * eps2) * inv_r;
        Self::new(r, eps1, eps2, eps12)
    }
}
