// [cite:math_scalars]
//! First-order zero-allocation forward-mode automatic differentiation number [`Dual`].

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::traits::{Scalar, powi_generic};

/// First-order dual number $a + b \varepsilon$ with $\varepsilon^2 = 0$.
///
/// Generic over any [`Scalar`] `T`, supporting both primitive floats (`Dual<f64>`, `Dual<f32>`)
/// and rigorous interval slope/derivative enclosures (`Dual<Interval<f64>>`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Dual<T> {
    /// Primal function value $f(x)$.
    pub val: T,
    /// First-order directional derivative $f'(x) \cdot \dot{x}$.
    pub eps: T,
}

impl<T: Scalar> Dual<T> {
    /// Constructs a dual number from explicit primal `val` and tangent `eps`.
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

    /// Constructs a seeded independent variable `(val, 1)`.
    #[inline]
    #[must_use]
    pub const fn variable(val: T) -> Self {
        Self { val, eps: T::ONE }
    }

    #[inline]
    fn chain(self, f0: T, df: T) -> Self {
        Self {
            val: f0,
            eps: self.eps * df,
        }
    }
}

impl<T: Scalar> Neg for Dual<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.val, -self.eps)
    }
}

impl<T: Scalar> Add for Dual<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.val + rhs.val, self.eps + rhs.eps)
    }
}

impl<T: Scalar> Sub for Dual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.val - rhs.val, self.eps - rhs.eps)
    }
}

impl<T: Scalar> Mul for Dual<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.val * rhs.val, self.eps * rhs.val + self.val * rhs.eps)
    }
}

impl<T: Scalar> Div for Dual<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = rhs.val.recip();
        Self::new(
            self.val * inv,
            (self.eps * rhs.val - self.val * rhs.eps) * (inv.sqr()),
        )
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
        self.val.is_finite_val() && self.eps.is_finite_val()
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
        Self::new(self.val.sqr(), T::TWO * self.val * self.eps)
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        self.chain(s, T::HALF / s)
    }
    #[inline]
    fn recip(self) -> Self {
        let r = self.val.recip();
        self.chain(r, -r.sqr())
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        powi_generic(self, n)
    }
    #[inline]
    fn powf(self, exp: Self) -> Self {
        if exp.eps == T::ZERO {
            let er = exp.val.real_part();
            if er.is_finite() && er >= f64::from(i32::MIN) && er <= f64::from(i32::MAX) {
                let n = er as i32;
                if exp.val == T::from_f64(f64::from(n)) {
                    return self.powi(n);
                }
            }
            let p = self.val.powf(exp.val);
            let dp = exp.val * self.val.powf(exp.val - T::ONE);
            self.chain(p, dp)
        } else {
            (exp * self.ln()).exp()
        }
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.val.exp();
        self.chain(e, e)
    }
    #[inline]
    fn ln(self) -> Self {
        self.chain(self.val.ln(), self.val.recip())
    }
    #[inline]
    fn log2(self) -> Self {
        self.chain(self.val.log2(), self.val.recip() * T::log2_e_scale())
    }
    #[inline]
    fn log10(self) -> Self {
        self.chain(self.val.log10(), self.val.recip() * T::log10_e_scale())
    }
    #[inline]
    fn sin(self) -> Self {
        let (s, c) = self.val.sin_cos();
        self.chain(s, c)
    }
    #[inline]
    fn cos(self) -> Self {
        let (s, c) = self.val.sin_cos();
        self.chain(c, -s)
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.val.sin_cos();
        (self.chain(s, c), self.chain(c, -s))
    }
    #[inline]
    fn tan(self) -> Self {
        let t = self.val.tan();
        self.chain(t, T::ONE + t.sqr())
    }
    #[inline]
    fn asin(self) -> Self {
        self.chain(self.val.asin(), (T::ONE - self.val.sqr()).sqrt().recip())
    }
    #[inline]
    fn acos(self) -> Self {
        self.chain(self.val.acos(), -(T::ONE - self.val.sqr()).sqrt().recip())
    }
    #[inline]
    fn atan(self) -> Self {
        self.chain(self.val.atan(), (T::ONE + self.val.sqr()).recip())
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        let r2 = self.val.sqr() + other.val.sqr();
        Self::new(
            self.val.atan2(other.val),
            (other.val * self.eps - self.val * other.eps) / r2,
        )
    }
    #[inline]
    fn sinh(self) -> Self {
        self.chain(self.val.sinh(), self.val.cosh())
    }
    #[inline]
    fn cosh(self) -> Self {
        self.chain(self.val.cosh(), self.val.sinh())
    }
    #[inline]
    fn tanh(self) -> Self {
        let th = self.val.tanh();
        self.chain(th, T::ONE - th.sqr())
    }
    #[inline]
    fn asinh(self) -> Self {
        self.chain(self.val.asinh(), (T::ONE + self.val.sqr()).sqrt().recip())
    }
    #[inline]
    fn acosh(self) -> Self {
        self.chain(self.val.acosh(), (self.val.sqr() - T::ONE).sqrt().recip())
    }
    #[inline]
    fn atanh(self) -> Self {
        self.chain(self.val.atanh(), (T::ONE - self.val.sqr()).recip())
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        let r = self.val.hypot(other.val);
        if r == T::ZERO {
            Self::constant(T::ZERO)
        } else {
            Self::new(r, (self.val * self.eps + other.val * other.eps) / r)
        }
    }
}
