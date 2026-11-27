// [cite:math_scalars]
//! Forward-mode dual number $a + b\varepsilon$ ($\varepsilon^2 = 0$) for exact automatic differentiation.

use crate::ieee::powi_field;
use crate::traits::{ComplexField, Field, RealField, RefOps, Ring};
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Forward-mode dual number $a + b\varepsilon$ satisfying $\varepsilon^2 = 0$.
///
/// Evaluating any generic [`ComplexField`] expression at `Dual::variable(x)` propagates
/// both the primal value $f(x)$ in `re` and the exact first derivative $f'(x)$ in `eps`
/// with zero truncation error and zero heap allocations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Dual<T = f64> {
    /// Primal value $f(x)$.
    pub re: T,
    /// Infinitesimal tangent / directional derivative $f'(x)$.
    pub eps: T,
}

impl<T> Dual<T> {
    /// Constructs a dual number `re + eps * ε`.
    #[inline]
    pub const fn new(re: T, eps: T) -> Self {
        Self { re, eps }
    }
}

impl<T: Ring> Dual<T> {
    /// Embeds a constant scalar `re + 0 * ε` with zero derivative.
    #[inline]
    pub fn constant(re: T) -> Self {
        Self { re, eps: T::zero() }
    }

    /// Seeds an active differentiation variable `re + 1 * ε`.
    #[inline]
    pub fn variable(re: T) -> Self {
        Self { re, eps: T::one() }
    }
}

impl<T: fmt::Display> fmt::Display for Dual<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}ε", self.re, self.eps)
    }
}

impl<T: Ring> Add for Dual<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.eps + rhs.eps)
    }
}

impl<T: Ring> Add<&Dual<T>> for &Dual<T> {
    type Output = Dual<T>;
    #[inline]
    fn add(self, rhs: &Dual<T>) -> Dual<T> {
        Dual::new(self.re.ref_add(&rhs.re), self.eps.ref_add(&rhs.eps))
    }
}

impl<T: Ring> Add<&Dual<T>> for Dual<T> {
    type Output = Dual<T>;
    #[inline]
    fn add(self, rhs: &Dual<T>) -> Dual<T> {
        Dual::new(self.re.ref_add(&rhs.re), self.eps.ref_add(&rhs.eps))
    }
}

impl<T: Ring> Add<Dual<T>> for &Dual<T> {
    type Output = Dual<T>;
    #[inline]
    fn add(self, rhs: Dual<T>) -> Dual<T> {
        Dual::new(self.re.ref_add(&rhs.re), self.eps.ref_add(&rhs.eps))
    }
}

impl<T: Ring> Sub for Dual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.eps - rhs.eps)
    }
}

impl<T: Ring> Sub<&Dual<T>> for &Dual<T> {
    type Output = Dual<T>;
    #[inline]
    fn sub(self, rhs: &Dual<T>) -> Dual<T> {
        Dual::new(self.re.ref_sub(&rhs.re), self.eps.ref_sub(&rhs.eps))
    }
}

impl<T: Ring> Sub<&Dual<T>> for Dual<T> {
    type Output = Dual<T>;
    #[inline]
    fn sub(self, rhs: &Dual<T>) -> Dual<T> {
        Dual::new(self.re.ref_sub(&rhs.re), self.eps.ref_sub(&rhs.eps))
    }
}

impl<T: Ring> Sub<Dual<T>> for &Dual<T> {
    type Output = Dual<T>;
    #[inline]
    fn sub(self, rhs: Dual<T>) -> Dual<T> {
        Dual::new(self.re.ref_sub(&rhs.re), self.eps.ref_sub(&rhs.eps))
    }
}

impl<T: Ring> Mul for Dual<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.ref_mul(&rhs)
    }
}

impl<T: Ring> Mul<&Dual<T>> for &Dual<T> {
    type Output = Dual<T>;
    #[inline]
    fn mul(self, rhs: &Dual<T>) -> Dual<T> {
        self.ref_mul(rhs)
    }
}

impl<T: Ring> Mul<&Dual<T>> for Dual<T> {
    type Output = Dual<T>;
    #[inline]
    fn mul(self, rhs: &Dual<T>) -> Dual<T> {
        self.ref_mul(rhs)
    }
}

impl<T: Ring> Mul<Dual<T>> for &Dual<T> {
    type Output = Dual<T>;
    #[inline]
    fn mul(self, rhs: Dual<T>) -> Dual<T> {
        self.ref_mul(&rhs)
    }
}

impl<T: Field> Div for Dual<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self.ref_div(&rhs)
    }
}

impl<T: Field> Div<&Dual<T>> for &Dual<T> {
    type Output = Dual<T>;
    #[inline]
    fn div(self, rhs: &Dual<T>) -> Dual<T> {
        self.ref_div(rhs)
    }
}

impl<T: Ring> Neg for Dual<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.eps)
    }
}

impl<T: Ring> Neg for &Dual<T> {
    type Output = Dual<T>;
    #[inline]
    fn neg(self) -> Dual<T> {
        self.ref_neg()
    }
}

impl<T: Ring> AddAssign for Dual<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.ref_add(&rhs);
    }
}

impl<T: Ring> SubAssign for Dual<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.ref_sub(&rhs);
    }
}

impl<T: Ring> MulAssign for Dual<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.ref_mul(&rhs);
    }
}

impl<T: Field> DivAssign for Dual<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = self.ref_div(&rhs);
    }
}

impl<T: Ring> Ring for Dual<T> {
    #[inline]
    fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }
    #[inline]
    fn one() -> Self {
        Self::new(T::one(), T::zero())
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.eps.is_zero()
    }
    #[inline]
    fn ring_ref_add(&self, rhs: &Self) -> Self {
        Self::new(self.re.ref_add(&rhs.re), self.eps.ref_add(&rhs.eps))
    }
    #[inline]
    fn ring_ref_sub(&self, rhs: &Self) -> Self {
        Self::new(self.re.ref_sub(&rhs.re), self.eps.ref_sub(&rhs.eps))
    }
    #[inline]
    fn ring_ref_mul(&self, rhs: &Self) -> Self {
        let p = self.re.ref_mul(&rhs.re);
        let d = self.re.ref_mul(&rhs.eps) + self.eps.ref_mul(&rhs.re);
        Self::new(p, d)
    }
    #[inline]
    fn ring_ref_neg(&self) -> Self {
        Self::new(self.re.ref_neg(), self.eps.ref_neg())
    }
}

impl<T: Field> Field for Dual<T> {
    #[inline]
    fn ref_div(&self, rhs: &Self) -> Self {
        let q = self.re.ref_div(&rhs.re);
        let num = self.eps.ref_mul(&rhs.re) - self.re.ref_mul(&rhs.eps);
        let den = rhs.re.ref_mul(&rhs.re);
        Self::new(q, num.ref_div(&den))
    }
    #[inline]
    fn try_inv(&self) -> Option<Self> {
        let inv_re = self.re.try_inv()?;
        let neg_eps = self.eps.ref_neg();
        let inv_sq = inv_re.ref_mul(&inv_re);
        Some(Self::new(inv_re, neg_eps.ref_mul(&inv_sq)))
    }
}

impl<T: RealField> ComplexField for Dual<T> {
    type Real = T;

    #[inline]
    fn from_real(re: T) -> Self {
        Self::new(re, T::ZERO)
    }
    #[inline]
    fn imag_unit() -> Self {
        Self::nan()
    }
    #[inline]
    fn nan() -> Self {
        Self::new(T::NAN, T::NAN)
    }
    #[inline]
    fn real(&self) -> T {
        self.re
    }
    #[inline]
    fn imag(&self) -> T {
        T::ZERO
    }
    #[inline]
    fn conj(&self) -> Self {
        *self
    }
    #[inline]
    fn norm_sqr(&self) -> T {
        self.re * self.re
    }
    #[inline]
    fn modulus(&self) -> T {
        self.re.abs()
    }
    #[inline]
    fn scale_real(&self, factor: T) -> Self {
        Self::new(self.re * factor, self.eps * factor)
    }
    #[inline]
    fn is_finite(&self) -> bool {
        self.re.is_finite() && self.eps.is_finite()
    }
    #[inline]
    fn is_nan(&self) -> bool {
        self.re.is_nan() || self.eps.is_nan()
    }
    #[inline]
    fn abs(self) -> Self {
        let sign = if self.re > T::ZERO {
            T::ONE
        } else if self.re < T::ZERO {
            -T::ONE
        } else {
            T::ZERO
        };
        Self::new(self.re.abs(), self.eps * sign)
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = self.re.sqrt();
        Self::new(s, self.eps / (T::TWO * s))
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.re.exp();
        Self::new(e, self.eps * e)
    }
    #[inline]
    fn ln(self) -> Self {
        Self::new(self.re.ln(), self.eps / self.re)
    }
    #[inline]
    fn log10(self) -> Self {
        let ln10 = T::from_f64(core::f64::consts::LN_10);
        Self::new(self.re.log10(), self.eps / (self.re * ln10))
    }
    #[inline]
    fn sin(self) -> Self {
        let (s, c) = self.re.sin_cos();
        Self::new(s, self.eps * c)
    }
    #[inline]
    fn cos(self) -> Self {
        let (s, c) = self.re.sin_cos();
        Self::new(c, -self.eps * s)
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos();
        (Self::new(s, self.eps * c), Self::new(c, -self.eps * s))
    }
    #[inline]
    fn tan(self) -> Self {
        let (s, c) = self.re.sin_cos();
        Self::new(s / c, self.eps / (c * c))
    }
    #[inline]
    fn sinh(self) -> Self {
        Self::new(self.re.sinh(), self.eps * self.re.cosh())
    }
    #[inline]
    fn cosh(self) -> Self {
        Self::new(self.re.cosh(), self.eps * self.re.sinh())
    }
    #[inline]
    fn tanh(self) -> Self {
        let ch = self.re.cosh();
        Self::new(self.re.tanh(), self.eps / (ch * ch))
    }
    #[inline]
    fn asin(self) -> Self {
        let d = (T::ONE - self.re * self.re).sqrt();
        Self::new(self.re.asin(), self.eps / d)
    }
    #[inline]
    fn acos(self) -> Self {
        let d = (T::ONE - self.re * self.re).sqrt();
        Self::new(self.re.acos(), -self.eps / d)
    }
    #[inline]
    fn atan(self) -> Self {
        let d = T::ONE + self.re * self.re;
        Self::new(self.re.atan(), self.eps / d)
    }
    #[inline]
    fn powf(self, exp: T) -> Self {
        if exp == T::ZERO {
            return Self::one();
        }
        let p = self.re.powf(exp);
        let dp = exp * self.re.powf(exp - T::ONE) * self.eps;
        Self::new(p, dp)
    }
    #[inline]
    fn powc(self, exp: Self) -> Self {
        if exp.eps == T::ZERO {
            return ComplexField::powf(self, exp.re);
        }
        let p = self.re.powf(exp.re);
        let dp = p * (exp.eps * self.re.ln() + exp.re * self.eps / self.re);
        Self::new(p, dp)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        powi_field(self, n)
    }
}
