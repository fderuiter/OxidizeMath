// [cite:math_commons]
use crate::traits::{ComplexField, Field, RealField, Ring, Scalar};
use core::cmp::Ordering;
use core::fmt;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// First-order forward-mode automatic differentiation dual number $a + b\varepsilon$ ($\varepsilon^2 = 0$).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Dual<T> {
    /// Primal real value $a = f(x)$.
    pub re: T,
    /// First-order tangent derivative $b = f'(x)$.
    pub eps: T,
}

impl<T: RealField> Dual<T> {
    /// Constructs a dual number $a + b\varepsilon$ from primal `re` and tangent `eps`.
    #[inline(always)]
    pub const fn new(re: T, eps: T) -> Self {
        Self { re, eps }
    }

    /// Embeds a constant real value $c + 0\varepsilon$.
    #[inline(always)]
    pub const fn constant(re: T) -> Self {
        Self { re, eps: T::ZERO }
    }

    /// Seeds an active differentiation variable $x + 1\varepsilon$.
    #[inline(always)]
    pub const fn variable(re: T) -> Self {
        Self { re, eps: T::ONE }
    }
}

impl<T: fmt::Display> fmt::Display for Dual<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}ε", self.re, self.eps)
    }
}

impl<T: PartialOrd> PartialOrd for Dual<T> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.re.partial_cmp(&other.re)
    }
}

macro_rules! impl_dual_linear_op {
    ($tr:ident, $m:ident, $op:tt) => {
        impl<T: RealField> $tr for Dual<T> {
            type Output = Self;
            #[inline(always)]
            fn $m(self, rhs: Self) -> Self {
                Self::new(self.re $op rhs.re, self.eps $op rhs.eps)
            }
        }
    };
}

impl_dual_linear_op!(Add, add, +);
impl_dual_linear_op!(Sub, sub, -);

impl<T: RealField> Mul for Dual<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.re * rhs.re, self.re * rhs.eps + self.eps * rhs.re)
    }
}

impl<T: RealField> Div for Dual<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let inv = T::ONE / rhs.re;
        let q = self.re * inv;
        Self::new(q, (self.eps - q * rhs.eps) * inv)
    }
}

impl<T: RealField> Rem for Dual<T> {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        let q = self.re / rhs.re;
        let k = if q >= T::ZERO { q.floor() } else { q.ceil() };
        Self::new(self.re % rhs.re, self.eps - k * rhs.eps)
    }
}

impl<T: RealField> Neg for Dual<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.eps)
    }
}

macro_rules! impl_dual_assign {
    ($tr:ident, $m:ident, $op:tt) => {
        impl<T: RealField> $tr for Dual<T> {
            #[inline(always)]
            fn $m(&mut self, rhs: Self) {
                *self = *self $op rhs;
            }
        }
    };
}

impl_dual_assign!(AddAssign, add_assign, +);
impl_dual_assign!(SubAssign, sub_assign, -);
impl_dual_assign!(MulAssign, mul_assign, *);
impl_dual_assign!(DivAssign, div_assign, /);
impl_dual_assign!(RemAssign, rem_assign, %);

impl<T: RealField> num_traits::Zero for Dual<T> {
    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.re == T::ZERO && self.eps == T::ZERO
    }
}

impl<T: RealField> num_traits::One for Dual<T> {
    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }
}

impl<T: RealField> num_traits::Num for Dual<T> {
    type FromStrRadixErr = T::FromStrRadixErr;
    #[inline]
    fn from_str_radix(str_val: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(str_val, radix).map(Self::constant)
    }
}

impl<T: Scalar> Scalar for Dual<T> {}

impl<T: RealField> Ring for Dual<T> {
    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }
    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }
}

impl<T: RealField> Field for Dual<T> {
    #[inline]
    fn checked_inverse(&self) -> Option<Self> {
        let inv_re = self.re.checked_inverse()?;
        Some(Self::new(inv_re, -self.eps * inv_re * inv_re))
    }
    #[inline(always)]
    fn multiplicative_inverse(&self) -> Self {
        Self::ONE / *self
    }
}

impl<T: RealField> ComplexField for Dual<T> {
    type Real = Self;

    #[inline(always)]
    fn from_real(re: Self) -> Self {
        re
    }
    #[inline(always)]
    fn nan() -> Self {
        Self::NAN
    }
    #[inline(always)]
    fn imag_unit() -> Self {
        Self::NAN
    }
    #[inline(always)]
    fn real(self) -> Self {
        self
    }
    #[inline(always)]
    fn imag(self) -> Self {
        Self::ZERO
    }
    #[inline(always)]
    fn conj(self) -> Self {
        self
    }
    #[inline(always)]
    fn norm_sqr(self) -> Self {
        self * self
    }
    #[inline(always)]
    fn abs(self) -> Self {
        if self.re > T::ZERO {
            self
        } else if self.re < T::ZERO {
            -self
        } else {
            Self::new(<T as ComplexField>::abs(self.re), T::ZERO)
        }
    }
    #[inline(always)]
    fn arg(self) -> Self {
        Self::constant(self.re.arg())
    }
    #[inline(always)]
    fn scale(self, factor: Self) -> Self {
        self * factor
    }
    #[inline(always)]
    fn is_all_finite(self) -> bool {
        self.re.is_finite() && self.eps.is_finite()
    }
    #[inline(always)]
    fn is_any_nan(self) -> bool {
        self.re.is_nan() || self.eps.is_nan()
    }
    #[inline(always)]
    fn csqrt(self) -> Self {
        RealField::sqrt(self)
    }
    #[inline]
    fn cbrt(self) -> Self {
        let c = self.re.cbrt();
        Self::new(c, self.eps / (T::from_f64(3.0) * c * c))
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
    fn log2(self) -> Self {
        Self::new(self.re.log2(), self.eps / (self.re * T::TWO.ln()))
    }
    #[inline]
    fn log10(self) -> Self {
        Self::new(
            self.re.log10(),
            self.eps / (self.re * T::from_f64(10.0).ln()),
        )
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
        let t = self.re.tan();
        Self::new(t, self.eps * (T::ONE + t * t))
    }
    #[inline]
    fn asin(self) -> Self {
        Self::new(
            self.re.asin(),
            self.eps / (T::ONE - self.re * self.re).sqrt(),
        )
    }
    #[inline]
    fn acos(self) -> Self {
        Self::new(
            self.re.acos(),
            -self.eps / (T::ONE - self.re * self.re).sqrt(),
        )
    }
    #[inline]
    fn atan(self) -> Self {
        Self::new(self.re.atan(), self.eps / (T::ONE + self.re * self.re))
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
        let th = self.re.tanh();
        Self::new(th, self.eps * (T::ONE - th * th))
    }
    #[inline]
    fn asinh(self) -> Self {
        Self::new(self.re.asinh(), self.eps / self.re.hypot(T::ONE))
    }
    #[inline]
    fn acosh(self) -> Self {
        Self::new(
            self.re.acosh(),
            self.eps / ((self.re - T::ONE).sqrt() * (self.re + T::ONE).sqrt()),
        )
    }
    #[inline]
    fn atanh(self) -> Self {
        Self::new(self.re.atanh(), self.eps / (T::ONE - self.re * self.re))
    }
    #[inline]
    fn powf(self, exp: Self) -> Self {
        let val = self.re.powf(exp.re);
        let d_base = if self.eps == T::ZERO {
            T::ZERO
        } else {
            exp.re * self.re.powf(exp.re - T::ONE) * self.eps
        };
        let d_exp = if exp.eps == T::ZERO {
            T::ZERO
        } else {
            val * self.re.ln() * exp.eps
        };
        Self::new(val, d_base + d_exp)
    }
    #[inline]
    fn powc(self, exp: Self) -> Self {
        self.powf(exp)
    }
    #[inline]
    fn powi(self, exp: i32) -> Self {
        if exp == 0 {
            return Self::ONE;
        }
        Self::new(
            self.re.powi(exp),
            T::from_i32(exp) * self.re.powi(exp - 1) * self.eps,
        )
    }
    #[inline]
    fn floor(self) -> Self {
        Self::constant(self.re.floor())
    }
    #[inline]
    fn ceil(self) -> Self {
        Self::constant(self.re.ceil())
    }
    #[inline]
    fn round(self) -> Self {
        Self::constant(self.re.round())
    }
    #[inline]
    fn signum(self) -> Self {
        Self::constant(self.re.signum())
    }
}

impl<T: RealField> RealField for Dual<T> {
    const ZERO: Self = Self::constant(T::ZERO);
    const ONE: Self = Self::constant(T::ONE);
    const TWO: Self = Self::constant(T::TWO);
    const HALF: Self = Self::constant(T::HALF);
    const PI: Self = Self::constant(T::PI);
    const FRAC_PI_2: Self = Self::constant(T::FRAC_PI_2);
    const TAU: Self = Self::constant(T::TAU);
    const E: Self = Self::constant(T::E);
    const EPSILON: Self = Self::constant(T::EPSILON);
    const MIN_POSITIVE: Self = Self::constant(T::MIN_POSITIVE);
    const MIN_VALUE: Self = Self::constant(T::MIN_VALUE);
    const MAX_VALUE: Self = Self::constant(T::MAX_VALUE);
    const INFINITY: Self = Self::constant(T::INFINITY);
    const NEG_INFINITY: Self = Self::constant(T::NEG_INFINITY);
    const NAN: Self = Self::new(T::NAN, T::NAN);
    const DEFAULT_TOL_SCALE: Self = Self::constant(T::DEFAULT_TOL_SCALE);

    #[inline(always)]
    fn from_f64(val: f64) -> Self {
        Self::constant(T::from_f64(val))
    }
    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        Self::constant(T::from_usize(n))
    }
    #[inline(always)]
    fn from_i32(n: i32) -> Self {
        Self::constant(T::from_i32(n))
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self.re.to_f64()
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.eps.is_finite()
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        self.re.is_nan() || self.eps.is_nan()
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = self.re.sqrt();
        let d = if s == T::ZERO && self.eps == T::ZERO {
            T::ZERO
        } else {
            self.eps / (T::TWO * s)
        };
        Self::new(s, d)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        let h = self.re.hypot(other.re);
        let d = if h == T::ZERO {
            T::ZERO
        } else {
            (self.re * self.eps + other.re * other.eps) / h
        };
        Self::new(h, d)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        let denom = self.re * self.re + other.re * other.re;
        let d = if denom == T::ZERO {
            T::ZERO
        } else {
            (other.re * self.eps - self.re * other.eps) / denom
        };
        Self::new(self.re.atan2(other.re), d)
    }
    #[inline]
    fn expm1(self) -> Self {
        Self::new(self.re.expm1(), self.eps * self.re.exp())
    }
    #[inline]
    fn ln_1p(self) -> Self {
        Self::new(self.re.ln_1p(), self.eps / (T::ONE + self.re))
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        let s = T::ONE.copysign(sign.re) * T::ONE.copysign(self.re);
        Self::new(self.re.copysign(sign.re), self.eps * s)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        if self.re.is_nan() {
            other
        } else if other.re.is_nan() || self.re <= other.re {
            self
        } else {
            other
        }
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        if self.re.is_nan() {
            other
        } else if other.re.is_nan() || self.re >= other.re {
            self
        } else {
            other
        }
    }
    #[inline]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        if self.re < lo.re {
            lo
        } else if self.re > hi.re {
            hi
        } else {
            self
        }
    }
    #[inline]
    fn rem_euclid(self, rhs: Self) -> Self {
        Self::new(self.re.rem_euclid(rhs.re), self.eps)
    }
    #[inline]
    fn total_cmp(&self, other: &Self) -> Ordering {
        self.re.total_cmp(&other.re)
    }
}
