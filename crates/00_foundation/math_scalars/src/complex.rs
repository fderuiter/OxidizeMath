// [cite:math_commons]
use crate::traits::{ComplexField, Field, RealField, Ring, Scalar};

pub use num_complex::{Complex, Complex32, Complex64};

impl<T: Scalar> Scalar for Complex<T> {}

impl<T: RealField> Ring for Complex<T> {
    #[inline(always)]
    fn zero() -> Self {
        Self::new(T::ZERO, T::ZERO)
    }

    #[inline(always)]
    fn one() -> Self {
        Self::new(T::ONE, T::ZERO)
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.re == T::ZERO && self.im == T::ZERO
    }
}

#[inline]
fn smith_inv<T: RealField>(z: Complex<T>) -> Complex<T> {
    let ar = <T as ComplexField>::abs(z.re);
    let ai = <T as ComplexField>::abs(z.im);
    if ar >= ai {
        if ar == T::ZERO {
            return Complex::new(T::NAN, T::NAN);
        }
        let ratio = z.im / z.re;
        let denom = z.re + z.im * ratio;
        Complex::new(T::ONE / denom, -ratio / denom)
    } else {
        let ratio = z.re / z.im;
        let denom = z.re * ratio + z.im;
        Complex::new(ratio / denom, -T::ONE / denom)
    }
}

impl<T: RealField> Field for Complex<T> {
    #[inline]
    fn checked_inverse(&self) -> Option<Self> {
        if (self.re == T::ZERO && self.im == T::ZERO)
            || !self.re.is_finite()
            || !self.im.is_finite()
        {
            return None;
        }
        let inv = smith_inv(*self);
        (inv.re.is_finite() && inv.im.is_finite()).then_some(inv)
    }

    #[inline]
    fn multiplicative_inverse(&self) -> Self {
        smith_inv(*self)
    }
}

#[inline]
fn complex_sqrt_principal<T: RealField>(z: Complex<T>) -> Complex<T> {
    if z.re == T::ZERO && z.im == T::ZERO {
        return Complex::new(T::ZERO, T::ZERO);
    }
    let modulus = z.re.hypot(z.im);
    if z.re >= T::ZERO {
        let t = (T::HALF * (modulus + z.re)).sqrt();
        let im = if t == T::ZERO {
            T::ZERO
        } else {
            T::HALF * z.im / t
        };
        Complex::new(t, im)
    } else {
        let t = (T::HALF * (modulus - z.re)).sqrt();
        let re = if t == T::ZERO {
            T::ZERO
        } else {
            <T as ComplexField>::abs(z.im) * T::HALF / t
        };
        Complex::new(re, t.copysign(z.im))
    }
}

impl<T: RealField> ComplexField for Complex<T> {
    type Real = T;

    #[inline(always)]
    fn from_real(re: T) -> Self {
        Self::new(re, T::ZERO)
    }
    #[inline(always)]
    fn nan() -> Self {
        Self::new(T::NAN, T::NAN)
    }
    #[inline(always)]
    fn imag_unit() -> Self {
        Self::new(T::ZERO, T::ONE)
    }
    #[inline(always)]
    fn real(self) -> T {
        self.re
    }
    #[inline(always)]
    fn imag(self) -> T {
        self.im
    }
    #[inline(always)]
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    #[inline(always)]
    fn norm_sqr(self) -> T {
        self.re * self.re + self.im * self.im
    }
    #[inline(always)]
    fn abs(self) -> T {
        self.re.hypot(self.im)
    }
    #[inline(always)]
    fn arg(self) -> T {
        self.im.atan2(self.re)
    }
    #[inline(always)]
    fn scale(self, factor: T) -> Self {
        Self::new(self.re * factor, self.im * factor)
    }
    #[inline(always)]
    fn is_all_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    #[inline(always)]
    fn is_any_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
    #[inline]
    fn csqrt(self) -> Self {
        complex_sqrt_principal(self)
    }
    #[inline]
    fn cbrt(self) -> Self {
        let r = self.re.hypot(self.im).cbrt();
        let theta = self.im.atan2(self.re) / T::from_f64(3.0);
        let (s, c) = theta.sin_cos();
        Self::new(r * c, r * s)
    }
    #[inline]
    fn exp(self) -> Self {
        let er = self.re.exp();
        let (s, c) = self.im.sin_cos();
        Self::new(er * c, er * s)
    }
    #[inline]
    fn ln(self) -> Self {
        Self::new(self.re.hypot(self.im).ln(), self.im.atan2(self.re))
    }
    #[inline]
    fn log2(self) -> Self {
        self.ln().scale(T::ONE / T::TWO.ln())
    }
    #[inline]
    fn log10(self) -> Self {
        self.ln().scale(T::ONE / T::from_f64(10.0).ln())
    }
    #[inline]
    fn sin(self) -> Self {
        let (sr, cr) = self.re.sin_cos();
        Self::new(sr * self.im.cosh(), cr * self.im.sinh())
    }
    #[inline]
    fn cos(self) -> Self {
        let (sr, cr) = self.re.sin_cos();
        Self::new(cr * self.im.cosh(), -sr * self.im.sinh())
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (sr, cr) = self.re.sin_cos();
        let sh = self.im.sinh();
        let ch = self.im.cosh();
        (Self::new(sr * ch, cr * sh), Self::new(cr * ch, -sr * sh))
    }
    #[inline]
    fn tan(self) -> Self {
        let (s, c) = self.sin_cos();
        s * smith_inv(c)
    }
    #[inline]
    fn asin(self) -> Self {
        let i = Self::imag_unit();
        let one = Self::one();
        -i * (i * self + (one - self * self).csqrt()).ln()
    }
    #[inline]
    fn acos(self) -> Self {
        let i = Self::imag_unit();
        let one = Self::one();
        -i * (self + i * (one - self * self).csqrt()).ln()
    }
    #[inline]
    fn atan(self) -> Self {
        let i = Self::imag_unit();
        let one = Self::one();
        let num = one + i * self;
        let den = one - i * self;
        (-i * (num * smith_inv(den)).ln()).scale(T::HALF)
    }
    #[inline]
    fn sinh(self) -> Self {
        let (si, ci) = self.im.sin_cos();
        Self::new(self.re.sinh() * ci, self.re.cosh() * si)
    }
    #[inline]
    fn cosh(self) -> Self {
        let (si, ci) = self.im.sin_cos();
        Self::new(self.re.cosh() * ci, self.re.sinh() * si)
    }
    #[inline]
    fn tanh(self) -> Self {
        self.sinh() * smith_inv(self.cosh())
    }
    #[inline]
    fn asinh(self) -> Self {
        (self + (self * self + Self::one()).csqrt()).ln()
    }
    #[inline]
    fn acosh(self) -> Self {
        let one = Self::one();
        (self + (self + one).csqrt() * (self - one).csqrt()).ln()
    }
    #[inline]
    fn atanh(self) -> Self {
        let one = Self::one();
        ((one + self) * smith_inv(one - self)).ln().scale(T::HALF)
    }
    #[inline]
    fn powf(self, exp: T) -> Self {
        if self.re == T::ZERO && self.im == T::ZERO {
            return if exp > T::ZERO {
                Self::zero()
            } else if exp == T::ZERO {
                Self::one()
            } else {
                Self::nan()
            };
        }
        (self.ln().scale(exp)).exp()
    }
    #[inline]
    fn powc(self, exp: Self) -> Self {
        if self.re == T::ZERO && self.im == T::ZERO {
            return if exp.re > T::ZERO {
                Self::zero()
            } else if exp.re == T::ZERO && exp.im == T::ZERO {
                Self::one()
            } else {
                Self::nan()
            };
        }
        (exp * self.ln()).exp()
    }
    #[inline]
    fn powi(self, exp: i32) -> Self {
        if exp == 0 {
            return Self::one();
        }
        let abs_exp = exp.unsigned_abs() as u64;
        let p = Ring::pow_u64(&self, abs_exp);
        if exp > 0 { p } else { smith_inv(p) }
    }
    #[inline]
    fn floor(self) -> Self {
        Self::new(self.re.floor(), self.im.floor())
    }
    #[inline]
    fn ceil(self) -> Self {
        Self::new(self.re.ceil(), self.im.ceil())
    }
    #[inline]
    fn round(self) -> Self {
        Self::new(self.re.round(), self.im.round())
    }
    #[inline]
    fn signum(self) -> Self {
        let r = self.re.hypot(self.im);
        if r == T::ZERO || !r.is_finite() {
            Self::nan()
        } else {
            Self::new(self.re / r, self.im / r)
        }
    }
}
