// [cite:math_scalars]
//! Multi-tangent stack-allocated forward-mode automatic differentiation [`DualVec`].

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::traits::{Scalar, powi_generic};

/// Multi-tangent forward-mode dual number $f(x) + \sum_{k=0}^{N-1} \partial_k f(x)\,\varepsilon_k$.
///
/// Propagates up to `N` partial derivatives simultaneously on the stack with zero heap allocations.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DualVec<T, const N: usize> {
    /// Primal function value $f(x)$.
    pub val: T,
    /// Tangent vector of `N` directional/partial derivatives.
    pub eps: [T; N],
}

impl<T: Scalar, const N: usize> Default for DualVec<T, N> {
    #[inline]
    fn default() -> Self {
        Self::constant(T::ZERO)
    }
}

impl<T: Scalar, const N: usize> DualVec<T, N> {
    /// Constructs a `DualVec` from primal `val` and tangent array `eps`.
    #[inline]
    #[must_use]
    pub const fn new(val: T, eps: [T; N]) -> Self {
        Self { val, eps }
    }

    /// Constructs a constant `DualVec` with zero tangent lanes.
    #[inline]
    #[must_use]
    pub const fn constant(val: T) -> Self {
        Self {
            val,
            eps: [T::ZERO; N],
        }
    }

    /// Constructs an independent coordinate variable seeded with `1` in lane `axis` (if `axis < N`).
    #[inline]
    #[must_use]
    pub fn variable(val: T, axis: usize) -> Self {
        debug_assert!(axis < N);
        let mut eps = [T::ZERO; N];
        if axis < N {
            eps[axis] = T::ONE;
        }
        Self { val, eps }
    }

    #[inline]
    fn chain(self, f0: T, df: T) -> Self {
        let mut eps = [T::ZERO; N];
        let mut i = 0;
        while i < N {
            eps[i] = self.eps[i] * df;
            i += 1;
        }
        Self { val: f0, eps }
    }
}

impl<T: Scalar, const N: usize> Neg for DualVec<T, N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut eps = [T::ZERO; N];
        let mut i = 0;
        while i < N {
            eps[i] = -self.eps[i];
            i += 1;
        }
        Self::new(-self.val, eps)
    }
}

impl<T: Scalar, const N: usize> Add for DualVec<T, N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut eps = [T::ZERO; N];
        let mut i = 0;
        while i < N {
            eps[i] = self.eps[i] + rhs.eps[i];
            i += 1;
        }
        Self::new(self.val + rhs.val, eps)
    }
}

impl<T: Scalar, const N: usize> Sub for DualVec<T, N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut eps = [T::ZERO; N];
        let mut i = 0;
        while i < N {
            eps[i] = self.eps[i] - rhs.eps[i];
            i += 1;
        }
        Self::new(self.val - rhs.val, eps)
    }
}

impl<T: Scalar, const N: usize> Mul for DualVec<T, N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut eps = [T::ZERO; N];
        let mut i = 0;
        while i < N {
            eps[i] = self.eps[i] * rhs.val + self.val * rhs.eps[i];
            i += 1;
        }
        Self::new(self.val * rhs.val, eps)
    }
}

impl<T: Scalar, const N: usize> Div for DualVec<T, N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = rhs.val.recip();
        let inv2 = inv.sqr();
        let mut eps = [T::ZERO; N];
        let mut i = 0;
        while i < N {
            eps[i] = (self.eps[i] * rhs.val - self.val * rhs.eps[i]) * inv2;
            i += 1;
        }
        Self::new(self.val * inv, eps)
    }
}

impl<T: Scalar, const N: usize> AddAssign for DualVec<T, N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: Scalar, const N: usize> SubAssign for DualVec<T, N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: Scalar, const N: usize> MulAssign for DualVec<T, N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: Scalar, const N: usize> DivAssign for DualVec<T, N> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: Scalar, const N: usize> Scalar for DualVec<T, N> {
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
        if !self.val.is_finite_val() {
            return false;
        }
        let mut i = 0;
        while i < N {
            if !self.eps[i].is_finite_val() {
                return false;
            }
            i += 1;
        }
        true
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
        self.chain(self.val.sqr(), T::TWO * self.val)
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
        let mut const_exp = true;
        let mut i = 0;
        while i < N {
            if exp.eps[i] != T::ZERO {
                const_exp = false;
                break;
            }
            i += 1;
        }
        if const_exp {
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
        let inv_r2 = (self.val.sqr() + other.val.sqr()).recip();
        let mut eps = [T::ZERO; N];
        let mut i = 0;
        while i < N {
            eps[i] = (other.val * self.eps[i] - self.val * other.eps[i]) * inv_r2;
            i += 1;
        }
        Self::new(self.val.atan2(other.val), eps)
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
            return Self::constant(T::ZERO);
        }
        let inv_r = r.recip();
        let mut eps = [T::ZERO; N];
        let mut i = 0;
        while i < N {
            eps[i] = (self.val * self.eps[i] + other.val * other.eps[i]) * inv_r;
            i += 1;
        }
        Self::new(r, eps)
    }
}
