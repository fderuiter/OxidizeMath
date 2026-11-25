//! First-order multi-tangent forward automatic differentiation (`DualVec<T, N>`).
// [cite:math_commons]

use crate::traits::Scalar;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Multi-tangent forward-mode dual number propagating `N` partial derivatives in a single pass.
///
/// Designed for zero-allocation stack gradients and Levenberg-Marquardt Jacobian rows.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DualVec<T, const N: usize> {
    /// Primal value $f(x)$.
    pub val: T,
    /// Tangent array $[\partial f / \partial x_0, \dots, \partial f / \partial x_{N-1}]$.
    pub eps: [T; N],
}

impl<T: Scalar, const N: usize> Default for DualVec<T, N> {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<T: Scalar, const N: usize> DualVec<T, N> {
    /// Constructs a multi-tangent dual number from primal `val` and tangent array `eps`.
    #[inline]
    #[must_use]
    pub const fn new(val: T, eps: [T; N]) -> Self {
        Self { val, eps }
    }

    /// Constructs a constant dual vector `(val, [0; N])`.
    #[inline]
    #[must_use]
    pub const fn constant(val: T) -> Self {
        Self {
            val,
            eps: [T::ZERO; N],
        }
    }

    /// Constructs the `index`-th active variable `(val, e_index)`.
    #[inline]
    #[must_use]
    pub fn variable(val: T, index: usize) -> Self {
        debug_assert!(index < N, "DualVec variable index must be < N");
        let mut eps = [T::ZERO; N];
        if index < N {
            eps[index] = T::ONE;
        }
        Self { val, eps }
    }

    /// Seeds an `N`-vector `[x_0, ..., x_{N-1}]` with the $N \times N$ identity tangent basis.
    #[inline]
    #[must_use]
    pub fn jacobian_seed(vals: [T; N]) -> [Self; N] {
        let mut out = [Self::ZERO; N];
        let mut i = 0;
        while i < N {
            out[i] = Self::variable(vals[i], i);
            i += 1;
        }
        out
    }

    #[inline]
    fn map_chain(self, val: T, deriv: T) -> Self {
        let mut eps = [T::ZERO; N];
        let mut i = 0;
        while i < N {
            eps[i] = deriv * self.eps[i];
            i += 1;
        }
        Self { val, eps }
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
        Self {
            val: self.val + rhs.val,
            eps,
        }
    }
}

impl<T: Scalar, const N: usize> Add<T> for DualVec<T, N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: T) -> Self {
        Self {
            val: self.val + rhs,
            eps: self.eps,
        }
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
        Self {
            val: self.val - rhs.val,
            eps,
        }
    }
}

impl<T: Scalar, const N: usize> Sub<T> for DualVec<T, N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: T) -> Self {
        Self {
            val: self.val - rhs,
            eps: self.eps,
        }
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
        Self {
            val: self.val * rhs.val,
            eps,
        }
    }
}

impl<T: Scalar, const N: usize> Mul<T> for DualVec<T, N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self {
        let mut eps = [T::ZERO; N];
        let mut i = 0;
        while i < N {
            eps[i] = self.eps[i] * rhs;
            i += 1;
        }
        Self {
            val: self.val * rhs,
            eps,
        }
    }
}

impl<T: Scalar, const N: usize> Div for DualVec<T, N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = T::ONE / rhs.val;
        let val = self.val * inv;
        let mut eps = [T::ZERO; N];
        let mut i = 0;
        while i < N {
            eps[i] = (self.eps[i] - val * rhs.eps[i]) * inv;
            i += 1;
        }
        Self { val, eps }
    }
}

impl<T: Scalar, const N: usize> Div<T> for DualVec<T, N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self {
        self * (T::ONE / rhs)
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
        Self {
            val: -self.val,
            eps,
        }
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
    const TAU: Self = Self::constant(T::TAU);

    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::constant(T::from_f64(v))
    }

    #[inline]
    fn is_finite(self) -> bool {
        if !self.val.is_finite() {
            return false;
        }
        let mut i = 0;
        while i < N {
            if !self.eps[i].is_finite() {
                return false;
            }
            i += 1;
        }
        true
    }

    #[inline]
    fn abs(self) -> Self {
        self.map_chain(self.val.abs(), self.val.signum())
    }

    #[inline]
    fn signum(self) -> Self {
        Self::constant(self.val.signum())
    }

    #[inline]
    fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        self.map_chain(s, T::HALF / s)
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
            self.map_chain(p_m1 * self.val, nf * p_m1)
        }
    }

    #[inline]
    fn powf(self, exp: Self) -> Self {
        let mut exp_const = true;
        let mut k = 0;
        while k < N {
            if exp.eps[k] != T::ZERO {
                exp_const = false;
                break;
            }
            k += 1;
        }
        let v = self.val.powf(exp.val);
        if exp_const {
            return self.map_chain(v, exp.val * self.val.powf(exp.val - T::ONE));
        }
        let ln_x = self.val.ln();
        let inv_x = T::ONE / self.val;
        let mut eps = [T::ZERO; N];
        let mut i = 0;
        while i < N {
            eps[i] = v * (exp.eps[i] * ln_x + exp.val * self.eps[i] * inv_x);
            i += 1;
        }
        Self { val: v, eps }
    }

    #[inline]
    fn exp(self) -> Self {
        let e = self.val.exp();
        self.map_chain(e, e)
    }

    #[inline]
    fn ln(self) -> Self {
        self.map_chain(self.val.ln(), T::ONE / self.val)
    }

    #[inline]
    fn sin(self) -> Self {
        let (s, c) = self.val.sin_cos();
        self.map_chain(s, c)
    }

    #[inline]
    fn cos(self) -> Self {
        let (s, c) = self.val.sin_cos();
        self.map_chain(c, -s)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.val.sin_cos();
        (self.map_chain(s, c), self.map_chain(c, -s))
    }

    #[inline]
    fn tan(self) -> Self {
        let t = self.val.tan();
        self.map_chain(t, T::ONE + t * t)
    }

    #[inline]
    fn asin(self) -> Self {
        let inv = T::ONE / (T::ONE - self.val * self.val).sqrt();
        self.map_chain(self.val.asin(), inv)
    }

    #[inline]
    fn acos(self) -> Self {
        let inv = T::ONE / (T::ONE - self.val * self.val).sqrt();
        self.map_chain(self.val.acos(), -inv)
    }

    #[inline]
    fn atan(self) -> Self {
        let inv = T::ONE / (T::ONE + self.val * self.val);
        self.map_chain(self.val.atan(), inv)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        let inv = T::ONE / (self.val * self.val + other.val * other.val);
        let mut eps = [T::ZERO; N];
        let mut i = 0;
        while i < N {
            eps[i] = (other.val * self.eps[i] - self.val * other.eps[i]) * inv;
            i += 1;
        }
        Self {
            val: self.val.atan2(other.val),
            eps,
        }
    }

    #[inline]
    fn sinh(self) -> Self {
        self.map_chain(self.val.sinh(), self.val.cosh())
    }

    #[inline]
    fn cosh(self) -> Self {
        self.map_chain(self.val.cosh(), self.val.sinh())
    }

    #[inline]
    fn tanh(self) -> Self {
        let th = self.val.tanh();
        self.map_chain(th, T::ONE - th * th)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        let r = self.val.hypot(other.val);
        if r == T::ZERO {
            return Self::ZERO;
        }
        let sx = self.val / r;
        let sy = other.val / r;
        let mut eps = [T::ZERO; N];
        let mut i = 0;
        while i < N {
            eps[i] = sx * self.eps[i] + sy * other.eps[i];
            i += 1;
        }
        Self { val: r, eps }
    }
}
