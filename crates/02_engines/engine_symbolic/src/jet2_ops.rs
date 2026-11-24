//! Arithmetic operator implementations for [`Jet2<N>`].
// [cite:engine_symbolic]

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::jet2::Jet2;

impl<const N: usize> Neg for Jet2<N> {
    type Output = Self;

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn neg(self) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        let mut i = 0;
        while i < N {
            grad[i] = -self.grad[i];
            let mut j = 0;
            while j < N {
                hess[i][j] = -self.hess[i][j];
                j += 1;
            }
            i += 1;
        }
        Self {
            val: -self.val,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Add for Jet2<N> {
    type Output = Self;

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn add(self, rhs: Self) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        let mut i = 0;
        while i < N {
            grad[i] = self.grad[i] + rhs.grad[i];
            let mut j = 0;
            while j < N {
                hess[i][j] = self.hess[i][j] + rhs.hess[i][j];
                j += 1;
            }
            i += 1;
        }
        Self {
            val: self.val + rhs.val,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Sub for Jet2<N> {
    type Output = Self;

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn sub(self, rhs: Self) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        let mut i = 0;
        while i < N {
            grad[i] = self.grad[i] - rhs.grad[i];
            let mut j = 0;
            while j < N {
                hess[i][j] = self.hess[i][j] - rhs.hess[i][j];
                j += 1;
            }
            i += 1;
        }
        Self {
            val: self.val - rhs.val,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Mul for Jet2<N> {
    type Output = Self;

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn mul(self, rhs: Self) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        let mut i = 0;
        while i < N {
            grad[i] = self.grad[i] * rhs.val + self.val * rhs.grad[i];
            let mut j = 0;
            while j < N {
                hess[i][j] = self.hess[i][j] * rhs.val
                    + self.grad[i] * rhs.grad[j]
                    + self.grad[j] * rhs.grad[i]
                    + self.val * rhs.hess[i][j];
                j += 1;
            }
            i += 1;
        }
        Self {
            val: self.val * rhs.val,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Div for Jet2<N> {
    type Output = Self;

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn div(self, rhs: Self) -> Self {
        let val = self.val / rhs.val;
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        let mut i = 0;
        while i < N {
            grad[i] = (self.grad[i] - val * rhs.grad[i]) / rhs.val;
            i += 1;
        }
        let mut r = 0;
        while r < N {
            let mut c = 0;
            while c < N {
                hess[r][c] = (self.hess[r][c]
                    - grad[r] * rhs.grad[c]
                    - grad[c] * rhs.grad[r]
                    - val * rhs.hess[r][c])
                    / rhs.val;
                c += 1;
            }
            r += 1;
        }
        Self { val, grad, hess }
    }
}

impl<const N: usize> Add<f64> for Jet2<N> {
    type Output = Self;

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn add(self, rhs: f64) -> Self {
        Self {
            val: self.val + rhs,
            grad: self.grad,
            hess: self.hess,
        }
    }
}

impl<const N: usize> Sub<f64> for Jet2<N> {
    type Output = Self;

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn sub(self, rhs: f64) -> Self {
        Self {
            val: self.val - rhs,
            grad: self.grad,
            hess: self.hess,
        }
    }
}

impl<const N: usize> Mul<f64> for Jet2<N> {
    type Output = Self;

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn mul(self, rhs: f64) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        let mut i = 0;
        while i < N {
            grad[i] = self.grad[i] * rhs;
            let mut j = 0;
            while j < N {
                hess[i][j] = self.hess[i][j] * rhs;
                j += 1;
            }
            i += 1;
        }
        Self {
            val: self.val * rhs,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Div<f64> for Jet2<N> {
    type Output = Self;

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn div(self, rhs: f64) -> Self {
        self * (1.0 / rhs)
    }
}

impl<const N: usize> AddAssign for Jet2<N> {
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize> SubAssign for Jet2<N> {
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const N: usize> MulAssign for Jet2<N> {
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const N: usize> DivAssign for Jet2<N> {
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
