#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

use crate::complex::Complex;
use crate::error::ScalarError;
use crate::fp::Fp;
use crate::traits::{FieldScalar, RealScalar};
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// Commutative field coefficient contract used by [`StackPoly`] for exact or numerical
/// univariate polynomial division and extended GCD.
pub trait PolyCoeff:
    Copy
    + Clone
    + fmt::Debug
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
    /// Additive identity $0$.
    const ZERO: Self;
    /// Multiplicative identity $1$.
    const ONE: Self;

    /// Returns `true` when `self` should be treated as zero when trimming leading coefficients.
    fn is_coeff_zero(self) -> bool;

    /// Fallible multiplicative inverse $c^{-1}$.
    ///
    /// # Errors
    /// Returns [`ScalarError::NonInvertible`] when `self` is zero.
    fn checked_coeff_inv(self) -> Result<Self, ScalarError>;

    /// Embeds small unsigned integer $k$ into the coefficient field ($k \cdot 1$).
    #[must_use]
    fn from_usize_count(mut k: usize) -> Self {
        let mut acc = Self::ZERO;
        let mut base = Self::ONE;
        while k > 0 {
            if (k & 1) == 1 {
                acc = acc + base;
            }
            base = base + base;
            k >>= 1;
        }
        acc
    }
}

impl<const P: u64> PolyCoeff for Fp<P> {
    const ZERO: Self = Fp::<P>::ZERO;
    const ONE: Self = Fp::<P>::ONE;

    #[inline]
    fn is_coeff_zero(self) -> bool {
        self.is_zero()
    }
    #[inline]
    fn checked_coeff_inv(self) -> Result<Self, ScalarError> {
        self.checked_inverse()
    }
}

impl PolyCoeff for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn is_coeff_zero(self) -> bool {
        self.abs() <= 1.0e-12
    }
    #[inline]
    fn checked_coeff_inv(self) -> Result<Self, ScalarError> {
        FieldScalar::try_inv(self)
    }
}

impl PolyCoeff for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn is_coeff_zero(self) -> bool {
        self.abs() <= 1.0e-5
    }
    #[inline]
    fn checked_coeff_inv(self) -> Result<Self, ScalarError> {
        FieldScalar::try_inv(self)
    }
}

impl<T: RealScalar> PolyCoeff for Complex<T> {
    const ZERO: Self = Complex::ZERO;
    const ONE: Self = Complex::ONE;

    #[inline]
    fn is_coeff_zero(self) -> bool {
        self.norm() <= T::DEFAULT_TOL_SCALE * T::EPSILON
    }
    #[inline]
    fn checked_coeff_inv(self) -> Result<Self, ScalarError> {
        self.checked_inv()
    }
}

/// Algebraic Euclidean domain supporting degree-bounded division with remainder
/// $a = q b + r$ where $r = 0$ or $\deg(r) < \deg(b)$.
pub trait EuclideanDomain: Sized {
    /// Returns `None` for the zero element and `Some(d)` for non-zero elements.
    fn euclidean_degree(&self) -> Option<usize>;

    /// Computes `(quotient, remainder)` satisfying `self == quotient * divisor + remainder`.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] if `divisor` is zero or [`ScalarError::CapacityExceeded`] on overflow.
    fn checked_div_rem(&self, divisor: &Self) -> Result<(Self, Self), ScalarError>;

    /// Computes the Euclidean quotient `self / divisor`.
    ///
    /// # Errors
    /// Propagates errors from [`Self::checked_div_rem`].
    fn checked_quo(&self, divisor: &Self) -> Result<Self, ScalarError> {
        self.checked_div_rem(divisor).map(|(q, _)| q)
    }

    /// Computes the Euclidean remainder `self % divisor`.
    ///
    /// # Errors
    /// Propagates errors from [`Self::checked_div_rem`].
    fn checked_rem(&self, divisor: &Self) -> Result<Self, ScalarError> {
        self.checked_div_rem(divisor).map(|(_, r)| r)
    }
}

/// Zero-allocation stack-backed univariate polynomial $\sum_{i=0}^{\text{len}-1} c_i x^i$
/// with compile-time coefficient capacity `CAP`.
#[derive(Clone, Copy, Debug)]
pub struct StackPoly<T: PolyCoeff, const CAP: usize> {
    coeffs: [T; CAP],
    len: usize,
}

impl<T: PolyCoeff, const CAP: usize> Default for StackPoly<T, CAP> {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl<T: PolyCoeff, const CAP: usize> PartialEq for StackPoly<T, CAP> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        let mut i = 0;
        while i < self.len {
            if self.coeffs[i] != other.coeffs[i] {
                return false;
            }
            i += 1;
        }
        true
    }
}

impl<T: PolyCoeff + Eq, const CAP: usize> Eq for StackPoly<T, CAP> {}

impl<T: PolyCoeff, const CAP: usize> StackPoly<T, CAP> {
    const VALID_CAP: () = assert!(CAP >= 1, "StackPoly CAP must be >= 1");

    /// Zero polynomial $0$.
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        let () = Self::VALID_CAP;
        Self {
            coeffs: [T::ZERO; CAP],
            len: 0,
        }
    }

    /// Constant polynomial $1$.
    #[must_use]
    pub fn one() -> Self {
        Self::constant(T::ONE)
    }

    /// Constant polynomial $c$.
    #[must_use]
    pub fn constant(c: T) -> Self {
        let mut out = Self::zero();
        if !c.is_coeff_zero() {
            out.coeffs[0] = c;
            out.len = 1;
        }
        out
    }

    /// Constructs a polynomial from ascending-degree slice `[c_0, c_1, ..., c_d]`, trimming trailing zeros.
    ///
    /// # Errors
    /// Returns [`ScalarError::CapacityExceeded`] if the trimmed degree requires more than `CAP` slots.
    pub fn from_slice(slice: &[T]) -> Result<Self, ScalarError> {
        let () = Self::VALID_CAP;
        let mut trimmed_len = slice.len();
        while trimmed_len > 0 && slice[trimmed_len - 1].is_coeff_zero() {
            trimmed_len -= 1;
        }
        if trimmed_len > CAP {
            return Err(ScalarError::CapacityExceeded);
        }
        let mut coeffs = [T::ZERO; CAP];
        coeffs[..trimmed_len].copy_from_slice(&slice[..trimmed_len]);
        Ok(Self {
            coeffs,
            len: trimmed_len,
        })
    }

    /// Active canonical coefficient slice `[c_0, ..., c_{len-1}]` (`&[]` for zero polynomial).
    #[inline]
    #[must_use]
    pub fn coeffs(&self) -> &[T] {
        &self.coeffs[..self.len]
    }

    /// Coefficient of $x^k$ (`T::ZERO` when $k \ge \text{len}$).
    #[inline]
    #[must_use]
    pub fn coeff(&self, k: usize) -> T {
        if k < self.len {
            self.coeffs[k]
        } else {
            T::ZERO
        }
    }

    /// Returns `true` for the zero polynomial.
    #[inline]
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.len == 0
    }

    /// Polynomial degree (`None` for the zero polynomial, `Some(0)` for non-zero constants).
    #[inline]
    #[must_use]
    pub const fn degree(&self) -> Option<usize> {
        if self.len == 0 {
            None
        } else {
            Some(self.len - 1)
        }
    }

    /// Leading coefficient (`T::ZERO` for the zero polynomial).
    #[inline]
    #[must_use]
    pub fn leading_coeff(&self) -> T {
        if self.len == 0 {
            T::ZERO
        } else {
            self.coeffs[self.len - 1]
        }
    }

    /// Evaluates $p(x)$ via Horner's rule with zero heap allocations.
    #[must_use]
    pub fn eval(&self, x: T) -> T {
        let mut acc = T::ZERO;
        let mut i = self.len;
        while i > 0 {
            i -= 1;
            acc = acc * x + self.coeffs[i];
        }
        acc
    }

    /// Formal algebraic derivative $p'(x) = \sum_{k=1}^d k c_k x^{k-1}$.
    #[must_use]
    pub fn derivative(&self) -> Self {
        if self.len <= 1 {
            return Self::zero();
        }
        let mut out = Self::zero();
        for k in 1..self.len {
            out.coeffs[k - 1] = T::from_usize_count(k) * self.coeffs[k];
        }
        out.len = self.len - 1;
        out.trim();
        out
    }

    /// Scales all coefficients by `scalar` and normalizes leading zeros.
    #[must_use]
    pub fn scale(&self, scalar: T) -> Self {
        if scalar.is_coeff_zero() || self.is_zero() {
            return Self::zero();
        }
        let mut out = *self;
        for c in &mut out.coeffs[..out.len] {
            *c = *c * scalar;
        }
        out.trim();
        out
    }

    /// Normalizes `self` to a monic polynomial by dividing by `leading_coeff()`.
    ///
    /// # Errors
    /// Returns [`ScalarError::NonInvertible`] when `self` is zero.
    pub fn checked_monic(&self) -> Result<Self, ScalarError> {
        let lc_inv = self.leading_coeff().checked_coeff_inv()?;
        Ok(self.scale(lc_inv))
    }

    /// Polynomial addition `self + rhs`.
    #[must_use]
    pub fn checked_add(&self, rhs: &Self) -> Self {
        let max_len = self.len.max(rhs.len);
        let mut out = Self::zero();
        for i in 0..max_len {
            out.coeffs[i] = self.coeff(i) + rhs.coeff(i);
        }
        out.len = max_len;
        out.trim();
        out
    }

    /// Polynomial subtraction `self - rhs`.
    #[must_use]
    pub fn checked_sub(&self, rhs: &Self) -> Self {
        let max_len = self.len.max(rhs.len);
        let mut out = Self::zero();
        for i in 0..max_len {
            out.coeffs[i] = self.coeff(i) - rhs.coeff(i);
        }
        out.len = max_len;
        out.trim();
        out
    }

    /// Checked polynomial product `self * rhs`.
    ///
    /// # Errors
    /// Returns [`ScalarError::CapacityExceeded`] if `deg(self) + deg(rhs) + 1 > CAP`.
    pub fn checked_mul(&self, rhs: &Self) -> Result<Self, ScalarError> {
        if self.is_zero() || rhs.is_zero() {
            return Ok(Self::zero());
        }
        let prod_len = self.len + rhs.len - 1;
        if prod_len > CAP {
            return Err(ScalarError::CapacityExceeded);
        }
        let mut out = Self::zero();
        for i in 0..self.len {
            for j in 0..rhs.len {
                out.coeffs[i + j] = out.coeffs[i + j] + self.coeffs[i] * rhs.coeffs[j];
            }
        }
        out.len = prod_len;
        out.trim();
        Ok(out)
    }

    /// Iterative extended Euclidean algorithm returning monic `(gcd, s, t)` with `s * self + t * rhs == gcd`.
    ///
    /// # Errors
    /// Returns [`ScalarError::NonInvertible`] if both inputs are zero, or [`ScalarError::CapacityExceeded`] if `CAP` overflows.
    pub fn xgcd(&self, rhs: &Self) -> Result<(Self, Self, Self), ScalarError> {
        if self.is_zero() && rhs.is_zero() {
            return Err(ScalarError::NonInvertible);
        }
        let mut old_r = *self;
        let mut r = *rhs;
        let mut old_s = Self::one();
        let mut s = Self::zero();
        let mut old_t = Self::zero();
        let mut t = Self::one();

        while !r.is_zero() {
            let (q, rem) = old_r.checked_div_rem(&r)?;
            let qs = q.checked_mul(&s)?;
            let qt = q.checked_mul(&t)?;
            let next_s = old_s.checked_sub(&qs);
            let next_t = old_t.checked_sub(&qt);
            old_r = r;
            r = rem;
            old_s = s;
            s = next_s;
            old_t = t;
            t = next_t;
        }
        let lc_inv = old_r.leading_coeff().checked_coeff_inv()?;
        Ok((
            old_r.scale(lc_inv),
            old_s.scale(lc_inv),
            old_t.scale(lc_inv),
        ))
    }

    fn trim(&mut self) {
        while self.len > 0 && self.coeffs[self.len - 1].is_coeff_zero() {
            self.coeffs[self.len - 1] = T::ZERO;
            self.len -= 1;
        }
    }
}

impl<T: PolyCoeff, const CAP: usize> EuclideanDomain for StackPoly<T, CAP> {
    #[inline]
    fn euclidean_degree(&self) -> Option<usize> {
        self.degree()
    }

    fn checked_div_rem(&self, divisor: &Self) -> Result<(Self, Self), ScalarError> {
        if divisor.is_zero() {
            return Err(ScalarError::DivisionByZero);
        }
        let div_deg = divisor.len - 1;
        if self.len < divisor.len {
            return Ok((Self::zero(), *self));
        }
        let inv_lead = divisor.leading_coeff().checked_coeff_inv()?;
        let mut rem = *self;
        let mut quot = Self::zero();
        quot.len = self.len - div_deg;

        while rem.len >= divisor.len && !rem.is_zero() {
            let shift = rem.len - divisor.len;
            let factor = rem.leading_coeff() * inv_lead;
            quot.coeffs[shift] = quot.coeffs[shift] + factor;
            for i in 0..divisor.len {
                rem.coeffs[i + shift] = rem.coeffs[i + shift] - factor * divisor.coeffs[i];
            }
            rem.coeffs[rem.len - 1] = T::ZERO;
            rem.trim();
        }
        quot.trim();
        Ok((quot, rem))
    }
}

impl<T: PolyCoeff, const CAP: usize> Add for StackPoly<T, CAP> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(&rhs)
    }
}

impl<T: PolyCoeff, const CAP: usize> Sub for StackPoly<T, CAP> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(&rhs)
    }
}

impl<T: PolyCoeff, const CAP: usize> Neg for StackPoly<T, CAP> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::zero().checked_sub(&self)
    }
}

impl<T: PolyCoeff, const CAP: usize> Mul for StackPoly<T, CAP> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.checked_mul(&rhs)
            .expect("StackPoly::mul capacity exceeded")
    }
}

impl<T: PolyCoeff, const CAP: usize> Div for StackPoly<T, CAP> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.checked_quo(&rhs).expect("StackPoly division by zero")
    }
}

impl<T: PolyCoeff, const CAP: usize> Rem for StackPoly<T, CAP> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        self.checked_rem(&rhs).expect("StackPoly remainder by zero")
    }
}
