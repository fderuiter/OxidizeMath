#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:modular_polynomials_review]
//! Zero-allocation stack-backed univariate polynomial ring $F\[x\]$ ([`StackPoly`]) with
//! Euclidean division (`checked_div_rem`), monic GCD, Extended GCD, Horner evaluation,
//! and formal differentiation.

use crate::complex::Complex;
use crate::error::ScalarError;
use crate::fp::Fp;
use crate::traits::{FieldScalar, RealScalar};
use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

/// Minimal `#![no_std]` field element contract unifying exact finite fields ([`Fp`]) and
/// approximate [`FieldScalar`] types (`f32`, `f64`, [`Complex`]) for [`StackPoly`].
pub trait PolyField:
    Copy
    + Clone
    + PartialEq
    + Default
    + fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
    /// Additive identity $0$.
    const ZERO: Self;
    /// Multiplicative identity $1$.
    const ONE: Self;

    /// Converts an unsigned integer index (used in formal derivatives) into `Self`.
    #[must_use]
    fn from_usize_ring(n: usize) -> Self;

    /// Fallible multiplicative inverse (`None` if zero or non-invertible).
    #[must_use]
    fn checked_inv(self) -> Option<Self>;
}

impl<const P: u64> PolyField for Fp<P> {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

    #[inline]
    fn from_usize_ring(n: usize) -> Self {
        Self::from_u64(n as u64)
    }

    #[inline]
    fn checked_inv(self) -> Option<Self> {
        Self::checked_inv(self)
    }
}

impl PolyField for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn from_usize_ring(n: usize) -> Self {
        n as f32
    }

    #[inline]
    fn checked_inv(self) -> Option<Self> {
        FieldScalar::checked_inv(self)
    }
}

impl PolyField for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn from_usize_ring(n: usize) -> Self {
        n as f64
    }

    #[inline]
    fn checked_inv(self) -> Option<Self> {
        FieldScalar::checked_inv(self)
    }
}

impl<T: RealScalar> PolyField for Complex<T> {
    const ZERO: Self = <Self as FieldScalar>::ZERO;
    const ONE: Self = <Self as FieldScalar>::ONE;

    #[inline]
    fn from_usize_ring(n: usize) -> Self {
        <Self as FieldScalar>::from_usize(n)
    }

    #[inline]
    fn checked_inv(self) -> Option<Self> {
        <Self as FieldScalar>::checked_inv(self)
    }
}

/// Fixed-capacity stack polynomial $\sum_{k=0}^{\text{len}-1} c_k x^k$ with zero heap allocation.
#[derive(Copy, Clone, Debug)]
pub struct StackPoly<T, const CAP: usize> {
    pub(crate) coeffs: [T; CAP],
    pub(crate) len: usize,
}

impl<T: PolyField, const CAP: usize> Default for StackPoly<T, CAP> {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl<T: PolyField, const CAP: usize> PartialEq for StackPoly<T, CAP> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.coeffs[..self.len] == other.coeffs[..other.len]
    }
}

impl<T: PolyField + Eq, const CAP: usize> Eq for StackPoly<T, CAP> {}

impl<T: PolyField, const CAP: usize> StackPoly<T, CAP> {
    /// Zero polynomial $0$.
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        Self {
            coeffs: [T::ZERO; CAP],
            len: 0,
        }
    }

    /// Constant polynomial $c$.
    pub fn constant(c: T) -> Result<Self, ScalarError> {
        if c == T::ZERO {
            return Ok(Self::zero());
        }
        if CAP == 0 {
            return Err(ScalarError::DegreeOverflow {
                capacity: 0,
                required: 1,
            });
        }
        let mut coeffs = [T::ZERO; CAP];
        coeffs[0] = c;
        Ok(Self { coeffs, len: 1 })
    }

    /// Monomial $c \cdot x^{\text{deg}}$.
    pub fn monomial(coeff: T, deg: usize) -> Result<Self, ScalarError> {
        if coeff == T::ZERO {
            return Ok(Self::zero());
        }
        let required = deg + 1;
        if required > CAP {
            return Err(ScalarError::DegreeOverflow {
                capacity: CAP,
                required,
            });
        }
        let mut coeffs = [T::ZERO; CAP];
        coeffs[deg] = coeff;
        Ok(Self {
            coeffs,
            len: required,
        })
    }

    /// Constructs a canonical trimmed `StackPoly` from ascending-degree slice `slice`.
    pub fn from_slice(slice: &[T]) -> Result<Self, ScalarError> {
        let mut trimmed_len = slice.len();
        while trimmed_len > 0 && slice[trimmed_len - 1] == T::ZERO {
            trimmed_len -= 1;
        }
        if trimmed_len > CAP {
            return Err(ScalarError::DegreeOverflow {
                capacity: CAP,
                required: trimmed_len,
            });
        }
        let mut coeffs = [T::ZERO; CAP];
        coeffs[..trimmed_len].copy_from_slice(&slice[..trimmed_len]);
        Ok(Self {
            coeffs,
            len: trimmed_len,
        })
    }

    fn normalize(&mut self) {
        while self.len > 0 && self.coeffs[self.len - 1] == T::ZERO {
            self.len -= 1;
        }
        for slot in &mut self.coeffs[self.len..] {
            *slot = T::ZERO;
        }
    }

    /// Canonical ascending-degree active coefficient slice (`&[]` for zero polynomial).
    #[inline]
    #[must_use]
    pub fn coeffs(&self) -> &[T] {
        &self.coeffs[..self.len]
    }

    /// Coefficient of $x^k$ (`T::ZERO` for `k >= self.len`).
    #[inline]
    #[must_use]
    pub fn coeff(&self, k: usize) -> T {
        if k < self.len {
            self.coeffs[k]
        } else {
            T::ZERO
        }
    }

    /// Returns `true` iff `self` is the zero polynomial.
    #[inline]
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.len == 0
    }

    /// Polynomial degree (`None` for the zero polynomial).
    #[inline]
    #[must_use]
    pub const fn degree(&self) -> Option<usize> {
        if self.len == 0 {
            None
        } else {
            Some(self.len - 1)
        }
    }

    /// Leading non-zero coefficient (`None` for the zero polynomial).
    #[inline]
    #[must_use]
    pub fn leading_coeff(&self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            Some(self.coeffs[self.len - 1])
        }
    }

    /// Evaluates $p(x)$ via Horner's scheme in $O(\deg p)$ operations.
    #[must_use]
    pub fn eval(&self, x: T) -> T {
        let mut acc = T::ZERO;
        for &c in self.coeffs[..self.len].iter().rev() {
            acc = acc * x + c;
        }
        acc
    }

    /// Formal derivative $p'(x) = \sum_{k=1}^{d} k c_k x^{k-1}$.
    #[must_use]
    pub fn formal_derivative(&self) -> Self {
        if self.len <= 1 {
            return Self::zero();
        }
        let mut out = Self::zero();
        for k in 1..self.len {
            out.coeffs[k - 1] = self.coeffs[k] * T::from_usize_ring(k);
        }
        out.len = self.len - 1;
        out.normalize();
        out
    }

    /// Scales every coefficient by `s`.
    #[must_use]
    pub fn scale(&self, s: T) -> Self {
        if s == T::ZERO || self.is_zero() {
            return Self::zero();
        }
        let mut out = *self;
        for slot in &mut out.coeffs[..out.len] {
            *slot = *slot * s;
        }
        out.normalize();
        out
    }

    /// Normalizes a non-zero polynomial to monic form (`leading_coeff == 1`).
    pub fn to_monic(&self) -> Result<Self, ScalarError> {
        let Some(lc) = self.leading_coeff() else {
            return Ok(Self::zero());
        };
        let inv = lc.checked_inv().ok_or(ScalarError::NonInvertible)?;
        Ok(self.scale(inv))
    }

    /// Checked polynomial addition.
    pub fn checked_add(&self, rhs: &Self) -> Result<Self, ScalarError> {
        let max_len = self.len.max(rhs.len);
        if max_len > CAP {
            return Err(ScalarError::DegreeOverflow {
                capacity: CAP,
                required: max_len,
            });
        }
        let mut out = Self::zero();
        for i in 0..max_len {
            out.coeffs[i] = self.coeff(i) + rhs.coeff(i);
        }
        out.len = max_len;
        out.normalize();
        Ok(out)
    }

    /// Checked polynomial subtraction.
    pub fn checked_sub(&self, rhs: &Self) -> Result<Self, ScalarError> {
        let max_len = self.len.max(rhs.len);
        let mut out = Self::zero();
        for i in 0..max_len {
            out.coeffs[i] = self.coeff(i) - rhs.coeff(i);
        }
        out.len = max_len;
        out.normalize();
        Ok(out)
    }

    /// Checked polynomial product returning [`ScalarError::DegreeOverflow`] if $\deg(a b) \ge \text{CAP}$.
    pub fn checked_mul(&self, rhs: &Self) -> Result<Self, ScalarError> {
        if self.is_zero() || rhs.is_zero() {
            return Ok(Self::zero());
        }
        let required = self.len + rhs.len - 1;
        if required > CAP {
            return Err(ScalarError::DegreeOverflow {
                capacity: CAP,
                required,
            });
        }
        let mut out = Self::zero();
        for i in 0..self.len {
            for j in 0..rhs.len {
                out.coeffs[i + j] = out.coeffs[i + j] + self.coeffs[i] * rhs.coeffs[j];
            }
        }
        out.len = required;
        out.normalize();
        Ok(out)
    }

    /// Euclidean polynomial division `self = q * divisor + r` with $\deg(r) < \deg(\text{divisor})$.
    pub fn checked_div_rem(&self, divisor: &Self) -> Result<(Self, Self), ScalarError> {
        let Some(div_deg) = divisor.degree() else {
            return Err(ScalarError::DivisionByZero);
        };
        let lc_inv = divisor.coeffs[div_deg]
            .checked_inv()
            .ok_or(ScalarError::NonInvertible)?;
        if self.len < divisor.len {
            return Ok((Self::zero(), *self));
        }
        let mut quot = Self::zero();
        let mut rem = *self;
        quot.len = self.len - divisor.len + 1;
        while let Some(rem_deg) = rem.degree() {
            if rem_deg < div_deg {
                break;
            }
            let shift = rem_deg - div_deg;
            let factor = rem.coeffs[rem_deg] * lc_inv;
            if factor == T::ZERO {
                rem.coeffs[rem_deg] = T::ZERO;
                rem.normalize();
                continue;
            }
            quot.coeffs[shift] = quot.coeffs[shift] + factor;
            rem.coeffs[rem_deg] = T::ZERO;
            for j in 0..div_deg {
                rem.coeffs[shift + j] = rem.coeffs[shift + j] - factor * divisor.coeffs[j];
            }
            rem.normalize();
        }
        quot.normalize();
        Ok((quot, rem))
    }

    /// Monic Euclidean greatest common divisor $\gcd(a, b) \in F\[x\]$.
    pub fn gcd(&self, other: &Self) -> Result<Self, ScalarError> {
        let mut a = *self;
        let mut b = *other;
        while !b.is_zero() {
            let (_, r) = a.checked_div_rem(&b)?;
            a = b;
            b = r;
        }
        a.to_monic()
    }

    /// Extended Euclidean algorithm returning monic `(g, s, t)` with $s \cdot a + t \cdot b = g$.
    pub fn extended_gcd(&self, other: &Self) -> Result<(Self, Self, Self), ScalarError> {
        let mut old_r = *self;
        let mut r = *other;
        let mut old_s = if CAP > 0 {
            Self::constant(T::ONE)?
        } else {
            Self::zero()
        };
        let mut s = Self::zero();
        let mut old_t = Self::zero();
        let mut t = if CAP > 0 {
            Self::constant(T::ONE)?
        } else {
            Self::zero()
        };
        while !r.is_zero() {
            let (q, rem) = old_r.checked_div_rem(&r)?;
            old_r = r;
            r = rem;
            let next_s = old_s.checked_sub(&q.checked_mul(&s)?)?;
            old_s = s;
            s = next_s;
            let next_t = old_t.checked_sub(&q.checked_mul(&t)?)?;
            old_t = t;
            t = next_t;
        }
        if let Some(lc) = old_r.leading_coeff() {
            let inv = lc.checked_inv().ok_or(ScalarError::NonInvertible)?;
            Ok((old_r.scale(inv), old_s.scale(inv), old_t.scale(inv)))
        } else {
            Ok((Self::zero(), Self::zero(), Self::zero()))
        }
    }
}

impl<T: PolyField, const CAP: usize> Add for StackPoly<T, CAP> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(&rhs)
            .expect("StackPoly addition fits within shared capacity CAP")
    }
}

impl<T: PolyField, const CAP: usize> Sub for StackPoly<T, CAP> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(&rhs)
            .expect("StackPoly subtraction fits within shared capacity CAP")
    }
}

impl<T: PolyField, const CAP: usize> Neg for StackPoly<T, CAP> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.scale(-T::ONE)
    }
}

impl<T: PolyField, const CAP: usize> Mul for StackPoly<T, CAP> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        self.checked_mul(&rhs)
            .expect("StackPoly product exceeded capacity CAP; use checked_mul for fallible bounds")
    }
}

impl<T: PolyField, const CAP: usize> Div for StackPoly<T, CAP> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div_rem(&rhs)
            .expect("StackPoly Euclidean division by zero")
            .0
    }
}

impl<T: PolyField, const CAP: usize> Rem for StackPoly<T, CAP> {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        self.checked_div_rem(&rhs)
            .expect("StackPoly Euclidean remainder by zero")
            .1
    }
}

impl<T: PolyField, const CAP: usize> AddAssign for StackPoly<T, CAP> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: PolyField, const CAP: usize> SubAssign for StackPoly<T, CAP> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: PolyField, const CAP: usize> MulAssign for StackPoly<T, CAP> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
