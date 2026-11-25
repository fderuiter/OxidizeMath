// [cite:math_commons]
use crate::error::ScalarDomainError;
use crate::traits::{Field, Ring};
use core::fmt;
use core::marker::PhantomData;

/// Explicit runtime-configured associative unital ring context $(R_C, +_C, \cdot_C, 0_C, 1_C)$.
///
/// Enables `#![no_std]` polynomial, number-theoretic, and arbitrary-precision arithmetic over domains
/// whose parameters (such as a runtime prime modulus $p$ or dynamic significand precision) cannot be
/// encoded in compile-time const generics and must not rely on `std` thread-local state.
pub trait DomainRing {
    /// Element representation operated on by this ring context.
    type Element: Clone + PartialEq + fmt::Debug;

    /// Returns the additive identity $0_C$.
    fn zero(&self) -> Self::Element;

    /// Returns the multiplicative identity $1_C$.
    fn one(&self) -> Self::Element;

    /// Returns `true` when `elem` is the additive identity in `self`.
    #[inline]
    fn is_zero(&self, elem: &Self::Element) -> bool {
        *elem == self.zero()
    }

    /// Computes $a +_C b$.
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;

    /// Computes $a -_C b$.
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;

    /// Computes $a \cdot_C b$.
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;

    /// Computes the additive inverse $-_C a$.
    fn neg(&self, a: &Self::Element) -> Self::Element;

    /// Computes binary exponentiation $a^{\text{exp}}$ in $O(\log \text{exp})$ ring multiplications.
    #[inline]
    fn pow_u64(&self, base: &Self::Element, mut exp: u64) -> Self::Element {
        let mut acc = self.one();
        let mut cur = base.clone();
        while exp > 0 {
            if exp & 1 == 1 {
                acc = self.mul(&acc, &cur);
            }
            exp >>= 1;
            if exp > 0 {
                cur = self.mul(&cur, &cur);
            }
        }
        acc
    }

    /// Evaluates ascending-degree polynomial $\sum_{k=0}^{n-1} c_k x^k$ at `x` via Horner's rule.
    #[inline]
    fn eval_horner(&self, coeffs: &[Self::Element], x: &Self::Element) -> Self::Element {
        let mut acc = self.zero();
        for coeff in coeffs.iter().rev() {
            let prod = self.mul(&acc, x);
            acc = self.add(&prod, coeff);
        }
        acc
    }
}

/// Explicit runtime-configured division ring / field context extending [`DomainRing`].
pub trait DomainField: DomainRing {
    /// Fallibly computes the multiplicative inverse $a^{-1}$ in `self`.
    fn try_inv(&self, a: &Self::Element) -> Result<Self::Element, ScalarDomainError>;

    /// Fallibly computes $a \cdot_C b^{-1}$ in `self`.
    #[inline]
    fn try_div(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ScalarDomainError> {
        let inv_b = self.try_inv(b)?;
        Ok(self.mul(a, &inv_b))
    }
}

/// Zero-sized (`size_of == 0`) inline adapter bridging stateless [`Ring`] and [`Field`] types into
/// [`DomainRing`] / [`DomainField`] algorithms with zero runtime dispatch or memory overhead.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StatelessContext<T>(PhantomData<fn() -> T>);

impl<T> StatelessContext<T> {
    /// Constructs a zero-sized [`StatelessContext`] token.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> Default for StatelessContext<T> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> fmt::Debug for StatelessContext<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("StatelessContext")
    }
}

impl<T: Ring> DomainRing for StatelessContext<T> {
    type Element = T;

    #[inline(always)]
    fn zero(&self) -> T {
        T::zero()
    }

    #[inline(always)]
    fn one(&self) -> T {
        T::one()
    }

    #[inline(always)]
    fn is_zero(&self, elem: &T) -> bool {
        elem.is_zero()
    }

    #[inline(always)]
    fn add(&self, a: &T, b: &T) -> T {
        a.clone() + b.clone()
    }

    #[inline(always)]
    fn sub(&self, a: &T, b: &T) -> T {
        a.clone() - b.clone()
    }

    #[inline(always)]
    fn mul(&self, a: &T, b: &T) -> T {
        a.clone() * b.clone()
    }

    #[inline(always)]
    fn neg(&self, a: &T) -> T {
        -a.clone()
    }
}

impl<T: Field> DomainField for StatelessContext<T> {
    #[inline(always)]
    fn try_inv(&self, a: &T) -> Result<T, ScalarDomainError> {
        a.try_inverse()
    }

    #[inline(always)]
    fn try_div(&self, a: &T, b: &T) -> Result<T, ScalarDomainError> {
        a.checked_div(b).ok_or(ScalarDomainError::DivisionByZero)
    }
}
