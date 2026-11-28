//! Ring and arithmetic operator implementations for [`LaurentPolynomial`].

// [cite:knot_braid_topology]

use super::laurent::{LaurentError, LaurentPolynomial, MAX_LAURENT_SPAN};
use crate::pure_math::algebra::traits::Ring;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

impl<T: Ring> Ring for LaurentPolynomial<T> {
    #[verified_engine::verified]
    fn zero() -> Self {
        let z = Self {
            min_exp: 0,
            coeffs: Vec::new(),
        };
        debug_assert_eq!(z.min_exp, 0);
        debug_assert!(z.coeffs.is_empty());
        z
    }

    #[verified_engine::verified]
    fn one() -> Self {
        let u = Self::monomial(T::one(), 0);
        debug_assert_eq!(u.min_exp, 0);
        u
    }

    #[inline]
    fn ring_ref_add(&self, rhs: &Self) -> Self {
        self + rhs
    }
    #[inline]
    fn ring_ref_sub(&self, rhs: &Self) -> Self {
        self - rhs
    }
    #[inline]
    fn ring_ref_mul(&self, rhs: &Self) -> Self {
        self * rhs
    }
    #[inline]
    fn ring_ref_neg(&self) -> Self {
        -self
    }
}

#[verified_engine::verified]
pub(super) fn try_combine_linear<T: Ring>(
    lhs: &LaurentPolynomial<T>,
    rhs: &LaurentPolynomial<T>,
    subtract: bool,
) -> Result<LaurentPolynomial<T>, LaurentError> {
    if lhs.coeffs.is_empty() {
        return Ok(if subtract { -rhs } else { rhs.clone() });
    }
    if rhs.coeffs.is_empty() {
        return Ok(lhs.clone());
    }
    let min_e = lhs.min_exp.min(rhs.min_exp);
    let max_lhs = lhs.max_degree().ok_or(LaurentError::ExponentOverflow)?;
    let max_rhs = rhs.max_degree().ok_or(LaurentError::ExponentOverflow)?;
    let max_e = max_lhs.max(max_rhs);
    let len = super::laurent_checked::checked_span_len(min_e, max_e)?;
    let mut out = vec![T::zero(); len];
    let off_l = usize::try_from(i64::from(lhs.min_exp) - i64::from(min_e))
        .map_err(|_| LaurentError::ExponentOverflow)?;
    for (i, c) in lhs.coeffs.iter().enumerate() {
        out[off_l + i] = c.clone();
    }
    let off_r = usize::try_from(i64::from(rhs.min_exp) - i64::from(min_e))
        .map_err(|_| LaurentError::ExponentOverflow)?;
    for (i, c) in rhs.coeffs.iter().enumerate() {
        out[off_r + i] = if subtract {
            out[off_r + i].ring_ref_sub(c)
        } else {
            out[off_r + i].ring_ref_add(c)
        };
    }
    LaurentPolynomial::try_new(min_e, out)
}

#[verified_engine::verified]
pub(super) fn checked_combine_i64(
    lhs: &LaurentPolynomial<i64>,
    rhs: &LaurentPolynomial<i64>,
    subtract: bool,
) -> Result<LaurentPolynomial<i64>, LaurentError> {
    if lhs.coeffs.is_empty() {
        return if subtract {
            rhs.checked_neg()
        } else {
            Ok(rhs.clone())
        };
    }
    if rhs.coeffs.is_empty() {
        return Ok(lhs.clone());
    }
    let min_e = lhs.min_exp.min(rhs.min_exp);
    let max_lhs = lhs.max_degree().ok_or(LaurentError::ExponentOverflow)?;
    let max_rhs = rhs.max_degree().ok_or(LaurentError::ExponentOverflow)?;
    let max_e = max_lhs.max(max_rhs);
    let len = super::laurent_checked::checked_span_len(min_e, max_e)?;
    let mut out = vec![0_i64; len];
    let off_l = usize::try_from(i64::from(lhs.min_exp) - i64::from(min_e))
        .map_err(|_| LaurentError::ExponentOverflow)?;
    out[off_l..(off_l + lhs.coeffs.len())].copy_from_slice(&lhs.coeffs);
    let off_r = usize::try_from(i64::from(rhs.min_exp) - i64::from(min_e))
        .map_err(|_| LaurentError::ExponentOverflow)?;
    for (i, &c) in rhs.coeffs.iter().enumerate() {
        let cur = out[off_r + i];
        out[off_r + i] = if subtract {
            cur.checked_sub(c)
                .ok_or(LaurentError::CoefficientOverflow)?
        } else {
            cur.checked_add(c)
                .ok_or(LaurentError::CoefficientOverflow)?
        };
    }
    LaurentPolynomial::try_new(min_e, out)
}

#[verified_engine::verified]
pub(super) fn try_convolve<T: Ring>(
    lhs: &LaurentPolynomial<T>,
    rhs: &LaurentPolynomial<T>,
) -> Result<LaurentPolynomial<T>, LaurentError> {
    if lhs.coeffs.is_empty() || rhs.coeffs.is_empty() {
        return Ok(LaurentPolynomial::zero());
    }
    let out_len = lhs
        .coeffs
        .len()
        .checked_add(rhs.coeffs.len())
        .and_then(|s| s.checked_sub(1))
        .ok_or(LaurentError::ExponentOverflow)?;
    if out_len > MAX_LAURENT_SPAN {
        return Err(LaurentError::SpanTooLarge {
            span: out_len,
            max: MAX_LAURENT_SPAN,
        });
    }
    let min_e = lhs
        .min_exp
        .checked_add(rhs.min_exp)
        .ok_or(LaurentError::ExponentOverflow)?;
    let mut out = vec![T::zero(); out_len];
    for (i, c1) in lhs.coeffs.iter().enumerate() {
        if c1.is_zero() {
            continue;
        }
        for (j, c2) in rhs.coeffs.iter().enumerate() {
            let prod = c1.ring_ref_mul(c2);
            out[i + j] = out[i + j].ring_ref_add(&prod);
        }
    }
    LaurentPolynomial::try_new(min_e, out)
}

#[verified_engine::verified]
pub(super) fn checked_convolve_i64(
    lhs: &LaurentPolynomial<i64>,
    rhs: &LaurentPolynomial<i64>,
) -> Result<LaurentPolynomial<i64>, LaurentError> {
    if lhs.coeffs.is_empty() || rhs.coeffs.is_empty() {
        return Ok(LaurentPolynomial::zero());
    }
    let out_len = lhs
        .coeffs
        .len()
        .checked_add(rhs.coeffs.len())
        .and_then(|s| s.checked_sub(1))
        .ok_or(LaurentError::ExponentOverflow)?;
    if out_len > MAX_LAURENT_SPAN {
        return Err(LaurentError::SpanTooLarge {
            span: out_len,
            max: MAX_LAURENT_SPAN,
        });
    }
    let min_e = lhs
        .min_exp
        .checked_add(rhs.min_exp)
        .ok_or(LaurentError::ExponentOverflow)?;
    let mut out = vec![0_i64; out_len];
    for (i, &c1) in lhs.coeffs.iter().enumerate() {
        if c1 == 0 {
            continue;
        }
        for (j, &c2) in rhs.coeffs.iter().enumerate() {
            let prod = c1
                .checked_mul(c2)
                .ok_or(LaurentError::CoefficientOverflow)?;
            out[i + j] = out[i + j]
                .checked_add(prod)
                .ok_or(LaurentError::CoefficientOverflow)?;
        }
    }
    LaurentPolynomial::try_new(min_e, out)
}

#[inline]
fn combine_linear<T: Ring>(
    lhs: &LaurentPolynomial<T>,
    rhs: &LaurentPolynomial<T>,
    subtract: bool,
) -> LaurentPolynomial<T> {
    try_combine_linear(lhs, rhs, subtract).expect("LaurentPolynomial linear combination overflow")
}

#[inline]
fn convolve<T: Ring>(
    lhs: &LaurentPolynomial<T>,
    rhs: &LaurentPolynomial<T>,
) -> LaurentPolynomial<T> {
    try_convolve(lhs, rhs).expect("LaurentPolynomial multiplication overflow")
}

impl<T: Ring> Add for LaurentPolynomial<T> {
    type Output = Self;
    #[verified_engine::verified]
    fn add(self, rhs: Self) -> Self {
        combine_linear(&self, &rhs, false)
    }
}

impl<'b, T: Ring> Add<&'b LaurentPolynomial<T>> for &LaurentPolynomial<T> {
    type Output = LaurentPolynomial<T>;
    #[verified_engine::verified]
    fn add(self, rhs: &'b LaurentPolynomial<T>) -> LaurentPolynomial<T> {
        combine_linear(self, rhs, false)
    }
}

impl<'b, T: Ring> Add<&'b LaurentPolynomial<T>> for LaurentPolynomial<T> {
    type Output = LaurentPolynomial<T>;
    #[verified_engine::verified]
    fn add(self, rhs: &'b LaurentPolynomial<T>) -> LaurentPolynomial<T> {
        combine_linear(&self, rhs, false)
    }
}

impl<T: Ring> Add<LaurentPolynomial<T>> for &LaurentPolynomial<T> {
    type Output = LaurentPolynomial<T>;
    #[verified_engine::verified]
    fn add(self, rhs: LaurentPolynomial<T>) -> LaurentPolynomial<T> {
        combine_linear(self, &rhs, false)
    }
}

impl<T: Ring> AddAssign for LaurentPolynomial<T> {
    #[verified_engine::verified]
    fn add_assign(&mut self, rhs: Self) {
        *self = combine_linear(self, &rhs, false);
    }
}

impl<T: Ring> AddAssign<&LaurentPolynomial<T>> for LaurentPolynomial<T> {
    #[verified_engine::verified]
    fn add_assign(&mut self, rhs: &Self) {
        *self = combine_linear(self, rhs, false);
    }
}

impl<T: Ring> Sub for LaurentPolynomial<T> {
    type Output = Self;
    #[verified_engine::verified]
    fn sub(self, rhs: Self) -> Self {
        combine_linear(&self, &rhs, true)
    }
}

impl<'b, T: Ring> Sub<&'b LaurentPolynomial<T>> for &LaurentPolynomial<T> {
    type Output = LaurentPolynomial<T>;
    #[verified_engine::verified]
    fn sub(self, rhs: &'b LaurentPolynomial<T>) -> LaurentPolynomial<T> {
        combine_linear(self, rhs, true)
    }
}

impl<'b, T: Ring> Sub<&'b LaurentPolynomial<T>> for LaurentPolynomial<T> {
    type Output = LaurentPolynomial<T>;
    #[verified_engine::verified]
    fn sub(self, rhs: &'b LaurentPolynomial<T>) -> LaurentPolynomial<T> {
        combine_linear(&self, rhs, true)
    }
}

impl<T: Ring> Sub<LaurentPolynomial<T>> for &LaurentPolynomial<T> {
    type Output = LaurentPolynomial<T>;
    #[verified_engine::verified]
    fn sub(self, rhs: LaurentPolynomial<T>) -> LaurentPolynomial<T> {
        combine_linear(self, &rhs, true)
    }
}

impl<T: Ring> SubAssign for LaurentPolynomial<T> {
    #[verified_engine::verified]
    fn sub_assign(&mut self, rhs: Self) {
        *self = combine_linear(self, &rhs, true);
    }
}

impl<T: Ring> SubAssign<&LaurentPolynomial<T>> for LaurentPolynomial<T> {
    #[verified_engine::verified]
    fn sub_assign(&mut self, rhs: &Self) {
        *self = combine_linear(self, rhs, true);
    }
}

impl<T: Ring> Mul for LaurentPolynomial<T> {
    type Output = Self;
    #[verified_engine::verified]
    fn mul(self, rhs: Self) -> Self {
        convolve(&self, &rhs)
    }
}

impl<'b, T: Ring> Mul<&'b LaurentPolynomial<T>> for &LaurentPolynomial<T> {
    type Output = LaurentPolynomial<T>;
    #[verified_engine::verified]
    fn mul(self, rhs: &'b LaurentPolynomial<T>) -> LaurentPolynomial<T> {
        convolve(self, rhs)
    }
}

impl<'b, T: Ring> Mul<&'b LaurentPolynomial<T>> for LaurentPolynomial<T> {
    type Output = LaurentPolynomial<T>;
    #[verified_engine::verified]
    fn mul(self, rhs: &'b LaurentPolynomial<T>) -> LaurentPolynomial<T> {
        convolve(&self, rhs)
    }
}

impl<T: Ring> Mul<LaurentPolynomial<T>> for &LaurentPolynomial<T> {
    type Output = LaurentPolynomial<T>;
    #[verified_engine::verified]
    fn mul(self, rhs: LaurentPolynomial<T>) -> LaurentPolynomial<T> {
        convolve(self, &rhs)
    }
}

impl<T: Ring> MulAssign for LaurentPolynomial<T> {
    #[verified_engine::verified]
    fn mul_assign(&mut self, rhs: Self) {
        *self = convolve(self, &rhs);
    }
}

impl<T: Ring> MulAssign<&LaurentPolynomial<T>> for LaurentPolynomial<T> {
    #[verified_engine::verified]
    fn mul_assign(&mut self, rhs: &Self) {
        *self = convolve(self, rhs);
    }
}

impl<T: Ring> Neg for LaurentPolynomial<T> {
    type Output = Self;
    #[verified_engine::verified]
    fn neg(mut self) -> Self {
        let orig_len = self.coeffs.len();
        for c in &mut self.coeffs {
            *c = c.ring_ref_neg();
        }
        debug_assert_eq!(self.coeffs.len(), orig_len);
        self
    }
}

impl<T: Ring> Neg for &LaurentPolynomial<T> {
    type Output = LaurentPolynomial<T>;
    #[verified_engine::verified]
    fn neg(self) -> LaurentPolynomial<T> {
        let res = LaurentPolynomial {
            min_exp: self.min_exp,
            coeffs: self.coeffs.iter().map(Ring::ring_ref_neg).collect(),
        };
        debug_assert_eq!(res.coeffs.len(), self.coeffs.len());
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laurent_ring_laws_and_reciprocal_checks() {
        let a = LaurentPolynomial::new(-2, vec![1_i64, -3, 0, 2]);
        let b = LaurentPolynomial::new(-1, vec![2_i64, 5, -1]);
        let c = LaurentPolynomial::new(0, vec![-1_i64, 4]);

        assert_eq!((&a + &b) + &c, &a + (&b + &c));
        assert_eq!(&a + &b, &b + &a);
        assert_eq!((&a * &b) * &c, &a * (&b * &c));
        assert_eq!(&a * &b, &b * &a);

        assert_eq!(&a * (&b + &c), (&a * &b) + (&a * &c));
        assert_eq!(&a + (-&a), LaurentPolynomial::zero());
        assert_eq!(&a * LaurentPolynomial::one(), a);

        let t3 = LaurentPolynomial::monomial(1_i64, 3);
        let tm3 = LaurentPolynomial::monomial(1_i64, -3);
        assert_eq!(&t3 * &tm3, LaurentPolynomial::one());
        assert_eq!(a.shift(4).shift(-4), a);

        let fig8 = LaurentPolynomial::new(-1, vec![-1_i64, 3, -1]);
        assert!(fig8.is_reciprocal());
        assert!(!fig8.shift(2).is_reciprocal());
        assert!(fig8.shift(2).is_reciprocal_up_to_shift());
        assert_eq!(fig8.shift(5).scale(&-1).normalize_alexander(), fig8);
    }
}
