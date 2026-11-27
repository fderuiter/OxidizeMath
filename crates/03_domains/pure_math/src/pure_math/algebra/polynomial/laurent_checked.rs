//! Checked arithmetic helpers and validated deserialization for [`LaurentPolynomial`].

// [cite:knot_braid_topology]

use super::laurent::LaurentPolynomial;
use crate::pure_math::algebra::traits::Ring;
use std::collections::BTreeMap;
use thiserror::Error;

/// Maximum dense coefficient span (`coeffs.len()`) allowed for a [`LaurentPolynomial`],
/// preventing multi-gigabyte allocations on adversarial sparse exponent gaps.
pub const MAX_LAURENT_SPAN: usize = 65_536;

/// Typed arithmetic and deserialization errors for [`LaurentPolynomial`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LaurentError {
    /// Exponent calculation overflowed signed 32-bit integer bounds.
    #[error("Laurent polynomial exponent overflowed signed 32-bit integer bounds")]
    ExponentOverflow,
    /// Dense degree span exceeded [`MAX_LAURENT_SPAN`].
    #[error("Laurent polynomial coefficient span {span} exceeds maximum {max}")]
    SpanTooLarge {
        /// Requested dense coefficient buffer length.
        span: usize,
        /// Configured maximum dense span.
        max: usize,
    },
    /// Signed integer coefficient arithmetic overflowed target bounds.
    #[error("Laurent polynomial integer coefficient overflowed signed bounds")]
    CoefficientOverflow,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLaurentPolynomial<T> {
    min_exp: i32,
    coeffs: Vec<T>,
}

impl<'de, T> serde::Deserialize<'de> for LaurentPolynomial<T>
where
    T: Ring + serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawLaurentPolynomial::<T>::deserialize(deserializer)?;
        if raw.coeffs.len() > MAX_LAURENT_SPAN {
            return Err(serde::de::Error::custom(LaurentError::SpanTooLarge {
                span: raw.coeffs.len(),
                max: MAX_LAURENT_SPAN,
            }));
        }
        Self::try_new(raw.min_exp, raw.coeffs).map_err(serde::de::Error::custom)
    }
}

#[verified_engine::verified]
pub(super) fn try_new_impl<T: Ring>(
    min_exp: i32,
    mut coeffs: Vec<T>,
) -> Result<LaurentPolynomial<T>, LaurentError> {
    while coeffs.last().is_some_and(Ring::is_zero) {
        coeffs.pop();
    }
    let lead_zeros = coeffs.iter().take_while(|c| c.is_zero()).count();
    if lead_zeros == coeffs.len() {
        return Ok(LaurentPolynomial::zero());
    }
    if lead_zeros > 0 {
        coeffs.drain(..lead_zeros);
    }
    if coeffs.len() > MAX_LAURENT_SPAN {
        return Err(LaurentError::SpanTooLarge {
            span: coeffs.len(),
            max: MAX_LAURENT_SPAN,
        });
    }
    let lead_shift = i32::try_from(lead_zeros).map_err(|_| LaurentError::ExponentOverflow)?;
    let adj_min = min_exp
        .checked_add(lead_shift)
        .ok_or(LaurentError::ExponentOverflow)?;
    let span_i32 = i32::try_from(coeffs.len() - 1).map_err(|_| LaurentError::ExponentOverflow)?;
    let _max_deg = adj_min
        .checked_add(span_i32)
        .ok_or(LaurentError::ExponentOverflow)?;
    debug_assert!(!coeffs[0].is_zero());
    Ok(LaurentPolynomial {
        min_exp: adj_min,
        coeffs,
    })
}

#[verified_engine::verified]
pub(super) fn checked_span_len(lo: i32, hi: i32) -> Result<usize, LaurentError> {
    let span_i64 = i64::from(hi) - i64::from(lo) + 1;
    if span_i64 > (MAX_LAURENT_SPAN as i64) {
        return Err(LaurentError::SpanTooLarge {
            span: usize::try_from(span_i64).unwrap_or(usize::MAX),
            max: MAX_LAURENT_SPAN,
        });
    }
    usize::try_from(span_i64).map_err(|_| LaurentError::ExponentOverflow)
}

#[verified_engine::verified]
pub(super) fn try_from_terms_impl<T: Ring>(
    terms: &[(i32, T)],
) -> Result<LaurentPolynomial<T>, LaurentError> {
    let mut accum: BTreeMap<i32, T> = BTreeMap::new();
    for (exp, coeff) in terms {
        if !coeff.is_zero() {
            accum
                .entry(*exp)
                .and_modify(|v| *v = v.ring_ref_add(coeff))
                .or_insert_with(|| coeff.clone());
        }
    }
    accum.retain(|_, v| !v.is_zero());
    let Some((&lo, _)) = accum.first_key_value() else {
        return Ok(LaurentPolynomial::zero());
    };
    let (&hi, _) = accum.last_key_value().expect("non-empty map");
    let len = checked_span_len(lo, hi)?;
    let mut coeffs = vec![T::zero(); len];
    for (exp, coeff) in accum {
        let idx = usize::try_from(i64::from(exp) - i64::from(lo))
            .map_err(|_| LaurentError::ExponentOverflow)?;
        coeffs[idx] = coeff;
    }
    LaurentPolynomial::try_new(lo, coeffs)
}

#[verified_engine::verified]
pub(super) fn try_shift_impl<T: Ring>(
    poly: &LaurentPolynomial<T>,
    delta_exp: i32,
) -> Result<LaurentPolynomial<T>, LaurentError> {
    if poly.coeffs.is_empty() {
        return Ok(LaurentPolynomial::zero());
    }
    let new_min = poly
        .min_exp
        .checked_add(delta_exp)
        .ok_or(LaurentError::ExponentOverflow)?;
    let span_i32 =
        i32::try_from(poly.coeffs.len() - 1).map_err(|_| LaurentError::ExponentOverflow)?;
    let _new_max = new_min
        .checked_add(span_i32)
        .ok_or(LaurentError::ExponentOverflow)?;
    debug_assert!(!poly.coeffs[0].is_zero());
    Ok(LaurentPolynomial {
        min_exp: new_min,
        coeffs: poly.coeffs.clone(),
    })
}

#[verified_engine::verified]
pub(super) fn try_invert_variable_impl<T: Ring>(
    poly: &LaurentPolynomial<T>,
) -> Result<LaurentPolynomial<T>, LaurentError> {
    let Some(max_deg) = poly.max_degree() else {
        return Ok(LaurentPolynomial::zero());
    };
    let new_min = max_deg
        .checked_neg()
        .ok_or(LaurentError::ExponentOverflow)?;
    let _new_max = poly
        .min_exp
        .checked_neg()
        .ok_or(LaurentError::ExponentOverflow)?;
    let mut rev = poly.coeffs.clone();
    rev.reverse();
    debug_assert!(!rev[0].is_zero());
    Ok(LaurentPolynomial {
        min_exp: new_min,
        coeffs: rev,
    })
}

#[verified_engine::verified]
pub(super) fn try_eval_at_one_impl(poly: &LaurentPolynomial<i64>) -> Result<i64, LaurentError> {
    let mut acc = 0_i64;
    for &c in &poly.coeffs {
        acc = acc
            .checked_add(c)
            .ok_or(LaurentError::CoefficientOverflow)?;
    }
    Ok(acc)
}

#[verified_engine::verified]
pub(super) fn try_normalize_alexander_impl(
    poly: &LaurentPolynomial<i64>,
) -> Result<LaurentPolynomial<i64>, LaurentError> {
    if poly.coeffs.is_empty() {
        return Ok(LaurentPolynomial::zero());
    }
    let span_i32 = i32::try_from(poly.span()).map_err(|_| LaurentError::ExponentOverflow)?;
    let target_min = -(span_i32 / 2);
    let sum_wide: i128 = poly.coeffs.iter().map(|&c| i128::from(c)).sum();
    let negate = if sum_wide != 0 {
        sum_wide < 0
    } else {
        poly.coeffs.last().is_some_and(|&c| c < 0)
    };
    let coeffs = if negate {
        let mut neg_coeffs = Vec::with_capacity(poly.coeffs.len());
        for &c in &poly.coeffs {
            neg_coeffs.push(c.checked_neg().ok_or(LaurentError::CoefficientOverflow)?);
        }
        neg_coeffs
    } else {
        poly.coeffs.clone()
    };
    LaurentPolynomial::try_new(target_min, coeffs)
}

impl<T: Ring> LaurentPolynomial<T> {
    /// Fallibly computes `self + rhs`, validating exponent bounds and [`MAX_LAURENT_SPAN`].
    ///
    /// # Errors
    /// Returns [`LaurentError`] if the combined degree span exceeds [`MAX_LAURENT_SPAN`] or `i32`.
    #[verified_engine::verified]
    pub fn try_add(&self, rhs: &Self) -> Result<Self, LaurentError> {
        super::laurent_ops::try_combine_linear(self, rhs, false)
    }

    /// Fallibly computes `self - rhs`, validating exponent bounds and [`MAX_LAURENT_SPAN`].
    ///
    /// # Errors
    /// Returns [`LaurentError`] if the combined degree span exceeds [`MAX_LAURENT_SPAN`] or `i32`.
    #[verified_engine::verified]
    pub fn try_sub(&self, rhs: &Self) -> Result<Self, LaurentError> {
        super::laurent_ops::try_combine_linear(self, rhs, true)
    }

    /// Fallibly computes `self * rhs`, validating exponent bounds and [`MAX_LAURENT_SPAN`].
    ///
    /// # Errors
    /// Returns [`LaurentError`] if the product exponents overflow `i32` or exceed [`MAX_LAURENT_SPAN`].
    #[verified_engine::verified]
    pub fn try_mul(&self, rhs: &Self) -> Result<Self, LaurentError> {
        super::laurent_ops::try_convolve(self, rhs)
    }
}

impl LaurentPolynomial<i64> {
    /// Fallibly constructs a [`LaurentPolynomial<i64>`] by summing `(exponent, coefficient)` terms
    /// with checked `i64` addition and `i32` exponent validation.
    ///
    /// # Errors
    /// Returns [`LaurentError`] on `i64` coefficient overflow, `i32` exponent overflow, or oversized span.
    #[verified_engine::verified]
    pub fn checked_from_terms(terms: &[(i32, i64)]) -> Result<Self, LaurentError> {
        let mut accum: BTreeMap<i32, i64> = BTreeMap::new();
        for &(exp, coeff) in terms {
            if coeff == 0 {
                continue;
            }
            let next = match accum.get(&exp) {
                Some(&prev) => prev
                    .checked_add(coeff)
                    .ok_or(LaurentError::CoefficientOverflow)?,
                None => coeff,
            };
            if next == 0 {
                accum.remove(&exp);
            } else {
                accum.insert(exp, next);
            }
        }
        let Some((&lo, _)) = accum.first_key_value() else {
            return Ok(Self::zero());
        };
        let (&hi, _) = accum.last_key_value().expect("non-empty map");
        let len = checked_span_len(lo, hi)?;
        let mut coeffs = vec![0_i64; len];
        for (exp, coeff) in accum {
            let idx = usize::try_from(i64::from(exp) - i64::from(lo))
                .map_err(|_| LaurentError::ExponentOverflow)?;
            coeffs[idx] = coeff;
        }
        Self::try_new(lo, coeffs)
    }

    /// Fallibly negates every `i64` coefficient.
    ///
    /// # Errors
    /// Returns [`LaurentError::CoefficientOverflow`] if any coefficient equals `i64::MIN`.
    #[verified_engine::verified]
    pub fn checked_neg(&self) -> Result<Self, LaurentError> {
        let mut out = Vec::with_capacity(self.coeffs.len());
        for &c in &self.coeffs {
            out.push(c.checked_neg().ok_or(LaurentError::CoefficientOverflow)?);
        }
        debug_assert_eq!(out.len(), self.coeffs.len());
        Ok(Self {
            min_exp: self.min_exp,
            coeffs: out,
        })
    }

    /// Fallibly computes `self + rhs` with checked `i64` coefficients and `i32` exponents.
    ///
    /// # Errors
    /// Returns [`LaurentError`] on coefficient, exponent, or span overflow.
    #[verified_engine::verified]
    pub fn checked_add(&self, rhs: &Self) -> Result<Self, LaurentError> {
        super::laurent_ops::checked_combine_i64(self, rhs, false)
    }

    /// Fallibly computes `self - rhs` with checked `i64` coefficients and `i32` exponents.
    ///
    /// # Errors
    /// Returns [`LaurentError`] on coefficient, exponent, or span overflow.
    #[verified_engine::verified]
    pub fn checked_sub(&self, rhs: &Self) -> Result<Self, LaurentError> {
        super::laurent_ops::checked_combine_i64(self, rhs, true)
    }

    /// Fallibly computes `self * rhs` with checked `i64` coefficients and `i32` exponents.
    ///
    /// # Errors
    /// Returns [`LaurentError`] on coefficient, exponent, or span overflow.
    #[verified_engine::verified]
    pub fn checked_mul(&self, rhs: &Self) -> Result<Self, LaurentError> {
        super::laurent_ops::checked_convolve_i64(self, rhs)
    }
}
