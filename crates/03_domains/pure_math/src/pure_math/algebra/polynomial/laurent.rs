//! Laurent polynomial ring $R[t, t^{-1}]$ with signed integer exponents.

// [cite:knot_braid_topology]

pub use super::laurent_checked::{LaurentError, MAX_LAURENT_SPAN};
use crate::pure_math::algebra::polynomial::Polynomial;
use crate::pure_math::algebra::traits::Ring;
use std::fmt;

/// A univariate Laurent polynomial $\sum_{k=m}^{M} c_k t^k$ with coefficients in a [`Ring`] `T`.
///
/// Canonical representation invariants:
/// - The zero Laurent polynomial has `min_exp == 0` and `coeffs.is_empty()`.
/// - Any non-zero Laurent polynomial has non-zero first (`min_exp`) and last
///   (`min_exp + coeffs.len() as i32 - 1`) coefficients.
///
/// # Example
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::algebra::LaurentPolynomial;
///
/// // Alexander polynomial of the trefoil knot: t^-1 - 1 + t
/// let trefoil = LaurentPolynomial::new(-1, vec![1_i64, -1, 1]);
/// assert_eq!(trefoil.min_degree(), Some(-1));
/// assert_eq!(trefoil.max_degree(), Some(1));
/// assert!(trefoil.is_reciprocal());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct LaurentPolynomial<T: Ring> {
    /// Lowest exponent $m \in \mathbb{Z}$ with non-zero coefficient (`0` when identically zero).
    pub(crate) min_exp: i32,
    /// Contiguous coefficient slice where `coeffs[i]` multiplies $t^{\text{min\_exp} + i}$.
    pub(crate) coeffs: Vec<T>,
}

impl<T: Ring> Default for LaurentPolynomial<T> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<T: Ring> LaurentPolynomial<T> {
    /// Returns the lowest stored exponent (`0` for the zero polynomial).
    #[must_use]
    #[verified_engine::verified]
    pub const fn min_exp(&self) -> i32 {
        self.min_exp
    }

    /// Returns a read-only slice of contiguous coefficients starting at [`Self::min_exp`].
    #[must_use]
    #[verified_engine::verified]
    pub fn coeffs(&self) -> &[T] {
        &self.coeffs
    }

    /// Fallibly constructs and normalizes a [`LaurentPolynomial`] starting at `min_exp`.
    ///
    /// # Errors
    /// Returns [`LaurentError::ExponentOverflow`] if trimming or `max_degree` overflows `i32`,
    /// or [`LaurentError::SpanTooLarge`] if the trimmed dense support exceeds [`MAX_LAURENT_SPAN`].
    #[verified_engine::verified]
    pub fn try_new(min_exp: i32, coeffs: Vec<T>) -> Result<Self, LaurentError> {
        super::laurent_checked::try_new_impl(min_exp, coeffs)
    }

    /// Constructs a canonicalized Laurent polynomial starting at `min_exp`.
    ///
    /// # Panics
    /// Panics if exponents overflow `i32` or the trimmed span exceeds [`MAX_LAURENT_SPAN`];
    /// prefer [`Self::try_new`] in fallible contexts.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(min_exp: i32, coeffs: Vec<T>) -> Self {
        Self::try_new(min_exp, coeffs).expect("LaurentPolynomial::new exponent or span overflow")
    }

    /// Fallibly constructs a [`LaurentPolynomial`] by summing `(exponent, coefficient)` terms.
    ///
    /// # Errors
    /// Returns [`LaurentError`] if the resulting exponent span exceeds [`MAX_LAURENT_SPAN`] or `i32`.
    #[verified_engine::verified]
    pub fn try_from_terms(terms: &[(i32, T)]) -> Result<Self, LaurentError> {
        super::laurent_checked::try_from_terms_impl(terms)
    }

    /// Constructs a Laurent polynomial by summing `(exponent, coefficient)` terms.
    ///
    /// # Panics
    /// Panics if the resulting degree span exceeds [`MAX_LAURENT_SPAN`] or `i32`;
    /// prefer [`Self::try_from_terms`] in fallible contexts.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_terms(terms: &[(i32, T)]) -> Self {
        Self::try_from_terms(terms)
            .expect("LaurentPolynomial::from_terms exponent or span overflow")
    }

    /// Fallibly embeds a non-negative-degree [`Polynomial`] into $R[t, t^{-1}]$.
    ///
    /// # Errors
    /// Propagates [`LaurentError`] from [`Self::try_new`].
    #[verified_engine::verified]
    pub fn try_from_polynomial(poly: Polynomial<T>) -> Result<Self, LaurentError> {
        Self::try_new(0, poly.coeffs)
    }

    /// Embeds a non-negative-degree [`Polynomial`] into $R[t, t^{-1}]$.
    ///
    /// # Panics
    /// Panics if `poly` exceeds [`MAX_LAURENT_SPAN`]; prefer [`Self::try_from_polynomial`].
    #[must_use]
    #[verified_engine::verified]
    pub fn from_polynomial(poly: Polynomial<T>) -> Self {
        Self::try_from_polynomial(poly).expect("LaurentPolynomial::from_polynomial span overflow")
    }

    /// Constructs a single monomial $c \cdot t^e$.
    #[must_use]
    #[verified_engine::verified]
    pub fn monomial(coeff: T, exp: i32) -> Self {
        if coeff.is_zero() {
            Self::zero()
        } else {
            Self {
                min_exp: exp,
                coeffs: vec![coeff],
            }
        }
    }

    /// Constructs a constant Laurent polynomial $c \cdot t^0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn constant(coeff: T) -> Self {
        Self::monomial(coeff, 0)
    }

    /// Lowest exponent with a non-zero coefficient, or `None` for the zero polynomial.
    #[must_use]
    #[verified_engine::verified]
    pub fn min_degree(&self) -> Option<i32> {
        (!self.coeffs.is_empty()).then_some(self.min_exp)
    }

    /// Highest exponent with a non-zero coefficient, or `None` for the zero polynomial.
    #[must_use]
    #[verified_engine::verified]
    pub fn max_degree(&self) -> Option<i32> {
        let last_offset = self.coeffs.len().checked_sub(1)?;
        let delta = i32::try_from(last_offset).ok()?;
        self.min_exp.checked_add(delta)
    }

    /// Exponent breadth `max_degree - min_degree` (`0` for zero or single-monomial polynomials).
    #[must_use]
    #[verified_engine::verified]
    pub fn span(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// Returns the coefficient of $t^{\text{exp}}$ (`T::zero()` outside support).
    #[must_use]
    #[verified_engine::verified]
    pub fn coeff(&self, exp: i32) -> T {
        if exp < self.min_exp {
            return T::zero();
        }
        let diff = i64::from(exp) - i64::from(self.min_exp);
        usize::try_from(diff)
            .ok()
            .and_then(|offset| self.coeffs.get(offset))
            .cloned()
            .unwrap_or_else(T::zero)
    }

    /// Fallibly multiplies by $t^{\Delta}$, shifting all exponents by `delta_exp`.
    ///
    /// # Errors
    /// Returns [`LaurentError::ExponentOverflow`] if `min_degree` or `max_degree` overflows `i32`.
    #[verified_engine::verified]
    pub fn try_shift(&self, delta_exp: i32) -> Result<Self, LaurentError> {
        super::laurent_checked::try_shift_impl(self, delta_exp)
    }

    /// Multiplies by $t^{\Delta}$, shifting all exponents by `delta_exp`.
    ///
    /// # Panics
    /// Panics if the shifted exponents overflow `i32`; prefer [`Self::try_shift`].
    #[must_use]
    #[verified_engine::verified]
    pub fn shift(&self, delta_exp: i32) -> Self {
        self.try_shift(delta_exp)
            .expect("LaurentPolynomial::shift exponent overflow")
    }

    /// Fallibly substitutes $t \mapsto t^{-1}$, sending $\sum c_k t^k$ to $\sum c_k t^{-k}$.
    ///
    /// # Errors
    /// Returns [`LaurentError::ExponentOverflow`] when `-min_degree` or `-max_degree` overflows `i32`.
    #[verified_engine::verified]
    pub fn try_invert_variable(&self) -> Result<Self, LaurentError> {
        super::laurent_checked::try_invert_variable_impl(self)
    }

    /// Substitutes $t \mapsto t^{-1}$, sending $\sum c_k t^k$ to $\sum c_k t^{-k}$.
    ///
    /// # Panics
    /// Panics if `-min_degree` or `-max_degree` overflows `i32`; prefer [`Self::try_invert_variable`].
    #[must_use]
    #[verified_engine::verified]
    pub fn invert_variable(&self) -> Self {
        self.try_invert_variable()
            .expect("LaurentPolynomial::invert_variable exponent overflow")
    }

    /// Returns `true` when $P(t) = P(t^{-1})$ identically (palindromic around degree `0`).
    #[must_use]
    #[verified_engine::verified]
    pub fn is_reciprocal(&self) -> bool {
        if self.coeffs.is_empty() {
            return true;
        }
        let Some(max_deg) = self.max_degree() else {
            return false;
        };
        i64::from(self.min_exp) + i64::from(max_deg) == 0 && self.is_reciprocal_up_to_shift()
    }

    /// Returns `true` when $P(t^{-1}) = t^k P(t)$ for some integer shift $k \in \mathbb{Z}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_reciprocal_up_to_shift(&self) -> bool {
        let n = self.coeffs.len();
        for i in 0..(n / 2) {
            if self.coeffs[i] != self.coeffs[n - 1 - i] {
                return false;
            }
        }
        true
    }

    /// Scales every coefficient by `scalar`.
    #[must_use]
    #[verified_engine::verified]
    pub fn scale(&self, scalar: &T) -> Self {
        if self.coeffs.is_empty() || scalar.is_zero() {
            return Self::zero();
        }
        let scaled = self.coeffs.iter().map(|c| c.ring_ref_mul(scalar)).collect();
        Self::new(self.min_exp, scaled)
    }

    /// Evaluates $P(x)$ given a ring unit `x` and its multiplicative inverse `x_inv`.
    #[must_use]
    #[verified_engine::verified]
    pub fn eval_with_inverse(&self, x: T, x_inv: T) -> T {
        if self.coeffs.is_empty() {
            return T::zero();
        }
        let mut horner = T::zero();
        for c in self.coeffs.iter().rev() {
            horner = horner.ring_ref_mul(&x).ring_ref_add(c);
        }
        let base = if self.min_exp >= 0 { x } else { x_inv };
        let factor = pow_ring_u32(base, self.min_exp.unsigned_abs());
        horner * factor
    }
}

#[verified_engine::verified]
fn pow_ring_u32<T: Ring>(mut base: T, mut exp: u32) -> T {
    let mut acc = T::one();
    while exp > 0 {
        if (exp & 1) == 1 {
            acc = acc.ring_ref_mul(&base);
        }
        exp >>= 1;
        if exp > 0 {
            base = base.ring_ref_mul(&base);
        }
    }
    acc
}

impl LaurentPolynomial<i64> {
    /// Evaluates $P(1) = \sum_k c_k$ using checked `i64` addition.
    ///
    /// # Errors
    /// Returns [`LaurentError::CoefficientOverflow`] if any partial sum overflows `i64`.
    #[verified_engine::verified]
    pub fn try_eval_at_one(&self) -> Result<i64, LaurentError> {
        super::laurent_checked::try_eval_at_one_impl(self)
    }

    /// Evaluates $P(1) = \sum_k c_k$ over `i64`.
    ///
    /// # Panics
    /// Panics if the sum overflows `i64`; prefer [`Self::try_eval_at_one`].
    #[must_use]
    #[verified_engine::verified]
    pub fn eval_at_one(&self) -> i64 {
        self.try_eval_at_one()
            .expect("LaurentPolynomial::eval_at_one coefficient sum overflow")
    }

    /// Returns `true` when $P(t^{-1}) = \pm t^k P(t)$ for some unit $\pm t^k \in \mathbb{Z}[t, t^{-1}]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_reciprocal_up_to_unit(&self) -> bool {
        if self.is_reciprocal_up_to_shift() {
            return true;
        }
        let n = self.coeffs.len();
        (0..n).all(|i| self.coeffs[n - 1 - i].checked_neg() == Some(self.coeffs[i]))
    }

    /// Fallibly normalizes a Laurent polynomial modulo units $\pm t^k \in \mathbb{Z}[t, t^{-1}]$.
    ///
    /// # Errors
    /// Returns [`LaurentError::CoefficientOverflow`] if sign canonicalization requires negating `i64::MIN`.
    #[verified_engine::verified]
    pub fn try_normalize_alexander(&self) -> Result<Self, LaurentError> {
        super::laurent_checked::try_normalize_alexander_impl(self)
    }

    /// Normalizes a Laurent polynomial modulo units $\pm t^k \in \mathbb{Z}[t, t^{-1}]$.
    ///
    /// Centers the support at `min_exp = -(span / 2)` and fixes the global sign so
    /// $P(1) > 0$ (or, when $P(1) = 0$, the highest-degree non-zero coefficient is positive).
    ///
    /// # Panics
    /// Panics if negating a `i64::MIN` coefficient overflows; prefer [`Self::try_normalize_alexander`].
    #[must_use]
    #[verified_engine::verified]
    pub fn normalize_alexander(&self) -> Self {
        self.try_normalize_alexander()
            .expect("LaurentPolynomial::normalize_alexander coefficient overflow")
    }

    /// Formats the Laurent polynomial in ascending powers of `var` (e.g. `"t^-1 - 1 + t"`).
    #[must_use]
    pub fn format_in_var(&self, var: &str) -> String {
        format_integer_laurent(&self.coeffs, self.min_exp, var, false)
    }

    /// Formats a half-integer Laurent polynomial where index `k` represents `var^(k/2)`.
    #[must_use]
    pub fn format_half_power_in_var(&self, var: &str) -> String {
        format_integer_laurent(&self.coeffs, self.min_exp, var, true)
    }
}

fn format_integer_laurent(coeffs: &[i64], min_exp: i32, var: &str, half_powers: bool) -> String {
    if coeffs.is_empty() {
        return "0".to_string();
    }
    let mut out = String::new();
    for (idx, &c) in coeffs.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let exp = i64::from(min_exp) + (idx as i64);
        let abs_c = c.unsigned_abs();
        if out.is_empty() {
            if c < 0 {
                out.push('-');
            }
        } else if c < 0 {
            out.push_str(" - ");
        } else {
            out.push_str(" + ");
        }
        append_laurent_term(&mut out, abs_c, exp, var, half_powers);
    }
    out
}

fn append_laurent_term(out: &mut String, abs_c: u64, exp: i64, var: &str, half_powers: bool) {
    if exp == 0 {
        out.push_str(&abs_c.to_string());
        return;
    }
    if abs_c != 1 {
        out.push_str(&abs_c.to_string());
    }
    out.push_str(var);
    if half_powers {
        if exp % 2 == 0 {
            let whole = exp / 2;
            if whole != 1 {
                out.push('^');
                out.push_str(&whole.to_string());
            }
        } else {
            out.push_str(&format!("^({exp}/2)"));
        }
    } else if exp != 1 {
        out.push('^');
        out.push_str(&exp.to_string());
    }
}

impl fmt::Display for LaurentPolynomial<i64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format_in_var("t"))
    }
}

impl<T: Ring> From<Polynomial<T>> for LaurentPolynomial<T> {
    #[verified_engine::verified]
    fn from(value: Polynomial<T>) -> Self {
        Self::from_polynomial(value)
    }
}
