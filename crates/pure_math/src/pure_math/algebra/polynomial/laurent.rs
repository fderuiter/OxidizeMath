//! Laurent polynomial ring $R[t, t^{-1}]$ with signed integer exponents.

// [cite:knot_braid_topology]

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
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LaurentPolynomial<T: Ring> {
    /// Lowest exponent $m \in \mathbb{Z}$ with non-zero coefficient (`0` when identically zero).
    pub min_exp: i32,
    /// Contiguous coefficient slice where `coeffs[i]` multiplies $t^{\text{min\_exp} + i}$.
    pub coeffs: Vec<T>,
}

impl<T: Ring> Default for LaurentPolynomial<T> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<T: Ring> LaurentPolynomial<T> {
    /// Constructs a canonicalized Laurent polynomial starting at `min_exp`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(min_exp: i32, coeffs: Vec<T>) -> Self {
        let mut poly = Self { min_exp, coeffs };
        poly.normalize_trim();
        debug_assert!(poly.coeffs.is_empty() || !poly.coeffs[0].is_zero());
        poly
    }

    /// Constructs a Laurent polynomial by summing `(exponent, coefficient)` terms.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_terms(terms: &[(i32, T)]) -> Self {
        let mut non_zero_bounds: Option<(i32, i32)> = None;
        for (exp, coeff) in terms {
            if !coeff.is_zero() {
                non_zero_bounds = Some(match non_zero_bounds {
                    None => (*exp, *exp),
                    Some((lo, hi)) => (lo.min(*exp), hi.max(*exp)),
                });
            }
        }
        let Some((lo, hi)) = non_zero_bounds else {
            return Self::zero();
        };
        let len = (hi - lo) as usize + 1;
        let mut coeffs = vec![T::zero(); len];
        for (exp, coeff) in terms {
            if !coeff.is_zero() {
                let idx = (*exp - lo) as usize;
                coeffs[idx] += coeff.clone();
            }
        }
        Self::new(lo, coeffs)
    }

    /// Embeds a non-negative-degree [`Polynomial`] into $R[t, t^{-1}]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_polynomial(poly: Polynomial<T>) -> Self {
        Self::new(0, poly.coeffs)
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

    /// Strips leading and trailing zero coefficients while preserving exponent alignment.
    #[verified_engine::verified]
    pub(crate) fn normalize_trim(&mut self) {
        while self.coeffs.last().is_some_and(Ring::is_zero) {
            self.coeffs.pop();
        }
        let lead_zeros = self.coeffs.iter().take_while(|c| c.is_zero()).count();
        if lead_zeros > 0 {
            self.coeffs.drain(..lead_zeros);
            self.min_exp = self.min_exp.saturating_add(lead_zeros as i32);
        }
        if self.coeffs.is_empty() {
            self.min_exp = 0;
        }
        debug_assert!(
            (self.coeffs.is_empty() && self.min_exp == 0)
                || (!self.coeffs[0].is_zero() && !self.coeffs[self.coeffs.len() - 1].is_zero())
        );
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
        (!self.coeffs.is_empty()).then(|| self.min_exp + (self.coeffs.len() as i32) - 1)
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
        let offset = (exp - self.min_exp) as usize;
        self.coeffs.get(offset).cloned().unwrap_or_else(T::zero)
    }

    /// Multiplies by $t^{\Delta}$, shifting all exponents by `delta_exp`.
    #[must_use]
    #[verified_engine::verified]
    pub fn shift(&self, delta_exp: i32) -> Self {
        let out = if self.coeffs.is_empty() {
            Self::zero()
        } else {
            Self {
                min_exp: self.min_exp.saturating_add(delta_exp),
                coeffs: self.coeffs.clone(),
            }
        };
        debug_assert_eq!(out.coeffs.len(), self.coeffs.len());
        out
    }

    /// Substitutes $t \mapsto t^{-1}$, sending $\sum c_k t^k$ to $\sum c_k t^{-k}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn invert_variable(&self) -> Self {
        let Some(max_deg) = self.max_degree() else {
            return Self::zero();
        };
        let mut rev = self.coeffs.clone();
        rev.reverse();
        let out = Self {
            min_exp: -max_deg,
            coeffs: rev,
        };
        debug_assert_eq!(out.span(), self.span());
        out
    }

    /// Returns `true` when $P(t) = P(t^{-1})$ identically (palindromic around degree `0`).
    #[must_use]
    #[verified_engine::verified]
    pub fn is_reciprocal(&self) -> bool {
        *self == self.invert_variable()
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
        let scaled = self
            .coeffs
            .iter()
            .map(|c| c.clone() * scalar.clone())
            .collect();
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
            horner = horner * x.clone() + c.clone();
        }
        if self.min_exp >= 0 {
            let mut pow = T::one();
            for _ in 0..self.min_exp {
                pow *= x.clone();
            }
            horner * pow
        } else {
            let mut pow_inv = T::one();
            for _ in 0..(-self.min_exp) {
                pow_inv *= x_inv.clone();
            }
            horner * pow_inv
        }
    }
}

impl LaurentPolynomial<i64> {
    /// Evaluates $P(1) = \sum_k c_k$ over `i64`.
    #[must_use]
    #[verified_engine::verified]
    pub fn eval_at_one(&self) -> i64 {
        self.coeffs.iter().copied().sum()
    }

    /// Returns `true` when $P(t^{-1}) = \pm t^k P(t)$ for some unit $\pm t^k \in \mathbb{Z}[t, t^{-1}]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_reciprocal_up_to_unit(&self) -> bool {
        if self.is_reciprocal_up_to_shift() {
            return true;
        }
        let n = self.coeffs.len();
        (0..n).all(|i| self.coeffs[i] == -self.coeffs[n - 1 - i])
    }

    /// Normalizes a Laurent polynomial modulo units $\pm t^k \in \mathbb{Z}[t, t^{-1}]$.
    ///
    /// Centers the support at `min_exp = -(span / 2)` and fixes the global sign so
    /// $P(1) > 0$ (or, when $P(1) = 0$, the highest-degree non-zero coefficient is positive).
    #[must_use]
    #[verified_engine::verified]
    pub fn normalize_alexander(&self) -> Self {
        if self.coeffs.is_empty() {
            return Self::zero();
        }
        let target_min = -((self.span() as i32) / 2);
        let mut centered = self.shift(target_min - self.min_exp);
        let sum = centered.eval_at_one();
        let negate = if sum != 0 {
            sum < 0
        } else {
            centered.coeffs.last().is_some_and(|&c| c < 0)
        };
        if negate {
            centered = -centered;
        }
        debug_assert_eq!(centered.min_exp, target_min);
        debug_assert!(centered.eval_at_one() >= 0);
        centered
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
        let exp = min_exp + idx as i32;
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

fn append_laurent_term(out: &mut String, abs_c: u64, exp: i32, var: &str, half_powers: bool) {
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
