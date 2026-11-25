//! Ring and arithmetic operator implementations for [`LaurentPolynomial`].

// [cite:knot_braid_topology]

use super::laurent::LaurentPolynomial;
use crate::pure_math::algebra::traits::{Ring, Scalar};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

impl<T: Ring> Scalar for LaurentPolynomial<T> {}

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
}

#[verified_engine::verified]
fn combine_linear<T: Ring>(
    lhs: &LaurentPolynomial<T>,
    rhs: &LaurentPolynomial<T>,
    subtract: bool,
) -> LaurentPolynomial<T> {
    if lhs.coeffs.is_empty() {
        return if subtract { -rhs } else { rhs.clone() };
    }
    if rhs.coeffs.is_empty() {
        return lhs.clone();
    }
    let min_e = lhs.min_exp.min(rhs.min_exp);
    let max_lhs = lhs.min_exp + (lhs.coeffs.len() as i32) - 1;
    let max_rhs = rhs.min_exp + (rhs.coeffs.len() as i32) - 1;
    let max_e = max_lhs.max(max_rhs);
    let len = (max_e - min_e) as usize + 1;
    let mut out = vec![T::zero(); len];

    let off_l = (lhs.min_exp - min_e) as usize;
    for (i, c) in lhs.coeffs.iter().enumerate() {
        out[off_l + i] += c.clone();
    }
    let off_r = (rhs.min_exp - min_e) as usize;
    for (i, c) in rhs.coeffs.iter().enumerate() {
        if subtract {
            out[off_r + i] -= c.clone();
        } else {
            out[off_r + i] += c.clone();
        }
    }
    let res = LaurentPolynomial::new(min_e, out);
    debug_assert!(res.coeffs.is_empty() || !res.coeffs[0].is_zero());
    debug_assert!(res.coeffs.len() <= len);
    res
}

#[verified_engine::verified]
fn convolve<T: Ring>(
    lhs: &LaurentPolynomial<T>,
    rhs: &LaurentPolynomial<T>,
) -> LaurentPolynomial<T> {
    if lhs.coeffs.is_empty() || rhs.coeffs.is_empty() {
        return LaurentPolynomial::zero();
    }
    let out_len = lhs.coeffs.len() + rhs.coeffs.len() - 1;
    let mut out = vec![T::zero(); out_len];
    for (i, c1) in lhs.coeffs.iter().enumerate() {
        if c1.is_zero() {
            continue;
        }
        for (j, c2) in rhs.coeffs.iter().enumerate() {
            out[i + j] += c1.clone() * c2.clone();
        }
    }
    let res = LaurentPolynomial::new(lhs.min_exp + rhs.min_exp, out);
    debug_assert!(res.coeffs.len() <= out_len);
    debug_assert!(res.coeffs.is_empty() || !res.coeffs[0].is_zero());
    res
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
            *c = -std::mem::replace(c, T::zero());
        }
        debug_assert_eq!(self.coeffs.len(), orig_len);
        self
    }
}

impl<T: Ring> Neg for &LaurentPolynomial<T> {
    type Output = LaurentPolynomial<T>;
    #[verified_engine::verified]
    fn neg(self) -> LaurentPolynomial<T> {
        let res = LaurentPolynomial::new(
            self.min_exp,
            self.coeffs.iter().map(|c| -c.clone()).collect(),
        );
        debug_assert!(res.coeffs.len() <= self.coeffs.len());
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

        // Associativity and commutativity
        assert_eq!((&a + &b) + &c, &a + (&b + &c));
        assert_eq!(&a + &b, &b + &a);
        assert_eq!((&a * &b) * &c, &a * (&b * &c));
        assert_eq!(&a * &b, &b * &a);

        // Distributivity, additive inverse, and multiplicative identity
        assert_eq!(&a * (&b + &c), (&a * &b) + (&a * &c));
        assert_eq!(&a + (-&a), LaurentPolynomial::zero());
        assert_eq!(&a * LaurentPolynomial::one(), a);

        // Monomial inverse identity t^3 * t^-3 == 1
        let t3 = LaurentPolynomial::monomial(1_i64, 3);
        let tm3 = LaurentPolynomial::monomial(1_i64, -3);
        assert_eq!(&t3 * &tm3, LaurentPolynomial::one());
        assert_eq!(a.shift(4).shift(-4), a);

        // Reciprocal check on figure-eight Alexander polynomial -t^-1 + 3 - t
        let fig8 = LaurentPolynomial::new(-1, vec![-1_i64, 3, -1]);
        assert!(fig8.is_reciprocal());
        assert!(!fig8.shift(2).is_reciprocal());
        assert!(fig8.shift(2).is_reciprocal_up_to_shift());
        assert_eq!(fig8.shift(5).scale(&-1).normalize_alexander(), fig8);
    }
}
