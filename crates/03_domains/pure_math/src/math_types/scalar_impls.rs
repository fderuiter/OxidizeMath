//! Unified [`math_scalars`] algebraic trait implementations for arbitrary-precision `dashu` types.

use super::{Float, Integer, Rational};
use crate::pure_math::algebra::traits::{EuclideanDomain, Field, Ring};
use dashu::float::FBig;
use dashu::integer::IBig;
use dashu::rational::RBig;
use std::ops::{AddAssign, DivAssign, MulAssign, Neg, SubAssign};

// --- Integer ---

impl Neg for Integer {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Neg for &Integer {
    type Output = Integer;
    #[inline]
    fn neg(self) -> Integer {
        Integer(-&self.0)
    }
}

impl Ring for Integer {
    #[inline]
    fn zero() -> Self {
        Self(IBig::ZERO)
    }
    #[inline]
    fn one() -> Self {
        Self(IBig::ONE)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    #[inline]
    fn ring_ref_add(&self, rhs: &Self) -> Self {
        Self(&self.0 + &rhs.0)
    }
    #[inline]
    fn ring_ref_sub(&self, rhs: &Self) -> Self {
        Self(&self.0 - &rhs.0)
    }
    #[inline]
    fn ring_ref_mul(&self, rhs: &Self) -> Self {
        Self(&self.0 * &rhs.0)
    }
    #[inline]
    fn ring_ref_neg(&self) -> Self {
        Self(-&self.0)
    }
}

impl EuclideanDomain for Integer {}

// --- Rational ---

impl Neg for Rational {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Neg for &Rational {
    type Output = Rational;
    #[inline]
    fn neg(self) -> Rational {
        Rational(-&self.0)
    }
}

impl AddAssign for Rational {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Rational {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl MulAssign for Rational {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl DivAssign for Rational {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl Ring for Rational {
    #[inline]
    fn zero() -> Self {
        Self(RBig::ZERO)
    }
    #[inline]
    fn one() -> Self {
        Self(RBig::ONE)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    #[inline]
    fn ring_ref_add(&self, rhs: &Self) -> Self {
        Self(&self.0 + &rhs.0)
    }
    #[inline]
    fn ring_ref_sub(&self, rhs: &Self) -> Self {
        Self(&self.0 - &rhs.0)
    }
    #[inline]
    fn ring_ref_mul(&self, rhs: &Self) -> Self {
        Self(&self.0 * &rhs.0)
    }
    #[inline]
    fn ring_ref_neg(&self) -> Self {
        Self(-&self.0)
    }
}

impl Field for Rational {
    #[inline]
    fn ref_div(&self, rhs: &Self) -> Self {
        Self(&self.0 / &rhs.0)
    }
    #[inline]
    fn try_inv(&self) -> Option<Self> {
        (!self.0.is_zero()).then(|| Self(&RBig::ONE / &self.0))
    }
}

// --- Float ---

impl Neg for Float {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Neg for &Float {
    type Output = Float;
    #[inline]
    fn neg(self) -> Float {
        Float(-self.0.clone())
    }
}

impl MulAssign for Float {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl DivAssign for Float {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl Ring for Float {
    #[inline]
    fn zero() -> Self {
        Self(FBig::ZERO)
    }
    #[inline]
    fn one() -> Self {
        Self(FBig::ONE)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
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

impl Field for Float {
    #[inline]
    fn ref_div(&self, rhs: &Self) -> Self {
        self / rhs
    }
    #[inline]
    fn try_inv(&self) -> Option<Self> {
        (!self.is_zero()).then(|| Self::one().ref_div(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::algebra::traits::{RefOps, Scalar};
    use crate::pure_math::algebra::{Fp, LaurentPolynomial, Polynomial};

    fn assert_scalar<T: Scalar>(_: &T) {}

    fn eval_horner_ref<T: Ring>(coeffs: &[T], x: &T) -> T {
        let mut acc = T::zero();
        for c in coeffs.iter().rev() {
            acc = acc.ref_mul(x).ref_add(c);
        }
        acc
    }

    #[test]
    fn non_copy_dashu_and_fp_domains_interoperate_under_unified_ring_and_field() {
        let z0 = Integer::from(5_i64);
        assert_scalar(&z0);
        let coeffs_z = [z0, Integer::from(-3_i64), Integer::from(2_i64)];
        let val_z = eval_horner_ref(&coeffs_z, &Integer::from(10_i64));
        assert_eq!(val_z, Integer::from(175_i64));

        let half = Rational::from((1, 2));
        let third = Rational::from((1, 3));
        assert_eq!(half.ref_add(&third), Rational::from((5, 6)));
        assert_eq!(Field::inv(&Rational::from((3, 7))), Rational::from((7, 3)));
        assert!(Field::try_inv(&Rational::zero()).is_none());

        let poly_q = Polynomial::new(vec![Rational::from((1, 4)), Rational::from((3, 2))]);
        assert_eq!(poly_q.eval(Rational::from((2, 3))), Rational::from((5, 4)));

        let a = Float::with_val(128, 2.5);
        let inv_a = Field::inv(&a);
        assert!((a.ref_mul(&inv_a).ref_sub(&Float::one())).abs() < 1e-15);
        let two = Float::with_val(128, 2.0);
        assert_eq!(two.ref_mul(&Field::inv(&two)), Float::one());

        let f3 = Fp::<7>::new(3);
        assert_eq!(f3.ref_mul(&Field::inv(&f3)), Fp::<7>::one());
        let lp = LaurentPolynomial::new(-1, vec![Integer::from(1_i64), Integer::from(-2_i64)]);
        assert!(!Ring::is_zero(&lp));
    }
}
