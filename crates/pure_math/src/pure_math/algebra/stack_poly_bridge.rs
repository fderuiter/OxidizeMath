// [cite:modular_polynomials_review]
//! Bridge implementations connecting stack-allocated [`StackPoly`] to [`Ring`] and [`EuclideanDomain`].

use crate::pure_math::algebra::traits::{EuclideanDomain, Ring};
use math_scalars::{PolyField, StackPoly};

impl<T: PolyField, const CAP: usize> Ring for StackPoly<T, CAP> {
    #[verified_engine::verified]
    fn zero() -> Self {
        Self::default()
    }

    #[verified_engine::verified]
    fn one() -> Self {
        Self::constant(T::ONE).expect("StackPoly::one requires CAP >= 1")
    }
}

impl<T: PolyField, const CAP: usize> EuclideanDomain for StackPoly<T, CAP> {}
