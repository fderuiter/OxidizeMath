// [cite:modular_polynomials_review]
//! # Finite Fields ($\mathbb{F}_P$ and Galois Extensions $\mathbb{F}_{P^n}$)
//!
//! Compile-time prime-validated finite fields [`Fp<const P: u64>`] backed by carry-safe
//! 128-bit Montgomery reduction up to $2^{64} - 59$ (with explicit $P = 2$ path) and
//! stack-allocated Galois quotient fields [`FpExt`] from `math_scalars`.

use crate::error::NumberTheoryError;
use crate::pure_math::algebra::traits::{Field, Ring};
pub use math_scalars::{Fp, FpExt, IrreducibleModulus, ScalarError};

impl<const P: u64> Ring for Fp<P> {
    #[verified_engine::verified]
    fn zero() -> Self {
        Self::ZERO
    }

    #[verified_engine::verified]
    fn one() -> Self {
        Self::ONE
    }
}

impl<const P: u64> Field for Fp<P> {
    #[verified_engine::verified]
    fn multiplicative_inverse(&self) -> Self {
        self.try_inverse()
            .expect("Division by zero in Fp<P>; use checked_inverse or try_inverse")
    }

    #[verified_engine::verified]
    fn checked_inverse(&self) -> Result<Self, NumberTheoryError> {
        self.try_inverse().map_err(|err| match err {
            ScalarError::DivisionByZero => NumberTheoryError::DivisionByZeroFieldElement,
            _ => NumberTheoryError::NoModularInverse,
        })
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Ring for FpExt<P, DEG, M> {
    #[verified_engine::verified]
    fn zero() -> Self {
        Self::ZERO
    }

    #[verified_engine::verified]
    fn one() -> Self {
        Self::ONE
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Field for FpExt<P, DEG, M> {
    #[verified_engine::verified]
    fn multiplicative_inverse(&self) -> Self {
        self.try_inverse().expect(
            "Non-invertible element in FpExt<P, DEG, M>; use checked_inverse or try_inverse",
        )
    }

    #[verified_engine::verified]
    fn checked_inverse(&self) -> Result<Self, NumberTheoryError> {
        self.try_inverse().map_err(|err| match err {
            ScalarError::DivisionByZero => NumberTheoryError::DivisionByZeroFieldElement,
            _ => NumberTheoryError::NoModularInverse,
        })
    }
}
