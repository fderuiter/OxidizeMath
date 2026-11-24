//! # Finite Fields ($\mathbb{F}_P$ and $\mathbb{F}_{P^n}$)
// [cite:math_scalars]
// [cite:advanced_linear_algebra]
//!
//! Re-exports compile-time validated prime fields [`Fp<const P: u64>`] backed by carry-safe
//! 128-bit Montgomery reduction and stack-allocated Galois extension fields [`FpExt`] from
//! [`math_scalars`], implementing [`Ring`] and [`Field`] for seamless integration with
//! [`crate::pure_math::algebra::Polynomial`] and elliptic-curve arithmetic.
//!
//! Composite moduli are rejected at compile time:
//!
//! ```compile_fail
//! use pure_math::pure_math::algebra::Fp;
//! let _bad = Fp::<15>::new(1);
//! ```

use crate::pure_math::algebra::traits::{Field, Ring};
pub use math_scalars::{Fp, FpExt, IrreducibleModulus, ScalarError, is_prime_u64};

impl<const P: u64> Ring for Fp<P> {
    #[verified_engine::verified]
    fn zero() -> Self {
        Self::ZERO
    }

    #[verified_engine::verified]
    fn one() -> Self {
        Self::ONE
    }

    #[verified_engine::verified]
    fn is_zero(&self) -> bool {
        Fp::is_zero(*self)
    }
}

impl<const P: u64> Field for Fp<P> {
    #[verified_engine::verified]
    fn multiplicative_inverse(&self) -> Self {
        self.checked_inverse().expect("Division by zero in Fp")
    }

    #[verified_engine::verified]
    fn try_multiplicative_inverse(&self) -> Result<Self, ScalarError> {
        self.checked_inverse()
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

    #[verified_engine::verified]
    fn is_zero(&self) -> bool {
        FpExt::is_zero(self)
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Field for FpExt<P, DEG, M> {
    #[verified_engine::verified]
    fn multiplicative_inverse(&self) -> Self {
        self.checked_inverse().expect("Division by zero in FpExt")
    }

    #[verified_engine::verified]
    fn try_multiplicative_inverse(&self) -> Result<Self, ScalarError> {
        self.checked_inverse()
    }
}
