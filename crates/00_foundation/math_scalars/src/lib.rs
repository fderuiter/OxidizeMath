#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]
// [cite:math_commons]

#[cfg(feature = "alloc")]
extern crate alloc;

/// Complex number trait implementations (`num_complex::Complex<T>`).
pub mod complex;
/// Explicit context-aware algebraic ring and field traits (`DomainRing`, `DomainField`, `StatelessContext`).
pub mod context;
/// Optional `alloc`-backed dynamic-precision `dashu` scalar and context implementations.
#[cfg(feature = "alloc")]
pub mod dashu_support;
/// Forward-mode automatic differentiation dual numbers (`Dual<T>`).
pub mod dual;
/// Stack-allocated `#![no_std]` error types for runtime-configured scalar domains.
pub mod error;
mod impls_float;
mod impls_int;
/// Runtime prime-modulus finite field context (`GF(p)` in `#![no_std]`).
pub mod mod_prime;
/// Stateless scalar and abstract-algebraic traits (`Scalar`, `Ring`, `Field`, `ComplexField`, `RealField`).
pub mod traits;

pub use complex::{Complex, Complex32, Complex64};
pub use context::{DomainField, DomainRing, StatelessContext};
#[cfg(feature = "alloc")]
pub use dashu_support::{DashuFloatContext, DashuModContext};
pub use dual::Dual;
pub use error::ScalarDomainError;
pub use mod_prime::ModPrimeContext;
pub use traits::{ComplexField, Field, RealField, Ring, Scalar};

/// Convenience prelude bringing both stateless and context-aware scalar traits into scope.
pub mod prelude {
    pub use crate::{
        Complex, Complex32, Complex64, ComplexField, DomainField, DomainRing, Dual, Field,
        ModPrimeContext, RealField, Ring, Scalar, ScalarDomainError, StatelessContext,
    };
    #[cfg(feature = "alloc")]
    pub use crate::{DashuFloatContext, DashuModContext};
}

#[cfg(test)]
mod tests;
