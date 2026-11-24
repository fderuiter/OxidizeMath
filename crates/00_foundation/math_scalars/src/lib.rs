#![no_std]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
#![doc = include_str!("../README.md")]
// [cite:math_scalars]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

/// Arbitrary-precision integers, exact rationals, and precision-preserving floats (`alloc`).
#[cfg(feature = "alloc")]
pub mod big;
/// Flat `#[repr(C)]` complex field elements `Complex<T>`.
pub mod complex;
/// Typed `#![no_std]` error enum for scalar, modular, polynomial, and $p$-adic operations.
pub mod error;
/// Compile-time validated prime fields `Fp<const P: u64>` via carry-safe Montgomery reduction.
pub mod fp;
/// Stack-polynomial Galois extension fields `FpExt` ($\mathbb{F}_{p^n}$).
pub mod galois;
/// Fixed-width 256-bit signed two's-complement integer `I256`.
pub mod int256;
/// Carry-safe runtime 64-bit modular arithmetic and Tonelli-Shanks square-root helpers.
pub mod modular;
/// Flat `#[repr(C)]` non-associative Cayley-Dickson octonion division algebra `Octonion<T>`.
pub mod octonion;
/// Fixed-precision stack $p$-adic numbers `Qp<const P: u64, const N: usize>` with Hensel lifting.
pub mod padic;
/// Stack-backed univariate polynomials `StackPoly` and `EuclideanDomain` division.
pub mod poly;
/// Flat `#[repr(C)]` Hamilton quaternion division ring `Quaternion<T>`.
pub mod quaternion;
/// Unordered `FieldScalar` and ordered `RealScalar` trait hierarchy.
pub mod traits;
/// Fixed-width 256-bit unsigned integer `U256`.
pub mod uint256;
/// Unit-norm 3D rotation quaternion `UnitQuaternion<T>`.
pub mod unit_quaternion;

#[cfg(feature = "alloc")]
pub use big::{BigFloat, BigInt, BigRational};
pub use complex::{Complex, Complex32, Complex64};
pub use error::ScalarError;
pub use fp::{Fp, is_prime_u64};
pub use galois::{FpExt, IrreducibleModulus};
pub use int256::I256;
pub use octonion::Octonion;
pub use padic::Qp;
pub use poly::{EuclideanDomain, PolyCoeff, StackPoly};
pub use quaternion::Quaternion;
pub use traits::{FieldScalar, RealScalar};
pub use uint256::U256;
pub use unit_quaternion::UnitQuaternion;
