#![no_std]
#![doc = include_str!("../README.md")]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:advanced_linear_algebra]
// [cite:modular_polynomials_review]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

/// Arbitrary-precision `alloc`-backed integer, rational, and floating-point types.
#[cfg(feature = "alloc")]
pub mod bigint;
/// Flat-coordinate `#[repr(C)]` complex algebra implementing [`FieldScalar`].
pub mod complex;
/// Stack-allocated `#![no_std]` error taxonomy.
pub mod error;
/// Stack-allocated 256-bit unsigned integer [`U256`].
pub mod fixed256;
/// Stack-allocated 256-bit two's-complement signed integer [`I256`].
pub mod fixed256_signed;
/// Compile-time validated prime field $\mathbb{F}_P$ using carry-safe 128-bit Montgomery REDC.
pub mod fp;
/// Stack-allocated Galois extension fields $\mathbb{F}_{P^{\text{DEG}}} \cong \mathbb{F}_P\[x\]/(m(x))$.
pub mod fp_ext;
/// Carry-safe 64-bit modular arithmetic, primality testing, and Tonelli-Shanks helpers.
pub mod modular;
/// Flat-coordinate `#[repr(C)]` Cayley-Dickson octonion alternative division algebra.
pub mod octonion;
/// Fixed-precision stack $p$-adic numbers $\mathbb{Q}_P$ and Newton-Hensel root lifting.
pub mod padic;
/// Stack-backed univariate polynomial ring with Euclidean division and extended GCD.
pub mod poly;
/// Flat-coordinate `#[repr(C)]` Hamilton quaternion division ring and $SO(3)$ [`UnitQuaternion`].
pub mod quaternion;
#[cfg(feature = "serde")]
mod serde_impl;
/// Unordered [`FieldScalar`] and ordered [`RealScalar`] scalar contracts.
pub mod traits;

#[cfg(feature = "alloc")]
pub use bigint::{BigFloat, BigInt, BigRational};
pub use complex::{Complex, Complex32, Complex64};
pub use error::ScalarError;
pub use fixed256::U256;
pub use fixed256_signed::I256;
pub use fp::Fp;
pub use fp_ext::{FpExt, IrreducibleModulus};
pub use modular::{
    MAX_PRIME_U64, is_prime_u64, mod_add_u64, mod_inv_u64, mod_mul_u64, mod_neg_u64, mod_pow_u64,
    mod_reduce_i64, mod_reduce_i128, mod_sub_u64, tonelli_shanks_u64,
};
pub use octonion::Octonion;
pub use padic::{Qp, eval_int_poly_qp, hensel_lift_root};
pub use poly::{PolyField, StackPoly};
pub use quaternion::{Quaternion, UnitQuaternion};
pub use traits::{FieldScalar, RealScalar};
