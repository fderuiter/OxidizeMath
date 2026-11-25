#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! Pure `#![no_std]` IEEE-754 and forward-AD real scalar contract unified on [`math_scalars::RealField`].

pub use math_scalars::RealField as Scalar;

/// Exact `0`-ULP `#![no_std]` IEEE-754 square root $\sqrt{x}$ for `f64` backed by [`math_scalars::RealField`].
/// `[cite:engine_linalg]`
#[inline(always)]
pub fn sqrt_f64(x: f64) -> f64 {
    <f64 as math_scalars::RealField>::sqrt(x)
}

/// Exact `#![no_std]` overflow- and underflow-safe Euclidean hypotenuse $\sqrt{x^2 + y^2}$ for `f64`.
/// `[cite:engine_linalg]`
#[inline(always)]
pub fn hypot_f64(x: f64, y: f64) -> f64 {
    <f64 as math_scalars::RealField>::hypot(x, y)
}
