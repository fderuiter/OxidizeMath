#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! Re-exports the unified `#![no_std]` [`math_scalars::RealField`] trait as [`Scalar`] alongside
//! `0`-ULP `libm`-backed square-root and hypotenuse primitives.

pub use math_scalars::{
    ComplexField, Field, RealField, RealField as Scalar, RefOps, Ring, hypot_f32, hypot_f64,
    sqrt_f32, sqrt_f64,
};
