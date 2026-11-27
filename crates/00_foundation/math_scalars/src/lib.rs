// [cite:math_scalars]
#![no_std]
#![doc = include_str!("../README.md")]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(any(test, feature = "std"))]
extern crate std;

/// `Complex<T>` implementations of [`RefOps`], [`Scalar`], [`Ring`], [`Field`], and [`ComplexField`].
pub mod complex;
/// Forward-mode dual number [`Dual`] for exact first-order automatic differentiation.
pub mod dual;
/// Stack-allocated `#![no_std]` [`ScalarError`] diagnostics.
pub mod error;
/// `0`-ULP `libm`-backed IEEE-754 kernels (`f32`, `f64`) and primitive integer [`Ring`] impls.
pub mod ieee;
/// Zero-copy 2D `(x, y)` coordinate series traits [`Sample2D`] and [`PointSeries`].
pub mod series;
/// Unified [`RefOps`], [`Scalar`], [`Ring`], [`Field`], [`ComplexField`], and [`RealField`] traits.
pub mod traits;
/// Caller-owned [`SolverWorkspace`], [`StepSize`], [`GridIndex`], [`Dimension`], and indexers.
pub mod workspace;

pub use complex::{Complex, Complex32, Complex64};
pub use dual::Dual;
pub use error::ScalarError;
pub use ieee::{hypot_f32, hypot_f64, sqrt_f32, sqrt_f64};
pub use series::{PointSeries, Sample2D};
pub use traits::{ComplexField, Field, RealField, RefOps, Ring, Scalar};
#[cfg(feature = "alloc")]
pub use workspace::OwnedWorkspace;
pub use workspace::{
    Dimension, GridIndex, SolverWorkspace, StepSize, flatten_2d_index, flatten_3d_index,
};

#[cfg(test)]
mod tests;
