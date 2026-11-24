#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]
// [cite:math_commons]

extern crate alloc;

#[cfg(feature = "std")]
#[allow(missing_docs)]
pub mod ast;
pub mod constants;
#[allow(missing_docs)]
pub mod error;
pub mod fit;
#[allow(missing_docs)]
pub mod math_kernel;
pub mod poi;
#[allow(missing_docs)]
pub mod primitives;
pub mod registry;
pub mod series;
pub mod stochastic;
#[allow(missing_docs)]
pub mod wigner;

#[cfg(test)]
mod serde_tests;
