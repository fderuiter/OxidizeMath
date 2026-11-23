#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]
// [cite:math_commons]

#[cfg(feature = "std")]
#[allow(missing_docs)]
pub mod ast;
pub mod constants;
#[cfg(feature = "std")]
#[allow(missing_docs)]
pub mod error;
#[cfg(feature = "std")]
pub mod fit;
#[allow(missing_docs)]
pub mod math_kernel;
#[cfg(feature = "std")]
pub mod poi;
#[cfg(feature = "std")]
#[allow(missing_docs)]
pub mod primitives;
pub mod registry;
#[cfg(feature = "std")]
pub mod series;
#[cfg(feature = "std")]
#[allow(missing_docs)]
pub mod wigner;
