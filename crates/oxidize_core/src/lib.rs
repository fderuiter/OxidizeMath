#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]
#[cfg(feature = "ast-analysis")]
#[allow(missing_docs)]
pub mod ast_visitor;
#[allow(missing_docs)]
pub mod boundary;
#[allow(missing_docs)]
pub mod iteration;
#[cfg(feature = "std")]
#[allow(missing_docs)]
pub mod mesh;
#[cfg(feature = "std")]
#[allow(missing_docs)]
pub mod path_utils;
#[allow(missing_docs)]
pub mod rng;
#[cfg(feature = "std")]
#[allow(missing_docs)]
pub mod traceability;
#[cfg(feature = "std")]
#[allow(missing_docs)]
pub mod vfs;
#[cfg(feature = "std")]
#[allow(missing_docs)]
pub mod workspace;
