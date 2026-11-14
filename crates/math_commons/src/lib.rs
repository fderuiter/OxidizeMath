#![doc = include_str!("../README.md")]

//! Common mathematical kernels, primitives, constants, and registry standards.
//!
//! # Example: Coordinate flattening calculation
//! ```
//! use math_commons::math_kernel::types::flatten_2d_index;
//!
//! let index = flatten_2d_index(3, 2, 10);
//! assert_eq!(index, 23);
//! ```
//!
//! # Example: Heatmap color mapping contract
//! ```
//! use math_commons::math_kernel::colormap::heatmap_color;
//!
//! let blue = heatmap_color(0.0);
//! assert_eq!(blue, [0, 0, 255]);
//! let red = heatmap_color(1.0);
//! assert_eq!(red, [255, 0, 0]);
//! ```

pub mod constants;
#[allow(missing_docs)]
pub mod error;
#[allow(missing_docs)]
pub mod math_kernel;
#[allow(missing_docs)]
pub mod primitives;
pub mod registry;
#[allow(missing_docs)]
pub mod wigner;
