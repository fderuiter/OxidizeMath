#![no_std]
#![doc = include_str!("../README.md")]
// [cite:math_scalars]
// [cite:advanced_linear_algebra]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod ball;
pub mod dual;
pub mod dual_vec;
pub mod hyper_dual;
pub mod interval;
mod interval_elem;
mod primitive_impls;
pub mod rounding;
pub mod traits;

#[cfg(test)]
mod tests;

pub use ball::Ball;
pub use dual::Dual;
pub use dual_vec::DualVec;
pub use hyper_dual::HyperDual;
pub use interval::Interval;
pub use rounding::{
    next_down_f32, next_down_f64, next_up_f32, next_up_f64, step_down_f32, step_down_f64,
    step_up_f32, step_up_f64, ulp_distance_f32, ulp_distance_f64,
};
pub use traits::{RealScalar, Scalar};
