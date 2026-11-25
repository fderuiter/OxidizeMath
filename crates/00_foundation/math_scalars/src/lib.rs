#![no_std]
#![doc = include_str!("../README.md")]
// [cite:math_commons]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(test)]
extern crate std;

pub mod ball;
pub mod directed;
pub mod dual;
pub mod dual_vec;
pub mod enclosure_cert;
pub mod hyper_dual;
pub mod interval;
mod interval_transcendental;
#[cfg(feature = "serde")]
mod serde_impls;
pub mod traits;
mod traits_f32;
pub mod ulp;

pub use ball::Ball;
pub use directed::DirectedRounding;
pub use dual::Dual;
pub use dual_vec::DualVec;
pub use enclosure_cert::{
    CertifiedBallRoot, CertifiedIntervalRoot, ReferenceEnclosureSummary, certify_reference_roots,
    interval_newton_enclose, krawczyk_enclose,
};
pub use hyper_dual::HyperDual;
pub use interval::Interval;
pub use traits::{RealScalar, Scalar};
pub use ulp::{
    next_down_f32, next_down_f64, next_up_f32, next_up_f64, pad_ulp_down_f32, pad_ulp_down_f64,
    pad_ulp_up_f32, pad_ulp_up_f64, ulp_distance_f32, ulp_distance_f64,
};

#[cfg(test)]
mod tests;
