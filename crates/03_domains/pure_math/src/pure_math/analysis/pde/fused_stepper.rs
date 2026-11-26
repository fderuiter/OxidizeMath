//! Re-exports unified finite-difference stencil and SPDE/Fourier steppers from [`engine_pde`].
// [cite:pure_math]

pub use engine_pde::*;

#[cfg(test)]
#[path = "fused_stepper_tests.rs"]
mod tests;
