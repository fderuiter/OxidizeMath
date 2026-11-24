//! Re-exports unified finite-difference stencil and SPDE/Fourier steppers from [`engine_pde`].
// [cite:pure_math]

pub use engine_pde::{
    BoundaryCondition, ExplicitSpdeStencilStepper, FusedStencilStepper, SemiImplicitFourierStepper,
    SpdeNoiseKind, SplitStepSpdeStencilStepper, StencilNeighbors3D, StencilOperators,
};

#[cfg(test)]
#[path = "fused_stepper_tests.rs"]
mod tests;
