#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]
// [cite:engine_pde]

#[cfg(feature = "alloc")]
extern crate alloc;

/// Boundary condition specifications and ghost-cell lookup helpers for 1D, 2D, and 3D grids.
pub mod boundary;
/// Stack-allocated `#![no_std]` error types for continuum PDE and SPDE solvers.
pub mod error;
/// `#![no_std]` zero-allocation radix-2 1D/2D Cooley-Tukey FFT and Orszag 2/3-rule dealiasing.
pub mod fft;
mod fourier_kernels;
/// Radix-2 semi-implicit Fourier SPDE stepper for Cahn-Hilliard, KPZ, and Edwards-Wilkinson fields.
pub mod fourier_stepper;
/// Deterministic coupled 1D, 2D, and 3D fused finite-difference stencil steppers.
pub mod fused_stepper;
/// Spatio-temporal additive and staggered-face conserved-flux SPDE noise injectors.
pub mod spde_noise;
/// Explicit and operator-split SPDE stencil steppers with additive and conserved-flux noise.
pub mod spde_stencil;
/// 1D, 2D, and 3D Laplacian, gradient, upwind, and public 2D Biharmonic stencil operators.
pub mod stencil;

pub use boundary::BoundaryCondition;
pub use error::PdeError;
pub use fft::{
    dealias_1d_spectrum, dealias_2d_spectrum, fft_1d_inplace, fft_2d_inplace,
    try_dealias_1d_spectrum, try_dealias_2d_spectrum, try_fft_1d_inplace, try_fft_2d_inplace,
};
pub use fourier_stepper::SemiImplicitFourierStepper;
pub use fused_stepper::FusedStencilStepper;
pub use math_rng::{BASE_SEED, OxidizeRng};
pub use spde_noise::{
    SpdeNoiseKind, inject_additive_noise_1d, inject_additive_noise_2d,
    inject_conserved_flux_noise_1d, inject_conserved_flux_noise_2d, try_inject_additive_noise_1d,
    try_inject_additive_noise_2d, try_inject_conserved_flux_noise_1d,
    try_inject_conserved_flux_noise_2d,
};
pub use spde_stencil::{ExplicitSpdeStencilStepper, SplitStepSpdeStencilStepper};
pub use stencil::{StencilNeighbors3D, StencilOperators};

#[cfg(feature = "alloc")]
/// Optional `alloc`-backed scratch buffer owner lending zero-allocation [`SolverWorkspace`](math_commons::math_kernel::types::SolverWorkspace) views.
#[derive(Debug, Clone)]
pub struct PdeOwnedWorkspace<T: math_commons::math_kernel::types::Scalar> {
    buf: alloc::vec::Vec<T>,
}

#[cfg(feature = "alloc")]
impl<T: math_commons::math_kernel::types::Scalar> PdeOwnedWorkspace<T> {
    /// Fallibly allocates a zero-initialized scratch buffer of `capacity` scalars.
    ///
    /// # Errors
    /// Returns [`PdeError::WorkspaceAllocationFailed`] if the allocator rejects `capacity`.
    #[verified_engine::verified]
    pub fn try_with_capacity(capacity: usize) -> Result<Self, PdeError> {
        let mut buf = alloc::vec::Vec::new();
        buf.try_reserve_exact(capacity)
            .map_err(|_| PdeError::WorkspaceAllocationFailed { capacity })?;
        buf.resize(capacity, T::ZERO);
        debug_assert_eq!(buf.len(), capacity);
        Ok(Self { buf })
    }

    /// Allocates a zero-initialized scratch buffer of `capacity` scalars once up front.
    ///
    /// # Panics
    /// Panics if heap allocation fails; prefer [`Self::try_with_capacity`] for fallible callers.
    #[verified_engine::verified]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::try_with_capacity(capacity).expect("PdeOwnedWorkspace::with_capacity failed")
    }

    /// Borrows this buffer as a zero-allocation [`SolverWorkspace`](math_commons::math_kernel::types::SolverWorkspace).
    #[inline]
    pub fn as_workspace(&mut self) -> math_commons::math_kernel::types::SolverWorkspace<'_, T> {
        math_commons::math_kernel::types::SolverWorkspace::new(&mut self.buf)
    }
}

#[cfg(test)]
mod serde_tests;
#[cfg(test)]
mod tests;
