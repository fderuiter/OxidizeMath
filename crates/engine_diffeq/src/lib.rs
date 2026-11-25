#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]
// [cite:engine_diffeq]

#[cfg(feature = "alloc")]
extern crate alloc;

/// Symmetric `B-A-O-A-B` underdamped Langevin dynamics solver.
pub mod baoab;
/// Stack-allocated `#![no_std]` error types for differential equation solvers.
pub mod error;
/// Multivariate Itô and Stratonovich Euler-Maruyama / Euler-Heun SDE solver.
pub mod euler_maruyama;
/// Zero-allocation discrete stochastic jump (Gillespie SSA) selection helpers.
pub mod jump;
/// Multivariate Itô and Stratonovich Milstein strong-order-1.0 SDE solver.
pub mod milstein;
/// Derivative-free strong-order-1.5 Stochastic Runge-Kutta (`SRIW1`) SDE solver.
pub mod sriw1;
/// Multivariate SDE and Langevin system traits and calculus definitions.
pub mod system;

pub use baoab::{BaoabLangevinSolver, BaoabParams};
pub use error::DiffeqError;
pub use euler_maruyama::EulerMaruyamaSolver;
pub use jump::ssa_sample_event;
pub use math_rng::{BASE_SEED, OxidizeRng};
pub use math_scalars::{Dual, DualVec, HyperDual, RealScalar, Scalar};
pub use milstein::MilsteinSolver;
pub use sriw1::{CorrelatedWienerIncrement, Sriw1Solver};
pub use system::{
    LangevinSystem, SdeSystem, StochasticCalculus, eval_ito_effective_drift,
    try_eval_ito_effective_drift,
};

#[cfg(feature = "alloc")]
/// Optional `alloc`-backed scratch buffer owner that lends zero-allocation [`SolverWorkspace`](math_commons::math_kernel::types::SolverWorkspace) views.
#[derive(Debug, Clone)]
pub struct OwnedWorkspace<T: math_scalars::Scalar> {
    buf: alloc::vec::Vec<T>,
}

#[cfg(feature = "alloc")]
impl<T: math_scalars::Scalar> OwnedWorkspace<T> {
    /// Fallibly allocates a reusable scratch buffer of `capacity` zero-initialized scalars.
    ///
    /// # Errors
    /// Returns [`DiffeqError::WorkspaceAllocationFailed`] if the allocator rejects `capacity`.
    #[verified_engine::verified]
    pub fn try_with_capacity(capacity: usize) -> Result<Self, DiffeqError> {
        let mut buf = alloc::vec::Vec::new();
        buf.try_reserve_exact(capacity)
            .map_err(|_| DiffeqError::WorkspaceAllocationFailed { capacity })?;
        buf.resize(capacity, T::ZERO);
        debug_assert_eq!(buf.len(), capacity);
        Ok(Self { buf })
    }

    /// Allocates a reusable scratch buffer of `capacity` zero-initialized scalars once up front.
    ///
    /// # Panics
    /// Panics if heap allocation fails; prefer [`Self::try_with_capacity`] for fallible callers.
    #[verified_engine::verified]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::try_with_capacity(capacity).expect("OwnedWorkspace::with_capacity allocation failed")
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
