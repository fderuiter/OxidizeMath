#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]
// [cite:engine_diffeq]

#[cfg(feature = "alloc")]
extern crate alloc;

/// Symmetric `B-A-O-A-B` underdamped Langevin dynamics solver.
pub mod baoab;
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
pub use euler_maruyama::EulerMaruyamaSolver;
pub use jump::ssa_sample_event;
pub use math_rng::{BASE_SEED, OxidizeRng};
pub use milstein::MilsteinSolver;
pub use sriw1::{CorrelatedWienerIncrement, Sriw1Solver};
pub use system::{
    LangevinSystem, SdeSystem, StochasticCalculus, eval_ito_effective_drift, ito_drift_scratch_len,
    try_eval_ito_effective_drift,
};

#[cfg(feature = "alloc")]
/// Optional `alloc`-backed scratch buffer owner that lends zero-allocation [`SolverWorkspace`](math_commons::math_kernel::types::SolverWorkspace) views.
#[derive(Debug, Clone)]
pub struct OwnedWorkspace<T: math_commons::math_kernel::types::Scalar> {
    buf: alloc::vec::Vec<T>,
}

#[cfg(feature = "alloc")]
impl<T: math_commons::math_kernel::types::Scalar> OwnedWorkspace<T> {
    /// Allocates a reusable scratch buffer of `capacity` zero-initialized scalars once up front.
    #[verified_engine::verified]
    pub fn with_capacity(capacity: usize) -> Self {
        let mut buf = alloc::vec::Vec::new();
        buf.resize(capacity, T::ZERO);
        assert_eq!(buf.len(), capacity);
        Self { buf }
    }

    /// Borrows this buffer as a zero-allocation [`SolverWorkspace`](math_commons::math_kernel::types::SolverWorkspace).
    #[inline]
    pub fn as_workspace(&mut self) -> math_commons::math_kernel::types::SolverWorkspace<'_, T> {
        math_commons::math_kernel::types::SolverWorkspace::new(&mut self.buf)
    }
}
