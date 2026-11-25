#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]
// [cite:engine_diffeq]

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
pub use math_memory::{ArenaScope, ByteArena, SolverWorkspace};
pub use math_rng::{BASE_SEED, OxidizeRng};
pub use milstein::MilsteinSolver;
pub use sriw1::{CorrelatedWienerIncrement, Sriw1Solver};
pub use system::{
    LangevinSystem, SdeSystem, StochasticCalculus, eval_ito_effective_drift,
    try_eval_ito_effective_drift,
};

#[cfg(test)]
mod serde_tests;
#[cfg(test)]
mod tests;
