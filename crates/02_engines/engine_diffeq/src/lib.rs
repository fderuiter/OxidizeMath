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
/// Zero-allocation 2-state Constant-Velocity scalar Kalman filter (`FastCvKalman`).
pub mod fast_cv;
/// Zero-allocation discrete stochastic jump (Gillespie SSA) selection helpers.
pub mod jump;
/// Generic discrete-time linear and Extended Kalman Filter (`EKF`) state estimation.
#[cfg(feature = "nalgebra")]
pub mod kalman;
/// Multivariate Itô and Stratonovich Milstein strong-order-1.0 SDE solver.
pub mod milstein;
/// Derivative-free strong-order-1.5 Stochastic Runge-Kutta (`SRIW1`) SDE solver.
pub mod sriw1;
/// Multivariate SDE and Langevin system traits and calculus definitions.
pub mod system;

pub use baoab::{BaoabLangevinSolver, BaoabParams};
pub use error::{DiffeqError, KalmanError};
pub use euler_maruyama::EulerMaruyamaSolver;
pub use fast_cv::FastCvKalman;
pub use jump::ssa_sample_event;
#[cfg(feature = "nalgebra")]
pub use kalman::{KalmanFilter, KalmanFilterBuilder, KalmanModel, KalmanSystem};
pub use math_commons::math_kernel::types::ScalarWorkspace;
#[cfg(feature = "alloc")]
pub use math_memory::OwnedByteArena;
pub use math_memory::{ArenaCheckpoint, ArenaError, ByteArena, ByteCursor, SolverWorkspace};
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
