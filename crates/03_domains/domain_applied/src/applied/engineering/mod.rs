//! Closed-loop stabilization and control primitives (`upo_control`).
// [cite:graph_parameters_rust]

/// Viswanath UPO reference trajectory and OGY Rayleigh-parameter EKF control primitives.
pub mod upo_control;

pub use upo_control::*;
