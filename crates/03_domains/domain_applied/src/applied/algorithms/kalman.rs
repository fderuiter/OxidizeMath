//! Generic Discrete-Time Linear and Extended Kalman Filter Re-exports and Plant Models.
//!
//! Re-exports [`FastCvKalman`], [`KalmanError`], [`KalmanFilter`], [`KalmanFilterBuilder`],
//! [`KalmanModel`], and [`KalmanSystem`] from Tier 2 `engine_diffeq` alongside stack `FastEkf`
//! and nonlinear ODE observer models.
// [cite:algorithms]

/// Zero-allocation stack Extended Kalman Filter (`FastEkf`) and RK4 variational tangent integrator.
pub mod ekf;
/// Nonlinear `KalmanSystem` observers and analytical ODE Jacobians for Lorenz, SEIR, and PK systems.
pub mod models;

pub use ekf::{FastEkf, MIN_MEASUREMENT_VARIANCE, rk4_step_with_jacobian};
pub use engine_diffeq::kalman::{KalmanFilter, KalmanFilterBuilder, KalmanModel, KalmanSystem};
pub use engine_diffeq::{FastCvKalman, KalmanError, fast_cv};
pub use models::{
    LorenzKalmanModel, PkKalmanModel, SeirKalmanModel, lorenz_rhs_and_jac, pk_rhs_and_jac,
    seir_rhs_and_jac,
};
