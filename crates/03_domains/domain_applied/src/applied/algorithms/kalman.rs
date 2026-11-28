//! Generic discrete-time linear and Extended Kalman Filter (`EKF`) re-exports from [`pure_math`],
//! plus stack-resident `FastEkf`, `FastCvKalman`, and closed-loop nonlinear plant observers.
// [cite:algorithms]

/// Zero-allocation stack Extended Kalman Filter (`FastEkf`) and RK4 variational tangent integrator.
pub mod ekf;
/// Zero-allocation 2-state Constant-Velocity Kalman filter (`FastCvKalman`).
pub mod fast_cv;
/// Nonlinear `KalmanSystem` observers and analytical ODE Jacobians for Lorenz, SEIR, and PK systems.
pub mod models;

pub use ekf::{FastEkf, MIN_MEASUREMENT_VARIANCE, rk4_step_with_jacobian};
pub use fast_cv::FastCvKalman;
pub use models::{
    LorenzKalmanModel, PkKalmanModel, SeirKalmanModel, lorenz_rhs_and_jac, pk_rhs_and_jac,
    seir_rhs_and_jac,
};
pub use pure_math::pure_math::analysis::kalman::{
    KalmanError, KalmanFilter, KalmanFilterBuilder, KalmanModel, KalmanSystem,
};
