//! Generic Discrete-Time Linear and Extended Kalman Filter Re-exports.
//!
//! Re-exports [`KalmanError`], [`KalmanFilter`], [`KalmanFilterBuilder`],
//! [`KalmanModel`], and [`KalmanSystem`] from Tier 2 `engine_diffeq`.

pub use engine_diffeq::KalmanError;
pub use engine_diffeq::kalman::{KalmanFilter, KalmanFilterBuilder, KalmanModel, KalmanSystem};
