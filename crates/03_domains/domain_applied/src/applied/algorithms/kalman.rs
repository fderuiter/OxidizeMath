//! Generic discrete-time linear and Extended Kalman Filter (`EKF`) re-exports from [`pure_math`].
// [cite:algorithms]

pub use pure_math::pure_math::analysis::kalman::{
    KalmanError, KalmanFilter, KalmanFilterBuilder, KalmanModel, KalmanSystem,
};
