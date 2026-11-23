//! Temporal Tracking with Kalman Filter.
//!
//! Implements a **Discrete Kalman Filter** using the **Strategy Pattern** to allow interchangeable
//! physics models (e.g., Constant Velocity, Constant Acceleration).
//!
//! **Update (2026-06-XX):** This module now uses the generic `KalmanFilter` implementation from
//! `applied::algorithms::kalman`, adapted for the specific domain of Radar Gating.

use crate::error::RadarError;
use domain_applied::applied::algorithms::kalman::{KalmanError, KalmanFilter, KalmanModel};
use nalgebra::{DMatrix, DVector};

/// Constant Velocity (CV) Model.
/// State: [Position, Velocity]
#[derive(Debug, Clone)]
pub struct ConstantVelocityModel {
    #[allow(missing_docs)]
    pub process_noise_var: f64,
    #[allow(missing_docs)]
    pub measurement_noise_var: f64,
}

impl ConstantVelocityModel {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(process_noise_var: f64, measurement_noise_var: f64) -> Self {
        Self {
            process_noise_var,
            measurement_noise_var,
        }
    }
}

impl KalmanModel<f64> for ConstantVelocityModel {
    #[verified_engine::verified]
    fn transition_matrix(&self, dt: f64) -> DMatrix<f64> {
        // [1, dt]
        // [0, 1 ]
        DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0])
    }

    #[verified_engine::verified]
    fn measurement_matrix(&self) -> DMatrix<f64> {
        // [1, 0] -> Measure position only
        DMatrix::from_row_slice(1, 2, &[1.0, 0.0])
    }

    #[verified_engine::verified]
    fn process_noise(&self, _dt: f64) -> DMatrix<f64> {
        // Simplified Q: I * var
        DMatrix::identity(2, 2) * self.process_noise_var
    }

    #[verified_engine::verified]
    fn measurement_noise(&self) -> DMatrix<f64> {
        // R is 1x1 matrix for scalar measurement
        DMatrix::from_row_slice(1, 1, &[self.measurement_noise_var])
    }
}

/// A wrapper around the generic Kalman Filter for 1D position and velocity tracking.
///
/// This maintains the original API of the radar gating module while using the unified core algorithm.
#[derive(Debug, Clone)]
pub struct TrackingFilter {
    inner: KalmanFilter<f64, ConstantVelocityModel>,
}

impl TrackingFilter {
    /// Fallibly constructs a 1D radar-gating Kalman filter.
    ///
    /// Returns [`RadarError::InvalidTrackingParameter`] if `initial_amplitude` is non-finite,
    /// `dt <= 0.0`, `model.process_noise_var < 0.0`, or `model.measurement_noise_var < 0.0`.
    #[verified_engine::verified]
    pub fn try_new(
        initial_amplitude: f64,
        dt: f64,
        model: ConstantVelocityModel,
    ) -> Result<Self, RadarError> {
        if !initial_amplitude.is_finite() {
            return Err(RadarError::InvalidTrackingParameter {
                parameter: "initial_amplitude",
                value: initial_amplitude,
            });
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(RadarError::InvalidTrackingParameter {
                parameter: "dt",
                value: dt,
            });
        }
        if !model.process_noise_var.is_finite() || model.process_noise_var < 0.0 {
            return Err(RadarError::InvalidTrackingParameter {
                parameter: "process_noise_var",
                value: model.process_noise_var,
            });
        }
        if !model.measurement_noise_var.is_finite() || model.measurement_noise_var < 0.0 {
            return Err(RadarError::InvalidTrackingParameter {
                parameter: "measurement_noise_var",
                value: model.measurement_noise_var,
            });
        }

        let initial_state = DVector::from_vec(vec![initial_amplitude, 0.0]);
        let initial_covariance = DMatrix::identity(2, 2);

        let inner = KalmanFilter::builder(model, dt)
            .initial_state(initial_state)
            .initial_covariance(initial_covariance)
            .build()
            .map_err(|err| match err {
                KalmanError::Math(math_err) => RadarError::Math(math_err),
                other => RadarError::InvalidConfiguration(other.to_string()),
            })?;

        Ok(Self { inner })
    }

    /// Creates a new Kalman Filter with a specific physics model.
    ///
    /// # Arguments
    /// * `initial_amplitude` - Initial position guess.
    /// * `dt` - Time step in seconds.
    /// * `model` - The physics model implementation (e.g., `ConstantVelocityModel`).
    #[verified_engine::verified]
    pub fn new(initial_amplitude: f64, dt: f64, model: ConstantVelocityModel) -> Self {
        Self::try_new(initial_amplitude, dt, model)
            .unwrap_or_else(|e| panic!("TrackingFilter encountered invalid parameters: {e}"))
    }

    /// Predicts the next state.
    #[verified_engine::verified]
    pub fn predict(&mut self) {
        self.inner.predict();
    }

    /// Updates the state with a new measurement.
    #[verified_engine::verified]
    pub fn update(&mut self, measured_amplitude: f64) {
        // Optimization: Use from_column_slice to avoid extra allocation if vector was larger,
        // though here we just need a 1-element vector. from_element is cleaner.
        let measurement = DVector::from_element(1, measured_amplitude);
        // Error handling: In this controlled domain (scalar update), inversion should rarely fail given R > 0.
        // If it fails (KalmanError::MatrixInversionError), we simply skip the update to maintain robustness
        // rather than crashing the real-time tracking loop.
        if let Err(e) = self.inner.update(&measurement) {
            // In a real system, we might log this: warn!("Kalman update failed: {:?}", e);
            let _ = e;
        }
    }

    /// Returns the current estimated amplitude (position).
    #[verified_engine::verified]
    pub fn amplitude(&self) -> f64 {
        self.inner.state[0]
    }

    /// Returns the current estimated velocity.
    #[verified_engine::verified]
    pub fn velocity(&self) -> f64 {
        self.inner.state[1]
    }

    /// Returns the current state covariance matrix.
    ///
    /// Useful for visualizing uncertainty or debugging filter convergence.
    #[verified_engine::verified]
    pub fn covariance(&self) -> &DMatrix<f64> {
        &self.inner.covariance
    }
}
