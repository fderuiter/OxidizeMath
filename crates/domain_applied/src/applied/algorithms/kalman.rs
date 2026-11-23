//! Generic Kalman Filter Implementation.
//!
//! A dimension-agnostic implementation of the Discrete Kalman Filter using `nalgebra::DMatrix`.
//! This module allows for state estimation of linear systems with arbitrary state and measurement dimensions.
use nalgebra::{DMatrix, DVector, RealField};
use thiserror::Error;
#[derive(Error, Debug, PartialEq, Clone)]
#[allow(missing_docs)]
pub enum KalmanError {
    #[error("Failed to invert innovation covariance matrix (singular)")]
    #[allow(missing_docs)]
    MatrixInversionError,
    #[allow(missing_docs)]
    #[error("Mathematical error: {0}")]
    Math(#[from] math_commons::error::MathError),
    #[error("Missing required initialization field: {0}")]
    #[allow(missing_docs)]
    InitializationError(String),
}
/// Defines the physics/dynamics model for the Kalman Filter.
///
/// Implementors provide the system matrices that define how the state evolves and how it is measured.
///
/// # Mathematical Model
///
/// State Evolution: $x_k = F_k x_{k-1} + w_k$
/// Measurement: $z_k = H_k x_k + v_k$
///
/// Where:
/// - $F_k$: Transition Matrix
/// - $H_k$: Measurement Matrix
/// - $w_k \sim N(0, Q_k)$: Process Noise
/// - $v_k \sim N(0, R_k)$: Measurement Noise
pub trait KalmanModel<T: RealField + Copy> {
    /// Returns the State Transition Matrix ($F_k$) of size $(n \times n)$.
    #[verified_engine::verified]
    fn transition_matrix(&self, dt: T) -> DMatrix<T>;
    /// Returns the Measurement Matrix ($H_k$) of size $(m \times n)$.
    #[verified_engine::verified]
    fn measurement_matrix(&self) -> DMatrix<T>;
    /// Returns the Process Noise Covariance ($Q_k$) of size $(n \times n)$.
    #[verified_engine::verified]
    fn process_noise(&self, dt: T) -> DMatrix<T>;
    /// Returns the Measurement Noise Covariance ($R_k$) of size $(m \times m)$.
    #[verified_engine::verified]
    fn measurement_noise(&self) -> DMatrix<T>;
}
/// Defines a generalized Kalman System (Linear or Non-Linear).
///
/// This trait supports both Standard Kalman Filters (Linear) and Extended Kalman Filters (EKF).
///
/// # Mathematical Model
///
/// State Evolution: $x_k = f(x_{k-1}) + w_k$
/// Measurement: $z_k = h(x_k) + v_k$
///
/// Where:
/// - $f(x)$: State Transition Function
/// - $h(x)$: Measurement Function
/// - $F_k = \frac{\partial f}{\partial x}$: Jacobian of State Transition
/// - $H_k = \frac{\partial h}{\partial x}$: Jacobian of Measurement
pub trait KalmanSystem<T: RealField + Copy> {
    /// Predicts the next state $x_k$ and returns the transition Jacobian $F_k$.
    ///
    /// # Arguments
    /// * `state` - The current state estimate $\hat{x}_{k-1|k-1}$.
    /// * `dt` - The time step $\Delta t$.
    ///
    /// # Returns
    /// A tuple `(predicted_state, transition_jacobian)`.
    #[verified_engine::verified]
    fn predict_state(&self, state: &DVector<T>, dt: T) -> (DVector<T>, DMatrix<T>);
    /// Predicts the measurement $z_k$ and returns the measurement Jacobian $H_k$.
    ///
    /// # Arguments
    /// * `state` - The predicted state estimate $\hat{x}_{k|k-1}$.
    ///
    /// # Returns
    /// A tuple `(predicted_measurement, measurement_jacobian)`.
    #[verified_engine::verified]
    fn predict_measurement(&self, state: &DVector<T>) -> (DVector<T>, DMatrix<T>);
    /// Returns the Process Noise Covariance $Q_k$.
    #[verified_engine::verified]
    fn process_noise(&self, dt: T) -> DMatrix<T>;
    /// Returns the Measurement Noise Covariance $R_k$.
    #[verified_engine::verified]
    fn measurement_noise(&self) -> DMatrix<T>;
}
impl<T: RealField + Copy, M: KalmanModel<T>> KalmanSystem<T> for M {
    #[verified_engine::verified]
    fn predict_state(&self, state: &DVector<T>, dt: T) -> (DVector<T>, DMatrix<T>) {
        let f = self.transition_matrix(dt);
        let x_pred = &f * state;
        (x_pred, f)
    }
    #[verified_engine::verified]
    fn predict_measurement(&self, state: &DVector<T>) -> (DVector<T>, DMatrix<T>) {
        let h = self.measurement_matrix();
        let z_pred = &h * state;
        (z_pred, h)
    }
    #[verified_engine::verified]
    fn process_noise(&self, dt: T) -> DMatrix<T> {
        KalmanModel::process_noise(self, dt)
    }
    #[verified_engine::verified]
    fn measurement_noise(&self) -> DMatrix<T> {
        KalmanModel::measurement_noise(self)
    }
}
/// A generic Discrete Kalman Filter.
///
/// Uses the **Strategy Pattern** via the `KalmanSystem` trait to decouple the estimation algorithm
/// from the specific system dynamics.
#[derive(Debug, Clone, PartialEq)]
pub struct KalmanFilter<T: RealField + Copy, M: KalmanSystem<T>> {
    /// State vector estimate ($\hat{x}$).
    pub state: DVector<T>,
    /// State covariance matrix ($P$).
    pub covariance: DMatrix<T>,
    /// The physics/system model strategy.
    pub model: M,
    /// Time step for the filter (if fixed).
    pub dt: T,
}
/// Builder for `KalmanFilter`.
///
/// Ensures valid initialization by enforcing dimension checks between state and covariance.
#[derive(Debug, Clone, PartialEq)]
pub struct KalmanFilterBuilder<T: RealField + Copy, M: KalmanSystem<T>> {
    state: Option<DVector<T>>,
    covariance: Option<DMatrix<T>>,
    model: M,
    dt: T,
}
impl<T: RealField + Copy, M: KalmanSystem<T>> KalmanFilterBuilder<T, M> {
    #[verified_engine::verified]
    fn new(model: M, dt: T) -> Self {
        Self {
            state: None,
            covariance: None,
            model,
            dt,
        }
    }
    /// Sets the initial state vector.
    #[verified_engine::verified]
    pub fn initial_state(mut self, state: DVector<T>) -> Self {
        self.state = Some(state);
        self
    }
    /// Sets the initial covariance matrix.
    #[verified_engine::verified]
    pub fn initial_covariance(mut self, covariance: DMatrix<T>) -> Self {
        self.covariance = Some(covariance);
        self
    }
    /// Builds the `KalmanFilter`, validating dimensions.
    ///
    /// # Errors
    /// Returns `KalmanError::Math` if state and covariance dimensions do not align.
    /// Returns `KalmanError::InitializationError` if state or covariance are missing.
    #[verified_engine::verified]
    pub fn build(self) -> Result<KalmanFilter<T, M>, KalmanError> {
        let state = self
            .state
            .ok_or_else(|| KalmanError::InitializationError("initial_state".to_string()))?;
        let covariance = self
            .covariance
            .ok_or_else(|| KalmanError::InitializationError("initial_covariance".to_string()))?;
        if state.len() != covariance.nrows() || covariance.nrows() != covariance.ncols() {
            return Err(KalmanError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(state.len()),
                    actual: math_commons::math_kernel::types::Dimension(
                        covariance.nrows() * covariance.ncols(),
                    ),
                },
            ));
        }
        Ok(KalmanFilter {
            state,
            covariance,
            model: self.model,
            dt: self.dt,
        })
    }
}
impl<T: RealField + Copy, M: KalmanSystem<T>> KalmanFilter<T, M> {
    /// Returns a new builder for constructing a `KalmanFilter`.
    #[verified_engine::verified]
    pub fn builder(model: M, dt: T) -> KalmanFilterBuilder<T, M> {
        KalmanFilterBuilder::new(model, dt)
    }
    /// Performs the **Prediction Step**.
    ///
    /// Projects the current state estimate and covariance forward in time.
    ///
    /// $$ \hat{x}_{k|k-1} = f(\hat{x}_{k-1|k-1}) $$
    /// $$ P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k $$
    #[verified_engine::verified]
    pub fn predict(&mut self) {
        // Generalized Prediction
        let (x_pred, f) = self.model.predict_state(&self.state, self.dt);
        let q = self.model.process_noise(self.dt);
        self.state = x_pred;
        self.covariance = &f * &self.covariance * f.transpose() + q;
    }
    /// Performs the **Update Step** with a new measurement.
    ///
    /// Incorporates the new observation $z_k$ to refine the state estimate.
    ///
    /// $$ y_k = z_k - h(\hat{x}_{k|k-1}) $$
    /// $$ S_k = H_k P_{k|k-1} H_k^T + R_k $$
    /// $$ K_k = P_{k|k-1} H_k^T S_k^{-1} $$
    /// $$ \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k y_k $$
    /// $$ P_{k|k} = (I - K_k H_k) P_{k|k-1} $$
    ///
    /// # Arguments
    ///
    /// * `measurement` - The measurement vector $z_k$.
    #[verified_engine::verified]
    pub fn update(&mut self, measurement: &DVector<T>) -> Result<(), KalmanError> {
        // Generalized Update
        let (z_pred, h) = self.model.predict_measurement(&self.state);
        let r = self.model.measurement_noise();
        // Innovation
        let y = measurement - z_pred;
        // Cache transpose outside the expression
        let h_t = h.transpose();
        // Innovation Covariance S = H P H^T + R
        let s = &h * &self.covariance * &h_t + r;
        // Invert S.
        // For 1D measurements, this is trivial. For nD, we need matrix inversion.
        // Kalman Filter requires S to be invertible (positive definite).
        let s_inv = s.try_inverse().ok_or(KalmanError::MatrixInversionError)?;
        // Kalman Gain K = P H^T S^-1
        let k = &self.covariance * &h_t * s_inv;
        // Update State
        self.state = &self.state + &k * y;
        // Update Covariance P = (I - K H) P
        let identity = DMatrix::identity(self.state.len(), self.state.len());
        self.covariance = (identity - &k * &h) * &self.covariance;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    // Mock Model for testing 1D constant velocity
    #[derive(Debug)]
    struct MockCvModel {
        process_noise: f64,
        measurement_noise: f64,
    }
    impl KalmanModel<f64> for MockCvModel {
        #[verified_engine::verified]
        fn transition_matrix(&self, dt: f64) -> DMatrix<f64> {
            // [1, dt]
            // [0, 1]
            DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0])
        }
        #[verified_engine::verified]
        fn measurement_matrix(&self) -> DMatrix<f64> {
            // [1, 0]
            DMatrix::from_row_slice(1, 2, &[1.0, 0.0])
        }
        #[verified_engine::verified]
        fn process_noise(&self, _dt: f64) -> DMatrix<f64> {
            DMatrix::identity(2, 2) * self.process_noise
        }
        #[verified_engine::verified]
        fn measurement_noise(&self) -> DMatrix<f64> {
            DMatrix::from_element(1, 1, self.measurement_noise)
        }
    }
    #[test]
    #[verified_engine::verified]
    fn test_kalman_predict_logic() -> Result<(), KalmanError> {
        let dt = 1.0;
        let model = MockCvModel {
            process_noise: 0.0,
            measurement_noise: 1.0,
        };
        let x_init = DVector::from_vec(vec![0.0, 10.0]); // Pos=0, Vel=10
        let p_init = DMatrix::identity(2, 2);
        let mut kf = KalmanFilter::builder(model, dt)
            .initial_state(x_init)
            .initial_covariance(p_init)
            .build()?;
        kf.predict();
        // New Pos = 0 + 10*1 = 10
        // New Vel = 10
        assert!((kf.state[0] - 10.0).abs() < math_commons::registry::TOLERANCE_FAST);
        assert!((kf.state[1] - 10.0).abs() < math_commons::registry::TOLERANCE_FAST);
        Ok(())
    }
    #[test]
    #[verified_engine::verified]
    fn test_extended_kalman_filter() -> Result<(), KalmanError> {
        // Define a non-linear model: x_{k} = sqrt(x_{k-1})
        // Measurement: z_k = x_k^2
        struct NonLinearModel;
        impl KalmanSystem<f64> for NonLinearModel {
            #[verified_engine::verified]
            fn predict_state(
                &self,
                state: &DVector<f64>,
                _dt: f64,
            ) -> (DVector<f64>, DMatrix<f64>) {
                let x = state[0];
                let x_pred = x.sqrt();
                // Derivative of sqrt(x) is 1 / (2 * sqrt(x))
                let f_jacobian = 0.5 / x.sqrt();
                (
                    DVector::from_element(1, x_pred),
                    DMatrix::from_element(1, 1, f_jacobian),
                )
            }
            #[verified_engine::verified]
            fn predict_measurement(&self, state: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
                let x = state[0];
                let z_pred = x.powi(2);
                // Derivative of x^2 is 2x
                let h_jacobian = 2.0 * x;
                (
                    DVector::from_element(1, z_pred),
                    DMatrix::from_element(1, 1, h_jacobian),
                )
            }
            #[verified_engine::verified]
            fn process_noise(&self, _dt: f64) -> DMatrix<f64> {
                DMatrix::from_element(1, 1, 0.1)
            }
            #[verified_engine::verified]
            fn measurement_noise(&self) -> DMatrix<f64> {
                DMatrix::from_element(1, 1, 0.1)
            }
        }
        let model = NonLinearModel;
        // Start at x=100.
        // Predict -> sqrt(100) = 10.
        // Update with z=100 (which corresponds to x=10).
        let initial_state = DVector::from_element(1, 100.0);
        let initial_covariance = DMatrix::from_element(1, 1, 1.0);
        let mut kf = KalmanFilter::builder(model, 1.0)
            .initial_state(initial_state)
            .initial_covariance(initial_covariance)
            .build()?;
        kf.predict();
        assert!(
            (kf.state[0] - 10.0).abs() < math_commons::registry::TOLERANCE_FAST,
            "Prediction step failed"
        );
        // Measurement z=100 (perfect measurement for x=10)
        let measurement = DVector::from_element(1, 100.0);
        kf.update(&measurement)?;
        // Should stay close to 10
        assert!(
            (kf.state[0] - 10.0).abs() < 0.5,
            "Update step diverged: {}",
            kf.state[0]
        );
        Ok(())
    }
    #[test]
    #[verified_engine::verified]
    fn test_singular_covariance_error() -> Result<(), KalmanError> {
        // Create a model where measurement noise R is zero and H is zero, leading to singular S.
        // S = HPH' + R. If H=0 and R=0, S=0, which is singular.
        struct SingularModel;
        impl KalmanSystem<f64> for SingularModel {
            #[verified_engine::verified]
            fn predict_state(
                &self,
                state: &DVector<f64>,
                _dt: f64,
            ) -> (DVector<f64>, DMatrix<f64>) {
                (state.clone(), DMatrix::identity(1, 1))
            }
            #[verified_engine::verified]
            fn predict_measurement(&self, _state: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
                (DVector::from_element(1, 0.0), DMatrix::zeros(1, 1)) // H = 0
            }
            #[verified_engine::verified]
            fn process_noise(&self, _dt: f64) -> DMatrix<f64> {
                DMatrix::zeros(1, 1)
            }
            #[verified_engine::verified]
            fn measurement_noise(&self) -> DMatrix<f64> {
                DMatrix::zeros(1, 1) // R = 0
            }
        }
        let mut kf = KalmanFilter::builder(SingularModel, 1.0)
            .initial_state(DVector::from_element(1, 0.0))
            .initial_covariance(DMatrix::identity(1, 1))
            .build()?;
        let measurement = DVector::from_element(1, 1.0);
        let result = kf.update(&measurement);
        assert_eq!(result, Err(KalmanError::MatrixInversionError));
        Ok(())
    }
    #[test]
    #[verified_engine::verified]
    fn test_dimension_mismatch() {
        let model = MockCvModel {
            process_noise: 0.0,
            measurement_noise: 1.0,
        };
        let state = DVector::from_element(2, 0.0); // 2x1
        let cov = DMatrix::from_element(3, 3, 1.0); // 3x3 - Mismatch with state
        let result = KalmanFilter::builder(model, 1.0)
            .initial_state(state)
            .initial_covariance(cov)
            .build();
        match result {
            Err(KalmanError::Math(math_commons::error::MathError::DimensionMismatch {
                expected,
                actual,
            })) => {
                assert_eq!(expected.0, 2);
                assert_eq!(actual.0, 9);
            }
            _ => panic!("Expected Math::DimensionMismatch error, got {:?}", result),
        }
    }
    #[test]
    #[verified_engine::verified]
    fn test_builder_missing_fields() {
        let model = MockCvModel {
            process_noise: 0.0,
            measurement_noise: 1.0,
        };
        let result = KalmanFilter::builder(model, 1.0).build();
        match result {
            Err(KalmanError::InitializationError(field)) => {
                assert_eq!(field, "initial_state");
            }
            _ => panic!("Expected InitializationError, got {:?}", result),
        }
    }
}
