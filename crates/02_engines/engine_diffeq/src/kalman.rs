//! Generic Discrete-Time Linear and Extended Kalman Filter (`EKF`).
//!
//! Dimension-agnostic state estimation for linear and nonlinear state-space systems
//! backed by `nalgebra::DMatrix` and `nalgebra::DVector` under `alloc`/`no_std`.
// [cite:engine_diffeq]

use crate::error::KalmanError;
pub use crate::fast_cv::FastCvKalman;
use nalgebra::{DMatrix, DVector, RealField};

/// Defines the linear state-space dynamics model for the discrete-time Kalman Filter.
///
/// State Evolution: $\mathbf{x}_k = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{w}_k$, Measurement: $\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k$,
/// with $\mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$ and $\mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$.
///
/// Reference: Kalman (1960), DOI: 10.1115/1.3662552; see
/// [`docs/math_specs/domain_applied.md`](../../../../docs/math_specs/domain_applied.md).
///
/// # Stability
/// Positive-semidefinite $\mathbf{Q}_k \succeq 0$ and strictly positive-definite $\mathbf{R}_k \succ 0$
/// keep the innovation covariance $\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k$ invertible.
///
/// # Domain of Validity
/// Finite matrices of conformable state dimension $n \ge 1$ and measurement dimension $m \ge 1$,
/// with positive time step `dt` $> 0$.
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

/// Minimum-variance discrete-time linear and Extended Kalman Filter (`EKF`) estimator.
///
/// Propagates state mean $\hat{\mathbf{x}}$ and error covariance $\mathbf{P}$ via predict-update recursion:
///
/// $$\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1|k-1}), \quad \mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1|k-1}\mathbf{F}_k^\top + \mathbf{Q}_k, \quad \mathbf{K}_k = \mathbf{P}_{k|k-1}\mathbf{H}_k^\top\left(\mathbf{H}_k\mathbf{P}_{k|k-1}\mathbf{H}_k^\top + \mathbf{R}_k\right)^{-1}$$
///
/// Reference: Kalman (1960), DOI: 10.1115/1.3662552; see
/// [`docs/math_specs/domain_applied.md`](../../../../docs/math_specs/domain_applied.md).
///
/// # Stability
/// Exact MMSE optimal for linear-Gaussian systems; returns [`KalmanError::MatrixInversionError`] if
/// $\mathbf{S}_k$ becomes singular, and requires small linearization error over `dt` in EKF mode.
///
/// # Domain of Validity
/// State vector `state` $\in \mathbb{R}^n$ aligned with symmetric positive-semidefinite
/// `covariance` $\mathbf{P} \in \mathbb{R}^{n \times n}$ and invertible innovation covariance $\mathbf{S}_k \succ 0$.
#[derive(Debug, Clone)]
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

/// Builder for [`KalmanFilter`].
///
/// Ensures valid initialization by enforcing dimension checks between state and covariance.
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

    /// Builds the [`KalmanFilter`], validating dimensions.
    ///
    /// # Errors
    /// Returns [`KalmanError::Math`] if state and covariance dimensions do not align.
    /// Returns [`KalmanError::InitializationError`] if state or covariance are missing.
    #[verified_engine::verified]
    pub fn build(self) -> Result<KalmanFilter<T, M>, KalmanError> {
        let state = self
            .state
            .ok_or(KalmanError::InitializationError("initial_state"))?;
        let covariance = self
            .covariance
            .ok_or(KalmanError::InitializationError("initial_covariance"))?;
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
    /// Returns a new builder for constructing a [`KalmanFilter`].
    #[verified_engine::verified]
    pub fn builder(model: M, dt: T) -> KalmanFilterBuilder<T, M> {
        KalmanFilterBuilder::new(model, dt)
    }

    /// Performs the **Prediction Step**.
    ///
    /// Projects the current state estimate and covariance forward in time:
    /// $$ \hat{x}_{k|k-1} = f(\hat{x}_{k-1|k-1}), \quad P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k $$
    #[verified_engine::verified]
    pub fn predict(&mut self) {
        let (x_pred, f) = self.model.predict_state(&self.state, self.dt);
        let q = self.model.process_noise(self.dt);
        self.state = x_pred;
        self.covariance = &f * &self.covariance * f.transpose() + q;
    }

    /// Performs the **Update Step** with a new measurement $z_k$.
    ///
    /// # Errors
    /// Returns [`KalmanError::MatrixInversionError`] if the innovation covariance $S_k$ is singular.
    #[verified_engine::verified]
    pub fn update(&mut self, measurement: &DVector<T>) -> Result<(), KalmanError> {
        let (z_pred, h) = self.model.predict_measurement(&self.state);
        let r = self.model.measurement_noise();
        let y = measurement - z_pred;
        let h_t = h.transpose();
        let s = &h * &self.covariance * &h_t + r;
        let s_inv = s.try_inverse().ok_or(KalmanError::MatrixInversionError)?;
        let k = &self.covariance * &h_t * s_inv;
        self.state = &self.state + &k * y;
        let identity = DMatrix::identity(self.state.len(), self.state.len());
        self.covariance = (identity - &k * &h) * &self.covariance;
        Ok(())
    }
}

#[cfg(test)]
#[path = "kalman_tests.rs"]
mod tests;
