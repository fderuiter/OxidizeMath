//! Generic discrete-time linear and Extended Kalman Filter (`EKF`) state estimation.
// [cite:pure_math]

use nalgebra::{DMatrix, DVector, RealField};
use thiserror::Error;

/// Errors returned during Kalman filter initialization and measurement updates.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum KalmanError {
    /// Innovation covariance matrix $\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k$ was singular.
    #[error("Failed to invert innovation covariance matrix (singular)")]
    MatrixInversionError,
    /// Underlying dimension mismatch or linear algebra error from `math_commons`.
    #[error("Mathematical error: {0}")]
    Math(#[from] math_commons::error::MathError),
    /// Required builder field (`initial_state` or `initial_covariance`) was omitted.
    #[error("Missing required initialization field: {0}")]
    InitializationError(String),
}

/// Defines the linear state-space dynamics model for the discrete-time Kalman Filter.
///
/// State Evolution: $\mathbf{x}_k = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{w}_k$, Measurement: $\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k$,
/// with $\mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$ and $\mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$.
///
/// Reference: Kalman (1960), DOI: 10.1115/1.3662552 (`kalman1960new`); see
/// [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Positive-semidefinite $\mathbf{Q}_k \succeq 0$ and strictly positive-definite $\mathbf{R}_k \succ 0$
/// keep the innovation covariance $\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k$ invertible.
///
/// # Domain of Validity
/// Finite matrices of conformable state dimension $n \ge 1$ and measurement dimension $m \ge 1$,
/// with positive time step `dt` $> 0$.
pub trait KalmanModel<T: RealField + Copy> {
    /// Returns the State Transition Matrix ($\mathbf{F}_k$) of size $(n \times n)$.
    #[verified_engine::verified]
    fn transition_matrix(&self, dt: T) -> DMatrix<T>;
    /// Returns the Measurement Matrix ($\mathbf{H}_k$) of size $(m \times n)$.
    #[verified_engine::verified]
    fn measurement_matrix(&self) -> DMatrix<T>;
    /// Returns the Process Noise Covariance ($\mathbf{Q}_k$) of size $(n \times n)$.
    #[verified_engine::verified]
    fn process_noise(&self, dt: T) -> DMatrix<T>;
    /// Returns the Measurement Noise Covariance ($\mathbf{R}_k$) of size $(m \times m)$.
    #[verified_engine::verified]
    fn measurement_noise(&self) -> DMatrix<T>;
}

/// Defines a generalized linear or nonlinear Extended Kalman Filter (`EKF`) state-space system.
///
/// Propagates nonlinear dynamics $\mathbf{x}_k = f(\mathbf{x}_{k-1}) + \mathbf{w}_k$ and observations
/// $\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k$ via first-order Jacobians
/// $\mathbf{F}_k = \left.\frac{\partial f}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}_{k-1|k-1}}$
/// and $\mathbf{H}_k = \left.\frac{\partial h}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}_{k|k-1}}$.
///
/// Reference: Kalman (1960), DOI: 10.1115/1.3662552 (`kalman1960new`); see
/// [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Local EKF covariance propagation remains accurate when higher-order Hessian terms of $f$ and $h$
/// are small over the uncertainty ellipsoid $\mathbf{P}_{k|k-1}$ and step `dt`.
///
/// # Domain of Validity
/// Differentiable state transition $f: \mathbb{R}^n \to \mathbb{R}^n$ and measurement map
/// $h: \mathbb{R}^n \to \mathbb{R}^m$ with finite Jacobians and positive-definite innovation covariance.
pub trait KalmanSystem<T: RealField + Copy> {
    /// Predicts the next state $\hat{\mathbf{x}}_{k|k-1}$ and returns `(predicted_state, F_k)`.
    #[verified_engine::verified]
    fn predict_state(&self, state: &DVector<T>, dt: T) -> (DVector<T>, DMatrix<T>);
    /// Predicts the measurement $\hat{\mathbf{z}}_{k|k-1}$ and returns `(predicted_measurement, H_k)`.
    #[verified_engine::verified]
    fn predict_measurement(&self, state: &DVector<T>) -> (DVector<T>, DMatrix<T>);
    /// Returns the Process Noise Covariance $\mathbf{Q}_k$.
    #[verified_engine::verified]
    fn process_noise(&self, dt: T) -> DMatrix<T>;
    /// Returns the Measurement Noise Covariance $\mathbf{R}_k$.
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
/// Reference: Kalman (1960), DOI: 10.1115/1.3662552 (`kalman1960new`); see
/// [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
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
    /// State vector estimate ($\hat{\mathbf{x}}$).
    pub state: DVector<T>,
    /// State covariance matrix ($\mathbf{P}$).
    pub covariance: DMatrix<T>,
    /// System dynamics and observation model strategy.
    pub model: M,
    /// Fixed integration time step $\Delta t$.
    pub dt: T,
}

/// Dimension-validating builder for [`KalmanFilter`].
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

    /// Sets the initial state vector $\hat{\mathbf{x}}_{0|0}$.
    #[verified_engine::verified]
    pub fn initial_state(mut self, state: DVector<T>) -> Self {
        self.state = Some(state);
        self
    }

    /// Sets the initial error covariance matrix $\mathbf{P}_{0|0}$.
    #[verified_engine::verified]
    pub fn initial_covariance(mut self, covariance: DMatrix<T>) -> Self {
        self.covariance = Some(covariance);
        self
    }

    /// Builds the [`KalmanFilter`], validating conformable state and square covariance dimensions.
    ///
    /// # Errors
    /// Returns [`KalmanError::InitializationError`] if `initial_state` or `initial_covariance` is
    /// missing, or [`KalmanError::Math`] if `covariance` is not $(n \times n)$ for state length $n$.
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
    /// Returns a new [`KalmanFilterBuilder`] for `model` and fixed step `dt`.
    #[verified_engine::verified]
    pub fn builder(model: M, dt: T) -> KalmanFilterBuilder<T, M> {
        KalmanFilterBuilder::new(model, dt)
    }

    /// Projects the current state estimate and covariance forward by `self.dt`:
    /// $\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1|k-1})$,
    /// $\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^\top + \mathbf{Q}_k$.
    #[verified_engine::verified]
    pub fn predict(&mut self) {
        let (x_pred, f) = self.model.predict_state(&self.state, self.dt);
        let q = self.model.process_noise(self.dt);
        self.state = x_pred;
        self.covariance = &f * &self.covariance * f.transpose() + q;
    }

    /// Incorporates measurement vector $\mathbf{z}_k$ via optimal Kalman gain $\mathbf{K}_k = \mathbf{P}_{k|k-1}\mathbf{H}_k^\top \mathbf{S}_k^{-1}$.
    ///
    /// # Errors
    /// Returns [`KalmanError::MatrixInversionError`] when innovation covariance $\mathbf{S}_k$ is singular.
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
