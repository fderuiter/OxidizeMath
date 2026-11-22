//! Optimization Algorithms.
//!
//! Provides structures and traits for mathematical optimization problems, including
//! first-order stochastic optimizers (`SGD`, `Adam`), damped Levenberg-Marquardt
//! nonlinear least squares ([`least_squares`]), and RK4 ODE trajectory shooting
//! calibration ([`ode_shooting`]).

/// Damped Levenberg-Marquardt nonlinear least-squares solver and pre-allocated workspace.
pub mod least_squares;
/// Pre-allocated Runge-Kutta 4 ODE trajectory shooting objective for parameter calibration.
pub mod ode_shooting;

pub use least_squares::*;
pub use ode_shooting::*;

use super::evolution::{EvolutionEngine, EvolutionError};
use nalgebra::{DMatrix, DVector, RealField};
use rand::RngCore;
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during mathematical optimization and nonlinear calibration.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum OptimizationError {
    /// Failed to convert a generic scalar or numerical value.
    #[error("Failed to convert numerical value")]
    ConversionError,
    /// Parameter count is zero or exceeds [`MAX_CALIBRATION_PARAMS`].
    #[error("Invalid parameter count {count} (expected 1..={max})")]
    InvalidParameterCount {
        /// Supplied parameter count.
        count: usize,
        /// Maximum supported parameter count.
        max: usize,
    },
    /// Observation count is smaller than parameter count / 2-row minimum or exceeds [`MAX_CALIBRATION_OBSERVATIONS`].
    #[error("Invalid observation count {count} (expected {min}..={max})")]
    InvalidObservationCount {
        /// Supplied observation count.
        count: usize,
        /// Minimum required observations.
        min: usize,
        /// Maximum supported observations.
        max: usize,
    },
    /// Box bounds slice has mismatched length or non-finite / inverted `[lower, upper]` limits.
    #[error("Invalid parameter box bounds at index {index}")]
    InvalidBounds {
        /// Index of the invalid bound entry.
        index: usize,
    },
    /// Pre-allocated [`LevenbergMarquardtWorkspace`] lacks capacity for `(observations, parameters)`.
    #[error("Workspace capacity ({max_obs}, {max_params}) is smaller than ({obs}, {params})")]
    WorkspaceCapacityExceeded {
        /// Requested observation count.
        obs: usize,
        /// Requested parameter count.
        params: usize,
        /// Pre-allocated observation capacity.
        max_obs: usize,
        /// Pre-allocated parameter capacity.
        max_params: usize,
    },
    /// Objective evaluation, Jacobian finite differencing, or linear step encountered non-finite values.
    #[error("Numerical instability or non-finite model state during optimization")]
    NumericalInstability,
}

impl From<EvolutionError> for OptimizationError {
    fn from(_: EvolutionError) -> Self {
        OptimizationError::ConversionError
    }
}

#[allow(missing_docs)]
pub struct L1RegularizedLeastSquares {
    lambda: f64,
}

impl L1RegularizedLeastSquares {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn cost(&mut self, a: &DMatrix<f64>, x: &DVector<f64>, y: &DVector<f64>) -> f64 {
        let residual = y - (a * x);
        let l2_term = 0.5 * residual.norm_squared();
        let l1_term = x.iter().map(|v| v.abs()).sum::<f64>();

        l2_term + self.lambda * l1_term
    }
}

/// Strategy for updating parameters.
/// Reimplemented via the Unified EvolutionEngine.
pub trait Optimizer<T: RealField + Copy, Key = u64> {
    #[allow(missing_docs)]
    fn update_matrix(
        &mut self,
        key: Key,
        param: &mut DMatrix<T>,
        grad: &DMatrix<T>,
    ) -> Result<(), OptimizationError>;
    #[allow(missing_docs)]
    fn update_vector(
        &mut self,
        key: Key,
        param: &mut DVector<T>,
        grad: &DVector<T>,
    ) -> Result<(), OptimizationError>;
}

#[allow(missing_docs)]
pub struct SGD<T> {
    #[allow(missing_docs)]
    pub learning_rate: T,
}

impl<T: RealField + Copy> SGD<T> {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(learning_rate: T) -> Self {
        Self { learning_rate }
    }
}

impl<T: RealField + Copy> EvolutionEngine<DMatrix<T>, DMatrix<T>> for SGD<T> {
    fn step<R: RngCore + ?Sized>(
        &mut self,
        state: &mut DMatrix<T>,
        aux: &mut DMatrix<T>,
        _rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError> {
        let lr = T::from_f64(dt).unwrap_or(self.learning_rate);
        *state -= &*aux * lr;
        Ok(())
    }
}

impl<T: RealField + Copy> EvolutionEngine<DVector<T>, DVector<T>> for SGD<T> {
    fn step<R: RngCore + ?Sized>(
        &mut self,
        state: &mut DVector<T>,
        aux: &mut DVector<T>,
        _rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError> {
        let lr = T::from_f64(dt).unwrap_or(self.learning_rate);
        *state -= &*aux * lr;
        Ok(())
    }
}

impl<T: RealField + Copy, Key> Optimizer<T, Key> for SGD<T> {
    #[verified_engine::verified]
    fn update_vector(
        &mut self,
        _key: Key,
        param: &mut DVector<T>,
        grad: &DVector<T>,
    ) -> Result<(), OptimizationError> {
        let mut aux = grad.clone();
        let mut rng = rand::rngs::mock::StepRng::new(0, 1);
        self.step(param, &mut aux, &mut rng, 0.01)
            .map_err(Into::into)
    }

    #[verified_engine::verified]
    fn update_matrix(
        &mut self,
        _key: Key,
        param: &mut DMatrix<T>,
        grad: &DMatrix<T>,
    ) -> Result<(), OptimizationError> {
        let mut aux = grad.clone();
        let mut rng = rand::rngs::mock::StepRng::new(0, 1);
        self.step(param, &mut aux, &mut rng, 0.01)
            .map_err(Into::into)
    }
}

#[allow(missing_docs)]
pub struct AdamState<T> {
    #[allow(missing_docs)]
    pub m: DMatrix<T>,
    #[allow(missing_docs)]
    pub v: DMatrix<T>,
    #[allow(missing_docs)]
    pub t: i32,
}

#[allow(missing_docs)]
pub struct Adam<T, Key>
where
    Key: Eq + std::hash::Hash,
{
    #[allow(missing_docs)]
    pub learning_rate: T,
    #[allow(missing_docs)]
    pub beta1: T,
    #[allow(missing_docs)]
    pub beta2: T,
    #[allow(missing_docs)]
    pub epsilon: T,
    #[allow(missing_docs)]
    pub states: HashMap<Key, AdamState<T>>,
}

impl<T: RealField + Copy, Key> Adam<T, Key>
where
    Key: Eq + std::hash::Hash + Clone,
{
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(lr: T) -> Result<Self, OptimizationError> {
        Ok(Self {
            learning_rate: lr,
            beta1: T::from_f64(0.9).ok_or(OptimizationError::ConversionError)?,
            beta2: T::from_f64(0.999).ok_or(OptimizationError::ConversionError)?,
            epsilon: T::from_f64(1e-8).ok_or(OptimizationError::ConversionError)?,
            states: HashMap::new(),
        })
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn get_state(&mut self, key: Key, shape: (usize, usize)) -> &mut AdamState<T> {
        self.states.entry(key).or_insert_with(|| AdamState {
            m: DMatrix::zeros(shape.0, shape.1),
            v: DMatrix::zeros(shape.0, shape.1),
            t: 0,
        })
    }
}

impl<T: RealField + Copy, Key: Eq + std::hash::Hash>
    EvolutionEngine<DMatrix<T>, (DMatrix<T>, &mut AdamState<T>)> for Adam<T, Key>
{
    fn step<R: RngCore + ?Sized>(
        &mut self,
        state: &mut DMatrix<T>,
        aux: &mut (DMatrix<T>, &mut AdamState<T>),
        _rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError> {
        let grad = &aux.0;
        let adam_state = &mut aux.1;
        let lr = if dt > 0.0 {
            T::from_f64(dt).unwrap_or(self.learning_rate)
        } else {
            self.learning_rate
        };

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let epsilon = self.epsilon;
        let one = T::one();

        adam_state.t += 1;
        let t_val = T::from_i32(adam_state.t).ok_or(EvolutionError::ConversionError)?;

        adam_state.m = &adam_state.m * beta1 + grad * (one - beta1);
        let grad_sq = grad.map(|g| g * g);
        adam_state.v = &adam_state.v * beta2 + grad_sq * (one - beta2);

        let m_hat = &adam_state.m / (one - beta1.powf(t_val));
        let v_hat = &adam_state.v / (one - beta2.powf(t_val));

        let update = m_hat.component_div(&v_hat.map(|v| v.sqrt() + epsilon));
        *state -= update * lr;
        Ok(())
    }
}

impl<T: RealField + Copy, Key: Eq + std::hash::Hash>
    EvolutionEngine<DVector<T>, (DVector<T>, &mut AdamState<T>)> for Adam<T, Key>
{
    fn step<R: RngCore + ?Sized>(
        &mut self,
        state: &mut DVector<T>,
        aux: &mut (DVector<T>, &mut AdamState<T>),
        _rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError> {
        let grad = &aux.0;
        let adam_state = &mut aux.1;
        let lr = if dt > 0.0 {
            T::from_f64(dt).unwrap_or(self.learning_rate)
        } else {
            self.learning_rate
        };

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let epsilon = self.epsilon;
        let one = T::one();

        let rows = state.len();
        let cols = 1;

        adam_state.t += 1;
        let t_val = T::from_i32(adam_state.t).ok_or(EvolutionError::ConversionError)?;

        let grad_mat = DMatrix::from_column_slice(rows, cols, grad.as_slice());

        adam_state.m = &adam_state.m * beta1 + &grad_mat * (one - beta1);
        let grad_sq = grad_mat.map(|g| g * g);
        adam_state.v = &adam_state.v * beta2 + grad_sq * (one - beta2);

        let m_hat = &adam_state.m / (one - beta1.powf(t_val));
        let v_hat = &adam_state.v / (one - beta2.powf(t_val));

        let update_mat = m_hat.component_div(&v_hat.map(|v| v.sqrt() + epsilon));
        let update_vec = DVector::from_column_slice(update_mat.as_slice());
        *state -= update_vec * lr;
        Ok(())
    }
}

impl<T: RealField + Copy, Key> Optimizer<T, Key> for Adam<T, Key>
where
    Key: Eq + std::hash::Hash + Clone,
{
    #[verified_engine::verified]
    fn update_matrix(
        &mut self,
        key: Key,
        param: &mut DMatrix<T>,
        grad: &DMatrix<T>,
    ) -> Result<(), OptimizationError> {
        let shape = (param.nrows(), param.ncols());
        let mut rng = rand::rngs::mock::StepRng::new(0, 1);

        let mut state_val = self.states.remove(&key).unwrap_or_else(|| AdamState {
            m: DMatrix::zeros(shape.0, shape.1),
            v: DMatrix::zeros(shape.0, shape.1),
            t: 0,
        });

        let mut aux = (grad.clone(), &mut state_val);
        EvolutionEngine::step(self, param, &mut aux, &mut rng, 0.0)?;

        self.states.insert(key, state_val);
        Ok(())
    }

    #[verified_engine::verified]
    fn update_vector(
        &mut self,
        key: Key,
        param: &mut DVector<T>,
        grad: &DVector<T>,
    ) -> Result<(), OptimizationError> {
        let shape = (param.len(), 1);
        let mut rng = rand::rngs::mock::StepRng::new(0, 1);

        let mut state_val = self.states.remove(&key).unwrap_or_else(|| AdamState {
            m: DMatrix::zeros(shape.0, shape.1),
            v: DMatrix::zeros(shape.0, shape.1),
            t: 0,
        });

        let mut aux = (grad.clone(), &mut state_val);
        EvolutionEngine::step(self, param, &mut aux, &mut rng, 0.0)?;

        self.states.insert(key, state_val);
        Ok(())
    }
}

/// Identifies the type of parameter being updated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamType {
    #[allow(missing_docs)]
    Weight,
    #[allow(missing_docs)]
    Bias,
}

/// Adapter Trait for model optimizers.
pub trait ModelOptimizer<T: RealField + Copy>: Optimizer<T, (usize, ParamType)> {}

impl<T: RealField + Copy, O> ModelOptimizer<T> for O where O: Optimizer<T, (usize, ParamType)> {}

/// Defines a Trainable Model that can perform forward passes and handle backpropagation.
pub trait Trainable<T: nalgebra::RealField + Copy = f64> {
    /// Performs a forward pass through the network.
    /// Returns the logits (pre-softmax outputs).
    #[verified_engine::verified]
    fn forward(&self, x: &nalgebra::DVector<T>) -> nalgebra::DVector<T>;

    /// Performs the backward pass and updates parameters.
    #[verified_engine::verified]
    fn backward_update(
        &mut self,
        x: &nalgebra::DVector<T>,
        loss_grad: &nalgebra::DVector<T>,
        optimizer: &mut dyn ModelOptimizer<T>,
    ) -> Result<(), OptimizationError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_optimizer_respects_configured_learning_rate() {
        let mut low_lr = Adam::<f64, usize>::new(0.001).expect("low adam");
        let mut high_lr = Adam::<f64, usize>::new(0.05).expect("high adam");

        let mut p_low = DMatrix::from_element(2, 2, 1.0_f64);
        let mut p_high = DMatrix::from_element(2, 2, 1.0_f64);
        let grad = DMatrix::from_element(2, 2, 0.5_f64);

        low_lr.update_matrix(0, &mut p_low, &grad).expect("low lr");
        high_lr
            .update_matrix(0, &mut p_high, &grad)
            .expect("high lr");

        let delta_low = (1.0 - p_low[(0, 0)]).abs();
        let delta_high = (1.0 - p_high[(0, 0)]).abs();
        assert!((delta_high / delta_low - 50.0).abs() < 1e-6);

        let mut v_low = DVector::from_element(2, 1.0);
        let mut v_high = DVector::from_element(2, 1.0);
        let v_grad = DVector::from_element(2, 0.5);
        low_lr
            .update_vector(1, &mut v_low, &v_grad)
            .expect("low vec");
        high_lr
            .update_vector(1, &mut v_high, &v_grad)
            .expect("high vec");
        assert!(((1.0 - v_high[0]).abs() / (1.0 - v_low[0]).abs() - 50.0).abs() < 1e-6);
    }
}
