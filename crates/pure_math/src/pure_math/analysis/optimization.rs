//! Optimization Algorithms.
//!
//! Provides structures and traits for mathematical optimization problems.

use super::evolution::{EvolutionEngine, EvolutionError};
use nalgebra::{DMatrix, DVector, RealField, Vector2};
use rand::RngCore;
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during optimization.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OptimizationError {
    #[error("Failed to convert numerical value")]
    #[allow(missing_docs)]
    ConversionError,
}

impl From<EvolutionError> for OptimizationError {
    fn from(_: EvolutionError) -> Self {
        OptimizationError::ConversionError
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

impl<T: RealField + Copy> EvolutionEngine<Vector2<T>, Vector2<T>> for SGD<T> {
    #[verified_engine::verified]
    fn step<R: RngCore + ?Sized>(
        &mut self,
        state: &mut Vector2<T>,
        aux: &mut Vector2<T>,
        _rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError> {
        let lr = if dt > 0.0 {
            T::from_f64(dt).unwrap_or(self.learning_rate)
        } else {
            self.learning_rate
        };
        *state -= *aux * lr;
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

#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "T: nalgebra::Scalar + serde::Serialize",
        deserialize = "T: nalgebra::Scalar + serde::Deserialize<'de>"
    ))
)]
#[allow(missing_docs)]
pub struct AdamState<T> {
    #[allow(missing_docs)]
    pub m: DMatrix<T>,
    #[allow(missing_docs)]
    pub v: DMatrix<T>,
    #[allow(missing_docs)]
    pub t: i32,
}

#[cfg(feature = "serde")]
#[path = "optimization_serde.rs"]
mod serde_adam_states;

/// Zero-allocation stack moment accumulators for 2D `Adam` parameter updates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "T: nalgebra::Scalar + serde::Serialize",
        deserialize = "T: nalgebra::Scalar + serde::Deserialize<'de>"
    ))
)]
pub struct AdamState2D<T> {
    /// First moment (exponentially decaying gradient mean).
    pub m: Vector2<T>,
    /// Second raw moment (exponentially decaying uncentered gradient variance).
    pub v: Vector2<T>,
    /// 1-indexed optimization step counter.
    pub t: i32,
}

impl<T: RealField + Copy> Default for AdamState2D<T> {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: RealField + Copy> AdamState2D<T> {
    /// Creates a zeroed 2D Adam moment accumulator on the stack.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self {
            m: Vector2::zeros(),
            v: Vector2::zeros(),
            t: 0,
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "T: nalgebra::Scalar + serde::Serialize, Key: Eq + std::hash::Hash + serde::Serialize",
        deserialize = "T: nalgebra::Scalar + serde::Deserialize<'de>, Key: Eq + std::hash::Hash + serde::Deserialize<'de>"
    ))
)]
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
    #[cfg_attr(feature = "serde", serde(with = "serde_adam_states"))]
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

impl<T: RealField + Copy, Key: Eq + std::hash::Hash>
    EvolutionEngine<Vector2<T>, (Vector2<T>, &mut AdamState2D<T>)> for Adam<T, Key>
{
    #[verified_engine::verified]
    fn step<R: RngCore + ?Sized>(
        &mut self,
        state: &mut Vector2<T>,
        aux: &mut (Vector2<T>, &mut AdamState2D<T>),
        _rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError> {
        let grad = aux.0;
        let adam_state = &mut *aux.1;
        let lr = if dt > 0.0 {
            T::from_f64(dt).unwrap_or(self.learning_rate)
        } else {
            self.learning_rate
        };
        let one = T::one();
        adam_state.t += 1;
        let t_val = T::from_i32(adam_state.t).ok_or(EvolutionError::ConversionError)?;
        adam_state.m = adam_state.m * self.beta1 + grad * (one - self.beta1);
        let grad_sq = Vector2::new(grad.x * grad.x, grad.y * grad.y);
        adam_state.v = adam_state.v * self.beta2 + grad_sq * (one - self.beta2);
        let m_hat = adam_state.m / (one - self.beta1.powf(t_val));
        let v_hat = adam_state.v / (one - self.beta2.powf(t_val));
        let dx = m_hat.x / (v_hat.x.sqrt() + self.epsilon);
        let dy = m_hat.y / (v_hat.y.sqrt() + self.epsilon);
        *state -= Vector2::new(dx, dy) * lr;
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
        // extract state first to avoid borrow conflicts
        let shape = (param.nrows(), param.ncols());
        let mut rng = rand::rngs::mock::StepRng::new(0, 1);

        // This is safe because we use the internal state, keeping the interface unchanged
        // while fulfilling the EvolutionEngine pattern.
        // Get the state first:
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    ///
    /// # Arguments
    /// * `x` - Input vector.
    /// * `loss_grad` - Gradient of the loss with respect to the output logits (dJ/dz).
    /// * `optimizer` - The optimizer strategy to update parameters.
    ///
    /// # Errors
    /// Returns an `OptimizationError` if the parameter updates fail during optimization.
    #[verified_engine::verified]
    fn backward_update(
        &mut self,
        x: &nalgebra::DVector<T>,
        loss_grad: &nalgebra::DVector<T>,
        optimizer: &mut dyn ModelOptimizer<T>,
    ) -> Result<(), OptimizationError>;
}

#[cfg(test)]
#[path = "optimization_tests.rs"]
mod tests;
