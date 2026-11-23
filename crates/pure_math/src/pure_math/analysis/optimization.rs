//! Optimization Algorithms.
//!
//! Provides structures and traits for mathematical optimization problems.

use super::evolution::{EvolutionEngine, EvolutionError};
use nalgebra::{DMatrix, DVector, RealField, Vector2};
use rand::RngCore;
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during optimization.
#[derive(Debug, Clone, PartialEq, Error)]
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

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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

fn sgd_step_slice<T: RealField + Copy>(param: &mut [T], grad: &[T], lr: T) {
    assert_eq!(param.len(), grad.len());
    for (p, &g) in param.iter_mut().zip(grad.iter()) {
        *p -= g * lr;
    }
}

#[derive(Clone, Copy)]
struct AdamHyperparams<T> {
    lr: T,
    beta1: T,
    beta2: T,
    epsilon: T,
}

fn adam_step_slices<T: RealField + Copy>(
    param: &mut [T],
    grad: &[T],
    adam_state: &mut AdamState<T>,
    hyper: AdamHyperparams<T>,
) -> Result<(), EvolutionError> {
    assert_eq!(param.len(), grad.len());
    assert_eq!(adam_state.m.as_slice().len(), param.len());
    assert_eq!(adam_state.v.as_slice().len(), param.len());

    let one = T::one();
    let next_t = adam_state
        .t
        .checked_add(1)
        .ok_or(EvolutionError::ConversionError)?;
    let t_val = T::from_i32(next_t).ok_or(EvolutionError::ConversionError)?;
    adam_state.t = next_t;

    let one_minus_b1 = one - hyper.beta1;
    let one_minus_b2 = one - hyper.beta2;
    let bias_corr1 = one - hyper.beta1.powf(t_val);
    let bias_corr2 = one - hyper.beta2.powf(t_val);

    for (((p, &g), m), v) in param
        .iter_mut()
        .zip(grad.iter())
        .zip(adam_state.m.as_mut_slice().iter_mut())
        .zip(adam_state.v.as_mut_slice().iter_mut())
    {
        *m = *m * hyper.beta1 + g * one_minus_b1;
        *v = *v * hyper.beta2 + (g * g) * one_minus_b2;
        let m_hat = *m / bias_corr1;
        let v_hat = *v / bias_corr2;
        let update = m_hat / (v_hat.sqrt() + hyper.epsilon);
        *p -= update * hyper.lr;
    }
    Ok(())
}

impl<T: RealField + Copy> EvolutionEngine<DMatrix<T>, DMatrix<T>> for SGD<T> {
    fn step<R: RngCore + ?Sized>(
        &mut self,
        state: &mut DMatrix<T>,
        aux: &mut DMatrix<T>,
        _rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError> {
        assert_eq!(state.shape(), aux.shape());
        let lr = T::from_f64(dt).unwrap_or(self.learning_rate);
        sgd_step_slice(state.as_mut_slice(), aux.as_slice(), lr);
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
        assert_eq!(state.len(), aux.len());
        let lr = T::from_f64(dt).unwrap_or(self.learning_rate);
        sgd_step_slice(state.as_mut_slice(), aux.as_slice(), lr);
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
        assert_eq!(param.len(), grad.len());
        let lr = T::from_f64(0.01).unwrap_or(self.learning_rate);
        sgd_step_slice(param.as_mut_slice(), grad.as_slice(), lr);
        Ok(())
    }

    #[verified_engine::verified]
    fn update_matrix(
        &mut self,
        _key: Key,
        param: &mut DMatrix<T>,
        grad: &DMatrix<T>,
    ) -> Result<(), OptimizationError> {
        assert_eq!(param.shape(), grad.shape());
        let lr = T::from_f64(0.01).unwrap_or(self.learning_rate);
        sgd_step_slice(param.as_mut_slice(), grad.as_slice(), lr);
        Ok(())
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub struct AdamState<T> {
    #[allow(missing_docs)]
    pub m: DMatrix<T>,
    #[allow(missing_docs)]
    pub v: DMatrix<T>,
    #[allow(missing_docs)]
    pub t: i32,
}

/// Zero-allocation stack moment accumulators for 2D `Adam` parameter updates.
#[derive(Debug, Clone, Copy, PartialEq)]
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

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
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

impl<T: RealField + Copy, Key: Eq + std::hash::Hash> Adam<T, Key> {
    fn step_hyperparams(&self, dt: f64) -> AdamHyperparams<T> {
        let lr = if dt > 0.0 {
            T::from_f64(dt).unwrap_or(self.learning_rate)
        } else {
            self.learning_rate
        };
        AdamHyperparams {
            lr,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
        }
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
        assert_eq!(state.shape(), aux.0.shape());
        let hyper = self.step_hyperparams(dt);
        adam_step_slices(state.as_mut_slice(), aux.0.as_slice(), aux.1, hyper)
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
        assert_eq!(state.len(), aux.0.len());
        let hyper = self.step_hyperparams(dt);
        adam_step_slices(state.as_mut_slice(), aux.0.as_slice(), aux.1, hyper)
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
        assert_eq!(param.shape(), grad.shape());
        let shape = (param.nrows(), param.ncols());
        let hyper = self.step_hyperparams(0.0);
        let adam_state = self.states.entry(key).or_insert_with(|| AdamState {
            m: DMatrix::zeros(shape.0, shape.1),
            v: DMatrix::zeros(shape.0, shape.1),
            t: 0,
        });
        adam_step_slices(param.as_mut_slice(), grad.as_slice(), adam_state, hyper)
            .map_err(Into::into)
    }

    #[verified_engine::verified]
    fn update_vector(
        &mut self,
        key: Key,
        param: &mut DVector<T>,
        grad: &DVector<T>,
    ) -> Result<(), OptimizationError> {
        assert_eq!(param.len(), grad.len());
        let shape = (param.len(), 1);
        let hyper = self.step_hyperparams(0.0);
        let adam_state = self.states.entry(key).or_insert_with(|| AdamState {
            m: DMatrix::zeros(shape.0, shape.1),
            v: DMatrix::zeros(shape.0, shape.1),
            t: 0,
        });
        adam_step_slices(param.as_mut_slice(), grad.as_slice(), adam_state, hyper)
            .map_err(Into::into)
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
