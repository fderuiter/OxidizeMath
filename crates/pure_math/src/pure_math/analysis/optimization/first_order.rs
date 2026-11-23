//! First-order stochastic and adaptive gradient optimizers (`SGD`, `Adam`).

use super::super::evolution::{EvolutionEngine, EvolutionError};
use super::{OptimizationError, Optimizer};
use nalgebra::{DMatrix, DVector, RealField, Vector2};
use rand::RngCore;
use std::collections::HashMap;

/// Stochastic Gradient Descent optimizer with configurable learning rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SGD<T> {
    /// Step size multiplier applied to gradients during parameter updates.
    pub learning_rate: T,
}

impl<T: RealField + Copy> SGD<T> {
    /// Creates a new `SGD` optimizer with the specified `learning_rate`.
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
        sgd_step_slice(param.as_mut_slice(), grad.as_slice(), self.learning_rate);
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
        sgd_step_slice(param.as_mut_slice(), grad.as_slice(), self.learning_rate);
        Ok(())
    }
}

/// Dynamic matrix moment accumulators for `Adam` parameter updates.
pub struct AdamState<T> {
    /// First moment accumulator (exponentially decaying gradient mean).
    pub m: DMatrix<T>,
    /// Second raw moment accumulator (exponentially decaying squared gradient).
    pub v: DMatrix<T>,
    /// 1-indexed optimization step counter.
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

/// Adaptive Moment Estimation (`Adam`) first-order optimizer.
pub struct Adam<T, Key>
where
    Key: Eq + std::hash::Hash,
{
    /// Base step size learning rate.
    pub learning_rate: T,
    /// Exponential decay factor for first moment estimates.
    pub beta1: T,
    /// Exponential decay factor for second moment estimates.
    pub beta2: T,
    /// Numerical stability term added to denominator.
    pub epsilon: T,
    /// Per-parameter moment state table.
    pub states: HashMap<Key, AdamState<T>>,
}

impl<T: RealField + Copy, Key> Adam<T, Key>
where
    Key: Eq + std::hash::Hash + Clone,
{
    /// Creates a new `Adam` optimizer with standard `(0.9, 0.999, 1e-8)` defaults.
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

    /// Returns or initializes the moment state associated with `key`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_and_sgd_respect_configured_learning_rate() {
        let mut low_lr = Adam::<f64, usize>::new(0.001).expect("low adam");
        let mut high_lr = Adam::<f64, usize>::new(0.05).expect("high adam");
        let mut p_low = DMatrix::from_element(2, 2, 1.0_f64);
        let mut p_high = DMatrix::from_element(2, 2, 1.0_f64);
        let grad = DMatrix::from_element(2, 2, 0.5_f64);

        low_lr.update_matrix(0, &mut p_low, &grad).expect("low lr");
        high_lr
            .update_matrix(0, &mut p_high, &grad)
            .expect("high lr");
        assert!(((1.0 - p_high[(0, 0)]).abs() / (1.0 - p_low[(0, 0)]).abs() - 50.0).abs() < 1e-6);

        let mut sgd = SGD::new(0.25_f64);
        let mut mat = DMatrix::from_element(2, 2, 2.0_f64);
        let mut vec = DVector::from_element(3, 1.5_f64);
        let v_grad = DVector::from_element(3, 0.4_f64);
        Optimizer::<f64, usize>::update_matrix(&mut sgd, 0, &mut mat, &grad).expect("sgd mat");
        Optimizer::<f64, usize>::update_vector(&mut sgd, 1, &mut vec, &v_grad).expect("sgd vec");
        assert!((mat[(0, 0)] - 1.875).abs() < 1e-12);
        assert!((vec[0] - 1.4).abs() < 1e-12);
    }
}
