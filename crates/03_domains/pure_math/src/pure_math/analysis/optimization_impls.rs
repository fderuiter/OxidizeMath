//! Private validation helpers and `Optimizer` / `EvolutionEngine` trait implementations
//! for [`SGD`] and [`Adam`].

use super::super::evolution::{EvolutionEngine, EvolutionError};
use super::{Adam, AdamState, AdamState2D, OptimizationError, Optimizer, SGD};
use nalgebra::{DMatrix, DVector, RealField, Vector2};
use rand::RngCore;

#[inline]
pub(super) fn check_vector_dims(expected: usize, actual: usize) -> Result<(), OptimizationError> {
    if expected != actual {
        return Err(OptimizationError::DimensionMismatch { expected, actual });
    }
    Ok(())
}

#[inline]
pub(super) fn check_matrix_dims(
    expected_shape: (usize, usize),
    actual_shape: (usize, usize),
) -> Result<(), OptimizationError> {
    if expected_shape.0 != actual_shape.0 {
        return Err(OptimizationError::DimensionMismatch {
            expected: expected_shape.0,
            actual: actual_shape.0,
        });
    }
    if expected_shape.1 != actual_shape.1 {
        return Err(OptimizationError::DimensionMismatch {
            expected: expected_shape.1,
            actual: actual_shape.1,
        });
    }
    Ok(())
}

#[inline]
pub(super) fn validate_positive_lr<T: RealField + Copy>(lr: T) -> Result<(), OptimizationError> {
    if !lr.is_finite() || lr.partial_cmp(&T::zero()) != Some(std::cmp::Ordering::Greater) {
        return Err(OptimizationError::InvalidInput {
            reason: "learning rate must be finite and strictly positive".to_owned(),
        });
    }
    Ok(())
}

#[inline]
pub(super) fn validate_finite_slices<T: RealField + Copy>(
    param: &[T],
    grad: &[T],
) -> Result<(), OptimizationError> {
    if !param.iter().all(|v| v.is_finite()) || !grad.iter().all(|v| v.is_finite()) {
        return Err(OptimizationError::InvalidInput {
            reason: "parameter and gradient entries must be finite".to_owned(),
        });
    }
    Ok(())
}

#[inline]
fn resolve_evolution_lr<T: RealField + Copy>(dt: f64, fallback_lr: T) -> Result<T, EvolutionError> {
    if !dt.is_finite() || dt < 0.0 {
        return Err(EvolutionError::InvalidState);
    }
    let lr = if dt > 0.0 {
        T::from_f64(dt).ok_or(EvolutionError::ConversionError)?
    } else {
        fallback_lr
    };
    if !lr.is_finite() || lr.partial_cmp(&T::zero()) != Some(std::cmp::Ordering::Greater) {
        return Err(EvolutionError::InvalidState);
    }
    Ok(lr)
}

fn sgd_step_slice<T: RealField + Copy>(param: &mut [T], grad: &[T], lr: T) {
    debug_assert_eq!(param.len(), grad.len());
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
    if param.len() != grad.len()
        || adam_state.m.as_slice().len() != param.len()
        || adam_state.v.as_slice().len() != param.len()
    {
        return Err(EvolutionError::InvalidState);
    }

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
        if state.shape() != aux.shape()
            || !state.iter().all(|v| v.is_finite())
            || !aux.iter().all(|v| v.is_finite())
        {
            return Err(EvolutionError::InvalidState);
        }
        let lr = resolve_evolution_lr(dt, self.learning_rate)?;
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
        if state.len() != aux.len()
            || !state.iter().all(|v| v.is_finite())
            || !aux.iter().all(|v| v.is_finite())
        {
            return Err(EvolutionError::InvalidState);
        }
        let lr = resolve_evolution_lr(dt, self.learning_rate)?;
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
        if !state.x.is_finite() || !state.y.is_finite() || !aux.x.is_finite() || !aux.y.is_finite()
        {
            return Err(EvolutionError::InvalidState);
        }
        let lr = resolve_evolution_lr(dt, self.learning_rate)?;
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
        check_vector_dims(param.len(), grad.len())?;
        validate_positive_lr(self.learning_rate)?;
        validate_finite_slices(param.as_slice(), grad.as_slice())?;
        sgd_step_slice(param.as_mut_slice(), grad.as_slice(), self.learning_rate);
        debug_assert!(param.iter().all(|v| v.is_finite()));
        Ok(())
    }

    #[verified_engine::verified]
    fn update_matrix(
        &mut self,
        _key: Key,
        param: &mut DMatrix<T>,
        grad: &DMatrix<T>,
    ) -> Result<(), OptimizationError> {
        check_matrix_dims(param.shape(), grad.shape())?;
        validate_positive_lr(self.learning_rate)?;
        validate_finite_slices(param.as_slice(), grad.as_slice())?;
        sgd_step_slice(param.as_mut_slice(), grad.as_slice(), self.learning_rate);
        debug_assert!(param.iter().all(|v| v.is_finite()));
        Ok(())
    }
}

impl<T: RealField + Copy, Key: Eq + std::hash::Hash> Adam<T, Key> {
    fn step_hyperparams(&self, dt: f64) -> Result<AdamHyperparams<T>, EvolutionError> {
        let lr = resolve_evolution_lr(dt, self.learning_rate)?;
        Ok(AdamHyperparams {
            lr,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
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
        if state.shape() != aux.0.shape()
            || !state.iter().all(|v| v.is_finite())
            || !aux.0.iter().all(|v| v.is_finite())
        {
            return Err(EvolutionError::InvalidState);
        }
        let hyper = self.step_hyperparams(dt)?;
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
        if state.len() != aux.0.len()
            || !state.iter().all(|v| v.is_finite())
            || !aux.0.iter().all(|v| v.is_finite())
        {
            return Err(EvolutionError::InvalidState);
        }
        let hyper = self.step_hyperparams(dt)?;
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
        if !state.x.is_finite()
            || !state.y.is_finite()
            || !grad.x.is_finite()
            || !grad.y.is_finite()
        {
            return Err(EvolutionError::InvalidState);
        }
        let adam_state = &mut *aux.1;
        let lr = resolve_evolution_lr(dt, self.learning_rate)?;
        let one = T::one();
        let next_t = adam_state
            .t
            .checked_add(1)
            .ok_or(EvolutionError::ConversionError)?;
        adam_state.t = next_t;
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
        check_matrix_dims(param.shape(), grad.shape())?;
        validate_positive_lr(self.learning_rate)?;
        validate_finite_slices(param.as_slice(), grad.as_slice())?;
        let shape = (param.nrows(), param.ncols());
        let hyper = self.step_hyperparams(0.0)?;
        let adam_state = self.states.entry(key).or_insert_with(|| AdamState {
            m: DMatrix::zeros(shape.0, shape.1),
            v: DMatrix::zeros(shape.0, shape.1),
            t: 0,
        });
        check_matrix_dims(adam_state.m.shape(), shape)?;
        check_matrix_dims(adam_state.v.shape(), shape)?;
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
        check_vector_dims(param.len(), grad.len())?;
        validate_positive_lr(self.learning_rate)?;
        validate_finite_slices(param.as_slice(), grad.as_slice())?;
        let shape = (param.len(), 1);
        let hyper = self.step_hyperparams(0.0)?;
        let adam_state = self.states.entry(key).or_insert_with(|| AdamState {
            m: DMatrix::zeros(shape.0, shape.1),
            v: DMatrix::zeros(shape.0, shape.1),
            t: 0,
        });
        check_vector_dims(adam_state.m.len(), param.len())?;
        check_vector_dims(adam_state.v.len(), param.len())?;
        adam_step_slices(param.as_mut_slice(), grad.as_slice(), adam_state, hyper)
            .map_err(Into::into)
    }
}
