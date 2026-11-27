//! Optimization Algorithms.
//!
//! Provides structures and traits for mathematical optimization problems.

use super::evolution::EvolutionError;
use nalgebra::{DMatrix, DVector, RealField, Vector2};
use std::collections::HashMap;
use thiserror::Error;

#[path = "optimization_impls.rs"]
mod optimization_impls;
use optimization_impls::validate_positive_lr;

/// Errors that can occur during optimization.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum OptimizationError {
    /// Variant `ConversionError` (ConversionError).
    #[error("Failed to convert numerical value")]
    ConversionError,
    /// Dimension mismatch between prediction and target tensors during optimization.
    #[error("Dimension mismatch during optimization: expected {expected}, actual {actual}")]
    DimensionMismatch {
        /// Expected vector/matrix dimension.
        expected: usize,
        /// Actual vector/matrix dimension encountered.
        actual: usize,
    },
    /// Non-finite or out-of-domain input encountered during optimization or loss evaluation.
    #[error("Invalid optimization input: {reason}")]
    InvalidInput {
        /// Description of the violated input invariant.
        reason: String,
    },
    /// Wrapped evolution engine error preserving causal source.
    #[error("Evolution solver error: {0}")]
    Evolution(#[from] EvolutionError),
}

#[doc = include_str!("../../../../../../docs/math_specs/snippets/opt_l1_lsq.md")]
pub struct L1RegularizedLeastSquares {
    lambda: f64,
}

impl L1RegularizedLeastSquares {
    /// Constructs an $L_1$-regularized least-squares objective with penalty weight `lambda`.
    #[verified_engine::verified]
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Evaluates $\frac{1}{2}\|A x - y\|_2^2 + \lambda \|x\|_1$.
    #[verified_engine::verified]
    pub fn cost(&mut self, a: &DMatrix<f64>, x: &DVector<f64>, y: &DVector<f64>) -> f64 {
        let residual = y - (a * x);
        let l2_term = 0.5 * residual.norm_squared();
        let l1_term = x.iter().map(|v| v.abs()).sum::<f64>();

        l2_term + self.lambda * l1_term
    }
}

#[doc = include_str!("../../../../../../docs/math_specs/snippets/opt_optimizer.md")]
pub trait Optimizer<T: RealField + Copy, Key = u64> {
    /// Computes `update_matrix` (update matrix).
    fn update_matrix(
        &mut self,
        key: Key,
        param: &mut DMatrix<T>,
        grad: &DMatrix<T>,
    ) -> Result<(), OptimizationError>;
    /// Computes `update_vector` (update vector).
    fn update_vector(
        &mut self,
        key: Key,
        param: &mut DVector<T>,
        grad: &DVector<T>,
    ) -> Result<(), OptimizationError>;
}

#[doc = include_str!("../../../../../../docs/math_specs/snippets/opt_sgd.md")]
pub struct SGD<T> {
    /// Value for `learning_rate` (learning rate).
    pub learning_rate: T,
}

impl<T: RealField + Copy> SGD<T> {
    /// Constructs a stochastic gradient descent optimizer with step size `learning_rate`.
    #[verified_engine::verified]
    pub fn new(learning_rate: T) -> Self {
        Self { learning_rate }
    }
}

/// Data structure `AdamState` representing AdamState state.
pub struct AdamState<T> {
    /// Value for `m` (m).
    pub m: DMatrix<T>,
    /// Value for `v` (v).
    pub v: DMatrix<T>,
    /// Value for `t` (t).
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

#[doc = include_str!("../../../../../../docs/math_specs/snippets/opt_adam.md")]
pub struct Adam<T, Key>
where
    Key: Eq + std::hash::Hash,
{
    /// Value for `learning_rate` (learning rate).
    pub learning_rate: T,
    /// Value for `beta1` (beta1).
    pub beta1: T,
    /// Value for `beta2` (beta2).
    pub beta2: T,
    /// Value for `epsilon` (epsilon).
    pub epsilon: T,
    /// Value for `states` (states).
    pub states: HashMap<Key, AdamState<T>>,
}

impl<T: RealField + Copy, Key> Adam<T, Key>
where
    Key: Eq + std::hash::Hash + Clone,
{
    /// Constructs an `Adam` optimizer with default moments ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$).
    #[verified_engine::verified]
    pub fn new(lr: T) -> Result<Self, OptimizationError> {
        validate_positive_lr(lr)?;
        Ok(Self {
            learning_rate: lr,
            beta1: T::from_f64(0.9).ok_or(OptimizationError::ConversionError)?,
            beta2: T::from_f64(0.999).ok_or(OptimizationError::ConversionError)?,
            epsilon: T::from_f64(1e-8).ok_or(OptimizationError::ConversionError)?,
            states: HashMap::new(),
        })
    }

    /// Returns or initializes zeroed first/second moment accumulators for parameter `key`.
    #[verified_engine::verified]
    pub fn get_state(&mut self, key: Key, shape: (usize, usize)) -> &mut AdamState<T> {
        self.states.entry(key).or_insert_with(|| AdamState {
            m: DMatrix::zeros(shape.0, shape.1),
            v: DMatrix::zeros(shape.0, shape.1),
            t: 0,
        })
    }
}

/// Identifies the type of parameter being updated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamType {
    /// Variant `Weight` (Weight).
    Weight,
    /// Variant `Bias` (Bias).
    Bias,
}

#[doc = include_str!("../../../../../../docs/math_specs/snippets/opt_model_opt.md")]
pub trait ModelOptimizer<T: RealField + Copy>: Optimizer<T, (usize, ParamType)> {}

impl<T: RealField + Copy, O> ModelOptimizer<T> for O where O: Optimizer<T, (usize, ParamType)> {}

/// Defines a Trainable Model that can perform forward passes and handle backpropagation.
pub trait Trainable<T: nalgebra::RealField + Copy = f64> {
    /// Performs a forward pass through the network and returns pre-softmax logits.
    #[verified_engine::verified]
    fn forward(&self, x: &nalgebra::DVector<T>) -> nalgebra::DVector<T>;

    /// Performs the backward pass and updates parameters via `optimizer`.
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
