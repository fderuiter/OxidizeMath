use crate::ai::sds::rendering::{NeRFModel, RayBundle};
use crate::error::AIError;
use math_commons::primitives::{PositiveFloat, UnitInterval};
use nalgebra::DMatrix;
use pure_math::pure_math::analysis::optimization::{Adam, AdamState, Optimizer as CoreOptimizer};
use std::collections::HashMap;

fn validate_param_grad_shapes(params: &DMatrix<f64>, grads: &DMatrix<f64>) -> Result<(), AIError> {
    if params.shape() != grads.shape() {
        return Err(math_commons::error::MathError::DimensionMismatch {
            expected: math_commons::math_kernel::types::Dimension(params.nrows() * params.ncols()),
            actual: math_commons::math_kernel::types::Dimension(grads.nrows() * grads.ncols()),
        }
        .into());
    }
    Ok(())
}

/// Module 5.1: Jacobian-Vector Product
/// Input: SDS Gradient delta_SDS, Rendered Image x_render, NeRF Weights theta.
/// Operation: We need dL/dtheta. By chain rule, this is delta_SDS * dx_render/dtheta.
/// Modern autodiff engines (PyTorch) handle this by calling .backward() on the rendered image with delta_SDS as the incoming gradient.
/// Output: Gradient accumulator for theta.
pub trait DifferentiableNeRF: NeRFModel {
    /// Computes the gradient of the loss with respect to the model parameters,
    /// given the gradient of the loss with respect to the output image.
    /// This corresponds to the `backward` pass in an autodiff framework.
    ///
    /// # Arguments
    /// * `bundle` - The ray bundle used for the forward pass.
    /// * `image_grad` - The gradient of the loss w.r.t the rendered image (delta_SDS).
    ///
    /// # Returns
    /// * `DMatrix<f64>` - The gradient vector for the model parameters (theta).
    #[verified_engine::verified]
    fn backward(&self, bundle: &RayBundle, image_grad: &DMatrix<f64>) -> DMatrix<f64>;
}

/// Module 5.2: Optimizer Step
/// Input: Weights theta, Gradients nabla_theta, Learning Rate eta.
/// Operation: Update weights (e.g., using Adam optimizer).
/// Output: Updated NeRF Model.
pub trait Optimizer {
    /// Performs a single optimization step.
    ///
    /// # Arguments
    /// * `params` - The current parameters.
    /// * `grads` - The gradients with respect to the parameters.
    ///
    /// # Returns
    /// * `Ok(DMatrix<f64>)` - The updated parameters.
    /// * `Err(AIError)` - If dimensions mismatch.
    #[verified_engine::verified]
    fn step(
        &mut self,
        params: &DMatrix<f64>,
        grads: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, AIError>;
}

/// Simplified Adam implementation for a single parameter tensor (e.g., NeRF weights).
/// theta_{t+1} = theta_t - eta * m_t / (sqrt(v_t) + epsilon)
#[derive(Debug, Clone, PartialEq)]
pub struct AdamOptimizer {
    #[allow(missing_docs)]
    pub learning_rate: PositiveFloat,
    #[allow(missing_docs)]
    pub beta1: UnitInterval,
    #[allow(missing_docs)]
    pub beta2: UnitInterval,
    #[allow(missing_docs)]
    pub epsilon: PositiveFloat,
    #[allow(missing_docs)]
    pub m: Option<DMatrix<f64>>,
    #[allow(missing_docs)]
    pub v: Option<DMatrix<f64>>,
    #[allow(missing_docs)]
    pub t: usize,
}

impl AdamOptimizer {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(learning_rate: PositiveFloat) -> Self {
        Self {
            learning_rate,
            beta1: UnitInterval::new(0.9).unwrap(),
            beta2: UnitInterval::new(0.999).unwrap(),
            epsilon: PositiveFloat::new(1e-8).unwrap(),
            m: None,
            v: None,
            t: 0,
        }
    }
}

impl Optimizer for AdamOptimizer {
    #[verified_engine::verified]
    fn step(
        &mut self,
        params: &DMatrix<f64>,
        grads: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, AIError> {
        validate_param_grad_shapes(params, grads)?;

        if self.m.is_none() {
            self.m = Some(DMatrix::zeros(params.nrows(), params.ncols()));
            self.v = Some(DMatrix::zeros(params.nrows(), params.ncols()));
        }

        let m_ref = self.m.as_ref().ok_or(AIError::UninitializedState {
            name: "m".to_string(),
        })?;
        let v_ref = self.v.as_ref().ok_or(AIError::UninitializedState {
            name: "v".to_string(),
        })?;
        validate_param_grad_shapes(params, m_ref)?;
        validate_param_grad_shapes(params, v_ref)?;

        let t_i32 =
            i32::try_from(self.t).map_err(|_| math_commons::error::MathError::ConversionError {
                reason: "Adam step counter exceeded i32::MAX".to_string(),
            })?;
        let m = self.m.take().ok_or(AIError::UninitializedState {
            name: "m".to_string(),
        })?;
        let v = self.v.take().ok_or(AIError::UninitializedState {
            name: "v".to_string(),
        })?;

        let mut states = HashMap::with_capacity(1);
        states.insert((), AdamState { m, v, t: t_i32 });
        let mut core_adam = Adam {
            learning_rate: self.learning_rate.value(),
            beta1: self.beta1.value(),
            beta2: self.beta2.value(),
            epsilon: self.epsilon.value(),
            states,
        };

        let mut updated = params.clone();
        let update_res = CoreOptimizer::update_matrix(&mut core_adam, (), &mut updated, grads);

        if let Some(state) = core_adam.states.remove(&()) {
            self.m = Some(state.m);
            self.v = Some(state.v);
            if let Ok(next_t) = usize::try_from(state.t) {
                self.t = next_t;
            }
        }

        update_res.map_err(|_| AIError::OptimizationDivergence)?;
        Ok(updated)
    }
}

/// Stochastic Gradient Descent with Momentum.
/// v_{t+1} = momentum * v_t + g_t
/// theta_{t+1} = theta_t - lr * v_{t+1}
#[derive(Debug, Clone, PartialEq)]
pub struct SgdOptimizer {
    #[allow(missing_docs)]
    pub learning_rate: PositiveFloat,
    #[allow(missing_docs)]
    pub momentum: UnitInterval,
    #[allow(missing_docs)]
    pub velocity: Option<DMatrix<f64>>,
}

impl SgdOptimizer {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(learning_rate: PositiveFloat, momentum: UnitInterval) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: None,
        }
    }
}

impl Optimizer for SgdOptimizer {
    #[verified_engine::verified]
    fn step(
        &mut self,
        params: &DMatrix<f64>,
        grads: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, AIError> {
        validate_param_grad_shapes(params, grads)?;

        // Initialize state if needed
        if self.velocity.is_none() {
            self.velocity = Some(DMatrix::zeros(params.nrows(), params.ncols()));
        }

        let v = self.velocity.as_mut().ok_or(AIError::UninitializedState {
            name: "velocity".to_string(),
        })?;
        validate_param_grad_shapes(params, v)?;

        // Update velocity: v_{t+1} = momentum * v_t + g_t
        // Note: Some implementations use v_{t+1} = momentum * v_t - lr * g_t.
        // We stick to the additive accumulation of gradients, then subtract.
        // v = mu * v + g
        *v = &*v * self.momentum.value() + grads;

        // Update parameters: theta = theta - lr * v
        Ok(params - &*v * self.learning_rate.value())
    }
}

#[cfg(test)]
#[path = "tests_training.rs"]
mod tests;
