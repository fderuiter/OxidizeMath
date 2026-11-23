use crate::ai::sds::rendering::{NeRFModel, RayBundle};
use crate::error::AIError;
use math_commons::primitives::{PositiveFloat, UnitInterval};
use nalgebra::DMatrix;

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
        if params.shape() != grads.shape() {
            return Err(math_commons::error::MathError::DimensionMismatch {
                expected: math_commons::math_kernel::types::Dimension(
                    params.nrows() * params.ncols(),
                ),
                actual: math_commons::math_kernel::types::Dimension(grads.nrows() * grads.ncols()),
            }
            .into());
        }

        self.t += 1;

        // Initialize state if needed
        if self.m.is_none() {
            self.m = Some(DMatrix::zeros(params.nrows(), params.ncols()));
            self.v = Some(DMatrix::zeros(params.nrows(), params.ncols()));
        }

        let m = self.m.as_mut().ok_or(AIError::UninitializedState {
            name: "m".to_string(),
        })?;
        let v = self.v.as_mut().ok_or(AIError::UninitializedState {
            name: "v".to_string(),
        })?;

        // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        *m = &*m * self.beta1.value() + grads * (1.0 - self.beta1.value());

        // Update biased second raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        // Element-wise square of gradients
        let grads_sq = grads.map(|x| x * x);
        *v = &*v * self.beta2.value() + grads_sq * (1.0 - self.beta2.value());

        // Compute bias-corrected first moment estimate
        let m_hat = &*m / (1.0 - self.beta1.value().powi(self.t as i32));

        // Compute bias-corrected second raw moment estimate
        let v_hat = &*v / (1.0 - self.beta2.value().powi(self.t as i32));

        // Update parameters: theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)
        let update_term = m_hat.component_div(&v_hat.map(|x| x.sqrt() + self.epsilon.value()));

        Ok(params - update_term * self.learning_rate.value())
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
        if params.shape() != grads.shape() {
            return Err(math_commons::error::MathError::DimensionMismatch {
                expected: math_commons::math_kernel::types::Dimension(
                    params.nrows() * params.ncols(),
                ),
                actual: math_commons::math_kernel::types::Dimension(grads.nrows() * grads.ncols()),
            }
            .into());
        }

        // Initialize state if needed
        if self.velocity.is_none() {
            self.velocity = Some(DMatrix::zeros(params.nrows(), params.ncols()));
        }

        let v = self.velocity.as_mut().ok_or(AIError::UninitializedState {
            name: "velocity".to_string(),
        })?;

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
