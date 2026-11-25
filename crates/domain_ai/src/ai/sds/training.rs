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

/// First-order parameter matrix update interface for Score Distillation Sampling optimization.
///
/// Advances differentiable scene parameters $\boldsymbol{\theta}_t \in \mathbb{R}^{m \times n}$ along incoming score gradients $\mathbf{g}_t$:
///
/// $$\boldsymbol{\theta}_{t+1} = \mathcal{U}(\boldsymbol{\theta}_t, \mathbf{g}_t), \qquad \mathbf{g}_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\mathrm{SDS}}(\boldsymbol{\theta}_t)$$
///
/// Reference: Poole et al. (2022), arXiv:2209.14988; see
/// [`docs/math_specs/domain_ai.md`](../../../../../docs/math_specs/domain_ai.md).
///
/// # Stability
/// Implementations validate exact `(nrows, ncols)` shape agreement between `params`, `grads`, and
/// internal moment buffers before mutating optimizer state.
///
/// # Domain of Validity
/// Finite real matrices `params` and `grads` of identical dimensions `(m, n)`.
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

/// Adaptive Moment Estimation (`Adam`) optimizer over a matrix of differentiable scene parameters.
///
/// Maintains bias-corrected exponential moving averages of first and second gradient moments:
///
/// $$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\mathbf{g}_t, \quad \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\mathbf{g}_t^{\odot 2}, \quad \boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$
///
/// Reference: Kingma & Ba (2015), arXiv:1412.6980; see
/// [`docs/math_specs/domain_ai.md`](../../../../../docs/math_specs/domain_ai.md).
///
/// # Stability
/// Bias correction $(1 - \beta_1^t)^{-1}$ and $(1 - \beta_2^t)^{-1}$ prevents initial step bias,
/// while strictly positive $\epsilon > 0$ guards elementwise division against zero variance.
///
/// # Domain of Validity
/// Strictly positive `learning_rate` $\eta > 0$ and `epsilon` $\epsilon > 0$, decay factors
/// $\beta_1, \beta_2 \in [0, 1)$, step count $t < \text{i32::MAX}$, and finite matching parameter/gradient shapes.
#[derive(Debug, Clone, PartialEq)]
pub struct AdamOptimizer {
    /// Step-size hyperparameter $\eta > 0$.
    pub learning_rate: PositiveFloat,
    /// Exponential decay factor $\beta_1 \in [0, 1]$ for the first moment estimate.
    pub beta1: UnitInterval,
    /// Exponential decay factor $\beta_2 \in [0, 1]$ for the second raw moment estimate.
    pub beta2: UnitInterval,
    /// Numerical damping constant $\epsilon > 0$ added to the root-mean-square denominator.
    pub epsilon: PositiveFloat,
    /// First-moment running average matrix $\mathbf{m}_t$, lazily initialized on first step.
    pub m: Option<DMatrix<f64>>,
    /// Second-moment running average matrix $\mathbf{v}_t$, lazily initialized on first step.
    pub v: Option<DMatrix<f64>>,
    /// Completed optimization step counter $t$.
    pub t: usize,
}

impl AdamOptimizer {
    /// Creates an `AdamOptimizer` with standard defaults ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$).
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

/// Heavy-ball Stochastic Gradient Descent (`SGD`) optimizer with Polyak momentum.
///
/// Accumulates exponentially damped velocity before updating parameters:
///
/// $$\mathbf{v}_{t+1} = \mu \mathbf{v}_t + \mathbf{g}_t, \qquad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{v}_{t+1}$$
///
/// Reference: Polyak (1964), DOI: 10.1016/0041-5553(64)90137-5; Poole et al. (2022), arXiv:2209.14988;
/// see [`docs/math_specs/domain_ai.md`](../../../../../docs/math_specs/domain_ai.md).
///
/// # Stability
/// On $L$-smooth objectives, convergence requires $0 < \eta < 2(1 + \mu)/L$ with $\mu \in [0, 1)$;
/// setting $\mu = 1$ disables velocity damping and can cause unbounded oscillation.
///
/// # Domain of Validity
/// Strictly positive `learning_rate` $\eta > 0$, `momentum` $\mu \in [0, 1)$, and finite matching
/// parameter/gradient matrices.
#[derive(Debug, Clone, PartialEq)]
pub struct SgdOptimizer {
    /// Step-size hyperparameter $\eta > 0$.
    pub learning_rate: PositiveFloat,
    /// Polyak momentum damping coefficient $\mu \in [0, 1]$.
    pub momentum: UnitInterval,
    /// Accumulated velocity matrix $\mathbf{v}_t$, lazily initialized on first step.
    pub velocity: Option<DMatrix<f64>>,
}

impl SgdOptimizer {
    /// Constructs a momentum SGD optimizer with `learning_rate` and `momentum`.
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
