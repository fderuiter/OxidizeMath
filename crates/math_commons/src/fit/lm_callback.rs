//! Callback traits and reverse-mode autodiff adapters for [`super::levenberg_marquardt_numeric`].
// [cite:math_commons]

use super::lm::{MAX_FIT_PARAMS, numeric_central_jacobian};
use crate::ast::{CompiledExpr, SolverWorkspace};

/// Scalar value and parameter-Jacobian callback for [`super::levenberg_marquardt_numeric`].
///
/// Implemented for:
/// - Pre-compiled [`CompiledExpr`] / `&CompiledExpr` (`slot 0 = x`, `slots 1..=k = params`),
///   evaluating $\hat{f}(x; \theta)$ and all $k \le 6$ parameter partials in **1** reverse pass
///   per observation via [`CompiledExpr::eval_grad_into`].
/// - [`ReverseGradFit`] pairing value and reverse-gradient closures over a shared
///   [`SolverWorkspace`].
/// - Plain scalar closures `Fn(f64, &[f64]) -> Option<f64>` as a backward-compatible fallback.
pub trait NumericFitCallback {
    /// Evaluates $\hat{f}(x; \theta)$ at `x` for `params[..k]`.
    fn eval_point(&self, x: f64, params: &[f64]) -> Option<f64>;

    /// Evaluates $\hat{f}(x; \theta)$ and $\nabla_\theta \hat{f}(x; \theta)$ using `workspace`.
    fn eval_point_and_jac(
        &self,
        x: f64,
        theta: &[f64; MAX_FIT_PARAMS],
        k: usize,
        workspace: &mut SolverWorkspace,
    ) -> (f64, [f64; MAX_FIT_PARAMS]);
}

impl<F> NumericFitCallback for F
where
    F: Fn(f64, &[f64]) -> Option<f64>,
{
    #[inline]
    fn eval_point(&self, x: f64, params: &[f64]) -> Option<f64> {
        self(x, params)
    }

    #[inline]
    fn eval_point_and_jac(
        &self,
        x: f64,
        theta: &[f64; MAX_FIT_PARAMS],
        k: usize,
        _workspace: &mut SolverWorkspace,
    ) -> (f64, [f64; MAX_FIT_PARAMS]) {
        numeric_central_jacobian(x, theta, k, self)
    }
}

#[verified_engine::verified]
fn eval_compiled_point_and_jac(
    compiled: &CompiledExpr,
    x: f64,
    theta: &[f64; MAX_FIT_PARAMS],
    k: usize,
    workspace: &mut SolverWorkspace,
) -> (f64, [f64; MAX_FIT_PARAMS]) {
    let active_k = k.min(MAX_FIT_PARAMS);
    let mut vars = [0.0_f64; MAX_FIT_PARAMS + 1];
    let mut grads = [0.0_f64; MAX_FIT_PARAMS + 1];
    vars[0] = x;
    vars[1..=active_k].copy_from_slice(&theta[..active_k]);
    let y = compiled.eval_grad_into(&vars[..=active_k], workspace, &mut grads[..=active_k]);
    if !y.is_finite() {
        return (f64::NAN, [0.0; MAX_FIT_PARAMS]);
    }
    let mut jac = [0.0_f64; MAX_FIT_PARAMS];
    jac[..active_k].copy_from_slice(&grads[1..=active_k]);
    (y, jac)
}

impl NumericFitCallback for CompiledExpr {
    #[inline]
    #[verified_engine::verified]
    fn eval_point(&self, x: f64, params: &[f64]) -> Option<f64> {
        let y = self.eval_1d(x, params);
        y.is_finite().then_some(y)
    }

    #[inline]
    #[verified_engine::verified]
    fn eval_point_and_jac(
        &self,
        x: f64,
        theta: &[f64; MAX_FIT_PARAMS],
        k: usize,
        workspace: &mut SolverWorkspace,
    ) -> (f64, [f64; MAX_FIT_PARAMS]) {
        eval_compiled_point_and_jac(self, x, theta, k, workspace)
    }
}

impl NumericFitCallback for &CompiledExpr {
    #[inline]
    #[verified_engine::verified]
    fn eval_point(&self, x: f64, params: &[f64]) -> Option<f64> {
        let y = self.eval_1d(x, params);
        y.is_finite().then_some(y)
    }

    #[inline]
    #[verified_engine::verified]
    fn eval_point_and_jac(
        &self,
        x: f64,
        theta: &[f64; MAX_FIT_PARAMS],
        k: usize,
        workspace: &mut SolverWorkspace,
    ) -> (f64, [f64; MAX_FIT_PARAMS]) {
        eval_compiled_point_and_jac(self, x, theta, k, workspace)
    }
}

/// Zero-allocation adapter pairing a primal value closure with a single-pass reverse-gradient
/// callback backed by [`SolverWorkspace`].
#[derive(Debug, Clone, Copy)]
pub struct ReverseGradFit<V, G> {
    /// Evaluates $\hat{f}(x; \theta)$ for residual line searches.
    pub eval_val: V,
    /// Evaluates $\hat{f}(x; \theta)$ and writes $\nabla_\theta \hat{f}(x; \theta)$ into `grad_out`.
    pub eval_grad: G,
}

impl<V, G> NumericFitCallback for ReverseGradFit<V, G>
where
    V: Fn(f64, &[f64]) -> Option<f64>,
    G: Fn(f64, &[f64], &mut [f64], &mut SolverWorkspace) -> Option<f64>,
{
    #[inline]
    fn eval_point(&self, x: f64, params: &[f64]) -> Option<f64> {
        (self.eval_val)(x, params)
    }

    #[inline]
    fn eval_point_and_jac(
        &self,
        x: f64,
        theta: &[f64; MAX_FIT_PARAMS],
        k: usize,
        workspace: &mut SolverWorkspace,
    ) -> (f64, [f64; MAX_FIT_PARAMS]) {
        let active_k = k.min(MAX_FIT_PARAMS);
        let mut jac = [0.0_f64; MAX_FIT_PARAMS];
        let Some(y) = (self.eval_grad)(x, &theta[..active_k], &mut jac[..active_k], workspace)
        else {
            return (f64::NAN, [0.0; MAX_FIT_PARAMS]);
        };
        if !y.is_finite() {
            return (f64::NAN, [0.0; MAX_FIT_PARAMS]);
        }
        (y, jac)
    }
}
