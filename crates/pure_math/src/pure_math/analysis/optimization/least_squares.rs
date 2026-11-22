//! Damped Levenberg-Marquardt nonlinear least-squares solver with box constraints
//! and pre-allocated workspace buffers.

use super::OptimizationError;
use crate::pure_math::algebra::linear_algebra::numerical::{
    moore_penrose_pseudoinverse, solve_normal_equation,
};
use nalgebra::{DMatrix, DVector};

/// Maximum number of calibrated scalar parameters supported by [`LevenbergMarquardtSolver`].
pub const MAX_CALIBRATION_PARAMS: usize = 16;

/// Maximum number of outer Levenberg-Marquardt iterations permitted per calibration run.
pub const MAX_CALIBRATION_ITERATIONS: usize = 100;

/// Maximum number of empirical observation rows permitted per calibration dataset.
pub const MAX_CALIBRATION_OBSERVATIONS: usize = 10_000;

/// Closed box bound `[lower, upper]` constraining a single scalar model parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParameterBounds {
    /// Inclusive lower bound.
    pub lower: f64,
    /// Inclusive upper bound.
    pub upper: f64,
}

impl ParameterBounds {
    /// Creates validated finite box bounds with `lower <= upper`.
    #[verified_engine::verified]
    pub fn new(lower: f64, upper: f64) -> Result<Self, OptimizationError> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(OptimizationError::InvalidBounds { index: 0 });
        }
        Ok(Self { lower, upper })
    }

    /// Clamps `value` into `[self.lower, self.upper]`.
    #[verified_engine::verified]
    pub fn clamp(&self, value: f64) -> f64 {
        value.clamp(self.lower, self.upper)
    }
}

/// Hyperparameters governing damped Levenberg-Marquardt convergence and finite differencing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeastSquaresConfig {
    /// Maximum outer iterations (clamped to `1..=MAX_CALIBRATION_ITERATIONS`).
    pub max_iterations: usize,
    /// Initial Marquardt damping factor $\lambda_0$.
    pub initial_lambda: f64,
    /// Multiplicative $\lambda$ increase on rejected trial step.
    pub lambda_up: f64,
    /// Multiplicative $\lambda$ decrease on accepted trial step.
    pub lambda_down: f64,
    /// Relative parameter step convergence tolerance.
    pub step_tolerance: f64,
    /// Infinity-norm gradient ($J^T r$) convergence tolerance.
    pub gradient_tolerance: f64,
    /// Relative central finite-difference perturbation scale.
    pub fd_epsilon: f64,
    /// Singular-value truncation threshold for pseudoinverse fallback and covariance.
    pub svd_epsilon: f64,
}

impl Default for LeastSquaresConfig {
    fn default() -> Self {
        Self {
            max_iterations: MAX_CALIBRATION_ITERATIONS,
            initial_lambda: 1e-3,
            lambda_up: 10.0,
            lambda_down: 0.2,
            step_tolerance: 1e-8,
            gradient_tolerance: 1e-9,
            fd_epsilon: 1e-6,
            svd_epsilon: 1e-10,
        }
    }
}

/// Summary statistics, confidence intervals, and residuals returned by nonlinear calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationSummary {
    /// Calibrated parameter estimates $\hat{\theta}$.
    pub parameters: Vec<f64>,
    /// Asymptotic parameter standard errors $\sqrt{\hat{\sigma}^2 [(J^T J)^{\dagger}]_{jj}}$.
    pub standard_errors: Vec<f64>,
    /// Half-widths of the two-sided 95% confidence intervals ($t_{0.975, \nu} \cdot \text{SE}_j$).
    pub ci_95_half_widths: Vec<f64>,
    /// Root-mean-square error $\sqrt{\text{RSS} / N}$.
    pub rmse: f64,
    /// Coefficient of determination $R^2 = 1 - \text{RSS} / \text{TSS}$.
    pub r_squared: f64,
    /// Gaussian Akaike Information Criterion $N \ln(\max(\text{RSS}/N, 10^{-30})) + 2p$.
    pub aic: f64,
    /// Number of Levenberg-Marquardt outer iterations executed.
    pub iterations: usize,
    /// Final residuals $r_i = y_{\text{observed}, i} - y_{\text{fitted}, i}$.
    pub residuals: Vec<f64>,
}

/// Stack-only metrics returned by [`LevenbergMarquardtSolver::solve_in_place`] under `VerifiedAllocator`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InPlaceCalibrationMetrics {
    /// Root-mean-square error $\sqrt{\text{RSS} / N}$.
    pub rmse: f64,
    /// Coefficient of determination $R^2$.
    pub r_squared: f64,
    /// Gaussian Akaike Information Criterion.
    pub aic: f64,
    /// Number of Levenberg-Marquardt iterations executed.
    pub iterations: usize,
    /// Active observation count $N$.
    pub observation_count: usize,
    /// Active parameter count $p$.
    pub parameter_count: usize,
}

/// Trait implemented by nonlinear residual objectives ($r_i(\theta) = y_i - \hat{y}_i(\theta)$).
pub trait LeastSquaresObjective {
    /// Total number of scalar residual rows $N$.
    fn observation_count(&self) -> usize;

    /// Evaluates $y_{\text{observed}}$ into `observed_out` so $R^2$ can compute total sum of squares.
    fn copy_observed(&self, observed_out: &mut [f64]);

    /// Evaluates residuals $r_i(\theta) = y_i - \hat{y}_i(\theta)$ in-place without heap allocations.
    ///
    /// Returns `false` if model evaluation encounters non-finite values or domain violations.
    fn eval_residuals(&mut self, params: &[f64], residuals_out: &mut [f64]) -> bool;
}

/// Zero-allocation adapter fitting a scalar curve $y = f(x; \theta)$ to `(x_obs, y_obs)` slices.
pub struct ClosureLeastSquaresObjective<'a, F> {
    x_obs: &'a [f64],
    y_obs: &'a [f64],
    model: F,
}

impl<'a, F> ClosureLeastSquaresObjective<'a, F>
where
    F: FnMut(f64, &[f64]) -> f64,
{
    /// Wraps `(x_obs, y_obs)` slices and a scalar prediction closure `model(x, params) -> y_hat`.
    pub fn new(x_obs: &'a [f64], y_obs: &'a [f64], model: F) -> Result<Self, OptimizationError> {
        let n = x_obs.len();
        if n != y_obs.len() || !(2..=MAX_CALIBRATION_OBSERVATIONS).contains(&n) {
            return Err(OptimizationError::InvalidObservationCount {
                count: n,
                min: 2,
                max: MAX_CALIBRATION_OBSERVATIONS,
            });
        }
        if x_obs.iter().any(|v| !v.is_finite()) || y_obs.iter().any(|v| !v.is_finite()) {
            return Err(OptimizationError::NumericalInstability);
        }
        Ok(Self {
            x_obs,
            y_obs,
            model,
        })
    }
}

impl<F> LeastSquaresObjective for ClosureLeastSquaresObjective<'_, F>
where
    F: FnMut(f64, &[f64]) -> f64,
{
    fn observation_count(&self) -> usize {
        self.y_obs.len()
    }

    fn copy_observed(&self, observed_out: &mut [f64]) {
        let n = self.y_obs.len().min(observed_out.len());
        observed_out[..n].copy_from_slice(&self.y_obs[..n]);
    }

    fn eval_residuals(&mut self, params: &[f64], residuals_out: &mut [f64]) -> bool {
        let n = self.x_obs.len();
        if residuals_out.len() < n || params.iter().any(|p| !p.is_finite()) {
            return false;
        }
        for ((&x_val, &y_val), out_r) in self
            .x_obs
            .iter()
            .zip(self.y_obs.iter())
            .zip(residuals_out.iter_mut())
        {
            let y_hat = (self.model)(x_val, params);
            let res = y_val - y_hat;
            if !y_hat.is_finite() || !res.is_finite() {
                return false;
            }
            *out_r = res;
        }
        true
    }
}

/// Pre-allocated buffers for zero-allocation Levenberg-Marquardt execution.
#[derive(Debug, Clone)]
pub struct LevenbergMarquardtWorkspace {
    max_obs: usize,
    max_params: usize,
    active_obs: usize,
    active_params: usize,
    params: Vec<f64>,
    trial_params: Vec<f64>,
    params_plus: Vec<f64>,
    params_minus: Vec<f64>,
    observed: Vec<f64>,
    residuals: Vec<f64>,
    trial_residuals: Vec<f64>,
    residuals_plus: Vec<f64>,
    residuals_minus: Vec<f64>,
    jacobian: Vec<f64>,
    jtj: Vec<f64>,
    damped_jtj: Vec<f64>,
    neg_jtr: Vec<f64>,
    step: Vec<f64>,
    covariance: Vec<f64>,
    standard_errors: Vec<f64>,
    ci_95_half_widths: Vec<f64>,
}

impl LevenbergMarquardtWorkspace {
    /// Pre-allocates workspace buffers for up to `max_obs` rows and `max_params` parameters.
    pub fn new(max_obs: usize, max_params: usize) -> Result<Self, OptimizationError> {
        if !(1..=MAX_CALIBRATION_PARAMS).contains(&max_params) {
            return Err(OptimizationError::InvalidParameterCount {
                count: max_params,
                max: MAX_CALIBRATION_PARAMS,
            });
        }
        if !(2..=MAX_CALIBRATION_OBSERVATIONS).contains(&max_obs) {
            return Err(OptimizationError::InvalidObservationCount {
                count: max_obs,
                min: 2,
                max: MAX_CALIBRATION_OBSERVATIONS,
            });
        }
        Ok(Self {
            max_obs,
            max_params,
            active_obs: 0,
            active_params: 0,
            params: vec![0.0; max_params],
            trial_params: vec![0.0; max_params],
            params_plus: vec![0.0; max_params],
            params_minus: vec![0.0; max_params],
            observed: vec![0.0; max_obs],
            residuals: vec![0.0; max_obs],
            trial_residuals: vec![0.0; max_obs],
            residuals_plus: vec![0.0; max_obs],
            residuals_minus: vec![0.0; max_obs],
            jacobian: vec![0.0; max_obs * max_params],
            jtj: vec![0.0; max_params * max_params],
            damped_jtj: vec![0.0; max_params * max_params],
            neg_jtr: vec![0.0; max_params],
            step: vec![0.0; max_params],
            covariance: vec![0.0; max_params * max_params],
            standard_errors: vec![0.0; max_params],
            ci_95_half_widths: vec![0.0; max_params],
        })
    }

    /// Active calibrated parameter slice from the most recent solve.
    pub fn parameters(&self) -> &[f64] {
        &self.params[..self.active_params]
    }

    /// Active parameter standard errors from the most recent solve.
    pub fn standard_errors(&self) -> &[f64] {
        &self.standard_errors[..self.active_params]
    }

    /// Active 95% confidence interval half-widths from the most recent solve.
    pub fn ci_95_half_widths(&self) -> &[f64] {
        &self.ci_95_half_widths[..self.active_params]
    }

    /// Active residual slice $r_i = y_i - \hat{y}_i$ from the most recent solve.
    pub fn residuals(&self) -> &[f64] {
        &self.residuals[..self.active_obs]
    }

    /// Active $p \times p$ asymptotic parameter covariance matrix (row-major).
    pub fn covariance(&self) -> &[f64] {
        let pp = self.active_params * self.active_params;
        &self.covariance[..pp]
    }

    /// Materializes an owned [`CalibrationSummary`] from the completed workspace buffers.
    pub fn to_summary(&self, metrics: &InPlaceCalibrationMetrics) -> CalibrationSummary {
        CalibrationSummary {
            parameters: self.parameters().to_vec(),
            standard_errors: self.standard_errors().to_vec(),
            ci_95_half_widths: self.ci_95_half_widths().to_vec(),
            rmse: metrics.rmse,
            r_squared: metrics.r_squared,
            aic: metrics.aic,
            iterations: metrics.iterations,
            residuals: self.residuals().to_vec(),
        }
    }
}

/// Damped Levenberg-Marquardt nonlinear least-squares solver.
#[derive(Debug, Clone, Copy)]
pub struct LevenbergMarquardtSolver {
    config: LeastSquaresConfig,
}

impl Default for LevenbergMarquardtSolver {
    fn default() -> Self {
        Self::new(LeastSquaresConfig::default())
    }
}

#[path = "least_squares_kernel.rs"]
mod kernel;

impl LevenbergMarquardtSolver {
    /// Creates a solver with `config` (clamping `max_iterations` to `1..=100`).
    pub fn new(mut config: LeastSquaresConfig) -> Self {
        config.max_iterations = config.max_iterations.clamp(1, MAX_CALIBRATION_ITERATIONS);
        Self { config }
    }

    /// Solves a nonlinear least-squares problem entirely in-place with zero heap allocations.
    pub fn solve_in_place<O: LeastSquaresObjective>(
        &self,
        objective: &mut O,
        initial_params: &[f64],
        bounds: &[ParameterBounds],
        workspace: &mut LevenbergMarquardtWorkspace,
    ) -> Result<InPlaceCalibrationMetrics, OptimizationError> {
        kernel::run_lm_in_place(&self.config, objective, initial_params, bounds, workspace)
    }

    /// Calibrates `objective` and returns an owned [`CalibrationSummary`].
    pub fn fit<O: LeastSquaresObjective>(
        &self,
        objective: &mut O,
        initial_params: &[f64],
        bounds: &[ParameterBounds],
        workspace: &mut LevenbergMarquardtWorkspace,
    ) -> Result<CalibrationSummary, OptimizationError> {
        let metrics = self.solve_in_place(objective, initial_params, bounds, workspace)?;
        Ok(workspace.to_summary(&metrics))
    }

    /// Computes a damped Gauss-Newton linear step via `nalgebra` normal equations
    /// (`numerical::solve_normal_equation`) with SVD (`numerical::moore_penrose_pseudoinverse`) fallback.
    #[verified_engine::verified]
    pub fn solve_damped_step_matrix(
        jacobian: &DMatrix<f64>,
        residuals: &DVector<f64>,
        lambda: f64,
        svd_epsilon: f64,
    ) -> DVector<f64> {
        let m = jacobian.nrows();
        let p = jacobian.ncols();
        let mut aug_a = DMatrix::zeros(m + p, p);
        aug_a.view_mut((0, 0), (m, p)).copy_from(jacobian);
        let jtj = jacobian.transpose() * jacobian;
        let lam = lambda.max(0.0);
        for j in 0..p {
            let scale = jtj[(j, j)].abs().max(1e-15);
            aug_a[(m + j, j)] = (lam * scale).sqrt();
        }
        let mut aug_b = DVector::zeros(m + p);
        for i in 0..m {
            aug_b[i] = -residuals[i];
        }
        if let Ok(step) = solve_normal_equation(&aug_a, &aug_b)
            && step.iter().all(|v| v.is_finite())
        {
            return step;
        }
        moore_penrose_pseudoinverse(&aug_a, svd_epsilon) * aug_b
    }
}

#[cfg(test)]
#[path = "least_squares_tests.rs"]
mod tests;
