//! Proximal `L1`-regularized least squares (Lasso) solvers: ISTA and FISTA.

use super::OptimizationError;
use nalgebra::{DMatrix, DVector};

/// Scalar proximal soft-thresholding operator $S_\tau(v) = \operatorname{sign}(v)\max(|v| - \tau, 0)$.
#[verified_engine::verified]
pub fn soft_threshold_scalar(value: f64, threshold: f64) -> f64 {
    let tau = threshold.max(0.0);
    if value > tau {
        value - tau
    } else if value < -tau {
        value + tau
    } else {
        0.0
    }
}

/// Applies elementwise soft-thresholding in-place with zero heap allocations.
#[verified_engine::verified]
pub fn soft_threshold_in_place(v: &mut DVector<f64>, threshold: f64) {
    let tau = threshold.max(0.0);
    for elem in v.as_mut_slice() {
        *elem = soft_threshold_scalar(*elem, tau);
    }
}

/// Returns a new soft-thresholded vector $S_\tau(v)$.
#[verified_engine::verified]
pub fn soft_threshold(v: &DVector<f64>, threshold: f64) -> DVector<f64> {
    let mut out = v.clone();
    soft_threshold_in_place(&mut out, threshold);
    out
}

/// Stopping and step-size parameters for proximal ISTA/FISTA solvers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProximalConfig {
    /// Maximum proximal iterations.
    pub max_iterations: usize,
    /// Infinity-norm step convergence tolerance $\|x_{k+1} - x_k\|_\infty \le \text{tolerance}$.
    pub tolerance: f64,
    /// Optional fixed step size $\alpha > 0$; if `None`, Beck-Teboulle Lipschitz backtracking is used.
    pub step_size: Option<f64>,
}

impl Default for ProximalConfig {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_iterations: math_commons::registry::MAX_ITERATIONS,
            tolerance: math_commons::registry::TOLERANCE_FAST,
            step_size: None,
        }
    }
}

/// Solution summary returned by allocating ISTA/FISTA solvers.
#[derive(Debug, Clone, PartialEq)]
pub struct ProximalSolution {
    /// Converged sparse parameter vector $x^*$.
    pub x: DVector<f64>,
    /// Final composite objective $\frac{1}{2}\|A x^* - y\|_2^2 + \lambda \|x^*\|_1$.
    pub cost: f64,
    /// Number of iterations executed.
    pub iterations: usize,
}

/// Preallocated scratch buffers for zero-allocation ISTA and FISTA execution.
#[derive(Debug, Clone)]
pub struct ProximalWorkspace {
    residual: DVector<f64>,
    grad: DVector<f64>,
    x_next: DVector<f64>,
    y_extrap: DVector<f64>,
    trial_res: DVector<f64>,
}

impl ProximalWorkspace {
    /// Allocates reusable scratch vectors for an `nrows x ncols` linear model.
    #[verified_engine::verified]
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            residual: DVector::zeros(nrows),
            grad: DVector::zeros(ncols),
            x_next: DVector::zeros(ncols),
            y_extrap: DVector::zeros(ncols),
            trial_res: DVector::zeros(nrows),
        }
    }

    fn validate_dims(
        &self,
        a: &DMatrix<f64>,
        x: &DVector<f64>,
        y: &DVector<f64>,
    ) -> Result<(), OptimizationError> {
        if a.nrows() != y.len() {
            return Err(OptimizationError::DimensionMismatch {
                expected: a.nrows(),
                actual: y.len(),
            });
        }
        if a.ncols() != x.len() {
            return Err(OptimizationError::DimensionMismatch {
                expected: a.ncols(),
                actual: x.len(),
            });
        }
        if self.residual.len() != a.nrows() || self.grad.len() != a.ncols() {
            return Err(OptimizationError::DimensionMismatch {
                expected: a.ncols(),
                actual: self.grad.len(),
            });
        }
        Ok(())
    }
}

/// Composite $\ell_1$-regularized least squares problem
/// $\min_x \frac{1}{2}\|A x - y\|_2^2 + \lambda \|x\|_1$.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::{DMatrix, DVector};
/// use math_explorer::pure_math::analysis::optimization::{
///     L1RegularizedLeastSquares, ProximalConfig,
/// };
///
/// let a = DMatrix::identity(3, 3);
/// let y = DVector::from_vec(vec![2.5, -0.05, -1.8]);
/// let lasso = L1RegularizedLeastSquares::new(0.2);
/// let sol = lasso.solve_fista(&a, &y, &ProximalConfig::default()).unwrap();
/// assert!((sol.x[0] - 2.3).abs() < 1e-5);
/// assert!(sol.x[1].abs() < 1e-9);
/// assert!((sol.x[2] + 1.6).abs() < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct L1RegularizedLeastSquares {
    lambda: f64,
}

fn eval_smooth_and_grad(
    a: &DMatrix<f64>,
    point: &DVector<f64>,
    y: &DVector<f64>,
    residual: &mut DVector<f64>,
    grad: &mut DVector<f64>,
) -> f64 {
    residual.copy_from(y);
    residual.gemv(-1.0, a, point, 1.0);
    let smooth = 0.5 * residual.norm_squared();
    grad.gemv_tr(-1.0, a, residual, 0.0);
    smooth
}

fn prox_step_with_lipschitz(
    a: &DMatrix<f64>,
    y: &DVector<f64>,
    center: &DVector<f64>,
    lambda: f64,
    fixed_step: Option<f64>,
    lipschitz: &mut f64,
    ws: &mut ProximalWorkspace,
) -> Result<(), OptimizationError> {
    let f_center = eval_smooth_and_grad(a, center, y, &mut ws.residual, &mut ws.grad);
    if !f_center.is_finite() {
        return Err(OptimizationError::NonFiniteValue);
    }
    if let Some(alpha) = fixed_step {
        for i in 0..center.len() {
            ws.x_next[i] = soft_threshold_scalar(center[i] - alpha * ws.grad[i], alpha * lambda);
        }
        return Ok(());
    }
    for _ in 0..32 {
        let alpha = 1.0 / *lipschitz;
        let mut lin_dot = 0.0;
        let mut sq_diff = 0.0;
        for i in 0..center.len() {
            let xi = soft_threshold_scalar(center[i] - alpha * ws.grad[i], alpha * lambda);
            ws.x_next[i] = xi;
            let diff = xi - center[i];
            lin_dot += ws.grad[i] * diff;
            sq_diff += diff * diff;
        }
        ws.trial_res.copy_from(y);
        ws.trial_res.gemv(-1.0, a, &ws.x_next, 1.0);
        let f_next = 0.5 * ws.trial_res.norm_squared();
        if f_next <= f_center + lin_dot + 0.5 * (*lipschitz) * sq_diff + 1e-14 {
            return Ok(());
        }
        *lipschitz *= 2.0;
    }
    Ok(())
}

fn max_abs_diff(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    let mut max_d = 0.0_f64;
    for (&u, &v) in a.as_slice().iter().zip(b.as_slice().iter()) {
        let d = (u - v).abs();
        if d > max_d {
            max_d = d;
        }
    }
    max_d
}

impl L1RegularizedLeastSquares {
    /// Creates a new `L1RegularizedLeastSquares` objective with penalty weight `lambda`.
    #[verified_engine::verified]
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Returns the configured $\ell_1$ penalty coefficient $\lambda$.
    #[verified_engine::verified]
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Evaluates $\frac{1}{2}\|A x - y\|_2^2 + \lambda \|x\|_1$ (legacy mutable receiver kept for API stability).
    #[verified_engine::verified]
    pub fn cost(&mut self, a: &DMatrix<f64>, x: &DVector<f64>, y: &DVector<f64>) -> f64 {
        self.evaluate_cost(a, x, y)
    }

    /// Evaluates $\frac{1}{2}\|A x - y\|_2^2 + \lambda \|x\|_1$ without requiring mutable access.
    #[verified_engine::verified]
    pub fn evaluate_cost(&self, a: &DMatrix<f64>, x: &DVector<f64>, y: &DVector<f64>) -> f64 {
        let residual = y - (a * x);
        let l2_term = 0.5 * residual.norm_squared();
        let l1_term = x.iter().map(|v| v.abs()).sum::<f64>();
        l2_term + self.lambda * l1_term
    }

    fn validate_config(&self, config: &ProximalConfig) -> Result<(), OptimizationError> {
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(OptimizationError::InvalidParameter(
                "lambda must be finite and non-negative",
            ));
        }
        if config.tolerance <= 0.0 || !config.tolerance.is_finite() {
            return Err(OptimizationError::InvalidParameter(
                "tolerance must be positive and finite",
            ));
        }
        if let Some(step) = config.step_size
            && (!step.is_finite() || step <= 0.0)
        {
            return Err(OptimizationError::InvalidParameter(
                "step_size must be positive and finite",
            ));
        }
        Ok(())
    }

    /// Solves the Lasso objective in-place via ISTA using preallocated `workspace` (zero heap allocations).
    #[verified_engine::verified]
    pub fn solve_ista_in_place(
        &self,
        a: &DMatrix<f64>,
        y: &DVector<f64>,
        x: &mut DVector<f64>,
        config: &ProximalConfig,
        workspace: &mut ProximalWorkspace,
    ) -> Result<usize, OptimizationError> {
        self.validate_config(config)?;
        workspace.validate_dims(a, x, y)?;
        if x.is_empty() {
            return Ok(0);
        }
        let mut lipschitz = a.column(0).norm_squared().max(1e-6);
        for iter in 1..=config.max_iterations {
            prox_step_with_lipschitz(
                a,
                y,
                x,
                self.lambda,
                config.step_size,
                &mut lipschitz,
                workspace,
            )?;
            let step_err = max_abs_diff(&workspace.x_next, x);
            x.copy_from(&workspace.x_next);
            if step_err <= config.tolerance {
                return Ok(iter);
            }
        }
        Ok(config.max_iterations)
    }

    /// Solves the Lasso objective in-place via accelerated FISTA using preallocated `workspace` (zero heap allocations).
    #[verified_engine::verified]
    pub fn solve_fista_in_place(
        &self,
        a: &DMatrix<f64>,
        y: &DVector<f64>,
        x: &mut DVector<f64>,
        config: &ProximalConfig,
        workspace: &mut ProximalWorkspace,
    ) -> Result<usize, OptimizationError> {
        self.validate_config(config)?;
        workspace.validate_dims(a, x, y)?;
        if x.is_empty() {
            return Ok(0);
        }
        workspace.y_extrap.copy_from(x);
        let mut t = 1.0_f64;
        let mut lipschitz = a.column(0).norm_squared().max(1e-6);
        for iter in 1..=config.max_iterations {
            std::mem::swap(&mut workspace.y_extrap, x);
            // After swap: `x` holds `y_k`, `workspace.y_extrap` holds `x_k`.
            prox_step_with_lipschitz(
                a,
                y,
                x,
                self.lambda,
                config.step_size,
                &mut lipschitz,
                workspace,
            )?;
            let step_err = max_abs_diff(&workspace.x_next, &workspace.y_extrap);
            let mut restart_dot = 0.0;
            for i in 0..x.len() {
                let y_k = x[i];
                let x_kp1 = workspace.x_next[i];
                let x_k = workspace.y_extrap[i];
                restart_dot += (y_k - x_kp1) * (x_kp1 - x_k);
            }
            let (t_next, beta) = if restart_dot > 0.0 {
                (1.0, 0.0)
            } else {
                let tn = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
                (tn, (t - 1.0) / tn)
            };
            for i in 0..x.len() {
                let x_kp1 = workspace.x_next[i];
                let x_k = workspace.y_extrap[i];
                x[i] = x_kp1;
                workspace.y_extrap[i] = x_kp1 + beta * (x_kp1 - x_k);
            }
            t = t_next;
            if step_err <= config.tolerance {
                return Ok(iter);
            }
        }
        Ok(config.max_iterations)
    }

    /// Allocates workspace and solves the Lasso problem via ISTA.
    #[verified_engine::verified]
    pub fn solve_ista(
        &self,
        a: &DMatrix<f64>,
        y: &DVector<f64>,
        config: &ProximalConfig,
    ) -> Result<ProximalSolution, OptimizationError> {
        let mut x = DVector::zeros(a.ncols());
        let mut ws = ProximalWorkspace::new(a.nrows(), a.ncols());
        let iterations = self.solve_ista_in_place(a, y, &mut x, config, &mut ws)?;
        let cost = self.evaluate_cost(a, &x, y);
        Ok(ProximalSolution {
            x,
            cost,
            iterations,
        })
    }

    /// Allocates workspace and solves the Lasso problem via Nesterov-accelerated FISTA.
    #[verified_engine::verified]
    pub fn solve_fista(
        &self,
        a: &DMatrix<f64>,
        y: &DVector<f64>,
        config: &ProximalConfig,
    ) -> Result<ProximalSolution, OptimizationError> {
        let mut x = DVector::zeros(a.ncols());
        let mut ws = ProximalWorkspace::new(a.nrows(), a.ncols());
        let iterations = self.solve_fista_in_place(a, y, &mut x, config, &mut ws)?;
        let cost = self.evaluate_cost(a, &x, y);
        Ok(ProximalSolution {
            x,
            cost,
            iterations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ista_and_fista_match_orthogonal_closed_form() {
        let a = DMatrix::identity(4, 4);
        let y = DVector::from_vec(vec![3.0, -0.15, 0.08, -2.2]);
        let lasso = L1RegularizedLeastSquares::new(0.2);
        let cfg = ProximalConfig {
            tolerance: 1e-9,
            ..ProximalConfig::default()
        };
        let ista = lasso.solve_ista(&a, &y, &cfg).expect("ista");
        let fista = lasso.solve_fista(&a, &y, &cfg).expect("fista");
        let expected = [2.8, 0.0, 0.0, -2.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((ista.x[i] - exp).abs() < 1e-6);
            assert!((fista.x[i] - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fista_converges_on_correlated_design_matrix() {
        let a = DMatrix::from_row_slice(3, 3, &[2.0, 0.5, 0.0, 0.3, 1.5, 0.2, 0.0, 0.4, 1.2]);
        let y = DVector::from_vec(vec![2.1, -1.4, 0.05]);
        let lasso = L1RegularizedLeastSquares::new(0.35);
        let cfg = ProximalConfig {
            max_iterations: 2_000,
            tolerance: 1e-9,
            step_size: None,
        };
        let ista = lasso.solve_ista(&a, &y, &cfg).expect("ista");
        let fista = lasso.solve_fista(&a, &y, &cfg).expect("fista");
        assert!((ista.x - &fista.x).norm() < 1e-5);
        assert!(fista.iterations <= ista.iterations);
    }
}
