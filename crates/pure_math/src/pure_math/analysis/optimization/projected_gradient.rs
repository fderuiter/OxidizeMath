//! Box-bound projection primitives and spectral Projected Gradient solver.

use super::OptimizationError;
use nalgebra::DVector;

/// Coordinate-wise hyperrectangle bounds $[l, u] = \{x \in \mathbb{R}^n : l_i \le x_i \le u_i\}$.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxBounds {
    /// Coordinate-wise lower bounds $l \in \mathbb{R}^n$.
    pub lower: DVector<f64>,
    /// Coordinate-wise upper bounds $u \in \mathbb{R}^n$.
    pub upper: DVector<f64>,
}

impl BoxBounds {
    /// Creates validated [`BoxBounds`] satisfying $l_i \le u_i$ for all $i$.
    #[verified_engine::verified]
    pub fn new(lower: DVector<f64>, upper: DVector<f64>) -> Result<Self, OptimizationError> {
        if lower.len() != upper.len() {
            return Err(OptimizationError::DimensionMismatch {
                expected: lower.len(),
                actual: upper.len(),
            });
        }
        for (&l, &u) in lower.as_slice().iter().zip(upper.as_slice().iter()) {
            if l.is_nan() || u.is_nan() || l > u {
                return Err(OptimizationError::InvalidBounds);
            }
        }
        Ok(Self { lower, upper })
    }

    /// Creates uniform box bounds $[low, high]^n$.
    #[verified_engine::verified]
    pub fn uniform(dim: usize, low: f64, high: f64) -> Result<Self, OptimizationError> {
        Self::new(
            DVector::from_element(dim, low),
            DVector::from_element(dim, high),
        )
    }

    /// Creates unbounded limits $[-\infty, +\infty]^n$.
    #[verified_engine::verified]
    pub fn unbounded(dim: usize) -> Self {
        Self {
            lower: DVector::from_element(dim, f64::NEG_INFINITY),
            upper: DVector::from_element(dim, f64::INFINITY),
        }
    }

    /// Returns the ambient dimension $n$.
    #[verified_engine::verified]
    pub fn dim(&self) -> usize {
        self.lower.len()
    }

    /// Projects `x` onto $[l, u]$ in-place with zero heap allocations.
    #[verified_engine::verified]
    pub fn project_in_place(&self, x: &mut DVector<f64>) {
        assert_eq!(x.len(), self.lower.len());
        for ((xi, &li), &ui) in x
            .as_mut_slice()
            .iter_mut()
            .zip(self.lower.as_slice().iter())
            .zip(self.upper.as_slice().iter())
        {
            *xi = xi.clamp(li, ui);
        }
    }

    /// Returns the Euclidean projection $P_{[l, u]}(x)$.
    #[verified_engine::verified]
    pub fn project(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut out = x.clone();
        self.project_in_place(&mut out);
        out
    }

    /// Checks whether every coordinate satisfies $l_i - \text{tol} \le x_i \le u_i + \text{tol}$.
    #[verified_engine::verified]
    pub fn is_feasible(&self, x: &DVector<f64>, tol: f64) -> bool {
        if x.len() != self.lower.len() {
            return false;
        }
        for ((&xi, &li), &ui) in x
            .as_slice()
            .iter()
            .zip(self.lower.as_slice().iter())
            .zip(self.upper.as_slice().iter())
        {
            if xi < li - tol || xi > ui + tol {
                return false;
            }
        }
        true
    }

    /// Infinity norm of the projected gradient residual `||x - P_{[l, u]}(x - g)||_inf`.
    #[verified_engine::verified]
    pub fn projected_gradient_norm(&self, x: &DVector<f64>, grad: &DVector<f64>) -> f64 {
        let mut max_res = 0.0_f64;
        for (((&xi, &gi), &li), &ui) in x
            .as_slice()
            .iter()
            .zip(grad.as_slice().iter())
            .zip(self.lower.as_slice().iter())
            .zip(self.upper.as_slice().iter())
        {
            let proj = (xi - gi).clamp(li, ui);
            let res = (xi - proj).abs();
            if res > max_res {
                max_res = res;
            }
        }
        max_res
    }
}

/// Solution summary shared by box-constrained solvers (`ProjectedGradientSolver`, `LbfgsBSolver`).
#[derive(Debug, Clone, PartialEq)]
pub struct ConstrainedSolution {
    /// Converged primal solution $x^* \in [l, u]$.
    pub x: DVector<f64>,
    /// Final objective value $f(x^*)$.
    pub objective: f64,
    /// Infinity norm of final projected gradient $\|x^* - P_{[l, u]}(x^* - \nabla f(x^*))\|_\infty$.
    pub projected_grad_norm: f64,
    /// Number of outer iterations executed.
    pub iterations: usize,
    /// Recorded 2D primal trajectory `(x[0], x[1])`.
    pub iterate_path: Vec<[f64; 2]>,
}

/// Configuration for [`ProjectedGradientSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedGradientConfig {
    /// Maximum outer iterations.
    pub max_iterations: usize,
    /// Projected gradient stopping tolerance.
    pub tolerance: f64,
    /// Maximum Armijo backtracking reductions per iteration.
    pub max_line_search: usize,
    /// Initial trial step size.
    pub initial_step: f64,
}

impl Default for ProjectedGradientConfig {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_iterations: math_commons::registry::MAX_ITERATIONS,
            tolerance: math_commons::registry::TOLERANCE_FAST,
            max_line_search: 28,
            initial_step: 1.0,
        }
    }
}

/// Preallocated workspace for zero-allocation [`ProjectedGradientSolver`] execution.
#[derive(Debug, Clone)]
pub struct ProjectedGradientWorkspace {
    grad: DVector<f64>,
    grad_prev: DVector<f64>,
    x_prev: DVector<f64>,
    x_trial: DVector<f64>,
    iterate_path: Vec<[f64; 2]>,
}

impl ProjectedGradientWorkspace {
    /// Preallocates vectors for ambient dimension `dim`.
    #[verified_engine::verified]
    pub fn new(dim: usize, max_iterations: usize) -> Self {
        Self {
            grad: DVector::zeros(dim),
            grad_prev: DVector::zeros(dim),
            x_prev: DVector::zeros(dim),
            x_trial: DVector::zeros(dim),
            iterate_path: Vec::with_capacity(max_iterations + 2),
        }
    }

    /// Returns the recorded 2D iterate trajectory slice.
    #[verified_engine::verified]
    pub fn iterate_path_slice(&self) -> &[[f64; 2]] {
        &self.iterate_path
    }
}

/// Spectral Barzilai-Borwein Projected Gradient solver for smooth box-constrained minimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedGradientSolver {
    /// Stopping and line-search hyperparameters.
    pub config: ProjectedGradientConfig,
}

impl Default for ProjectedGradientSolver {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            config: ProjectedGradientConfig::default(),
        }
    }
}

fn spectral_bb_step(
    x: &DVector<f64>,
    x_prev: &DVector<f64>,
    grad: &DVector<f64>,
    grad_prev: &DVector<f64>,
    fallback: f64,
) -> f64 {
    let mut ss = 0.0;
    let mut sy = 0.0;
    for i in 0..x.len() {
        let s = x[i] - x_prev[i];
        let y = grad[i] - grad_prev[i];
        ss += s * s;
        sy += s * y;
    }
    if sy > 1e-14 && ss > 1e-20 {
        (ss / sy).clamp(1e-8, 1e4)
    } else {
        fallback
    }
}

fn projected_armijo_search<F: FnMut(&DVector<f64>, &mut DVector<f64>) -> f64>(
    fg: &mut F,
    bounds: &BoxBounds,
    x: &DVector<f64>,
    fx: f64,
    alpha0: f64,
    max_ls: usize,
    ws: &mut ProjectedGradientWorkspace,
) -> Result<(f64, bool), OptimizationError> {
    let mut alpha = alpha0;
    for _ in 0..max_ls {
        let mut dir_dot = 0.0;
        for i in 0..x.len() {
            let xi = (x[i] - alpha * ws.grad[i]).clamp(bounds.lower[i], bounds.upper[i]);
            ws.x_trial[i] = xi;
            dir_dot += ws.grad[i] * (xi - x[i]);
        }
        let f_trial = fg(&ws.x_trial, &mut ws.grad_prev);
        if !f_trial.is_finite() {
            alpha *= 0.5;
            continue;
        }
        if f_trial <= fx + 1e-4 * dir_dot + 1e-15 {
            return Ok((f_trial, true));
        }
        alpha *= 0.5;
    }
    Ok((fx, false))
}

impl ProjectedGradientSolver {
    /// Creates a new [`ProjectedGradientSolver`] with custom configuration.
    #[verified_engine::verified]
    pub fn new(config: ProjectedGradientConfig) -> Self {
        Self { config }
    }

    /// Minimizes `fg` over `bounds` in-place with zero heap allocations.
    #[verified_engine::verified]
    pub fn solve_in_place<F>(
        &self,
        mut fg: F,
        bounds: &BoxBounds,
        x: &mut DVector<f64>,
        workspace: &mut ProjectedGradientWorkspace,
    ) -> Result<(f64, f64, usize), OptimizationError>
    where
        F: FnMut(&DVector<f64>, &mut DVector<f64>) -> f64,
    {
        let n = x.len();
        if bounds.dim() != n || workspace.grad.len() != n {
            return Err(OptimizationError::DimensionMismatch {
                expected: bounds.dim(),
                actual: n,
            });
        }
        bounds.project_in_place(x);
        workspace.iterate_path.clear();
        let mut fx = fg(x, &mut workspace.grad);
        if !fx.is_finite() {
            return Err(OptimizationError::NonFiniteValue);
        }
        if n >= 2 && workspace.iterate_path.len() < workspace.iterate_path.capacity() {
            workspace.iterate_path.push([x[0], x[1]]);
        }
        let mut alpha = self.config.initial_step;
        let mut iters = 0;
        for iter in 1..=self.config.max_iterations {
            let pg = bounds.projected_gradient_norm(x, &workspace.grad);
            if pg <= self.config.tolerance {
                return Ok((fx, pg, iter - 1));
            }
            iters = iter;
            if iter > 1 {
                alpha = spectral_bb_step(
                    x,
                    &workspace.x_prev,
                    &workspace.grad,
                    &workspace.grad_prev,
                    self.config.initial_step,
                );
            }
            let (f_trial, accepted) = projected_armijo_search(
                &mut fg,
                bounds,
                x,
                fx,
                alpha,
                self.config.max_line_search,
                workspace,
            )?;
            if !accepted {
                break;
            }
            // Save old x & grad into x_prev & grad_prev, while new grad is temporarily in grad_prev:
            workspace.x_prev.copy_from(x);
            x.copy_from(&workspace.x_trial);
            std::mem::swap(&mut workspace.grad, &mut workspace.grad_prev);
            fx = f_trial;
            if n >= 2 && workspace.iterate_path.len() < workspace.iterate_path.capacity() {
                workspace.iterate_path.push([x[0], x[1]]);
            }
        }
        let pg = bounds.projected_gradient_norm(x, &workspace.grad);
        Ok((fx, pg, iters))
    }

    /// Allocates workspace and minimizes `fg` over `bounds` starting from `x0`.
    #[verified_engine::verified]
    pub fn solve<F>(
        &self,
        fg: F,
        bounds: &BoxBounds,
        x0: &DVector<f64>,
    ) -> Result<ConstrainedSolution, OptimizationError>
    where
        F: FnMut(&DVector<f64>, &mut DVector<f64>) -> f64,
    {
        let mut x = x0.clone();
        let mut ws = ProjectedGradientWorkspace::new(x.len(), self.config.max_iterations);
        let (objective, projected_grad_norm, iterations) =
            self.solve_in_place(fg, bounds, &mut x, &mut ws)?;
        Ok(ConstrainedSolution {
            x,
            objective,
            projected_grad_norm,
            iterations,
            iterate_path: ws.iterate_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projected_gradient_hits_active_boundary() {
        let bounds = BoxBounds::uniform(2, 0.0, 1.0).expect("valid bounds");
        let x0 = DVector::from_vec(vec![0.8, 0.8]);
        let sol = ProjectedGradientSolver::default()
            .solve(
                |x, g| {
                    g[0] = 2.0 * (x[0] - 1.5);
                    g[1] = 2.0 * (x[1] + 0.5);
                    (x[0] - 1.5).powi(2) + (x[1] + 0.5).powi(2)
                },
                &bounds,
                &x0,
            )
            .expect("converges");
        assert!((sol.x[0] - 1.0).abs() < 1e-8);
        assert!(sol.x[1].abs() < 1e-8);
    }
}
