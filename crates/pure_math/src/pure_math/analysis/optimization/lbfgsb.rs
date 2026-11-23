//! Box-constrained limited-memory BFGS (`L-BFGS-B`) solver with preallocated circular history.

use super::OptimizationError;
use super::projected_gradient::{BoxBounds, ConstrainedSolution};
use nalgebra::{DMatrix, DVector};

/// Configuration parameters for [`LbfgsBSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LbfgsBConfig {
    /// Number of curvature pairs $(s_k, y_k)$ retained (`memory >= 1`).
    pub memory: usize,
    /// Maximum quasi-Newton outer iterations.
    pub max_iterations: usize,
    /// Infinity-norm projected gradient tolerance.
    pub tolerance: f64,
    /// Maximum backtracking trials per outer iteration.
    pub max_line_search: usize,
}

impl Default for LbfgsBConfig {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            memory: 8,
            max_iterations: math_commons::registry::MAX_ITERATIONS,
            tolerance: math_commons::registry::TOLERANCE_FAST,
            max_line_search: 26,
        }
    }
}

/// Preallocated scratch and curvature-history buffers for zero-allocation `L-BFGS-B` solves.
#[derive(Debug, Clone)]
pub struct LbfgsBWorkspace {
    s_hist: DMatrix<f64>,
    y_hist: DMatrix<f64>,
    rho: Vec<f64>,
    alpha_buf: Vec<f64>,
    active_clamped: Vec<bool>,
    grad: DVector<f64>,
    grad_trial: DVector<f64>,
    direction: DVector<f64>,
    x_trial: DVector<f64>,
    s_vec: DVector<f64>,
    y_vec: DVector<f64>,
    iterate_path: Vec<[f64; 2]>,
    head: usize,
    count: usize,
}

impl LbfgsBWorkspace {
    /// Preallocates history and vector buffers for ambient dimension `dim` and history size `memory`.
    #[verified_engine::verified]
    pub fn new(dim: usize, memory: usize, max_iterations: usize) -> Self {
        let mem = memory.max(1);
        Self {
            s_hist: DMatrix::zeros(dim, mem),
            y_hist: DMatrix::zeros(dim, mem),
            rho: vec![0.0; mem],
            alpha_buf: vec![0.0; mem],
            active_clamped: vec![false; dim],
            grad: DVector::zeros(dim),
            grad_trial: DVector::zeros(dim),
            direction: DVector::zeros(dim),
            x_trial: DVector::zeros(dim),
            s_vec: DVector::zeros(dim),
            y_vec: DVector::zeros(dim),
            iterate_path: Vec::with_capacity(max_iterations + 2),
            head: 0,
            count: 0,
        }
    }

    /// Resets stored $(s_k, y_k)$ curvature pairs without deallocating memory.
    #[verified_engine::verified]
    pub fn reset_history(&mut self) {
        self.head = 0;
        self.count = 0;
        self.iterate_path.clear();
    }

    /// Returns the recorded 2D iterate trajectory slice.
    #[verified_engine::verified]
    pub fn iterate_path_slice(&self) -> &[[f64; 2]] {
        &self.iterate_path
    }
}

/// Box-constrained limited-memory BFGS quasi-Newton solver (`L-BFGS-B`).
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::DVector;
/// use math_explorer::pure_math::analysis::optimization::{BoxBounds, LbfgsBSolver};
///
/// // Rosenbrock constrained to [0.0, 0.8] x [0.0, 0.8] -> optimum on boundary x1 = 0.8, x2 = 0.64
/// let bounds = BoxBounds::uniform(2, 0.0, 0.8).unwrap();
/// let x0 = DVector::from_vec(vec![0.2, 0.2]);
/// let sol = LbfgsBSolver::default()
///     .solve(
///         |x, g| {
///             let r1 = x[0] - 1.0;
///             let r2 = x[1] - x[0] * x[0];
///             g[0] = 2.0 * r1 - 400.0 * x[0] * r2;
///             g[1] = 200.0 * r2;
///             r1 * r1 + 100.0 * r2 * r2
///         },
///         &bounds,
///         &x0,
///     )
///     .unwrap();
/// assert!((sol.x[0] - 0.8).abs() < 1e-5);
/// assert!((sol.x[1] - 0.64).abs() < 1e-4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LbfgsBSolver {
    /// Solver hyperparameters.
    pub config: LbfgsBConfig,
}

impl Default for LbfgsBSolver {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            config: LbfgsBConfig::default(),
        }
    }
}

fn identify_active_bounds(
    x: &DVector<f64>,
    bounds: &BoxBounds,
    ws: &mut LbfgsBWorkspace,
    tol: f64,
) {
    let cauchy_scale = 1e-2;
    for i in 0..x.len() {
        let g = ws.grad[i];
        let at_low = (x[i] - bounds.lower[i]).abs() <= tol && g >= 0.0;
        let at_high = (bounds.upper[i] - x[i]).abs() <= tol && g <= 0.0;
        let cauchy = (x[i] - cauchy_scale * g).clamp(bounds.lower[i], bounds.upper[i]);
        let cauchy_low =
            (cauchy - bounds.lower[i]).abs() <= tol && g > 0.0 && (x[i] - bounds.lower[i]) <= tol;
        let cauchy_high =
            (bounds.upper[i] - cauchy).abs() <= tol && g < 0.0 && (bounds.upper[i] - x[i]) <= tol;
        ws.active_clamped[i] = at_low || at_high || cauchy_low || cauchy_high;
    }
}

fn lbfgs_reverse_and_scale(ws: &mut LbfgsBWorkspace) {
    let n = ws.grad.len();
    let m = ws.s_hist.ncols();
    for k in (0..ws.count).rev() {
        let slot = (ws.head + m - ws.count + k) % m;
        let mut dot = 0.0;
        for i in 0..n {
            if !ws.active_clamped[i] {
                dot += ws.s_hist[(i, slot)] * ws.direction[i];
            }
        }
        let a = ws.rho[slot] * dot;
        ws.alpha_buf[slot] = a;
        for i in 0..n {
            if !ws.active_clamped[i] {
                ws.direction[i] -= a * ws.y_hist[(i, slot)];
            }
        }
    }
    let newest = (ws.head + m - 1) % m;
    let (mut sy, mut yy) = (0.0, 0.0);
    for i in 0..n {
        let (s, y) = (ws.s_hist[(i, newest)], ws.y_hist[(i, newest)]);
        sy += s * y;
        yy += y * y;
    }
    let gamma = if yy > 1e-20 && sy > 1e-20 {
        (sy / yy).clamp(1e-6, 1e6)
    } else {
        1.0
    };
    ws.direction *= gamma;
}

fn compute_lbfgs_subspace_direction(ws: &mut LbfgsBWorkspace) {
    let n = ws.grad.len();
    let m = ws.s_hist.ncols();
    for i in 0..n {
        ws.direction[i] = if ws.active_clamped[i] {
            0.0
        } else {
            -ws.grad[i]
        };
    }
    if ws.count == 0 {
        return;
    }
    lbfgs_reverse_and_scale(ws);
    for k in 0..ws.count {
        let slot = (ws.head + m - ws.count + k) % m;
        let mut dot = 0.0;
        for i in 0..n {
            if !ws.active_clamped[i] {
                dot += ws.y_hist[(i, slot)] * ws.direction[i];
            }
        }
        let diff = ws.alpha_buf[slot] - ws.rho[slot] * dot;
        for i in 0..n {
            if !ws.active_clamped[i] {
                ws.direction[i] += diff * ws.s_hist[(i, slot)];
            }
        }
    }
    for i in 0..n {
        if ws.active_clamped[i] {
            ws.direction[i] = 0.0;
        }
    }
    if ws.grad.dot(&ws.direction) >= -1e-16 {
        for i in 0..n {
            ws.direction[i] = if ws.active_clamped[i] {
                0.0
            } else {
                -ws.grad[i]
            };
        }
    }
}

fn projected_backtrack<F: FnMut(&DVector<f64>, &mut DVector<f64>) -> f64>(
    fg: &mut F,
    bounds: &BoxBounds,
    x: &DVector<f64>,
    fx: f64,
    max_ls: usize,
    ws: &mut LbfgsBWorkspace,
) -> (f64, bool) {
    let mut alpha = 1.0_f64;
    for _ in 0..max_ls {
        let mut gs = 0.0;
        let mut step_sq = 0.0;
        for i in 0..x.len() {
            let xi = (x[i] + alpha * ws.direction[i]).clamp(bounds.lower[i], bounds.upper[i]);
            ws.x_trial[i] = xi;
            let dx = xi - x[i];
            gs += ws.grad[i] * dx;
            step_sq += dx * dx;
        }
        if step_sq > 0.0 && gs < 0.0 {
            let f_trial = fg(&ws.x_trial, &mut ws.grad_trial);
            if f_trial.is_finite() && f_trial <= fx + 1e-4 * gs + 1e-15 {
                return (f_trial, true);
            }
        }
        alpha *= 0.5;
    }
    (fx, false)
}

fn push_curvature_pair(ws: &mut LbfgsBWorkspace) {
    let sy = ws.s_vec.dot(&ws.y_vec);
    let s_norm = ws.s_vec.norm();
    let y_norm = ws.y_vec.norm();
    if sy > 1e-12 * s_norm * y_norm && sy.is_finite() {
        let m = ws.s_hist.ncols();
        let slot = ws.head;
        ws.s_hist.set_column(slot, &ws.s_vec);
        ws.y_hist.set_column(slot, &ws.y_vec);
        ws.rho[slot] = 1.0 / sy;
        ws.head = (ws.head + 1) % m;
        if ws.count < m {
            ws.count += 1;
        }
    }
}

impl LbfgsBSolver {
    /// Creates a new [`LbfgsBSolver`] with custom configuration.
    #[verified_engine::verified]
    pub fn new(config: LbfgsBConfig) -> Self {
        Self { config }
    }

    /// Minimizes `fg` subject to `bounds` in-place with zero heap allocations.
    #[verified_engine::verified]
    pub fn solve_in_place<F>(
        &self,
        mut fg: F,
        bounds: &BoxBounds,
        x: &mut DVector<f64>,
        workspace: &mut LbfgsBWorkspace,
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
        workspace.reset_history();
        bounds.project_in_place(x);
        let mut fx = fg(x, &mut workspace.grad);
        if !fx.is_finite() || workspace.grad.iter().any(|g| !g.is_finite()) {
            return Err(OptimizationError::NonFiniteValue);
        }
        if n >= 2 && workspace.iterate_path.len() < workspace.iterate_path.capacity() {
            workspace.iterate_path.push([x[0], x[1]]);
        }
        let mut iters = 0;
        for iter in 1..=self.config.max_iterations {
            let pg = bounds.projected_gradient_norm(x, &workspace.grad);
            if pg <= self.config.tolerance {
                return Ok((fx, pg, iter - 1));
            }
            iters = iter;
            identify_active_bounds(x, bounds, workspace, 1e-10);
            compute_lbfgs_subspace_direction(workspace);

            let (mut f_trial, mut accepted) = projected_backtrack(
                &mut fg,
                bounds,
                x,
                fx,
                self.config.max_line_search,
                workspace,
            );
            if !accepted {
                // Fallback to unmasked projected steepest descent arc:
                workspace.count = 0;
                for i in 0..n {
                    workspace.direction[i] = -workspace.grad[i];
                }
                (f_trial, accepted) = projected_backtrack(
                    &mut fg,
                    bounds,
                    x,
                    fx,
                    self.config.max_line_search,
                    workspace,
                );
                if !accepted {
                    break;
                }
            }
            for i in 0..n {
                workspace.s_vec[i] = workspace.x_trial[i] - x[i];
                workspace.y_vec[i] = workspace.grad_trial[i] - workspace.grad[i];
                x[i] = workspace.x_trial[i];
                workspace.grad[i] = workspace.grad_trial[i];
            }
            push_curvature_pair(workspace);
            fx = f_trial;
            if n >= 2 && workspace.iterate_path.len() < workspace.iterate_path.capacity() {
                workspace.iterate_path.push([x[0], x[1]]);
            }
        }
        let pg = bounds.projected_gradient_norm(x, &workspace.grad);
        Ok((fx, pg, iters))
    }

    /// Allocates workspace and minimizes `fg` subject to `bounds` starting from `x0`.
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
        let mut ws = LbfgsBWorkspace::new(x.len(), self.config.memory, self.config.max_iterations);
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
    fn test_lbfgsb_interior_and_boundary_optima() {
        let solver = LbfgsBSolver::default();
        let bounds = BoxBounds::new(
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![0.75, 2.0]),
        )
        .expect("bounds");
        let x0 = DVector::from_vec(vec![0.1, 1.5]);
        let sol = solver
            .solve(
                |x, g| {
                    g[0] = 4.0 * (x[0] - 1.2);
                    g[1] = 2.0 * (x[1] - 0.6);
                    2.0 * (x[0] - 1.2).powi(2) + (x[1] - 0.6).powi(2)
                },
                &bounds,
                &x0,
            )
            .expect("converged");
        assert!((sol.x[0] - 0.75).abs() < 1e-7);
        assert!((sol.x[1] - 0.6).abs() < 1e-7);
    }
}
