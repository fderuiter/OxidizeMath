//! Shifted Powell-Hestenes-Rockafellar Augmented Lagrangian solver for smooth
//! nonlinear equality, inequality, and box-constrained programs.

use super::OptimizationError;
use super::lbfgsb::{LbfgsBConfig, LbfgsBSolver, LbfgsBWorkspace};
use super::projected_gradient::BoxBounds;
use nalgebra::{DMatrix, DVector};

/// Hyperparameters for [`AugmentedLagrangianSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AugmentedLagrangianConfig {
    /// Maximum outer multiplier/penalty updates.
    pub max_outer_iterations: usize,
    /// Maximum inner `L-BFGS-B` iterations per subproblem.
    pub max_inner_iterations: usize,
    /// Primal constraint violation tolerance $\max(\|h(x)\|_\infty, \|\max(0, g(x))\|_\infty)$.
    pub constraint_tolerance: f64,
    /// Stationarity tolerance on inner projected Lagrangian gradient.
    pub gradient_tolerance: f64,
    /// Initial penalty parameter $\rho_0 > 0$.
    pub initial_penalty: f64,
    /// Multiplicative penalty growth factor ($\gamma > 1$).
    pub penalty_growth: f64,
    /// Maximum penalty cap $\rho_{\max}$.
    pub max_penalty: f64,
}

impl Default for AugmentedLagrangianConfig {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_outer_iterations: 30,
            max_inner_iterations: 120,
            constraint_tolerance: 1e-6,
            gradient_tolerance: 1e-6,
            initial_penalty: 10.0,
            penalty_growth: 5.0,
            max_penalty: 1e7,
        }
    }
}

/// Primal-dual solution returned by [`AugmentedLagrangianSolver::solve`].
#[derive(Debug, Clone, PartialEq)]
pub struct AugmentedLagrangianSolution {
    /// Converged primal vector $x^*$.
    pub x: DVector<f64>,
    /// Final objective value $f(x^*)$.
    pub objective: f64,
    /// Equality Lagrange multipliers $\lambda^* \in \mathbb{R}^{m_{\text{eq}}}$.
    pub eq_multipliers: DVector<f64>,
    /// Inequality KKT multipliers $\mu^* \ge 0 \in \mathbb{R}^{m_{\text{ineq}}}$.
    pub ineq_multipliers: DVector<f64>,
    /// Infinity-norm primal feasibility violation at $x^*$.
    pub constraint_violation: f64,
    /// Outer multiplier updates executed.
    pub outer_iterations: usize,
    /// Recorded 2D outer/inner trajectory `(x[0], x[1])`.
    pub iterate_path: Vec<[f64; 2]>,
}

/// Preallocated buffers for zero-allocation [`AugmentedLagrangianSolver`] execution.
#[derive(Debug, Clone)]
pub struct AugmentedLagrangianWorkspace {
    grad_f: DVector<f64>,
    h_eq: DVector<f64>,
    jac_eq: DMatrix<f64>,
    g_ineq: DVector<f64>,
    jac_ineq: DMatrix<f64>,
    eq_mult: DVector<f64>,
    ineq_mult: DVector<f64>,
    inner_ws: LbfgsBWorkspace,
    iterate_path: Vec<[f64; 2]>,
    objective: f64,
    violation: f64,
}

impl AugmentedLagrangianWorkspace {
    /// Preallocates buffers for `dim` variables, `m_eq` equalities, and `m_ineq` inequalities.
    #[verified_engine::verified]
    pub fn new(dim: usize, m_eq: usize, m_ineq: usize, max_inner_iters: usize) -> Self {
        Self {
            grad_f: DVector::zeros(dim),
            h_eq: DVector::zeros(m_eq),
            jac_eq: DMatrix::zeros(m_eq, dim),
            g_ineq: DVector::zeros(m_ineq),
            jac_ineq: DMatrix::zeros(m_ineq, dim),
            eq_mult: DVector::zeros(m_eq),
            ineq_mult: DVector::zeros(m_ineq),
            inner_ws: LbfgsBWorkspace::new(dim, 8, max_inner_iters),
            iterate_path: Vec::with_capacity(64),
            objective: f64::NAN,
            violation: f64::INFINITY,
        }
    }

    /// Returns the equality multipliers $\lambda$ from the last solve.
    #[verified_engine::verified]
    pub fn eq_multipliers(&self) -> &DVector<f64> {
        &self.eq_mult
    }

    /// Returns the inequality multipliers $\mu \ge 0$ from the last solve.
    #[verified_engine::verified]
    pub fn ineq_multipliers(&self) -> &DVector<f64> {
        &self.ineq_mult
    }

    /// Returns `(objective, constraint_violation)` from the last solve.
    #[verified_engine::verified]
    pub fn summary(&self) -> (f64, f64) {
        (self.objective, self.violation)
    }

    /// Returns the recorded 2D outer iterate trajectory slice.
    #[verified_engine::verified]
    pub fn iterate_path_slice(&self) -> &[[f64; 2]] {
        &self.iterate_path
    }
}

/// Shifted PHR Augmented Lagrangian solver for smooth nonlinear constrained optimization.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::DVector;
/// use math_explorer::pure_math::analysis::optimization::{
///     AugmentedLagrangianSolver, BoxBounds,
/// };
///
/// // Minimize x1 + x2 subject to x1^2 + x2^2 - 2 = 0 (optimum at [-1, -1], f* = -2)
/// let solver = AugmentedLagrangianSolver::default();
/// let bounds = BoxBounds::uniform(2, -3.0, 3.0).unwrap();
/// let x0 = DVector::from_vec(vec![-0.5, -0.8]);
/// let sol = solver
///     .solve(
///         |x, gf, h, jh, _g, _jg| {
///             gf[0] = 1.0;
///             gf[1] = 1.0;
///             h[0] = x[0] * x[0] + x[1] * x[1] - 2.0;
///             jh[(0, 0)] = 2.0 * x[0];
///             jh[(0, 1)] = 2.0 * x[1];
///             x[0] + x[1]
///         },
///         &bounds,
///         &x0,
///         1,
///         0,
///     )
///     .unwrap();
/// assert!((sol.x[0] + 1.0).abs() < 1e-4);
/// assert!((sol.x[1] + 1.0).abs() < 1e-4);
/// assert!(sol.constraint_violation < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AugmentedLagrangianSolver {
    /// Outer and inner solver configuration.
    pub config: AugmentedLagrangianConfig,
}

impl Default for AugmentedLagrangianSolver {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            config: AugmentedLagrangianConfig::default(),
        }
    }
}

struct SubproblemBuffers<'a> {
    grad_f: &'a mut DVector<f64>,
    h_eq: &'a mut DVector<f64>,
    jac_eq: &'a mut DMatrix<f64>,
    g_ineq: &'a mut DVector<f64>,
    jac_ineq: &'a mut DMatrix<f64>,
    eq_mult: &'a DVector<f64>,
    ineq_mult: &'a DVector<f64>,
}

fn eval_phr_lagrangian<F>(
    prob: &mut F,
    x: &DVector<f64>,
    grad_out: &mut DVector<f64>,
    rho: f64,
    bufs: &mut SubproblemBuffers<'_>,
) -> f64
where
    F: FnMut(
        &DVector<f64>,
        &mut DVector<f64>,
        &mut DVector<f64>,
        &mut DMatrix<f64>,
        &mut DVector<f64>,
        &mut DMatrix<f64>,
    ) -> f64,
{
    let f_val = prob(
        x,
        bufs.grad_f,
        bufs.h_eq,
        bufs.jac_eq,
        bufs.g_ineq,
        bufs.jac_ineq,
    );
    grad_out.copy_from(bufs.grad_f);
    let mut aug = f_val;
    let inv_2rho = 0.5 / rho;
    let n = x.len();
    for i in 0..bufs.h_eq.len() {
        let shifted = bufs.eq_mult[i] + rho * bufs.h_eq[i];
        aug += inv_2rho * (shifted * shifted - bufs.eq_mult[i] * bufs.eq_mult[i]);
        for j in 0..n {
            grad_out[j] += shifted * bufs.jac_eq[(i, j)];
        }
    }
    for i in 0..bufs.g_ineq.len() {
        let shifted = (bufs.ineq_mult[i] + rho * bufs.g_ineq[i]).max(0.0);
        aug += inv_2rho * (shifted * shifted - bufs.ineq_mult[i] * bufs.ineq_mult[i]);
        if shifted > 0.0 {
            for j in 0..n {
                grad_out[j] += shifted * bufs.jac_ineq[(i, j)];
            }
        }
    }
    aug
}

fn compute_violation(
    h_eq: &DVector<f64>,
    g_ineq: &DVector<f64>,
    eq_mult: &DVector<f64>,
    ineq_mult: &DVector<f64>,
    rho: f64,
) -> f64 {
    let mut viol = 0.0_f64;
    for &h in h_eq.as_slice() {
        viol = viol.max(h.abs());
    }
    for (&g, &mu) in g_ineq.as_slice().iter().zip(ineq_mult.as_slice().iter()) {
        // Complementarity + primal feasibility residual:
        let res = g.max(-mu / rho).abs();
        viol = viol.max(res);
    }
    let _ = eq_mult;
    viol
}

impl AugmentedLagrangianSolver {
    /// Creates a new [`AugmentedLagrangianSolver`].
    #[verified_engine::verified]
    pub fn new(config: AugmentedLagrangianConfig) -> Self {
        Self { config }
    }

    /// Solves the nonlinear constrained program in-place with zero heap allocations.
    #[verified_engine::verified]
    pub fn solve_in_place<F>(
        &self,
        mut problem: F,
        bounds: &BoxBounds,
        x: &mut DVector<f64>,
        workspace: &mut AugmentedLagrangianWorkspace,
    ) -> Result<usize, OptimizationError>
    where
        F: FnMut(
            &DVector<f64>,
            &mut DVector<f64>,
            &mut DVector<f64>,
            &mut DMatrix<f64>,
            &mut DVector<f64>,
            &mut DMatrix<f64>,
        ) -> f64,
    {
        let n = x.len();
        if bounds.dim() != n || workspace.grad_f.len() != n {
            return Err(OptimizationError::DimensionMismatch {
                expected: bounds.dim(),
                actual: n,
            });
        }
        bounds.project_in_place(x);
        workspace.eq_mult.fill(0.0);
        workspace.ineq_mult.fill(0.0);
        workspace.iterate_path.clear();
        if n >= 2 && workspace.iterate_path.len() < workspace.iterate_path.capacity() {
            workspace.iterate_path.push([x[0], x[1]]);
        }

        let mut rho = self.config.initial_penalty.max(1e-3);
        let mut prev_viol = f64::INFINITY;
        let mut outer_done = 0;

        for outer in 1..=self.config.max_outer_iterations {
            outer_done = outer;
            let inner_tol = (1.0 / rho).clamp(self.config.gradient_tolerance, 1e-2);
            let inner_solver = LbfgsBSolver::new(LbfgsBConfig {
                memory: 8,
                max_iterations: self.config.max_inner_iterations,
                tolerance: inner_tol,
                max_line_search: 24,
            });
            let mut bufs = SubproblemBuffers {
                grad_f: &mut workspace.grad_f,
                h_eq: &mut workspace.h_eq,
                jac_eq: &mut workspace.jac_eq,
                g_ineq: &mut workspace.g_ineq,
                jac_ineq: &mut workspace.jac_ineq,
                eq_mult: &workspace.eq_mult,
                ineq_mult: &workspace.ineq_mult,
            };
            let (_, pg_norm, _) = inner_solver.solve_in_place(
                |xk, grad_out| eval_phr_lagrangian(&mut problem, xk, grad_out, rho, &mut bufs),
                bounds,
                x,
                &mut workspace.inner_ws,
            )?;

            workspace.objective = problem(
                x,
                &mut workspace.grad_f,
                &mut workspace.h_eq,
                &mut workspace.jac_eq,
                &mut workspace.g_ineq,
                &mut workspace.jac_ineq,
            );
            workspace.violation = compute_violation(
                &workspace.h_eq,
                &workspace.g_ineq,
                &workspace.eq_mult,
                &workspace.ineq_mult,
                rho,
            );
            if n >= 2 && workspace.iterate_path.len() < workspace.iterate_path.capacity() {
                workspace.iterate_path.push([x[0], x[1]]);
            }
            if workspace.violation <= self.config.constraint_tolerance
                && pg_norm <= self.config.gradient_tolerance * 10.0
            {
                break;
            }
            for i in 0..workspace.h_eq.len() {
                workspace.eq_mult[i] =
                    (workspace.eq_mult[i] + rho * workspace.h_eq[i]).clamp(-1e8, 1e8);
            }
            for i in 0..workspace.g_ineq.len() {
                workspace.ineq_mult[i] =
                    (workspace.ineq_mult[i] + rho * workspace.g_ineq[i]).clamp(0.0, 1e8);
            }
            if workspace.violation > 0.25 * prev_viol {
                rho = (rho * self.config.penalty_growth).min(self.config.max_penalty);
            }
            prev_viol = workspace.violation;
        }
        Ok(outer_done)
    }

    /// Allocates workspace and solves a general nonlinear constrained program.
    #[verified_engine::verified]
    pub fn solve<F>(
        &self,
        problem: F,
        bounds: &BoxBounds,
        x0: &DVector<f64>,
        m_eq: usize,
        m_ineq: usize,
    ) -> Result<AugmentedLagrangianSolution, OptimizationError>
    where
        F: FnMut(
            &DVector<f64>,
            &mut DVector<f64>,
            &mut DVector<f64>,
            &mut DMatrix<f64>,
            &mut DVector<f64>,
            &mut DMatrix<f64>,
        ) -> f64,
    {
        let mut x = x0.clone();
        let mut ws = AugmentedLagrangianWorkspace::new(
            x.len(),
            m_eq,
            m_ineq,
            self.config.max_inner_iterations,
        );
        let outer_iterations = self.solve_in_place(problem, bounds, &mut x, &mut ws)?;
        Ok(AugmentedLagrangianSolution {
            x,
            objective: ws.objective,
            eq_multipliers: ws.eq_mult,
            ineq_multipliers: ws.ineq_mult,
            constraint_violation: ws.violation,
            outer_iterations,
            iterate_path: ws.iterate_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_augmented_lagrangian_nonlinear_circle_inequality() {
        // Minimize (x0 - 2)^2 + (x1 - 2)^2 subject to x0^2 + x1^2 - 2 <= 0
        // Optimum at (1, 1), objective = 2, multiplier mu = 1.0
        let solver = AugmentedLagrangianSolver::default();
        let bounds = BoxBounds::uniform(2, -4.0, 4.0).expect("bounds");
        let x0 = DVector::from_vec(vec![0.4, 0.6]);
        let sol = solver
            .solve(
                |x, gf, _h, _jh, g, jg| {
                    gf[0] = 2.0 * (x[0] - 2.0);
                    gf[1] = 2.0 * (x[1] - 2.0);
                    g[0] = x[0] * x[0] + x[1] * x[1] - 2.0;
                    jg[(0, 0)] = 2.0 * x[0];
                    jg[(0, 1)] = 2.0 * x[1];
                    (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2)
                },
                &bounds,
                &x0,
                0,
                1,
            )
            .expect("converged");
        assert!((sol.x[0] - 1.0).abs() < 1e-4);
        assert!((sol.x[1] - 1.0).abs() < 1e-4);
        assert!((sol.ineq_multipliers[0] - 1.0).abs() < 1e-3);
    }
}
