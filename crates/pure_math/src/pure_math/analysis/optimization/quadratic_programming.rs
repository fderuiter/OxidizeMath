//! Active-Set Convex Quadratic Programming (`ActiveSetQpSolver`) with preallocated KKT workspace.

use super::OptimizationError;
use super::linear_programming::{ConstraintSense, LinearProgram, LpSense, SimplexWorkspace};
use nalgebra::{DMatrix, DVector};

/// Termination status of [`ActiveSetQpSolver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpStatus {
    /// KKT primal-dual optimal point found.
    Optimal,
    /// Linear constraint set $A_{\text{eq}} x = b_{\text{eq}},\; A_{\text{ineq}} x \le b_{\text{ineq}}$ is empty.
    Infeasible,
}

/// Convex quadratic program $\min_x \frac{1}{2} x^T Q x + c^T x$
/// subject to $A_{\text{eq}} x = b_{\text{eq}}$ and $A_{\text{ineq}} x \le b_{\text{ineq}}$.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::{DMatrix, DVector};
/// use math_explorer::pure_math::analysis::optimization::{
///     ActiveSetQpSolver, QpStatus, QuadraticProgram,
/// };
///
/// // Minimize 0.5 * ((x1 - 2)^2 + (x2 - 2)^2) <=> 0.5 * x^T I x - [2, 2]^T x (+ const)
/// // subject to x1 + x2 <= 2, -x1 <= 0, -x2 <= 0
/// let q = DMatrix::identity(2, 2);
/// let c = DVector::from_vec(vec![-2.0, -2.0]);
/// let a_ineq = DMatrix::from_row_slice(3, 2, &[1.0, 1.0, -1.0, 0.0, 0.0, -1.0]);
/// let b_ineq = DVector::from_vec(vec![2.0, 0.0, 0.0]);
/// let qp = QuadraticProgram::new(q, c, DMatrix::zeros(0, 2), DVector::zeros(0), a_ineq, b_ineq).unwrap();
/// let sol = ActiveSetQpSolver::default().solve(&qp, None).unwrap();
/// assert_eq!(sol.status, QpStatus::Optimal);
/// assert!((sol.x[0] - 1.0).abs() < 1e-8);
/// assert!((sol.x[1] - 1.0).abs() < 1e-8);
/// assert!((sol.ineq_multipliers[0] - 1.0).abs() < 1e-8);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct QuadraticProgram {
    /// Symmetric positive semi-definite Hessian $Q \in \mathbb{R}^{n \times n}$.
    pub q: DMatrix<f64>,
    /// Linear cost vector $c \in \mathbb{R}^n$.
    pub c: DVector<f64>,
    /// Equality constraint matrix $A_{\text{eq}} \in \mathbb{R}^{m_{\text{eq}} \times n}$.
    pub a_eq: DMatrix<f64>,
    /// Equality right-hand side $b_{\text{eq}} \in \mathbb{R}^{m_{\text{eq}}}$.
    pub b_eq: DVector<f64>,
    /// Inequality constraint matrix $A_{\text{ineq}} \in \mathbb{R}^{m_{\text{ineq}} \times n}$.
    pub a_ineq: DMatrix<f64>,
    /// Inequality right-hand side $b_{\text{ineq}} \in \mathbb{R}^{m_{\text{ineq}}}$.
    pub b_ineq: DVector<f64>,
}

impl QuadraticProgram {
    /// Validates and constructs a [`QuadraticProgram`].
    #[verified_engine::verified]
    pub fn new(
        q: DMatrix<f64>,
        c: DVector<f64>,
        a_eq: DMatrix<f64>,
        b_eq: DVector<f64>,
        a_ineq: DMatrix<f64>,
        b_ineq: DVector<f64>,
    ) -> Result<Self, OptimizationError> {
        let n = c.len();
        if q.nrows() != n || q.ncols() != n {
            return Err(OptimizationError::DimensionMismatch {
                expected: n,
                actual: q.nrows(),
            });
        }
        if (a_eq.nrows() > 0 && a_eq.ncols() != n) || a_eq.nrows() != b_eq.len() {
            return Err(OptimizationError::DimensionMismatch {
                expected: b_eq.len(),
                actual: a_eq.nrows(),
            });
        }
        if (a_ineq.nrows() > 0 && a_ineq.ncols() != n) || a_ineq.nrows() != b_ineq.len() {
            return Err(OptimizationError::DimensionMismatch {
                expected: b_ineq.len(),
                actual: a_ineq.nrows(),
            });
        }
        Ok(Self {
            q,
            c,
            a_eq,
            b_eq,
            a_ineq,
            b_ineq,
        })
    }

    /// Evaluates $\frac{1}{2} x^T Q x + c^T x$ without heap allocations.
    #[verified_engine::verified]
    pub fn evaluate_objective(&self, x: &DVector<f64>) -> f64 {
        let mut quad = 0.0;
        let n = x.len();
        for j in 0..n {
            let xj = x[j];
            for i in 0..n {
                quad += x[i] * self.q[(i, j)] * xj;
            }
        }
        0.5 * quad + self.c.dot(x)
    }
}

/// Primal-dual solution returned by [`ActiveSetQpSolver::solve`].
#[derive(Debug, Clone, PartialEq)]
pub struct QpSolution {
    /// Termination status (`Optimal` or `Infeasible`).
    pub status: QpStatus,
    /// Primal solution $x^* \in \mathbb{R}^n$.
    pub x: DVector<f64>,
    /// Optimal quadratic objective $\frac{1}{2} x^{*T} Q x^* + c^T x^*$.
    pub objective: f64,
    /// Equality Lagrange multipliers $\lambda^* \in \mathbb{R}^{m_{\text{eq}}}$.
    pub eq_multipliers: DVector<f64>,
    /// Inequality KKT multipliers $\mu^* \ge 0 \in \mathbb{R}^{m_{\text{ineq}}}$.
    pub ineq_multipliers: DVector<f64>,
    /// Indices of active inequality constraints at termination.
    pub active_inequalities: Vec<usize>,
    /// Number of active-set iterations executed.
    pub iterations: usize,
    /// Visited 2D primal iterates `(x[0], x[1])`.
    pub iterate_path: Vec<[f64; 2]>,
}

/// Preallocated KKT and feasibility scratch buffers for zero-allocation QP solves.
#[derive(Debug, Clone)]
pub struct QpWorkspace {
    pub(super) kkt_mat: DMatrix<f64>,
    pub(super) kkt_rhs: DVector<f64>,
    pub(super) kkt_sol: DVector<f64>,
    pub(super) grad: DVector<f64>,
    pub(super) step: DVector<f64>,
    pub(super) eq_mult: DVector<f64>,
    pub(super) ineq_mult: DVector<f64>,
    pub(super) active_mask: Vec<bool>,
    pub(super) working_set: Vec<usize>,
    pub(super) iterate_path: Vec<[f64; 2]>,
    pub(super) feas_lp: LinearProgram,
    pub(super) feas_ws: SimplexWorkspace,
    pub(super) status: QpStatus,
    pub(super) objective: f64,
    pub(super) iterations: usize,
}

impl QpWorkspace {
    /// Preallocates KKT and Phase-I feasibility buffers for `(n, m_eq, m_ineq)`.
    #[verified_engine::verified]
    pub fn new(n: usize, m_eq: usize, m_ineq: usize, max_iterations: usize) -> Self {
        let max_kkt = n + m_eq + m_ineq;
        // Phase-I elastic LP has `2*n + 1` variables (`x+, x-, s`) and `m_eq + m_ineq` rows.
        let lp_vars = 2 * n + 1;
        let lp_rows = m_eq + m_ineq;
        let mut senses = vec![ConstraintSense::Equal; m_eq];
        senses.resize(lp_rows, ConstraintSense::LessEqual);
        let feas_lp = LinearProgram {
            sense: LpSense::Minimize,
            c: DVector::zeros(lp_vars),
            a: DMatrix::zeros(lp_rows, lp_vars),
            b: DVector::zeros(lp_rows),
            senses,
        };
        Self {
            kkt_mat: DMatrix::zeros(max_kkt, max_kkt),
            kkt_rhs: DVector::zeros(max_kkt),
            kkt_sol: DVector::zeros(max_kkt),
            grad: DVector::zeros(n),
            step: DVector::zeros(n),
            eq_mult: DVector::zeros(m_eq),
            ineq_mult: DVector::zeros(m_ineq),
            active_mask: vec![false; m_ineq],
            working_set: Vec::with_capacity(m_ineq),
            iterate_path: Vec::with_capacity(max_iterations + 2),
            feas_lp,
            feas_ws: SimplexWorkspace::new(lp_vars, lp_rows, max_iterations.max(64)),
            status: QpStatus::Infeasible,
            objective: f64::NAN,
            iterations: 0,
        }
    }

    /// Returns `(status, objective, iterations)` from the most recent solve.
    #[verified_engine::verified]
    pub fn summary(&self) -> (QpStatus, f64, usize) {
        (self.status, self.objective, self.iterations)
    }

    /// Returns the inequality KKT multipliers from the most recent solve.
    #[verified_engine::verified]
    pub fn ineq_multipliers(&self) -> &DVector<f64> {
        &self.ineq_mult
    }

    /// Returns the equality Lagrange multipliers from the most recent solve.
    #[verified_engine::verified]
    pub fn eq_multipliers(&self) -> &DVector<f64> {
        &self.eq_mult
    }

    /// Returns the 2D iterate trajectory slice.
    #[verified_engine::verified]
    pub fn iterate_path_slice(&self) -> &[[f64; 2]] {
        &self.iterate_path
    }
}

/// Primal Active-Set solver for convex quadratic programs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActiveSetQpSolver {
    /// Maximum active-set iterations.
    pub max_iterations: usize,
    /// Primal feasibility and dual KKT tolerance.
    pub tolerance: f64,
}

impl Default for ActiveSetQpSolver {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-8,
        }
    }
}

use super::qp_kkt::{
    assemble_and_solve_kkt, blocking_step, find_feasible_start, is_point_feasible,
};

impl ActiveSetQpSolver {
    /// Creates a new [`ActiveSetQpSolver`].
    #[verified_engine::verified]
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    fn init_working_set(&self, qp: &QuadraticProgram, x: &DVector<f64>, ws: &mut QpWorkspace) {
        ws.active_mask.fill(false);
        ws.working_set.clear();
        let max_active = x.len().saturating_sub(qp.a_eq.nrows());
        for r in 0..qp.a_ineq.nrows() {
            if ws.working_set.len() >= max_active {
                break;
            }
            let mut ax = 0.0;
            for j in 0..x.len() {
                ax += qp.a_ineq[(r, j)] * x[j];
            }
            if (qp.b_ineq[r] - ax).abs() <= 10.0 * self.tolerance {
                ws.active_mask[r] = true;
                ws.working_set.push(r);
            }
        }
    }

    /// Solves `qp` in-place using `workspace` with zero heap allocations.
    #[verified_engine::verified]
    pub fn solve_in_place(
        &self,
        qp: &QuadraticProgram,
        x: &mut DVector<f64>,
        workspace: &mut QpWorkspace,
    ) -> Result<QpStatus, OptimizationError> {
        let n = qp.c.len();
        if x.len() != n || workspace.grad.len() != n {
            return Err(OptimizationError::DimensionMismatch {
                expected: n,
                actual: x.len(),
            });
        }
        workspace.iterate_path.clear();
        workspace.eq_mult.fill(0.0);
        workspace.ineq_mult.fill(0.0);
        workspace.iterations = 0;

        if !is_point_feasible(qp, x, self.tolerance)
            && !find_feasible_start(qp, x, workspace, self.tolerance)?
        {
            workspace.status = QpStatus::Infeasible;
            workspace.objective = f64::NAN;
            return Ok(QpStatus::Infeasible);
        }
        self.init_working_set(qp, x, workspace);
        if n >= 2 && workspace.iterate_path.len() < workspace.iterate_path.capacity() {
            workspace.iterate_path.push([x[0], x[1]]);
        }

        let m_eq = qp.a_eq.nrows();
        for iter in 1..=self.max_iterations {
            workspace.iterations = iter;
            assemble_and_solve_kkt(qp, x, workspace);
            let step_inf = workspace.step.amax();
            if step_inf <= self.tolerance {
                let mut min_mu = 0.0_f64;
                let mut drop_pos = None;
                for (pos, &idx) in workspace.working_set.iter().enumerate() {
                    let mu = workspace.kkt_sol[n + m_eq + pos];
                    if mu < min_mu {
                        min_mu = mu;
                        drop_pos = Some((pos, idx));
                    }
                }
                if min_mu >= -self.tolerance {
                    break;
                }
                if let Some((pos, idx)) = drop_pos {
                    workspace.active_mask[idx] = false;
                    workspace.working_set.remove(pos);
                }
            } else {
                let (alpha, blocker) = blocking_step(qp, x, workspace, self.tolerance);
                x.axpy(alpha, &workspace.step, 1.0);
                if n >= 2 && workspace.iterate_path.len() < workspace.iterate_path.capacity() {
                    workspace.iterate_path.push([x[0], x[1]]);
                }
                if let Some(b_idx) = blocker
                    && !workspace.active_mask[b_idx]
                {
                    workspace.active_mask[b_idx] = true;
                    workspace.working_set.push(b_idx);
                }
            }
        }
        assemble_and_solve_kkt(qp, x, workspace);
        for r in 0..m_eq {
            workspace.eq_mult[r] = workspace.kkt_sol[n + r];
        }
        workspace.ineq_mult.fill(0.0);
        for (pos, &idx) in workspace.working_set.iter().enumerate() {
            workspace.ineq_mult[idx] = workspace.kkt_sol[n + m_eq + pos].max(0.0);
        }
        workspace.objective = qp.evaluate_objective(x);
        workspace.status = QpStatus::Optimal;
        Ok(QpStatus::Optimal)
    }

    /// Allocates a [`QpWorkspace`] and solves `qp` (optionally warm-started from `x0`).
    #[verified_engine::verified]
    pub fn solve(
        &self,
        qp: &QuadraticProgram,
        x0: Option<&DVector<f64>>,
    ) -> Result<QpSolution, OptimizationError> {
        let n = qp.c.len();
        let mut x = x0.cloned().unwrap_or_else(|| DVector::zeros(n));
        let mut ws = QpWorkspace::new(n, qp.a_eq.nrows(), qp.a_ineq.nrows(), self.max_iterations);
        let status = self.solve_in_place(qp, &mut x, &mut ws)?;
        Ok(QpSolution {
            status,
            x,
            objective: ws.objective,
            eq_multipliers: ws.eq_mult,
            ineq_multipliers: ws.ineq_mult,
            active_inequalities: ws.working_set,
            iterations: ws.iterations,
            iterate_path: ws.iterate_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_set_qp_equality_and_inequality_kkt() {
        let q = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let c = DVector::from_vec(vec![-2.0, -5.0]);
        let a_ineq = DMatrix::from_row_slice(
            5,
            2,
            &[-1.0, 2.0, 1.0, 2.0, 1.0, -2.0, -1.0, 0.0, 0.0, -1.0],
        );
        let b_ineq = DVector::from_vec(vec![2.0, 6.0, 2.0, 0.0, 0.0]);
        let qp = QuadraticProgram::new(
            q,
            c,
            DMatrix::zeros(0, 2),
            DVector::zeros(0),
            a_ineq,
            b_ineq,
        )
        .expect("qp");
        let sol = ActiveSetQpSolver::default()
            .solve(&qp, Some(&DVector::from_vec(vec![2.0, 0.0])))
            .expect("solve");
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.4).abs() < 1e-7);
        assert!((sol.x[1] - 1.7).abs() < 1e-7);
        assert!((sol.ineq_multipliers[0] - 0.8).abs() < 1e-7);
    }
}
