//! Two-Phase Simplex Linear Programming solver with Bland's anti-cycling pivot rule
//! and exact dual shadow prices.

use super::OptimizationError;
use nalgebra::{DMatrix, DVector};

/// Optimization direction for a [`LinearProgram`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpSense {
    /// Maximize $c^T x$.
    Maximize,
    /// Minimize $c^T x$.
    Minimize,
}

/// Relational sense of each linear constraint row $a_i^T x \odot b_i$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintSense {
    /// $a_i^T x \le b_i$.
    LessEqual,
    /// $a_i^T x = b_i$.
    Equal,
    /// $a_i^T x \ge b_i$.
    GreaterEqual,
}

/// Termination status of [`SimplexSolver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpStatus {
    /// Finite primal-dual optimal vertex found.
    Optimal,
    /// Feasible region $A x \odot b,\; x \ge 0$ is empty.
    Infeasible,
    /// Objective improves without bound along a feasible extreme ray.
    Unbounded,
}

/// Linear program $\operatorname{opt}_{x \ge 0} c^T x$ subject to $A x \odot b$.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::{DMatrix, DVector};
/// use math_explorer::pure_math::analysis::optimization::{
///     LinearProgram, LpStatus, SimplexSolver,
/// };
///
/// let c = DVector::from_vec(vec![3.0, 2.0]);
/// let a = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 2.0, 1.0]);
/// let b = DVector::from_vec(vec![4.0, 5.0]);
/// let lp = LinearProgram::maximize(c, a, b).unwrap();
/// let sol = SimplexSolver::default().solve(&lp).unwrap();
/// assert_eq!(sol.status, LpStatus::Optimal);
/// assert!((sol.x[0] - 1.0).abs() < 1e-8);
/// assert!((sol.x[1] - 3.0).abs() < 1e-8);
/// assert!((sol.objective - 9.0).abs() < 1e-8);
/// assert!((sol.shadow_prices[0] - 1.0).abs() < 1e-8);
/// assert!((sol.shadow_prices[1] - 1.0).abs() < 1e-8);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LinearProgram {
    /// Objective sense (`Maximize` or `Minimize`).
    pub sense: LpSense,
    /// Objective vector $c \in \mathbb{R}^n$.
    pub c: DVector<f64>,
    /// Constraint matrix $A \in \mathbb{R}^{m \times n}$.
    pub a: DMatrix<f64>,
    /// Right-hand side vector $b \in \mathbb{R}^m$.
    pub b: DVector<f64>,
    /// Relational sense per constraint row.
    pub senses: Vec<ConstraintSense>,
}

impl LinearProgram {
    /// Constructs a general linear program with explicit constraint senses and $x \ge 0$.
    #[verified_engine::verified]
    pub fn new(
        sense: LpSense,
        c: DVector<f64>,
        a: DMatrix<f64>,
        b: DVector<f64>,
        senses: Vec<ConstraintSense>,
    ) -> Result<Self, OptimizationError> {
        if a.ncols() != c.len() {
            return Err(OptimizationError::DimensionMismatch {
                expected: c.len(),
                actual: a.ncols(),
            });
        }
        if a.nrows() != b.len() || senses.len() != b.len() {
            return Err(OptimizationError::DimensionMismatch {
                expected: b.len(),
                actual: senses.len(),
            });
        }
        if c.iter().any(|v| !v.is_finite())
            || b.iter().any(|v| !v.is_finite())
            || a.iter().any(|v| !v.is_finite())
        {
            return Err(OptimizationError::NonFiniteValue);
        }
        Ok(Self {
            sense,
            c,
            a,
            b,
            senses,
        })
    }

    /// Constructs $\max c^T x$ subject to $A x \le b,\; x \ge 0$.
    #[verified_engine::verified]
    pub fn maximize(
        c: DVector<f64>,
        a: DMatrix<f64>,
        b: DVector<f64>,
    ) -> Result<Self, OptimizationError> {
        let m = b.len();
        Self::new(
            LpSense::Maximize,
            c,
            a,
            b,
            vec![ConstraintSense::LessEqual; m],
        )
    }

    /// Constructs $\min c^T x$ subject to $A x \le b,\; x \ge 0$.
    #[verified_engine::verified]
    pub fn minimize(
        c: DVector<f64>,
        a: DMatrix<f64>,
        b: DVector<f64>,
    ) -> Result<Self, OptimizationError> {
        let m = b.len();
        Self::new(
            LpSense::Minimize,
            c,
            a,
            b,
            vec![ConstraintSense::LessEqual; m],
        )
    }
}

/// Solution summary returned by [`SimplexSolver::solve`].
#[derive(Debug, Clone, PartialEq)]
pub struct LpSolution {
    /// Solver status (`Optimal`, `Infeasible`, or `Unbounded`).
    pub status: LpStatus,
    /// Primal decision vector $x^* \in \mathbb{R}^n$ ($x^* \ge 0$).
    pub x: DVector<f64>,
    /// Optimal objective value $c^T x^*$ (`NaN` if infeasible, $\pm\infty$ if unbounded).
    pub objective: f64,
    /// Dual shadow prices $\partial z^* / \partial b_i \in \mathbb{R}^m$.
    pub shadow_prices: DVector<f64>,
    /// Total Simplex pivots across Phase I and Phase II.
    pub iterations: usize,
    /// Visited 2D primal vertices `(x[0], x[1])` across Phase II pivots.
    pub pivot_path: Vec<[f64; 2]>,
}

/// Preallocated tableau workspace for zero-allocation Two-Phase Simplex solves.
#[derive(Debug, Clone)]
pub struct SimplexWorkspace {
    // Max layout: (m + 1) rows by (n + 2*m + 1) cols:
    // [primal (n) | slack/surplus (m) | identity/artificial (m) | rhs (1)]
    pub(super) tableau: DMatrix<f64>,
    pub(super) basis: Vec<usize>,
    pub(super) row_signs: Vec<f64>,
    pub(super) col_active: Vec<bool>,
    pub(super) primal: DVector<f64>,
    pub(super) shadow_prices: DVector<f64>,
    pub(super) pivot_path: Vec<[f64; 2]>,
    pub(super) status: LpStatus,
    pub(super) objective: f64,
    pub(super) iterations: usize,
}

impl SimplexWorkspace {
    /// Preallocates a Simplex tableau for `num_vars` variables and `num_constraints` rows.
    #[verified_engine::verified]
    pub fn new(num_vars: usize, num_constraints: usize, max_iterations: usize) -> Self {
        let rows = num_constraints + 1;
        let cols = num_vars + 2 * num_constraints + 1;
        Self {
            tableau: DMatrix::zeros(rows, cols),
            basis: vec![0; num_constraints],
            row_signs: vec![1.0; num_constraints],
            col_active: vec![false; num_vars + 2 * num_constraints],
            primal: DVector::zeros(num_vars),
            shadow_prices: DVector::zeros(num_constraints),
            pivot_path: Vec::with_capacity(max_iterations + 2),
            status: LpStatus::Infeasible,
            objective: f64::NAN,
            iterations: 0,
        }
    }

    /// Returns the primal vector from the last solve.
    #[verified_engine::verified]
    pub fn primal(&self) -> &DVector<f64> {
        &self.primal
    }

    /// Returns the dual shadow price vector from the last solve.
    #[verified_engine::verified]
    pub fn shadow_prices(&self) -> &DVector<f64> {
        &self.shadow_prices
    }

    /// Returns the 2D pivot path recorded during Phase II.
    #[verified_engine::verified]
    pub fn pivot_path_slice(&self) -> &[[f64; 2]] {
        &self.pivot_path
    }

    /// Returns `(status, objective, iterations)` from the last solve.
    #[verified_engine::verified]
    pub fn summary(&self) -> (LpStatus, f64, usize) {
        (self.status, self.objective, self.iterations)
    }
}

/// Two-Phase Simplex solver using Bland's anti-cycling pivot rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimplexSolver {
    /// Maximum total pivots allowed across Phase I and Phase II.
    pub max_iterations: usize,
    /// Numerical pivot and optimality tolerance.
    pub tolerance: f64,
}

impl Default for SimplexSolver {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_iterations: math_commons::registry::MAX_ITERATIONS,
            tolerance: 1e-9,
        }
    }
}

use super::lp_tableau::{
    bland_entering_col, bland_leaving_row, drive_artificials_out, pivot_tableau,
    record_primal_vertex, setup_phase_one, setup_phase_two_objective,
};

impl SimplexSolver {
    /// Creates a new [`SimplexSolver`] with explicit iteration limit and numerical tolerance.
    #[verified_engine::verified]
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    fn run_simplex_loop(
        &self,
        ws: &mut SimplexWorkspace,
        m: usize,
        n: usize,
        total_cols: usize,
        rhs_col: usize,
        record_path: bool,
    ) -> Result<bool, OptimizationError> {
        while let Some(enter) = bland_entering_col(&ws.tableau, &ws.col_active, m, self.tolerance) {
            if ws.iterations >= self.max_iterations {
                return Err(OptimizationError::MaxIterationsExceeded(
                    self.max_iterations,
                ));
            }
            let Some(leave) =
                bland_leaving_row(&ws.tableau, &ws.basis, m, enter, rhs_col, self.tolerance)
            else {
                return Ok(false);
            };
            pivot_tableau(&mut ws.tableau, &mut ws.basis, leave, enter, total_cols);
            ws.iterations += 1;
            if record_path {
                record_primal_vertex(ws, m, n, rhs_col);
            }
        }
        Ok(true)
    }

    /// Solves `lp` in-place using preallocated `workspace` with zero heap allocations.
    #[verified_engine::verified]
    pub fn solve_in_place(
        &self,
        lp: &LinearProgram,
        workspace: &mut SimplexWorkspace,
    ) -> Result<LpStatus, OptimizationError> {
        let n = lp.c.len();
        let m = lp.b.len();
        if workspace.tableau.nrows() != m + 1 || workspace.tableau.ncols() != n + 2 * m + 1 {
            return Err(OptimizationError::DimensionMismatch {
                expected: m + 1,
                actual: workspace.tableau.nrows(),
            });
        }
        workspace.iterations = 0;
        workspace.pivot_path.clear();
        workspace.primal.fill(0.0);
        workspace.shadow_prices.fill(0.0);

        let (total_cols, rhs_col) = setup_phase_one(lp, workspace);
        let _ = self.run_simplex_loop(workspace, m, n, total_cols, rhs_col, false)?;
        if workspace.tableau[(m, rhs_col)] < -1e-7 {
            workspace.status = LpStatus::Infeasible;
            workspace.objective = f64::NAN;
            return Ok(LpStatus::Infeasible);
        }

        drive_artificials_out(workspace, n, m, total_cols, self.tolerance);
        setup_phase_two_objective(lp, workspace, total_cols);
        record_primal_vertex(workspace, m, n, rhs_col);

        let bounded = self.run_simplex_loop(workspace, m, n, total_cols, rhs_col, true)?;
        if !bounded {
            workspace.status = LpStatus::Unbounded;
            workspace.objective = match lp.sense {
                LpSense::Maximize => f64::INFINITY,
                LpSense::Minimize => f64::NEG_INFINITY,
            };
            return Ok(LpStatus::Unbounded);
        }

        record_primal_vertex(workspace, m, n, rhs_col);
        let dir = match lp.sense {
            LpSense::Maximize => 1.0,
            LpSense::Minimize => -1.0,
        };
        workspace.objective = dir * workspace.tableau[(m, rhs_col)];
        for i in 0..m {
            let b_inv_col = n + m + i;
            workspace.shadow_prices[i] =
                dir * workspace.row_signs[i] * workspace.tableau[(m, b_inv_col)];
        }
        workspace.status = LpStatus::Optimal;
        Ok(LpStatus::Optimal)
    }

    /// Allocates a [`SimplexWorkspace`] and solves `lp`.
    #[verified_engine::verified]
    pub fn solve(&self, lp: &LinearProgram) -> Result<LpSolution, OptimizationError> {
        let mut ws = SimplexWorkspace::new(lp.c.len(), lp.b.len(), self.max_iterations);
        let status = self.solve_in_place(lp, &mut ws)?;
        Ok(LpSolution {
            status,
            x: ws.primal,
            objective: ws.objective,
            shadow_prices: ws.shadow_prices,
            iterations: ws.iterations,
            pivot_path: ws.pivot_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_optimal_infeasible_and_unbounded() {
        let solver = SimplexSolver::default();
        let lp_opt = LinearProgram::maximize(
            DVector::from_vec(vec![5.0, 4.0]),
            DMatrix::from_row_slice(3, 2, &[6.0, 4.0, 1.0, 2.0, -1.0, 1.0]),
            DVector::from_vec(vec![24.0, 6.0, 1.0]),
        )
        .expect("valid lp");
        let sol = solver.solve(&lp_opt).expect("optimal");
        assert_eq!(sol.status, LpStatus::Optimal);
        assert!((sol.x[0] - 3.0).abs() < 1e-8);
        assert!((sol.x[1] - 1.5).abs() < 1e-8);
        assert!((sol.objective - 21.0).abs() < 1e-8);
        assert!((sol.shadow_prices[0] - 0.75).abs() < 1e-8);
        assert!((sol.shadow_prices[1] - 0.5).abs() < 1e-8);
        assert!(sol.shadow_prices[2].abs() < 1e-8);

        // Infeasible: x1 + x2 <= 1 AND x1 + x2 >= 3
        let lp_infeas = LinearProgram::new(
            LpSense::Maximize,
            DVector::from_vec(vec![1.0, 1.0]),
            DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]),
            DVector::from_vec(vec![1.0, 3.0]),
            vec![ConstraintSense::LessEqual, ConstraintSense::GreaterEqual],
        )
        .expect("infeas lp");
        assert_eq!(
            solver.solve(&lp_infeas).unwrap().status,
            LpStatus::Infeasible
        );

        // Unbounded: max x1 + x2 s.t. x1 - x2 <= 1, x >= 0
        let lp_unb = LinearProgram::maximize(
            DVector::from_vec(vec![1.0, 1.0]),
            DMatrix::from_row_slice(1, 2, &[1.0, -1.0]),
            DVector::from_vec(vec![1.0]),
        )
        .expect("unb lp");
        assert_eq!(solver.solve(&lp_unb).unwrap().status, LpStatus::Unbounded);
    }
}
