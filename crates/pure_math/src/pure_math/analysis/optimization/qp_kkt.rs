//! Internal Phase-I LP feasibility and KKT linear-system routines for Active-Set QP.

use super::OptimizationError;
use super::linear_programming::{ConstraintSense, LpStatus, SimplexSolver};
use super::quadratic_programming::{QpWorkspace, QuadraticProgram};
use nalgebra::{DMatrix, DVector};

#[verified_engine::verified]
pub(super) fn is_point_feasible(qp: &QuadraticProgram, x: &DVector<f64>, tol: f64) -> bool {
    for r in 0..qp.a_eq.nrows() {
        let mut dot = 0.0;
        for j in 0..x.len() {
            dot += qp.a_eq[(r, j)] * x[j];
        }
        if (dot - qp.b_eq[r]).abs() > tol {
            return false;
        }
    }
    for r in 0..qp.a_ineq.nrows() {
        let mut dot = 0.0;
        for j in 0..x.len() {
            dot += qp.a_ineq[(r, j)] * x[j];
        }
        if dot > qp.b_ineq[r] + tol {
            return false;
        }
    }
    true
}

pub(super) fn find_feasible_start(
    qp: &QuadraticProgram,
    x: &mut DVector<f64>,
    ws: &mut QpWorkspace,
    tol: f64,
) -> Result<bool, OptimizationError> {
    let n = x.len();
    let m_eq = qp.a_eq.nrows();
    let m_ineq = qp.a_ineq.nrows();
    if m_eq == 0 && m_ineq == 0 {
        return Ok(true);
    }
    ws.feas_lp.c.fill(0.0);
    ws.feas_lp.c[2 * n] = 1.0;
    ws.feas_lp.a.fill(0.0);
    for r in 0..m_eq {
        for j in 0..n {
            ws.feas_lp.a[(r, j)] = qp.a_eq[(r, j)];
            ws.feas_lp.a[(r, n + j)] = -qp.a_eq[(r, j)];
        }
        ws.feas_lp.b[r] = qp.b_eq[r];
        ws.feas_lp.senses[r] = ConstraintSense::Equal;
    }
    for r in 0..m_ineq {
        let row = m_eq + r;
        for j in 0..n {
            ws.feas_lp.a[(row, j)] = qp.a_ineq[(r, j)];
            ws.feas_lp.a[(row, n + j)] = -qp.a_ineq[(r, j)];
        }
        ws.feas_lp.a[(row, 2 * n)] = -1.0;
        ws.feas_lp.b[row] = qp.b_ineq[r];
        ws.feas_lp.senses[row] = ConstraintSense::LessEqual;
    }
    let lp_solver = SimplexSolver::new(ws.feas_ws.pivot_path_slice().len().max(200), 1e-9);
    let status = lp_solver.solve_in_place(&ws.feas_lp, &mut ws.feas_ws)?;
    let (_, s_opt, _) = ws.feas_ws.summary();
    if status != LpStatus::Optimal || s_opt > tol {
        return Ok(false);
    }
    let prim = ws.feas_ws.primal();
    for j in 0..n {
        x[j] = prim[j] - prim[n + j];
    }
    Ok(true)
}

fn solve_dense_in_place(
    mat: &mut DMatrix<f64>,
    rhs: &mut DVector<f64>,
    sol: &mut DVector<f64>,
    dim: usize,
) -> bool {
    for k in 0..dim {
        let mut piv_row = k;
        let mut max_val = mat[(k, k)].abs();
        for r in (k + 1)..dim {
            let v = mat[(r, k)].abs();
            if v > max_val {
                max_val = v;
                piv_row = r;
            }
        }
        if max_val < 1e-12 {
            mat[(k, k)] = 1e-10;
            piv_row = k;
        }
        if piv_row != k {
            for c in k..dim {
                let tmp = mat[(k, c)];
                mat[(k, c)] = mat[(piv_row, c)];
                mat[(piv_row, c)] = tmp;
            }
            let tmp_r = rhs[k];
            rhs[k] = rhs[piv_row];
            rhs[piv_row] = tmp_r;
        }
        let inv_diag = 1.0 / mat[(k, k)];
        for r in (k + 1)..dim {
            let factor = mat[(r, k)] * inv_diag;
            if factor.abs() > 0.0 {
                for c in (k + 1)..dim {
                    mat[(r, c)] -= factor * mat[(k, c)];
                }
                rhs[r] -= factor * rhs[k];
            }
        }
    }
    for i in (0..dim).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..dim {
            sum -= mat[(i, j)] * sol[j];
        }
        if mat[(i, i)].abs() < 1e-14 {
            return false;
        }
        sol[i] = sum / mat[(i, i)];
    }
    true
}

pub(super) fn assemble_and_solve_kkt(
    qp: &QuadraticProgram,
    x: &DVector<f64>,
    ws: &mut QpWorkspace,
) -> usize {
    let n = x.len();
    let m_eq = qp.a_eq.nrows();
    let m_w = ws.working_set.len();
    let total = n + m_eq + m_w;
    for r in 0..total {
        ws.kkt_rhs[r] = 0.0;
        for c in 0..total {
            ws.kkt_mat[(r, c)] = 0.0;
        }
    }
    ws.grad.copy_from(&qp.c);
    ws.grad.gemv(1.0, &qp.q, x, 1.0);
    for i in 0..n {
        ws.kkt_rhs[i] = -ws.grad[i];
        for j in 0..n {
            ws.kkt_mat[(i, j)] = qp.q[(i, j)];
        }
        ws.kkt_mat[(i, i)] += 1e-12;
    }
    for r in 0..m_eq {
        let k_idx = n + r;
        for j in 0..n {
            let a = qp.a_eq[(r, j)];
            ws.kkt_mat[(j, k_idx)] = a;
            ws.kkt_mat[(k_idx, j)] = a;
        }
    }
    for (pos, &ineq_idx) in ws.working_set.iter().enumerate() {
        let k_idx = n + m_eq + pos;
        for j in 0..n {
            let a = qp.a_ineq[(ineq_idx, j)];
            ws.kkt_mat[(j, k_idx)] = a;
            ws.kkt_mat[(k_idx, j)] = a;
        }
    }
    let _ = solve_dense_in_place(&mut ws.kkt_mat, &mut ws.kkt_rhs, &mut ws.kkt_sol, total);
    for i in 0..n {
        ws.step[i] = ws.kkt_sol[i];
    }
    total
}

pub(super) fn blocking_step(
    qp: &QuadraticProgram,
    x: &DVector<f64>,
    ws: &QpWorkspace,
    tol: f64,
) -> (f64, Option<usize>) {
    let mut alpha = 1.0_f64;
    let mut blocker = None;
    for r in 0..qp.a_ineq.nrows() {
        if ws.active_mask[r] {
            continue;
        }
        let mut ap = 0.0;
        let mut ax = 0.0;
        for j in 0..x.len() {
            let a = qp.a_ineq[(r, j)];
            ap += a * ws.step[j];
            ax += a * x[j];
        }
        if ap > tol {
            let dist = (qp.b_ineq[r] - ax).max(0.0) / ap;
            if dist < alpha {
                alpha = dist;
                blocker = Some(r);
            }
        }
    }
    (alpha, blocker)
}
