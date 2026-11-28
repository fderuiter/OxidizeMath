// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! Zero-allocation workspace-backed numerical linear algebra solvers.

use super::LinearAlgebraError;
use crate::{
    CsrMatrixView, LinalgError, SolverWorkspace, cg_solve, cg_workspace_len, cholesky_solve,
    cholesky_workspace_len, gmres_solve, gmres_workspace_len, lu_solve, lu_workspace_len, mat_idx,
    qr_or_svd_solve, qr_or_svd_workspace_len, svd_ops_workspace_len, svd_pseudoinverse,
    tikhonov_solve, tikhonov_workspace_len,
};
use math_memory::ArenaLayout;
use nalgebra::{DMatrix, DVector};

/// Computes conservative byte scratch capacity sufficient for dense LU, QR/SVD, Cholesky, Tikhonov, CG, or GMRES solves of shape `(nrows, ncols)`.
#[must_use]
pub fn linalg_solve_workspace_bytes(nrows: usize, ncols: usize, gmres_restart: usize) -> usize {
    let mn = nrows.saturating_mul(ncols);
    let max_n = nrows.max(ncols);
    let max_f64_scratch = lu_workspace_len(max_n)
        .max(qr_or_svd_workspace_len(nrows, ncols))
        .max(svd_ops_workspace_len(nrows, ncols))
        .max(cholesky_workspace_len(max_n))
        .max(tikhonov_workspace_len(nrows, ncols))
        .max(cg_workspace_len(max_n))
        .max(gmres_workspace_len(max_n, gmres_restart.max(1)));
    let mut layout = ArenaLayout::new();
    let _ = layout.reserve::<usize>(nrows.saturating_add(1).saturating_add(mn).max(max_n));
    let _ = layout.reserve::<f64>(mn.saturating_mul(2).saturating_add(max_f64_scratch));
    layout.total_bytes().saturating_add(128)
}

pub(crate) fn copy_dmatrix_to_row_major_slice(a: &DMatrix<f64>, out: &mut [f64]) {
    let (m, n) = a.shape();
    for r in 0..m {
        let row = &mut out[r * n..(r + 1) * n];
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = a[(r, c)];
        }
    }
}

fn count_finite_nnz(a: &DMatrix<f64>) -> Result<usize, LinearAlgebraError> {
    let mut nnz = 0usize;
    for &v in a.as_slice() {
        if !v.is_finite() {
            return Err(LinearAlgebraError::InvalidInput(
                "matrix contains non-finite entry",
            ));
        }
        if v != 0.0 {
            nnz += 1;
        }
    }
    Ok(nnz)
}

fn fill_csr_slices(
    a: &DMatrix<f64>,
    row_ptr: &mut [usize],
    col_ind: &mut [usize],
    values: &mut [f64],
) {
    let (m, n) = a.shape();
    row_ptr[0] = 0;
    let mut cursor = 0usize;
    for r in 0..m {
        for c in 0..n {
            let v = a[(r, c)];
            if v != 0.0 {
                col_ind[cursor] = c;
                values[cursor] = v;
                cursor += 1;
            }
        }
        row_ptr[r + 1] = cursor;
    }
}

/// Solves $A\mathbf{x} = \mathbf{b}$ into `out_x` via partial-pivoted LU factorization $PA = LU$ using `workspace` (`0` heap allocations).
///
/// Reference: Golub & Van Loan (2013), *Matrix Computations*, DOI: 10.56021/9781421407944; [`docs/math_specs/pure_math.md`](../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Partial pivoting bounds element growth in practice with backward error $O(\varepsilon_{\text{mach}})$ and forward error $O(\kappa_2(A)\varepsilon_{\text{mach}})$.
///
/// # Domain of Validity
/// Requires square non-singular $A \in \mathbb{R}^{n \times n}$, finite $\mathbf{b} \in \mathbb{R}^n$, and `out_x.len() == n`.
pub fn solve_linear_system_with_workspace(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    out_x: &mut DVector<f64>,
    workspace: &mut (impl SolverWorkspace + ?Sized),
) -> Result<(), LinearAlgebraError> {
    let (m, n) = a.shape();
    if m != n || n != b.len() || out_x.len() != n {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    if n == 0 {
        return Ok(());
    }
    let mut cur = workspace.cursor();
    let a_row = cur.alloc_slice::<f64>(m * n).map_err(LinalgError::from)?;
    copy_dmatrix_to_row_major_slice(a, a_row);
    let pivots = cur.alloc_slice::<usize>(n).map_err(LinalgError::from)?;
    let mut sub = cur.into_arena();
    lu_solve(
        a_row,
        b.as_slice(),
        out_x.as_mut_slice(),
        n,
        pivots,
        &mut sub,
    )?;
    Ok(())
}

/// Solves $\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2$ into `out_x` via pivoted Householder QR / SVD fallback using `workspace` (`0` heap allocations).
///
/// Reference: Golub & Van Loan (2013), *Matrix Computations*, DOI: 10.56021/9781421407944; [`docs/math_specs/pure_math.md`](../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Orthogonal Householder reflections avoid squaring $\kappa_2(A)$ in explicit normal equations, falling back to truncated SVD on rank-deficient designs.
///
/// # Domain of Validity
/// Requires non-empty $A \in \mathbb{R}^{m \times n}$ with positive rank, $\mathbf{b} \in \mathbb{R}^m$, and `out_x.len() == n`.
pub fn solve_normal_equation_with_workspace(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    out_x: &mut DVector<f64>,
    workspace: &mut (impl SolverWorkspace + ?Sized),
) -> Result<(), LinearAlgebraError> {
    let (m, n) = a.shape();
    if m != b.len() || out_x.len() != n {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    if m == 0 || n == 0 {
        return Err(LinearAlgebraError::SingularMatrix);
    }
    let mut cur = workspace.cursor();
    let a_row = cur.alloc_slice::<f64>(m * n).map_err(LinalgError::from)?;
    copy_dmatrix_to_row_major_slice(a, a_row);
    let perm = cur.alloc_slice::<usize>(n).map_err(LinalgError::from)?;
    let mut sub = cur.into_arena();
    let rank = qr_or_svd_solve(
        a_row,
        m,
        n,
        b.as_slice(),
        out_x.as_mut_slice(),
        perm,
        &mut sub,
    )?;
    if rank == 0 {
        return Err(LinearAlgebraError::SingularMatrix);
    }
    Ok(())
}

/// Computes Moore-Penrose pseudoinverse $A^+ = V \Sigma^+ U^T$ into `out_pinv` (`ncols x nrows`) using `workspace` (`0` heap allocations).
pub fn try_moore_penrose_pseudoinverse_with_workspace(
    matrix: &DMatrix<f64>,
    epsilon: f64,
    out_pinv: &mut DMatrix<f64>,
    workspace: &mut (impl SolverWorkspace + ?Sized),
) -> Result<(), LinearAlgebraError> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(LinearAlgebraError::InvalidInput(
            "singular value tolerance epsilon must be finite and non-negative",
        ));
    }
    let (m, n) = matrix.shape();
    if out_pinv.shape() != (n, m) {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let mut cur = workspace.cursor();
    let a_row = cur.alloc_slice::<f64>(m * n).map_err(LinalgError::from)?;
    copy_dmatrix_to_row_major_slice(matrix, a_row);
    let pinv_row = cur.alloc_slice::<f64>(n * m).map_err(LinalgError::from)?;
    let mut sub = cur.into_arena();
    svd_pseudoinverse(a_row, m, n, epsilon, pinv_row, &mut sub)?;
    for r in 0..n {
        for c in 0..m {
            out_pinv[(r, c)] = pinv_row[mat_idx(r, c, m)];
        }
    }
    Ok(())
}

/// Solves SPD system $A\mathbf{x} = \mathbf{b}$ into `out_x` via Cholesky factorization $A = LL^\top$ using `workspace` (`0` heap allocations).
///
/// Reference: Golub & Van Loan (2013), *Matrix Computations*, DOI: 10.56021/9781421407944; [`docs/math_specs/pure_math.md`](../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Unconditionally backward-stable without pivoting on SPD matrices, failing closed on non-positive pivots.
///
/// # Domain of Validity
/// Requires symmetric positive-definite $A = A^\top \succ 0$ in $\mathbb{R}^{n \times n}$, $\mathbf{b} \in \mathbb{R}^n$, and `out_x.len() == n`.
pub fn solve_cholesky_system_with_workspace(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    out_x: &mut DVector<f64>,
    workspace: &mut (impl SolverWorkspace + ?Sized),
) -> Result<(), LinearAlgebraError> {
    let (m, n) = a.shape();
    if m != n || n != b.len() || out_x.len() != n {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let mut cur = workspace.cursor();
    let a_row = cur.alloc_slice::<f64>(m * n).map_err(LinalgError::from)?;
    copy_dmatrix_to_row_major_slice(a, a_row);
    let mut sub = cur.into_arena();
    cholesky_solve(a_row, b.as_slice(), out_x.as_mut_slice(), n, &mut sub)?;
    Ok(())
}

/// Solves Tikhonov ridge regression $\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2^2 + \lambda^2 \|\mathbf{x}\|_2^2$ into `out_x` using `workspace` (`0` heap allocations).
///
/// Reference: Hansen (1998), *Rank-Deficient and Discrete Ill-Posed Problems*, DOI: 10.1137/1.9780898719697; [`docs/math_specs/pure_math.md`](../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Spectral Tikhonov damping caps the effective condition number at $\max_i \sigma_i / \lambda$ for $\lambda > 0$.
///
/// # Domain of Validity
/// Requires finite $A \in \mathbb{R}^{m \times n}$, $\mathbf{b} \in \mathbb{R}^m$, `out_x.len() == n`, and finite $\lambda \ge 0$.
pub fn solve_tikhonov_system_with_workspace(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    lambda: f64,
    out_x: &mut DVector<f64>,
    workspace: &mut (impl SolverWorkspace + ?Sized),
) -> Result<(), LinearAlgebraError> {
    let (m, n) = a.shape();
    if m != b.len() || out_x.len() != n {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let mut cur = workspace.cursor();
    let a_row = cur.alloc_slice::<f64>(m * n).map_err(LinalgError::from)?;
    copy_dmatrix_to_row_major_slice(a, a_row);
    let mut sub = cur.into_arena();
    tikhonov_solve(
        a_row,
        m,
        n,
        b.as_slice(),
        lambda,
        out_x.as_mut_slice(),
        &mut sub,
    )?;
    Ok(())
}

/// Solves SPD system $A\mathbf{x} = \mathbf{b}$ into `out_x` via Conjugate Gradient using `workspace` (`0` heap allocations).
///
/// Reference: Hestenes & Stiefel (1952), *J. Res. Nat. Bur. Stand.* 49(6), 409–436, DOI: 10.6028/jres.049.044; [`docs/math_specs/pure_math.md`](../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// $A$-norm error contracts at rate $2((\sqrt{\kappa}-1)/(\sqrt{\kappa}+1))^k$ for $\kappa = \kappa_2(A)$; non-positive curvature fails closed.
///
/// # Domain of Validity
/// Requires symmetric positive-definite $A \in \mathbb{R}^{n \times n}$, finite $\mathbf{b} \in \mathbb{R}^n$, `out_x.len() == n`, `tol > 0`, and `max_iter >= 1`.
pub fn solve_conjugate_gradient_with_workspace(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    tol: f64,
    max_iter: usize,
    out_x: &mut DVector<f64>,
    workspace: &mut (impl SolverWorkspace + ?Sized),
) -> Result<(), LinearAlgebraError> {
    let (m, n) = a.shape();
    if m != n || n != b.len() || out_x.len() != n {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let nnz = count_finite_nnz(a)?;
    let mut cur = workspace.cursor();
    let row_ptr = cur.alloc_slice::<usize>(m + 1).map_err(LinalgError::from)?;
    let col_ind = cur.alloc_slice::<usize>(nnz).map_err(LinalgError::from)?;
    let values = cur.alloc_slice::<f64>(nnz).map_err(LinalgError::from)?;
    fill_csr_slices(a, row_ptr, col_ind, values);
    let csr = CsrMatrixView::new(n, n, row_ptr, col_ind, values)?;
    out_x.fill(0.0);
    let mut sub = cur.into_arena();
    cg_solve(
        &csr,
        b.as_slice(),
        out_x.as_mut_slice(),
        tol,
        max_iter,
        &mut sub,
    )?;
    Ok(())
}

/// Solves $A\mathbf{x} = \mathbf{b}$ into `out_x` via restarted $\text{GMRES}(m)$ using `workspace` (`0` heap allocations).
///
/// Reference: Saad & Schultz (1986), *SIAM J. Sci. Stat. Comput.* 7(3), 856–869, DOI: 10.1137/0907058; [`docs/math_specs/pure_math.md`](../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Modified Gram-Schmidt Arnoldi orthogonalization with progressive Givens rotations monotonically decreases $\|\mathbf{r}_k\|_2$ within each cycle.
///
/// # Domain of Validity
/// Requires non-singular square $A \in \mathbb{R}^{n \times n}$, finite $\mathbf{b} \in \mathbb{R}^n$, `out_x.len() == n`, `restart >= 1`, `tol > 0`, and `max_restarts >= 1`.
pub fn solve_gmres_restarted_with_workspace(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    restart: usize,
    tol: f64,
    max_restarts: usize,
    out_x: &mut DVector<f64>,
    workspace: &mut (impl SolverWorkspace + ?Sized),
) -> Result<(), LinearAlgebraError> {
    let (m, n) = a.shape();
    if m != n || n != b.len() || out_x.len() != n {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let nnz = count_finite_nnz(a)?;
    let mut cur = workspace.cursor();
    let row_ptr = cur.alloc_slice::<usize>(m + 1).map_err(LinalgError::from)?;
    let col_ind = cur.alloc_slice::<usize>(nnz).map_err(LinalgError::from)?;
    let values = cur.alloc_slice::<f64>(nnz).map_err(LinalgError::from)?;
    fill_csr_slices(a, row_ptr, col_ind, values);
    let csr = CsrMatrixView::new(n, n, row_ptr, col_ind, values)?;
    out_x.fill(0.0);
    let mut sub = cur.into_arena();
    gmres_solve(
        &csr,
        b.as_slice(),
        out_x.as_mut_slice(),
        restart,
        tol,
        max_restarts,
        &mut sub,
    )?;
    Ok(())
}
