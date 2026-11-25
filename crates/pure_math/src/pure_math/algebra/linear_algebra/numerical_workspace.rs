//! Zero-allocation workspace-accepting numerical linear algebra entry points.
// [cite:advanced_linear_algebra]

use super::numerical::{LinearAlgebraError, map_linalg_error};
use engine_linalg::{
    CsrMatrixView, LinalgError, SolverWorkspace, cg_solve, cholesky_solve, gmres_solve, lu_solve,
    qr_or_svd_solve, svd_pseudoinverse, tikhonov_solve,
};
use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::{DMatrix, DVector, RealField};

fn copy_row_major<T: RealField + Copy + engine_linalg::Scalar>(a: &DMatrix<T>, out: &mut [T]) {
    let (m, n) = a.shape();
    debug_assert_eq!(out.len(), m * n);
    for r in 0..m {
        for c in 0..n {
            out[flatten_2d_index(c, r, n)] = a[(r, c)];
        }
    }
}

/// Solves dense square linear system $A x = b$ into `x_out` using `ws` with zero heap allocations.
#[verified_engine::verified]
pub fn solve_linear_system_with_workspace<T, W>(
    a: &DMatrix<T>,
    b: &DVector<T>,
    x_out: &mut DVector<T>,
    ws: &mut W,
) -> Result<(), LinearAlgebraError>
where
    T: RealField + Copy + engine_linalg::Scalar,
    W: SolverWorkspace,
{
    let (m, n) = a.shape();
    if m != n || b.len() != n || x_out.len() != n {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    if n == 0 {
        return Ok(());
    }
    let mut scope = ws.reborrow();
    let len = n
        .checked_mul(n)
        .ok_or(LinearAlgebraError::DimensionMismatch)?;
    let a_row = scope
        .alloc_slice::<T>(len)
        .map_err(|e| map_linalg_error(LinalgError::from(e)))?;
    let pivots = scope
        .alloc_slice::<usize>(n)
        .map_err(|e| map_linalg_error(LinalgError::from(e)))?;
    copy_row_major(a, a_row);
    lu_solve(
        a_row,
        b.as_slice(),
        x_out.as_mut_slice(),
        n,
        pivots,
        &mut scope,
    )
    .map_err(map_linalg_error)?;
    debug_assert_eq!(x_out.len(), n);
    Ok(())
}

/// Solves overdetermined least-squares system $\min_x \|A x - b\|_2$ into `x_out` using `ws`
/// with zero heap allocations.
#[verified_engine::verified]
pub fn solve_normal_equation_with_workspace<T, W>(
    a: &DMatrix<T>,
    b: &DVector<T>,
    x_out: &mut DVector<T>,
    ws: &mut W,
) -> Result<(), LinearAlgebraError>
where
    T: RealField + Copy + engine_linalg::Scalar,
    W: SolverWorkspace,
{
    let (m, n) = a.shape();
    if b.len() != m || x_out.len() != n {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    if m == 0 || n == 0 {
        return Err(LinearAlgebraError::SingularMatrix);
    }
    let mut scope = ws.reborrow();
    let len = m
        .checked_mul(n)
        .ok_or(LinearAlgebraError::DimensionMismatch)?;
    let a_row = scope
        .alloc_slice::<T>(len)
        .map_err(|e| map_linalg_error(LinalgError::from(e)))?;
    let perm = scope
        .alloc_slice::<usize>(n)
        .map_err(|e| map_linalg_error(LinalgError::from(e)))?;
    copy_row_major(a, a_row);
    let rank = qr_or_svd_solve(
        a_row,
        m,
        n,
        b.as_slice(),
        x_out.as_mut_slice(),
        perm,
        &mut scope,
    )
    .map_err(map_linalg_error)?;
    if rank == 0 {
        return Err(LinearAlgebraError::SingularMatrix);
    }
    debug_assert!(rank <= m.min(n));
    Ok(())
}

/// Computes the Moore-Penrose pseudoinverse $A^+$ (`n x m`) into `pinv_out` using `ws`
/// with zero heap allocations.
#[verified_engine::verified]
pub fn try_moore_penrose_pseudoinverse_with_workspace<T, W>(
    a: &DMatrix<T>,
    tolerance: T,
    pinv_out: &mut DMatrix<T>,
    ws: &mut W,
) -> Result<(), LinearAlgebraError>
where
    T: RealField + Copy + engine_linalg::Scalar,
    W: SolverWorkspace,
{
    if !<T as engine_linalg::Scalar>::is_finite(tolerance) || tolerance < T::zero() {
        return Err(LinearAlgebraError::InvalidInput(
            "tolerance must be finite and non-negative",
        ));
    }
    let (m, n) = a.shape();
    if pinv_out.shape() != (n, m) {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let len = m
        .checked_mul(n)
        .ok_or(LinearAlgebraError::DimensionMismatch)?;
    let mut scope = ws.reborrow();
    let a_row = scope
        .alloc_slice::<T>(len)
        .map_err(|e| map_linalg_error(LinalgError::from(e)))?;
    let p_row = scope
        .alloc_slice::<T>(len)
        .map_err(|e| map_linalg_error(LinalgError::from(e)))?;
    copy_row_major(a, a_row);
    svd_pseudoinverse(a_row, m, n, tolerance, p_row, &mut scope).map_err(map_linalg_error)?;
    for r in 0..n {
        for c in 0..m {
            pinv_out[(r, c)] = p_row[flatten_2d_index(c, r, m)];
        }
    }
    debug_assert_eq!(pinv_out.shape(), (n, m));
    Ok(())
}

/// Solves symmetric positive-definite system $A x = b$ via Cholesky into `x_out` using `ws`
/// with zero heap allocations.
#[verified_engine::verified]
pub fn solve_cholesky_system_with_workspace<T, W>(
    a: &DMatrix<T>,
    b: &DVector<T>,
    x_out: &mut DVector<T>,
    ws: &mut W,
) -> Result<(), LinearAlgebraError>
where
    T: RealField + Copy + engine_linalg::Scalar,
    W: SolverWorkspace,
{
    let (m, n) = a.shape();
    if m != n || b.len() != n || x_out.len() != n {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let len = n
        .checked_mul(n)
        .ok_or(LinearAlgebraError::DimensionMismatch)?;
    let mut scope = ws.reborrow();
    let a_row = scope
        .alloc_slice::<T>(len)
        .map_err(|e| map_linalg_error(LinalgError::from(e)))?;
    copy_row_major(a, a_row);
    cholesky_solve(a_row, b.as_slice(), x_out.as_mut_slice(), n, &mut scope)
        .map_err(map_linalg_error)?;
    debug_assert_eq!(x_out.len(), n);
    Ok(())
}

/// Solves damped least-squares $\min_x \|A x - b\|_2^2 + \lambda^2 \|x\|_2^2$ into `x_out`
/// using `ws` with zero heap allocations.
#[verified_engine::verified]
pub fn solve_tikhonov_system_with_workspace<T, W>(
    a: &DMatrix<T>,
    b: &DVector<T>,
    lambda: T,
    x_out: &mut DVector<T>,
    ws: &mut W,
) -> Result<(), LinearAlgebraError>
where
    T: RealField + Copy + engine_linalg::Scalar,
    W: SolverWorkspace,
{
    let (m, n) = a.shape();
    if b.len() != m || x_out.len() != n {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let len = m
        .checked_mul(n)
        .ok_or(LinearAlgebraError::DimensionMismatch)?;
    let mut scope = ws.reborrow();
    let a_row = scope
        .alloc_slice::<T>(len)
        .map_err(|e| map_linalg_error(LinalgError::from(e)))?;
    copy_row_major(a, a_row);
    tikhonov_solve(
        a_row,
        m,
        n,
        b.as_slice(),
        lambda,
        x_out.as_mut_slice(),
        &mut scope,
    )
    .map_err(map_linalg_error)?;
    debug_assert_eq!(x_out.len(), n);
    Ok(())
}

fn count_nonzeros_checked<T: RealField + Copy + engine_linalg::Scalar>(
    a: &DMatrix<T>,
) -> Result<usize, LinearAlgebraError> {
    let (m, n) = a.shape();
    let mut nnz = 0usize;
    for r in 0..m {
        for c in 0..n {
            let v = a[(r, c)];
            if !<T as engine_linalg::Scalar>::is_finite(v) {
                return Err(LinearAlgebraError::InvalidInput(
                    "non-finite matrix entry in sparse conversion",
                ));
            }
            if v != T::zero() {
                nnz += 1;
            }
        }
    }
    debug_assert!(nnz <= m * n);
    Ok(nnz)
}

fn fill_csr<T: RealField + Copy>(
    a: &DMatrix<T>,
    row_ptr: &mut [usize],
    col_ind: &mut [usize],
    values: &mut [T],
) {
    let (m, n) = a.shape();
    debug_assert_eq!(row_ptr.len(), m + 1);
    let mut idx = 0usize;
    row_ptr[0] = 0;
    for r in 0..m {
        for c in 0..n {
            let v = a[(r, c)];
            if v != T::zero() {
                col_ind[idx] = c;
                values[idx] = v;
                idx += 1;
            }
        }
        row_ptr[r + 1] = idx;
    }
    debug_assert_eq!(idx, values.len());
}

/// Solves SPD sparse/dense system $A x = b$ via Conjugate Gradient into `x_out` staging CSR
/// buffers inside `ws` with zero heap allocations.
#[verified_engine::verified]
pub fn solve_conjugate_gradient_with_workspace<T, W>(
    a: &DMatrix<T>,
    b: &DVector<T>,
    x0: &DVector<T>,
    tol: T,
    max_iter: usize,
    x_out: &mut DVector<T>,
    ws: &mut W,
) -> Result<usize, LinearAlgebraError>
where
    T: RealField + Copy + engine_linalg::Scalar,
    W: SolverWorkspace,
{
    let (m, n) = a.shape();
    if m != n || b.len() != n || x0.len() != n || x_out.len() != n {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let nnz = count_nonzeros_checked(a)?;
    let mut scope = ws.reborrow();
    let row_ptr = scope
        .alloc_slice::<usize>(n + 1)
        .map_err(|e| map_linalg_error(LinalgError::from(e)))?;
    let col_ind = scope
        .alloc_slice::<usize>(nnz)
        .map_err(|e| map_linalg_error(LinalgError::from(e)))?;
    let values = scope
        .alloc_slice::<T>(nnz)
        .map_err(|e| map_linalg_error(LinalgError::from(e)))?;
    fill_csr(a, row_ptr, col_ind, values);
    let csr = CsrMatrixView::new(n, n, row_ptr, col_ind, values).map_err(map_linalg_error)?;
    x_out.copy_from(x0);
    let report = cg_solve(
        &csr,
        b.as_slice(),
        x_out.as_mut_slice(),
        tol,
        max_iter,
        &mut scope,
    )
    .map_err(map_linalg_error)?;
    debug_assert!(report.iterations <= max_iter);
    Ok(report.iterations)
}

/// Solves general square system $A x = b$ via restarted GMRES($m$) into `x_out` staging CSR
/// buffers inside `ws` with zero heap allocations.
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn solve_gmres_restarted_with_workspace<T, W>(
    a: &DMatrix<T>,
    b: &DVector<T>,
    x0: &DVector<T>,
    restart: usize,
    tol: T,
    max_restarts: usize,
    x_out: &mut DVector<T>,
    ws: &mut W,
) -> Result<usize, LinearAlgebraError>
where
    T: RealField + Copy + engine_linalg::Scalar,
    W: SolverWorkspace,
{
    let (m, n) = a.shape();
    if m != n || b.len() != n || x0.len() != n || x_out.len() != n {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let nnz = count_nonzeros_checked(a)?;
    let mut scope = ws.reborrow();
    let row_ptr = scope
        .alloc_slice::<usize>(n + 1)
        .map_err(|e| map_linalg_error(LinalgError::from(e)))?;
    let col_ind = scope
        .alloc_slice::<usize>(nnz)
        .map_err(|e| map_linalg_error(LinalgError::from(e)))?;
    let values = scope
        .alloc_slice::<T>(nnz)
        .map_err(|e| map_linalg_error(LinalgError::from(e)))?;
    fill_csr(a, row_ptr, col_ind, values);
    let csr = CsrMatrixView::new(n, n, row_ptr, col_ind, values).map_err(map_linalg_error)?;
    x_out.copy_from(x0);
    let report = gmres_solve(
        &csr,
        b.as_slice(),
        x_out.as_mut_slice(),
        restart,
        tol,
        max_restarts,
        &mut scope,
    )
    .map_err(map_linalg_error)?;
    debug_assert_eq!(x_out.len(), n);
    Ok(report.iterations)
}
