//! Zero-allocation workspace-accepting numerical linear algebra entry points.
// [cite:advanced_linear_algebra]

use super::numerical::LinearAlgebraError;
use engine_linalg::{
    CsrMatrixView, SolverWorkspace, cg_solve, cholesky_solve, gmres_solve, lu_solve,
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

/// Solves dense square linear system $A\mathbf{x} = \mathbf{b}$ into `x_out` via partial-pivoted LU factorization $PA = LU$ using `ws` with zero heap allocations.
///
/// Reference: Golub & Van Loan (2013), *Matrix Computations*, DOI: 10.56021/9781421407944; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Partial pivoting bounds element growth in practice with backward error $O(\varepsilon_{\text{mach}})$ and forward error $O(\kappa_2(A)\varepsilon_{\text{mach}})$.
///
/// # Domain of Validity
/// Requires non-singular square $A \in \mathbb{R}^{n \times n}$ with finite entries, compatible $\mathbf{b}, \mathbf{x}_{\text{out}} \in \mathbb{R}^n$, and sufficient capacity in `ws`.
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
    let a_row = scope.alloc_slice::<T>(len)?;
    let pivots = scope.alloc_slice::<usize>(n)?;
    copy_row_major(a, a_row);
    lu_solve(
        a_row,
        b.as_slice(),
        x_out.as_mut_slice(),
        n,
        pivots,
        &mut scope,
    )?;
    debug_assert_eq!(x_out.len(), n);
    Ok(())
}

/// Solves linear least-squares $\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2$ into `x_out` via column-pivoted Householder QR with SVD fallback using `ws` with zero heap allocations.
///
/// Reference: Golub & Van Loan (2013), *Matrix Computations*, DOI: 10.56021/9781421407944; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Orthogonal Householder reflections avoid squaring $\kappa_2(A)$ in explicit normal equations, falling back to truncated SVD on rank-deficient designs.
///
/// # Domain of Validity
/// Requires non-empty $A \in \mathbb{R}^{m \times n}$ with finite entries and positive numerical rank, compatible $\mathbf{b} \in \mathbb{R}^m$, $\mathbf{x}_{\text{out}} \in \mathbb{R}^n$, and sufficient capacity in `ws`.
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
    let a_row = scope.alloc_slice::<T>(len)?;
    let perm = scope.alloc_slice::<usize>(n)?;
    copy_row_major(a, a_row);
    let rank = qr_or_svd_solve(
        a_row,
        m,
        n,
        b.as_slice(),
        x_out.as_mut_slice(),
        perm,
        &mut scope,
    )?;
    if rank == 0 {
        return Err(LinearAlgebraError::SingularMatrix);
    }
    debug_assert!(rank <= m.min(n));
    Ok(())
}

/// Computes the Moore-Penrose pseudoinverse $A^+ = V \Sigma^+ U^\top$ (`n x m`) into `pinv_out` using `ws` with zero heap allocations.
///
/// Reference: Golub & Van Loan (2013), *Matrix Computations*, DOI: 10.56021/9781421407944; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Singular values below `tolerance` are zeroed before inversion so noise along near-null directions does not blow up $\|A^+\|_2$.
///
/// # Domain of Validity
/// Requires finite $A \in \mathbb{R}^{m \times n}$, finite `tolerance >= 0`, preallocated `pinv_out` of shape `(n, m)`, and sufficient capacity in `ws`.
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
    let a_row = scope.alloc_slice::<T>(len)?;
    let p_row = scope.alloc_slice::<T>(len)?;
    copy_row_major(a, a_row);
    svd_pseudoinverse(a_row, m, n, tolerance, p_row, &mut scope)?;
    for r in 0..n {
        for c in 0..m {
            pinv_out[(r, c)] = p_row[flatten_2d_index(c, r, m)];
        }
    }
    debug_assert_eq!(pinv_out.shape(), (n, m));
    Ok(())
}

/// Solves symmetric positive-definite system $A\mathbf{x} = \mathbf{b}$ via Cholesky factorization $A = LL^\top$ into `x_out` using `ws` with zero heap allocations.
///
/// Reference: Golub & Van Loan (2013), *Matrix Computations*, DOI: 10.56021/9781421407944; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Unconditionally backward-stable without pivoting on SPD matrices, satisfying $\|LL^\top - A\|_2 = O(\varepsilon_{\text{mach}}\|A\|_2)$ and failing closed on non-positive pivots.
///
/// # Domain of Validity
/// Requires symmetric positive-definite $A = A^\top \succ 0$ in $\mathbb{R}^{n \times n}$, finite $\mathbf{b}, \mathbf{x}_{\text{out}} \in \mathbb{R}^n$, and sufficient capacity in `ws`.
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
    let a_row = scope.alloc_slice::<T>(len)?;
    copy_row_major(a, a_row);
    cholesky_solve(a_row, b.as_slice(), x_out.as_mut_slice(), n, &mut scope)?;
    debug_assert_eq!(x_out.len(), n);
    Ok(())
}

/// Solves Tikhonov ridge regression $\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2^2 + \lambda^2 \|\mathbf{x}\|_2^2$ into `x_out` using `ws` with zero heap allocations.
///
/// Reference: Hansen (1998), *Rank-Deficient and Discrete Ill-Posed Problems*, DOI: 10.1137/1.9780898719697; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Spectral Tikhonov filter factors $\sigma_i / (\sigma_i^2 + \lambda^2)$ bound the effective condition number at $\max_i \sigma_i / \lambda$ for $\lambda > 0$.
///
/// # Domain of Validity
/// Requires finite $A \in \mathbb{R}^{m \times n}$, compatible $\mathbf{b} \in \mathbb{R}^m$ and $\mathbf{x}_{\text{out}} \in \mathbb{R}^n$, finite $\lambda \ge 0$, and sufficient capacity in `ws`.
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
    let a_row = scope.alloc_slice::<T>(len)?;
    copy_row_major(a, a_row);
    tikhonov_solve(
        a_row,
        m,
        n,
        b.as_slice(),
        lambda,
        x_out.as_mut_slice(),
        &mut scope,
    )?;
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

/// Solves SPD sparse/dense system $A\mathbf{x} = \mathbf{b}$ via Conjugate Gradient minimization of $\frac{1}{2}\mathbf{x}^\top A\mathbf{x} - \mathbf{b}^\top\mathbf{x}$ into `x_out` staging CSR buffers inside `ws` with zero heap allocations.
///
/// Reference: Hestenes & Stiefel (1952), *J. Res. Nat. Bur. Stand.* 49(6), 409–436, DOI: 10.6028/jres.049.044; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// $A$-norm error contracts at rate $2((\sqrt{\kappa}-1)/(\sqrt{\kappa}+1))^k$ for $\kappa = \kappa_2(A)$; non-positive curvature $\mathbf{p}^\top A\mathbf{p} \le 0$ fails closed.
///
/// # Domain of Validity
/// Requires symmetric positive-definite $A \in \mathbb{R}^{n \times n}$, finite $\mathbf{b}, \mathbf{x}_0, \mathbf{x}_{\text{out}} \in \mathbb{R}^n$, `tol > 0`, `max_iter >= 1`, and sufficient capacity in `ws`.
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
    let row_ptr = scope.alloc_slice::<usize>(n + 1)?;
    let col_ind = scope.alloc_slice::<usize>(nnz)?;
    let values = scope.alloc_slice::<T>(nnz)?;
    fill_csr(a, row_ptr, col_ind, values);
    let csr = CsrMatrixView::new(n, n, row_ptr, col_ind, values)?;
    x_out.copy_from(x0);
    let report = cg_solve(
        &csr,
        b.as_slice(),
        x_out.as_mut_slice(),
        tol,
        max_iter,
        &mut scope,
    )?;
    debug_assert!(report.iterations <= max_iter);
    Ok(report.iterations)
}

/// Solves general square system $A\mathbf{x} = \mathbf{b}$ via restarted $\text{GMRES}(m)$ minimizing $\|\mathbf{b} - A\mathbf{x}_k\|_2$ into `x_out` staging CSR buffers inside `ws` with zero heap allocations.
///
/// Reference: Saad & Schultz (1986), *SIAM J. Sci. Stat. Comput.* 7(3), 856–869, DOI: 10.1137/0907058; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Modified Gram-Schmidt Arnoldi orthogonalization with progressive Givens rotations monotonically decreases residual norm $\|\mathbf{r}_k\|_2$ within each Krylov cycle.
///
/// # Domain of Validity
/// Requires non-singular square $A \in \mathbb{R}^{n \times n}$, finite $\mathbf{b}, \mathbf{x}_0, \mathbf{x}_{\text{out}} \in \mathbb{R}^n$, `restart >= 1`, `tol > 0`, `max_restarts >= 1`, and sufficient capacity in `ws`.
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
    let row_ptr = scope.alloc_slice::<usize>(n + 1)?;
    let col_ind = scope.alloc_slice::<usize>(nnz)?;
    let values = scope.alloc_slice::<T>(nnz)?;
    fill_csr(a, row_ptr, col_ind, values);
    let csr = CsrMatrixView::new(n, n, row_ptr, col_ind, values)?;
    x_out.copy_from(x0);
    let report = gmres_solve(
        &csr,
        b.as_slice(),
        x_out.as_mut_slice(),
        restart,
        tol,
        max_restarts,
        &mut scope,
    )?;
    debug_assert_eq!(x_out.len(), n);
    Ok(report.iterations)
}
