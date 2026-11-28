#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! LU decomposition with partial pivoting, linear system solvers, determinant, and matrix inverse.

use crate::scalar::Scalar;
use crate::types::{LinalgError, SolverWorkspace, carve_scalars, check_all_finite, mat_idx};

/// Workspace scalar capacity required by [`lu_solve`] for an `n x n` matrix.
/// `[cite:engine_linalg]`
#[inline]
pub const fn lu_workspace_len(n: usize) -> usize {
    n * n
}

/// Workspace scalar capacity required by [`lu_inverse`] for an `n x n` matrix.
/// `[cite:engine_linalg]`
#[inline]
pub const fn lu_inverse_workspace_len(n: usize) -> usize {
    n * n + 2 * n
}

fn swap_rows<T: Scalar>(mat: &mut [T], n: usize, r1: usize, r2: usize) {
    if r1 == r2 {
        return;
    }
    for c in 0..n {
        mat.swap(mat_idx(r1, c, n), mat_idx(r2, c, n));
    }
}

/// Factors row-major `lu` (`n x n`) in-place as $P A = L U$ with partial pivoting.
///
/// Records the row permutation in `pivots` (`pivots.len() == n`) and returns the permutation
/// parity (`+1` for even row swaps, `-1` for odd row swaps).
/// `[cite:engine_linalg]`
pub fn lu_factor_in_place<T: Scalar>(
    lu: &mut [T],
    n: usize,
    pivots: &mut [usize],
    tol: T,
) -> Result<i32, LinalgError> {
    if n.checked_mul(n) != Some(lu.len()) || pivots.len() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(lu)?;
    for (i, slot) in pivots.iter_mut().enumerate() {
        *slot = i;
    }
    if n == 0 {
        return Ok(1);
    }

    let mut max_abs = T::ZERO;
    for &v in lu.iter() {
        max_abs = max_abs.max(v.abs());
    }
    let eff_tol = tol.max(T::from_usize(n) * T::EPSILON * max_abs);
    let mut parity = 1_i32;

    for k in 0..n {
        let mut best_row = k;
        let mut best_val = lu[mat_idx(k, k, n)].abs();
        for i in (k + 1)..n {
            let cand = lu[mat_idx(i, k, n)].abs();
            if cand > best_val {
                best_val = cand;
                best_row = i;
            }
        }
        if best_val <= eff_tol || best_val == T::ZERO {
            return Err(LinalgError::SingularMatrix);
        }
        if best_row != k {
            swap_rows(lu, n, k, best_row);
            pivots.swap(k, best_row);
            parity = -parity;
        }
        let pivot = lu[mat_idx(k, k, n)];
        let (top, bottom) = lu.split_at_mut((k + 1) * n);
        let pivot_tail = &top[k * n + (k + 1)..(k + 1) * n];
        for row in bottom.chunks_exact_mut(n) {
            let lu_ik = row[k];
            if lu_ik == T::ZERO {
                continue;
            }
            let mult = lu_ik / pivot;
            row[k] = mult;
            for (dst, &upper) in row[k + 1..].iter_mut().zip(pivot_tail.iter()) {
                *dst -= mult * upper;
            }
        }
    }
    Ok(parity)
}

/// Solves $A x = b$ given an in-place $P A = L U$ factorization from [`lu_factor_in_place`].
///
/// Reference: Higham (2002), *Accuracy and Stability of Numerical Algorithms*, DOI: 10.1137/1.9780898718027;
/// [`docs/math_specs/engine_linalg.md`](../../../../docs/math_specs/engine_linalg.md). `[cite:engine_linalg]`
///
/// # Stability
/// Partial pivoting bounds unit-lower multipliers $|L_{ij}| \le 1$, yielding backward error
/// $\|\Delta A\|_\infty \le O(n\,\rho_n\,\varepsilon_{\text{mach}})\|A\|_\infty$.
///
/// # Domain of Validity
/// Requires `lu.len() == n * n`, `pivots.len() == n` (valid permutation indices), finite `b` (`n`),
/// `x` (`n`), and non-zero upper-triangular pivots $U_{ii} \ne 0$ (`LinalgError::SingularMatrix`).
pub fn lu_solve_factored<T: Scalar>(
    lu: &[T],
    n: usize,
    pivots: &[usize],
    b: &[T],
    x: &mut [T],
) -> Result<(), LinalgError> {
    if n.checked_mul(n) != Some(lu.len()) || pivots.len() != n || b.len() != n || x.len() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(b)?;

    for i in 0..n {
        let mut sum = b[pivots[i]];
        for j in 0..i {
            sum -= lu[mat_idx(i, j, n)] * x[j];
        }
        x[i] = sum;
    }

    for i in (0..n).rev() {
        let mut sum = x[i];
        for j in (i + 1)..n {
            sum -= lu[mat_idx(i, j, n)] * x[j];
        }
        let diag = lu[mat_idx(i, i, n)];
        if diag == T::ZERO {
            return Err(LinalgError::SingularMatrix);
        }
        x[i] = sum / diag;
    }
    Ok(())
}

/// Solves the transposed system $A^T x = b$ given an in-place $P A = L U$ factorization.
///
/// Since $P A = L U$, $A^T = U^T L^T P$: solves $U^T y = b$, $L^T z = y$, and sets $x = P^T z$.
/// Reference: Higham (2002), DOI: 10.1137/1.9780898718027;
/// [`docs/math_specs/engine_linalg.md`](../../../../docs/math_specs/engine_linalg.md). `[cite:engine_linalg]`
///
/// # Stability
/// Shares the partial-pivoting multiplier bound $|L_{ij}| \le 1$ of [`lu_factor_in_place`], providing
/// backward-stable transposed triangular solves for Hager-Higham condition estimation.
///
/// # Domain of Validity
/// Requires `lu.len() == n * n`, `pivots.len() == n`, finite `b` (`n`), `x` (`n`), `work.len() >= n`,
/// and non-zero diagonal pivots $U_{ii} \ne 0$.
pub fn lu_solve_transpose_factored<T: Scalar>(
    lu: &[T],
    n: usize,
    pivots: &[usize],
    b: &[T],
    x: &mut [T],
    work: &mut [T],
) -> Result<(), LinalgError> {
    if n.checked_mul(n) != Some(lu.len())
        || pivots.len() != n
        || b.len() != n
        || x.len() != n
        || work.len() < n
    {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(b)?;

    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= lu[mat_idx(j, i, n)] * work[j];
        }
        let diag = lu[mat_idx(i, i, n)];
        if diag == T::ZERO {
            return Err(LinalgError::SingularMatrix);
        }
        work[i] = sum / diag;
    }

    for i in (0..n).rev() {
        let mut sum = work[i];
        for j in (i + 1)..n {
            sum -= lu[mat_idx(j, i, n)] * work[j];
        }
        work[i] = sum;
    }

    x.fill(T::ZERO);
    for i in 0..n {
        x[pivots[i]] = work[i];
    }
    Ok(())
}

/// Solves a square linear system $A x = b$ using $P A = L U$ with partial pivoting in `ws`.
///
/// Reference: Higham (2002), DOI: 10.1137/1.9780898718027;
/// [`docs/math_specs/engine_linalg.md`](../../../../docs/math_specs/engine_linalg.md). `[cite:engine_linalg]`
///
/// # Stability
/// Partial row pivoting bounds $|L_{ij}| \le 1$ and rejects singular or near-singular pivots below
/// $n\,\varepsilon_{\text{mach}}\max_{i,j}|A_{ij}|$.
///
/// # Domain of Validity
/// Requires finite nonsingular `a` (`n * n`), finite `b` (`n`), `x` (`n`), `pivots` (`n`), and `ws`
/// capacity at least [`lu_workspace_len`]`(n)`.
pub fn lu_solve<T: Scalar>(
    a: &[T],
    b: &[T],
    x: &mut [T],
    n: usize,
    pivots: &mut [usize],
    ws: &mut (impl SolverWorkspace + ?Sized),
) -> Result<(), LinalgError> {
    if n.checked_mul(n) != Some(a.len()) || b.len() != n || x.len() != n || pivots.len() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    let mut sub = ws.reborrow();
    let mut cur = sub.cursor();
    let lu_buf = carve_scalars::<T>(&mut cur, lu_workspace_len(n))?;
    lu_buf.copy_from_slice(a);
    lu_factor_in_place(lu_buf, n, pivots, T::ZERO)?;
    lu_solve_factored(lu_buf, n, pivots, b, x)
}

/// Computes $\det(A)$ from an in-place [`lu_factor_in_place`] factorization and swap `parity`.
/// `[cite:engine_linalg]`
pub fn lu_determinant<T: Scalar>(lu: &[T], n: usize, parity: i32) -> Result<T, LinalgError> {
    if n.checked_mul(n) != Some(lu.len()) {
        return Err(LinalgError::DimensionMismatch);
    }
    let mut prod = if parity < 0 { -T::ONE } else { T::ONE };
    for i in 0..n {
        prod *= lu[mat_idx(i, i, n)];
    }
    Ok(prod)
}

/// Computes the matrix inverse $A^{-1}$ of an `n x n` matrix via LU with partial pivoting.
/// `[cite:engine_linalg]`
pub fn lu_inverse<T: Scalar>(
    a: &[T],
    inv: &mut [T],
    n: usize,
    pivots: &mut [usize],
    ws: &mut (impl SolverWorkspace + ?Sized),
) -> Result<(), LinalgError> {
    if n.checked_mul(n) != Some(a.len()) || inv.len() != a.len() || pivots.len() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    let mut sub = ws.reborrow();
    let mut cur = sub.cursor();
    let lu_buf = carve_scalars::<T>(&mut cur, n * n)?;
    let rhs = carve_scalars::<T>(&mut cur, n)?;
    let col = carve_scalars::<T>(&mut cur, n)?;
    lu_buf.copy_from_slice(a);
    lu_factor_in_place(lu_buf, n, pivots, T::ZERO)?;

    for j in 0..n {
        rhs.fill(T::ZERO);
        rhs[j] = T::ONE;
        lu_solve_factored(lu_buf, n, pivots, rhs, col)?;
        for i in 0..n {
            inv[mat_idx(i, j, n)] = col[i];
        }
    }
    Ok(())
}
