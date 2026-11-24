//! Shared zero-allocation dense linear system adapters for nonlinear root-finding
//! and implicit/Rosenbrock ODE integrators, backed by [`engine_linalg::lu`].

/// Normalizes IEEE-754 negative zero (`-0.0`) to canonical positive zero (`0.0`).
#[verified_engine::verified]
#[inline]
pub(crate) fn canonical_zero(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x }
}

/// Checks whether `a` and `b` are finite and have strictly opposite signs without calling `signum`.
#[verified_engine::verified]
#[inline]
pub(crate) fn has_strict_sign_change(a: f64, b: f64) -> bool {
    (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0)
}

/// Computes an in-place `PA = LU` factorization of the `n x n` row-major matrix `lu`,
/// recording row permutations in `pivots[..n]` via [`engine_linalg::lu::lu_factor_in_place`].
#[verified_engine::verified]
pub(crate) fn lu_factor_in_place(
    lu: &mut [f64],
    pivots: &mut [usize],
    n: usize,
    pivot_tol: f64,
) -> bool {
    if n == 0 || lu.len() < n * n || pivots.len() < n || !pivot_tol.is_finite() {
        return false;
    }
    engine_linalg::lu::lu_factor_in_place(&mut lu[..n * n], n, &mut pivots[..n], pivot_tol).is_ok()
}

/// Solves `A * x = b` in-place using a pre-factored `PA = LU` decomposition from [`lu_factor_in_place`].
///
/// `scratch[..n]` buffers the input right-hand side before [`engine_linalg::lu::lu_solve_factored`]
/// writes the solution into `rhs[..n]`.
#[verified_engine::verified]
pub(crate) fn lu_solve_in_place(
    lu: &[f64],
    pivots: &[usize],
    rhs: &mut [f64],
    scratch: &mut [f64],
    n: usize,
) -> bool {
    if n == 0 || lu.len() < n * n || pivots.len() < n || rhs.len() < n || scratch.len() < n {
        return false;
    }
    scratch[..n].copy_from_slice(&rhs[..n]);
    if engine_linalg::lu::lu_solve_factored(
        &lu[..n * n],
        n,
        &pivots[..n],
        &scratch[..n],
        &mut rhs[..n],
    )
    .is_err()
    {
        return false;
    }
    rhs[..n].iter().all(|v| v.is_finite())
}

/// Solves the `n x n` row-major linear system `mat * x = rhs` in-place via `PA = LU`
/// partial pivoting, writing the solution vector into `rhs[..n]`.
#[verified_engine::verified]
pub(crate) fn solve_linear_in_place(
    mat: &mut [f64],
    rhs: &mut [f64],
    pivots: &mut [usize],
    scratch: &mut [f64],
    n: usize,
    pivot_tol: f64,
) -> bool {
    lu_factor_in_place(mat, pivots, n, pivot_tol) && lu_solve_in_place(mat, pivots, rhs, scratch, n)
}

// [cite:advanced_linear_algebra]
