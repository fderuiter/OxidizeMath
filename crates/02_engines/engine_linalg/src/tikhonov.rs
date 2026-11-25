#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! Tikhonov / ridge regularization ($\min \|A x - b\|_2^2 + \lambda^2 \|x\|_2^2$) and damped normal solvers.

use crate::lu::{lu_solve, lu_workspace_len};
use crate::scalar::Scalar;
use crate::svd_bidiag::{svd_bidiagonal, svd_bidiagonal_workspace_len};
use crate::types::{LinalgError, SolverWorkspace, carve_scalars, check_all_finite, mat_idx};

/// Workspace scalar capacity required by [`tikhonov_solve`] for an `m x n` matrix.
/// `[cite:engine_linalg]`
#[inline]
pub const fn tikhonov_workspace_len(m: usize, n: usize) -> usize {
    let k = if m < n { m } else { n };
    m * k + k + k * n + svd_bidiagonal_workspace_len(m, n)
}

/// Workspace scalar capacity required by [`solve_damped_normal`] for an `n x n` normal matrix.
/// `[cite:engine_linalg]`
#[inline]
pub const fn damped_normal_workspace_len(n: usize) -> usize {
    n * n + lu_workspace_len(n)
}

/// Solves Tikhonov ridge regression $\min_x \|A x - b\|_2^2 + \lambda^2 \|x\|_2^2$ via SVD filter factors.
///
/// Uses $x = \sum_i \frac{\sigma_i}{\sigma_i^2 + \lambda^2} (u_i^T b) v_i$ directly on $A$, avoiding
/// condition-number squaring in $A^T A + \lambda^2 I$.
/// Reference: Golub, Hansen, & O'Leary (1999), DOI: 10.1137/S0895479897326432;
/// [`docs/math_specs/engine_linalg.md`](../../../../docs/math_specs/engine_linalg.md). `[cite:advanced_linear_algebra]`
///
/// # Stability
/// Spectral filter weights $\sigma_i / (\sigma_i^2 + \lambda^2)$ are bounded by $1 / (2\lambda)$ for
/// $\lambda > 0$ and truncate $\sigma_i \le \max(m,n)\varepsilon_{\text{mach}}\sigma_1$ when $\lambda = 0$.
///
/// # Domain of Validity
/// Requires finite `a` (`m * n`), finite `b` (`m`), finite regularization `lambda >= 0`, `x` (`n`),
/// and `ws` capacity at least [`tikhonov_workspace_len`]`(m, n)`.
pub fn tikhonov_solve<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    b: &[T],
    lambda: T,
    x: &mut [T],
    ws: &mut impl SolverWorkspace,
) -> Result<(), LinalgError> {
    if !lambda.is_finite() || lambda < T::ZERO {
        return Err(LinalgError::InvalidInput);
    }
    if m.checked_mul(n) != Some(a.len()) || b.len() != m || x.len() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(a)?;
    check_all_finite(b)?;

    let k = m.min(n);
    x.fill(T::ZERO);
    if k == 0 {
        return Ok(());
    }

    let mut scope = ws.reborrow();
    let u = carve_scalars::<T>(&mut scope, m * k)?;
    let s = carve_scalars::<T>(&mut scope, k)?;
    let vt = carve_scalars::<T>(&mut scope, k * n)?;
    svd_bidiagonal(a, m, n, u, s, vt, &mut scope)?;

    let lam_sq = lambda * lambda;
    let zero_cut = T::from_usize(m.max(n)) * T::EPSILON * s[0];
    for idx in 0..k {
        let sigma = s[idx];
        let denom = sigma * sigma + lam_sq;
        if denom == T::ZERO || (lambda == T::ZERO && sigma <= zero_cut) {
            continue;
        }
        let mut ut_b = T::ZERO;
        for r in 0..m {
            ut_b += u[mat_idx(r, idx, k)] * b[r];
        }
        let weight = (sigma * ut_b) / denom;
        for c in 0..n {
            x[c] += weight * vt[mat_idx(idx, c, n)];
        }
    }
    Ok(())
}

/// Diagonal regularization strategy for [`solve_damped_normal`].
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DampingStyle<T> {
    /// Isotropic additive diagonal damping $H_{ii} + \lambda$.
    Isotropic(T),
    /// Multiplicative Marquardt diagonal damping $H_{ii} + \lambda \max(|H_{ii}|, \text{diag\_floor})$.
    Marquardt {
        /// Damping multiplier $\lambda \ge 0$.
        lambda: T,
        /// Minimum diagonal scale floor $\ge 0$.
        diag_floor: T,
    },
}

/// Solves a damped Levenberg-Marquardt / ridge normal system $(H + \lambda D) \delta = g$ via pivoted LU.
///
/// Reference: Marquardt (1963), DOI: 10.1137/0111030;
/// [`docs/math_specs/engine_linalg.md`](../../../../docs/math_specs/engine_linalg.md). `[cite:engine_linalg]`
///
/// # Stability
/// Positive diagonal shifts (`Isotropic` or scale-invariant `Marquardt` with `diag_floor`) lift
/// singular normal-matrix eigenvalues away from zero prior to partial-pivoting LU factorization.
///
/// # Domain of Validity
/// Requires finite `jtj` (`n * n`), finite `jtr` (`n`), finite non-negative damping parameters
/// (`lambda >= 0`, `diag_floor >= 0`), `delta` (`n`), `pivots` (`n`), and `ws` capacity at least
/// [`damped_normal_workspace_len`]`(n)`.
pub fn solve_damped_normal<T: Scalar>(
    jtj: &[T],
    jtr: &[T],
    n: usize,
    damping: DampingStyle<T>,
    delta: &mut [T],
    pivots: &mut [usize],
    ws: &mut impl SolverWorkspace,
) -> Result<(), LinalgError> {
    match damping {
        DampingStyle::Isotropic(lambda) => {
            if !lambda.is_finite() || lambda < T::ZERO {
                return Err(LinalgError::InvalidInput);
            }
        }
        DampingStyle::Marquardt { lambda, diag_floor } => {
            if !lambda.is_finite()
                || lambda < T::ZERO
                || !diag_floor.is_finite()
                || diag_floor < T::ZERO
            {
                return Err(LinalgError::InvalidInput);
            }
        }
    }
    if n.checked_mul(n) != Some(jtj.len())
        || jtr.len() != n
        || delta.len() != n
        || pivots.len() != n
    {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(jtj)?;
    check_all_finite(jtr)?;

    let mut scope = ws.reborrow();
    let damped = carve_scalars::<T>(&mut scope, n * n)?;
    damped.copy_from_slice(jtj);
    for i in 0..n {
        let ii = mat_idx(i, i, n);
        let d_ii = damped[ii];
        damped[ii] = match damping {
            DampingStyle::Isotropic(lambda) => d_ii + lambda,
            DampingStyle::Marquardt { lambda, diag_floor } => {
                let base = d_ii.abs().max(diag_floor);
                d_ii + lambda * base
            }
        };
    }
    lu_solve(damped, jtr, delta, n, pivots, &mut scope)
}
