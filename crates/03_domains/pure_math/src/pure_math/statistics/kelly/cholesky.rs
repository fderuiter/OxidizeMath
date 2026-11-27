//! Zero-allocation row-major Cholesky factorization and triangular solve for Kelly portfolios.

use crate::error::KellyError;
use math_scalars::mat_idx;
use nalgebra::DMatrix;

pub(super) const CHOL_PIVOT_TOL: f64 = 1e-12;

/// Computes in-place lower-triangular Cholesky factor `L` into row-major `chol_flat[..n*n]`.
#[verified_engine::verified]
pub(crate) fn cholesky_into_flat(
    covariance: &DMatrix<f64>,
    n: usize,
    sym_flat: &mut [f64],
    chol_flat: &mut [f64],
) -> Result<(), KellyError> {
    chol_flat[..n * n].fill(0.0);
    for i in 0..n {
        sym_flat[mat_idx(i, i, n)] = covariance[(i, i)];
        for j in (i + 1)..n {
            let avg = 0.5 * (covariance[(i, j)] + covariance[(j, i)]);
            sym_flat[mat_idx(i, j, n)] = avg;
            sym_flat[mat_idx(j, i, n)] = avg;
        }
    }
    for i in 0..n {
        for j in 0..=i {
            let mut sum = sym_flat[mat_idx(i, j, n)];
            for k in 0..j {
                sum -= chol_flat[mat_idx(i, k, n)] * chol_flat[mat_idx(j, k, n)];
            }
            if i == j {
                if !sum.is_finite() || sum <= CHOL_PIVOT_TOL * CHOL_PIVOT_TOL {
                    return Err(KellyError::NotPositiveDefinite);
                }
                let diag = sum.sqrt();
                if diag <= CHOL_PIVOT_TOL {
                    return Err(KellyError::NotPositiveDefinite);
                }
                chol_flat[mat_idx(i, i, n)] = diag;
            } else {
                chol_flat[mat_idx(i, j, n)] = sum / chol_flat[mat_idx(j, j, n)];
            }
        }
    }
    Ok(())
}

/// Solves `(L * L^T) * out = rhs` in-place from a row-major lower-triangular Cholesky factor `L`.
#[verified_engine::verified]
pub(super) fn cholesky_solve_flat(
    chol_flat: &[f64],
    rhs: &[f64],
    out: &mut [f64],
    n: usize,
) -> Result<(), KellyError> {
    for i in 0..n {
        let diag = chol_flat[mat_idx(i, i, n)];
        if !diag.is_finite() || diag <= CHOL_PIVOT_TOL {
            return Err(KellyError::NotPositiveDefinite);
        }
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= chol_flat[mat_idx(i, k, n)] * out[k];
        }
        out[i] = sum / diag;
    }
    for i in (0..n).rev() {
        let diag = chol_flat[mat_idx(i, i, n)];
        let mut sum = out[i];
        for k in (i + 1)..n {
            sum -= chol_flat[mat_idx(k, i, n)] * out[k];
        }
        out[i] = sum / diag;
    }
    Ok(())
}
