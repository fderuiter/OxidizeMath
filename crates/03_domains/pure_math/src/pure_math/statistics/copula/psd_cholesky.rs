//! Zero-allocation symmetric Jacobi PSD check and positive-semidefinite Cholesky factorization.

use crate::error::CopulaError;
use math_memory::SolverWorkspace;
use math_scalars::mat_idx;

const MAX_JACOBI_SWEEPS: usize = 64;
const JACOBI_TOL_SCALE: f64 = 16.0;

#[inline]
#[verified_engine::verified]
pub(super) const fn psd_cholesky_scratch_len(n: usize) -> usize {
    n.saturating_mul(n)
}

#[verified_engine::verified]
fn jacobi_eigenvalue_sweep(work: &mut [f64], n: usize, tol: f64) -> f64 {
    let mut max_off = 0.0_f64;
    for p in 0..n {
        for q in (p + 1)..n {
            let apq = work[mat_idx(p, q, n)];
            max_off = max_off.max(apq.abs());
            if apq.abs() <= tol {
                continue;
            }
            let app = work[mat_idx(p, p, n)];
            let aqq = work[mat_idx(q, q, n)];
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + 1.0_f64.hypot(tau))
            } else {
                -1.0 / (-tau + 1.0_f64.hypot(tau))
            };
            let c = 1.0 / 1.0_f64.hypot(t);
            let s = t * c;

            work[mat_idx(p, p, n)] = app - t * apq;
            work[mat_idx(q, q, n)] = aqq + t * apq;
            work[mat_idx(p, q, n)] = 0.0;
            work[mat_idx(q, p, n)] = 0.0;
            for r in 0..n {
                if r != p && r != q {
                    let arp = work[mat_idx(r, p, n)];
                    let arq = work[mat_idx(r, q, n)];
                    let new_rp = c * arp - s * arq;
                    let new_rq = s * arp + c * arq;
                    work[mat_idx(r, p, n)] = new_rp;
                    work[mat_idx(p, r, n)] = new_rp;
                    work[mat_idx(r, q, n)] = new_rq;
                    work[mat_idx(q, r, n)] = new_rq;
                }
            }
        }
    }
    max_off
}

#[verified_engine::verified]
fn verify_min_eigenvalue_nonnegative(
    sym_row: &[f64],
    work: &mut [f64],
    n: usize,
    psd_tol: f64,
) -> Result<(), CopulaError> {
    work[..n * n].copy_from_slice(&sym_row[..n * n]);
    if n <= 1 {
        if n == 1 && work[0] < -psd_tol {
            return Err(CopulaError::NotPositiveDefinite);
        }
        return Ok(());
    }
    let mut scale = 0.0_f64;
    for &v in &work[..n * n] {
        scale = scale.max(v.abs());
    }
    let tol = JACOBI_TOL_SCALE * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
    let mut converged = false;
    for _ in 0..MAX_JACOBI_SWEEPS {
        if jacobi_eigenvalue_sweep(work, n, tol) <= tol {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(CopulaError::NotPositiveDefinite);
    }
    for i in 0..n {
        if work[mat_idx(i, i, n)] < -psd_tol {
            return Err(CopulaError::NotPositiveDefinite);
        }
    }
    Ok(())
}

#[verified_engine::verified]
pub(super) fn factor_psd_cholesky_in_workspace(
    sym_row: &[f64],
    lower: &mut [f64],
    n: usize,
    psd_tol: f64,
    workspace: &mut (impl SolverWorkspace + ?Sized),
) -> Result<(), CopulaError> {
    let mut cur = workspace.cursor();
    let work = cur
        .alloc_slice::<f64>(psd_cholesky_scratch_len(n))
        .map_err(|_| CopulaError::NotPositiveDefinite)?;
    verify_min_eigenvalue_nonnegative(sym_row, work, n, psd_tol)?;

    let snap_tol = psd_tol.min(1e-12);
    lower[..n * n].fill(0.0);
    for j in 0..n {
        let mut diag = sym_row[mat_idx(j, j, n)];
        for k in 0..j {
            let l_jk = lower[mat_idx(j, k, n)];
            diag -= l_jk * l_jk;
        }
        if diag < -psd_tol {
            return Err(CopulaError::NotPositiveDefinite);
        }
        let l_jj = if diag <= snap_tol { 0.0 } else { diag.sqrt() };
        lower[mat_idx(j, j, n)] = l_jj;
        if l_jj > 0.0 {
            for i in (j + 1)..n {
                let mut sum = sym_row[mat_idx(i, j, n)];
                for k in 0..j {
                    sum -= lower[mat_idx(i, k, n)] * lower[mat_idx(j, k, n)];
                }
                lower[mat_idx(i, j, n)] = sum / l_jj;
            }
        }
    }
    Ok(())
}
