//! Numerical Linear Algebra Solvers.
//!
//! Provides implementations for solving linear systems, least squares, and matrix operations
//! commonly used in engineering and physics problems.

use nalgebra::{DMatrix, DVector};
use thiserror::Error;

/// Errors related to Linear Algebra operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum LinearAlgebraError {
    /// The matrix is singular and cannot be inverted or decomposed.
    #[error("Matrix is singular")]
    SingularMatrix,
    /// Dimensions of operands are incompatible.
    #[error("Matrix/Vector dimensions are incompatible")]
    DimensionMismatch,
    /// Solution could not be found.
    #[error("Solution could not be found")]
    SolutionNotFound,
}

/// Solves a linear system $Ax = b$ using LU decomposition (or best available method).
///
/// # Arguments
///
/// * `a` - The matrix $A$.
/// * `b` - The vector $b$.
///
/// # Returns
///
/// * `Result<DVector<f64>, LinearAlgebraError>` - The solution vector $x$.
#[verified_engine::verified]
pub fn solve_linear_system(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<DVector<f64>, LinearAlgebraError> {
    a.clone()
        .lu()
        .solve(b)
        .ok_or(LinearAlgebraError::SolutionNotFound)
}

/// Solves the Normal Equation for Least Squares.
///
/// $$ \mathbf{c} = (A^T A)^{-1} A^T \mathbf{b} $$
///
/// Used for finding the optimal coefficients that minimize the error squared.
///
/// # Arguments
///
/// * `a` - Design matrix $A$.
/// * `b` - Observation vector $b$.
///
/// # Returns
///
/// * `Result<DVector<f64>, LinearAlgebraError>` - The coefficient vector $\mathbf{c}$.
#[verified_engine::verified]
pub fn solve_normal_equation(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<DVector<f64>, LinearAlgebraError> {
    let a_t = a.transpose();
    let ata = &a_t * a;
    let atb = &a_t * b;

    // Invert (A^T A)
    // Note: Cholesky decomposition is more efficient for symmetric positive definite matrices (like A^T A)
    match ata.clone().cholesky() {
        Some(cholesky) => Ok(cholesky.solve(&atb)),
        None => ata
            .try_inverse()
            .map(|inv| &inv * &atb)
            .ok_or(LinearAlgebraError::SingularMatrix),
    }
}

/// Calculates the Moore-Penrose Pseudoinverse ($J^{\dagger}$).
///
/// $$ \dot{\gamma} = J^{\dagger} \tau_{cmd} $$
///
/// # Arguments
///
/// * `matrix` - The matrix $J$.
/// * `epsilon` - Tolerance for singular values (to handle near-singularities).
///
/// # Returns
///
/// * `DMatrix<f64>` - The pseudoinverse matrix.
#[verified_engine::verified]
pub fn moore_penrose_pseudoinverse(matrix: &DMatrix<f64>, epsilon: f64) -> DMatrix<f64> {
    match matrix.clone().svd(true, true).pseudo_inverse(epsilon) {
        Ok(pinv) => pinv,
        Err(_) => DMatrix::zeros(matrix.ncols(), matrix.nrows()), // Should generally not happen with SVD
    }
}

/// Maximum dimension supported by zero-allocation stack-buffered calibration solvers.
pub const MAX_IN_PLACE_DIM: usize = 16;

/// Solves a symmetric positive-definite linear system $A x = b$ of dimension `n <= 16`
/// in-place via Cholesky factorization $A = L L^T$ without heap allocations.
///
/// Returns `false` if `n == 0`, `n > MAX_IN_PLACE_DIM`, slice lengths are insufficient,
/// or $A$ is not numerically positive definite.
#[verified_engine::verified]
pub fn solve_symmetric_cholesky_in_place(
    sym: &[f64],
    rhs: &[f64],
    n: usize,
    out_x: &mut [f64],
) -> bool {
    if n == 0 || n > MAX_IN_PLACE_DIM || sym.len() < n * n || rhs.len() < n || out_x.len() < n {
        return false;
    }
    let mut lower = [0.0_f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM];
    if !cholesky_decompose_stack(sym, n, &mut lower) {
        return false;
    }
    cholesky_backsolve_stack(&lower, rhs, n, out_x)
}

#[verified_engine::verified]
fn cholesky_decompose_stack(
    sym: &[f64],
    n: usize,
    lower: &mut [f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM],
) -> bool {
    for i in 0..n {
        let row_i = i * n;
        for j in 0..=i {
            let row_j = j * n;
            let sym_idx = row_i + j;
            let mut sum = sym[sym_idx];
            for k in 0..j {
                let ik = row_i + k;
                let jk = row_j + k;
                sum -= lower[ik] * lower[jk];
            }
            if i == j {
                if !sum.is_finite() || sum <= 1e-15 {
                    return false;
                }
                lower[sym_idx] = sum.sqrt();
            } else {
                let jj = row_j + j;
                let val = sum / lower[jj];
                if !val.is_finite() {
                    return false;
                }
                lower[sym_idx] = val;
            }
        }
    }
    true
}

#[verified_engine::verified]
fn cholesky_backsolve_stack(
    lower: &[f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM],
    rhs: &[f64],
    n: usize,
    out_x: &mut [f64],
) -> bool {
    let mut y = [0.0_f64; MAX_IN_PLACE_DIM];
    for i in 0..n {
        let row_i = i * n;
        let mut sum = rhs[i];
        for (j, &yj) in y.iter().enumerate().take(i) {
            let ij = row_i + j;
            sum -= lower[ij] * yj;
        }
        let ii = row_i + i;
        let yi = sum / lower[ii];
        if !yi.is_finite() {
            return false;
        }
        y[i] = yi;
    }
    for rev in 0..n {
        let i = n - 1 - rev;
        let mut sum = y[i];
        for (j, &xj) in out_x.iter().enumerate().take(n).skip(i + 1) {
            let ji = j * n;
            let idx = ji + i;
            sum -= lower[idx] * xj;
        }
        let ii = i * n + i;
        let xi = sum / lower[ii];
        if !xi.is_finite() {
            return false;
        }
        out_x[i] = xi;
    }
    true
}

/// Computes the Moore-Penrose pseudoinverse $A^{\dagger}$ of a real symmetric $n \times n$
/// matrix (`n <= 16`) in-place via cyclic Jacobi eigendecomposition with zero heap allocations.
///
/// Singular values satisfying $|\lambda_k| \le \max(\epsilon, \epsilon \cdot \max_j |\lambda_j|)$
/// are truncated to zero.
#[verified_engine::verified]
pub fn moore_penrose_pseudoinverse_symmetric_in_place(
    sym: &[f64],
    n: usize,
    epsilon: f64,
    out_pinv: &mut [f64],
) -> bool {
    let nn = n * n;
    if n == 0 || n > MAX_IN_PLACE_DIM || sym.len() < nn || out_pinv.len() < nn {
        return false;
    }
    let mut work = [0.0_f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM];
    let mut vecs = [0.0_f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM];
    init_jacobi_buffers(sym, n, &mut work, &mut vecs);
    if !run_jacobi_sweeps(&mut work, &mut vecs, n) {
        return false;
    }
    assemble_symmetric_pinv(&work, &vecs, n, epsilon, out_pinv)
}

#[verified_engine::verified]
fn init_jacobi_buffers(
    sym: &[f64],
    n: usize,
    work: &mut [f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM],
    vecs: &mut [f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM],
) {
    for i in 0..n {
        let row_i = i * n;
        for j in 0..n {
            let idx = row_i + j;
            let rev = j * n + i;
            work[idx] = 0.5 * (sym[idx] + sym[rev]);
            vecs[idx] = if i == j { 1.0 } else { 0.0 };
        }
    }
}

#[verified_engine::verified]
fn run_jacobi_sweeps(
    work: &mut [f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM],
    vecs: &mut [f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM],
    n: usize,
) -> bool {
    for _sweep in 0..48 {
        let mut max_off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let pq = p * n + q;
                let apq = work[pq];
                if !apq.is_finite() {
                    return false;
                }
                if apq.abs() > max_off {
                    max_off = apq.abs();
                }
                jacobi_rotate_pair(work, vecs, n, p, q);
            }
        }
        if max_off <= 1e-14 {
            break;
        }
    }
    true
}

#[verified_engine::verified]
fn jacobi_rotate_pair(
    work: &mut [f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM],
    vecs: &mut [f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM],
    n: usize,
    p: usize,
    q: usize,
) {
    let pp = p * n + p;
    let qq = q * n + q;
    let pq = p * n + q;
    let qp = q * n + p;
    let apq = work[pq];
    let scale = (work[pp].abs() + work[qq].abs()).max(1.0);
    if apq.abs() <= 1e-15 * scale {
        return;
    }
    let tau = (work[qq] - work[pp]) / (2.0 * apq);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;
    apply_jacobi_similarity(work, vecs, n, p, q, c, s, t, apq);
    work[pq] = 0.0;
    work[qp] = 0.0;
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn apply_jacobi_similarity(
    work: &mut [f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM],
    vecs: &mut [f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM],
    n: usize,
    p: usize,
    q: usize,
    c: f64,
    s: f64,
    t: f64,
    apq: f64,
) {
    let pp = p * n + p;
    let qq = q * n + q;
    work[pp] -= t * apq;
    work[qq] += t * apq;
    for r in 0..n {
        if r != p && r != q {
            let rp = r * n + p;
            let pr = p * n + r;
            let rq = r * n + q;
            let qr = q * n + r;
            let arp = work[rp];
            let arq = work[rq];
            let new_rp = c * arp - s * arq;
            let new_rq = s * arp + c * arq;
            work[rp] = new_rp;
            work[pr] = new_rp;
            work[rq] = new_rq;
            work[qr] = new_rq;
        }
        let vrp = r * n + p;
        let vrq = r * n + q;
        let vp = vecs[vrp];
        let vq = vecs[vrq];
        vecs[vrp] = c * vp - s * vq;
        vecs[vrq] = s * vp + c * vq;
    }
}

#[verified_engine::verified]
fn assemble_symmetric_pinv(
    work: &[f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM],
    vecs: &[f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM],
    n: usize,
    epsilon: f64,
    out_pinv: &mut [f64],
) -> bool {
    let mut max_eig = 0.0_f64;
    for k in 0..n {
        let kk = k * n + k;
        let eig = work[kk];
        if !eig.is_finite() {
            return false;
        }
        if eig.abs() > max_eig {
            max_eig = eig.abs();
        }
    }
    let cutoff = epsilon.max(epsilon * max_eig);
    for val in out_pinv.iter_mut().take(n * n) {
        *val = 0.0;
    }
    for k in 0..n {
        let kk = k * n + k;
        let eig = work[kk];
        if eig.abs() > cutoff {
            let inv_eig = 1.0 / eig;
            accumulate_rank_one(vecs, n, k, inv_eig, out_pinv);
        }
    }
    true
}

#[verified_engine::verified]
fn accumulate_rank_one(
    vecs: &[f64; MAX_IN_PLACE_DIM * MAX_IN_PLACE_DIM],
    n: usize,
    col: usize,
    weight: f64,
    out_pinv: &mut [f64],
) {
    for i in 0..n {
        let ik = i * n + col;
        let vi = vecs[ik] * weight;
        let row_i = i * n;
        for j in 0..n {
            let jk = j * n + col;
            let ij = row_i + j;
            out_pinv[ij] += vi * vecs[jk];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_normal_equation() {
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
        let b = DVector::from_column_slice(&[1.0, 2.0, 3.0]);

        let c_vec = solve_normal_equation(&a, &b).unwrap();
        assert!(c_vec[0].abs() < math_commons::registry::TOLERANCE_FAST);
        assert!((c_vec[1] - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    fn test_in_place_cholesky_and_pinv_match_nalgebra_zero_alloc() {
        let sym = [4.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 6.0];
        let rhs = [7.0, 9.0, 9.0];
        let mut x_in_place = [0.0; 3];
        let mut pinv_in_place = [0.0; 9];

        let (ok, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            let c_ok = solve_symmetric_cholesky_in_place(&sym, &rhs, 3, &mut x_in_place);
            let p_ok =
                moore_penrose_pseudoinverse_symmetric_in_place(&sym, 3, 1e-12, &mut pinv_in_place);
            c_ok && p_ok
        });
        assert!(ok);
        for &v in &x_in_place {
            assert!((v - 1.0).abs() < 1e-12);
        }

        let dmat = DMatrix::from_row_slice(3, 3, &sym);
        let expected_pinv = moore_penrose_pseudoinverse(&dmat, 1e-12);
        for i in 0..3 {
            for j in 0..3 {
                assert!((pinv_in_place[i * 3 + j] - expected_pinv[(i, j)]).abs() < 1e-11);
            }
        }
    }

    #[test]
    fn test_in_place_symmetric_pinv_handles_rank_deficiency() {
        let singular = [1.0, 2.0, 2.0, 4.0];
        let mut pinv = [0.0; 4];
        assert!(moore_penrose_pseudoinverse_symmetric_in_place(
            &singular, 2, 1e-10, &mut pinv
        ));
        let expected =
            moore_penrose_pseudoinverse(&DMatrix::from_row_slice(2, 2, &singular), 1e-10);
        for i in 0..2 {
            for j in 0..2 {
                assert!((pinv[i * 2 + j] - expected[(i, j)]).abs() < 1e-12);
            }
        }
    }
}
