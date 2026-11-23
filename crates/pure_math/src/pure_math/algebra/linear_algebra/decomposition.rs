//! # Matrix Decomposition, Conditioning, and Norms
//!
//! Factorizing matrices into orthogonal, triangular, and diagonal factors enables backward-stable
//! least-squares estimation, spectral conditioning diagnostics, and low-rank approximation.
//!
//! ## Supported Factorizations & Invariants
//! * **Householder QR (`qr_decomposition`):** $A = QR$ with $Q^T Q = I$ and upper trapezoidal $R$.
//! * **Cholesky (`cholesky_decomposition`):** $A = L L^T$ for symmetric positive-definite matrices.
//! * **Thin SVD (`svd_decomposition`):** $A = U \operatorname{diag}(\sigma) V^T$ with $\sigma_1 \ge \dots \ge \sigma_r \ge 0$.
//! * **Conditioning & Norms (`condition_number`, `matrix_norm`):** $\kappa_2(A) = \sigma_{\max} / \sigma_{\min}$ and $\|A\|_{1, 2, \infty, F}$.

use super::numerical::{LinearAlgebraError, bounded_svd, ensure_finite_matrix};
use nalgebra::{DMatrix, DVector};

#[cfg(test)]
#[path = "decomposition_tests.rs"]
mod tests;

/// Orthogonal-triangular factors `(q, r)` produced by [`qr_decomposition`].
pub type QrFactors = (DMatrix<f64>, DMatrix<f64>);

/// Singular-value factors `(u, singular_values, v_t)` produced by [`svd_decomposition`].
pub type SvdFactors = (DMatrix<f64>, DVector<f64>, DMatrix<f64>);

/// Matrix norm selector for [`matrix_norm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixNorm {
    /// Induced $L_1$ operator norm (maximum absolute column sum): $\|A\|_1 = \max_j \sum_i |a_{ij}|$.
    One,
    /// Induced $L_2$ spectral norm (largest singular value): $\|A\|_2 = \sigma_{\max}(A)$.
    Spectral,
    /// Induced $L_\infty$ operator norm (maximum absolute row sum): $\|A\|_\infty = \max_i \sum_j |a_{ij}|$.
    Infinity,
    /// Frobenius (Hilbert-Schmidt) norm: $\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$.
    Frobenius,
}

/// Computes the effective rank of a matrix using Singular Value Decomposition (SVD).
///
/// The effective rank is determined by counting singular values greater than a given tolerance.
/// Returns `0` if the matrix is empty or contains non-finite values.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::algebra::linear_algebra::decomposition::effective_rank;
/// use math_explorer::nalgebra::DMatrix;
///
/// let diag = DMatrix::from_diagonal(&math_explorer::nalgebra::DVector::from_vec(vec![1.0, 1e-11, 3.0]));
/// assert_eq!(effective_rank(&diag, Some(1e-5)), 2);
/// ```
#[verified_engine::verified]
pub fn effective_rank(matrix: &DMatrix<f64>, tolerance: Option<f64>) -> usize {
    let tol = tolerance.unwrap_or(math_commons::registry::TOLERANCE_HIGH);
    if !tol.is_finite() {
        return 0;
    }
    let Ok(svd) = bounded_svd(matrix, false, false) else {
        return 0;
    };
    svd.singular_values
        .iter()
        .filter(|&&sigma| sigma > tol)
        .count()
}

/// Computes a rank-`k` Eckart-Young-Mirsky approximation $A_k$ using thin SVD.
///
/// # Arguments
/// * `matrix` - Input matrix $A \in \mathbb{R}^{m \times n}$.
/// * `k` - Target approximation rank.
///
/// # Returns
/// `Result<DMatrix<f64>, LinearAlgebraError>` containing $A_k = \sum_{i=1}^{\min(k, r)} \sigma_i u_i v_i^T$.
#[verified_engine::verified]
pub fn low_rank_approximation(
    matrix: &DMatrix<f64>,
    k: usize,
) -> Result<DMatrix<f64>, LinearAlgebraError> {
    let (u, s, v_t) = svd_decomposition(matrix)?;
    let rank = k.min(s.len());
    let mut u_scaled = DMatrix::zeros(u.nrows(), u.ncols());
    for i in 0..rank {
        u_scaled.set_column(i, &(u.column(i) * s[i]));
    }
    Ok(u_scaled * v_t)
}

/// Computes the thin Householder $QR$ factorization $A = QR$.
///
/// Returns `(q, r)` where $Q \in \mathbb{R}^{m \times \min(m, n)}$ has orthonormal columns
/// and $R \in \mathbb{R}^{\min(m, n) \times n}$ is upper triangular/trapezoidal.
#[verified_engine::verified]
pub fn qr_decomposition(matrix: &DMatrix<f64>) -> Result<QrFactors, LinearAlgebraError> {
    ensure_finite_matrix(matrix)?;
    Ok(matrix.clone().qr().unpack())
}

/// Computes the Cholesky factorization $A = L L^T$ of a real symmetric positive-definite matrix.
///
/// Returns `Err(LinearAlgebraError::DimensionMismatch)` if `matrix` is empty or non-square,
/// `Err(LinearAlgebraError::NonFiniteInput)` if any entry is `NaN`/`Inf`, and
/// `Err(LinearAlgebraError::NotPositiveDefinite)` if `matrix` is asymmetric or not positive definite.
#[verified_engine::verified]
pub fn cholesky_decomposition(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>, LinearAlgebraError> {
    ensure_finite_matrix(matrix)?;
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let max_abs = matrix.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let sym_tol = f64::EPSILON * (n as f64) * 64.0 * max_abs.max(1.0);
    for i in 0..n {
        for j in (i + 1)..n {
            if (matrix[(i, j)] - matrix[(j, i)]).abs() > sym_tol {
                return Err(LinearAlgebraError::NotPositiveDefinite);
            }
        }
    }
    matrix
        .clone()
        .cholesky()
        .map(|chol| chol.unpack())
        .ok_or(LinearAlgebraError::NotPositiveDefinite)
}

fn orthogonalize_against_prefix(vec: &mut DVector<f64>, basis: &DMatrix<f64>, count: usize) {
    for _ in 0..2 {
        for i in 0..count {
            let col = basis.column(i);
            let proj = col.dot(vec);
            *vec -= col * proj;
        }
    }
}

fn orthonormal_null_column(
    u_raw: &DMatrix<f64>,
    u_built: &DMatrix<f64>,
    col_idx: usize,
) -> DVector<f64> {
    let mut cand = u_raw.column(col_idx).into_owned();
    orthogonalize_against_prefix(&mut cand, u_built, col_idx);
    let norm = cand.norm();
    if norm > 1e-12 {
        return cand / norm;
    }
    let m = u_raw.nrows();
    for axis in 0..m {
        let mut e = DVector::zeros(m);
        e[axis] = 1.0;
        orthogonalize_against_prefix(&mut e, u_built, col_idx);
        let enorm = e.norm();
        if enorm > 1e-12 {
            return e / enorm;
        }
    }
    cand
}

fn sort_svd_descending(u: &mut DMatrix<f64>, s: &mut DVector<f64>, v_t: &mut DMatrix<f64>) {
    let r = s.len();
    let mut order: Vec<usize> = (0..r).collect();
    order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
    if order.iter().enumerate().all(|(i, &idx)| i == idx) {
        return;
    }
    let (u_old, s_old, vt_old) = (u.clone(), s.clone(), v_t.clone());
    for (new_idx, &old_idx) in order.iter().enumerate() {
        s[new_idx] = s_old[old_idx];
        u.set_column(new_idx, &u_old.column(old_idx));
        v_t.set_row(new_idx, &vt_old.row(old_idx));
    }
}

fn refine_tall_svd(matrix: &DMatrix<f64>) -> Result<SvdFactors, LinearAlgebraError> {
    let (m, n) = (matrix.nrows(), matrix.ncols());
    let svd = bounded_svd(matrix, true, true)?;
    let u_raw = svd.u.ok_or(LinearAlgebraError::ConvergenceFailed)?;
    let mut v_t = svd.v_t.ok_or(LinearAlgebraError::ConvergenceFailed)?;
    let max_sigma = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let rank_tol = f64::EPSILON * (m as f64) * max_sigma;
    let av = matrix * v_t.transpose();

    let mut u = DMatrix::zeros(m, n);
    let mut s = DVector::zeros(n);
    for j in 0..n {
        let mut w = av.column(j).into_owned();
        orthogonalize_against_prefix(&mut w, &u, j);
        let sigma_j = w.norm();
        if sigma_j > rank_tol {
            s[j] = sigma_j;
            u.set_column(j, &(w / sigma_j));
        } else {
            s[j] = sigma_j.min(svd.singular_values[j]);
            u.set_column(j, &orthonormal_null_column(&u_raw, &u, j));
        }
    }
    sort_svd_descending(&mut u, &mut s, &mut v_t);
    Ok((u, s, v_t))
}

/// Computes the thin Singular Value Decomposition $A = U \operatorname{diag}(\sigma) V^T$.
///
/// Returns `(u, singular_values, v_t)` with singular values sorted in descending order.
#[verified_engine::verified]
pub fn svd_decomposition(matrix: &DMatrix<f64>) -> Result<SvdFactors, LinearAlgebraError> {
    ensure_finite_matrix(matrix)?;
    if matrix.nrows() < matrix.ncols() {
        let (u_t, s, vt_t) = refine_tall_svd(&matrix.transpose())?;
        return Ok((vt_t.transpose(), s, u_t.transpose()));
    }
    refine_tall_svd(matrix)
}

/// Computes the spectral $L_2$ condition number $\kappa_2(A) = \sigma_{\max}(A) / \sigma_{\min}(A)$.
///
/// Returns `Ok(f64::INFINITY)` when the smallest singular value is zero.
#[verified_engine::verified]
pub fn condition_number(matrix: &DMatrix<f64>) -> Result<f64, LinearAlgebraError> {
    let svd = bounded_svd(matrix, false, false)?;
    let sigma_max = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let sigma_min = svd
        .singular_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    if sigma_max == 0.0 || sigma_min == 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok(sigma_max / sigma_min)
}

/// Evaluates the requested operator or Frobenius norm of `matrix`.
#[verified_engine::verified]
pub fn matrix_norm(matrix: &DMatrix<f64>, norm: MatrixNorm) -> Result<f64, LinearAlgebraError> {
    ensure_finite_matrix(matrix)?;
    match norm {
        MatrixNorm::One => {
            let max_col = (0..matrix.ncols())
                .map(|j| matrix.column(j).iter().map(|v| v.abs()).sum::<f64>())
                .fold(0.0_f64, f64::max);
            Ok(max_col)
        }
        MatrixNorm::Infinity => {
            let max_row = (0..matrix.nrows())
                .map(|i| matrix.row(i).iter().map(|v| v.abs()).sum::<f64>())
                .fold(0.0_f64, f64::max);
            Ok(max_row)
        }
        MatrixNorm::Frobenius => Ok(matrix.norm()),
        MatrixNorm::Spectral => {
            let svd = bounded_svd(matrix, false, false)?;
            Ok(svd.singular_values.iter().copied().fold(0.0_f64, f64::max))
        }
    }
}

// [cite:advanced_linear_algebra]
