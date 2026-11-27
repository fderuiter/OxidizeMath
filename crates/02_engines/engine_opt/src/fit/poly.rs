//! Stack-allocated centered polynomial ordinary least-squares regression (degrees 1–4).
// [cite:engine_opt]

use super::{CurveFit, FitModel};
use crate::error::FitError;
use math_scalars::series::PointSeries;
use math_scalars::{ComplexField, RealField};

/// Fits a polynomial of `degree` (`1..=4`) in linear coordinates with zero heap allocations.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn fit_polynomial<P: PointSeries + ?Sized>(
    points: &P,
    degree: usize,
) -> Result<CurveFit, FitError> {
    fit_polynomial_scaled(points, degree, false, false)
}

/// Fits a polynomial of `degree` (`1..=4`) with optional $\log_{10}$ axis transforms.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn fit_polynomial_scaled<P: PointSeries + ?Sized>(
    points: &P,
    degree: usize,
    log_x: bool,
    log_y: bool,
) -> Result<CurveFit, FitError> {
    let fit_model = FitModel::from_degree(degree)?;
    let required = degree + 1;
    let (finite_count, valid_count, sum_tx, sum_ty, min_tx, max_tx) =
        scan_poly_domain(points, log_x, log_y);

    if finite_count < required {
        return Err(FitError::InsufficientPoints {
            required,
            actual: finite_count,
        });
    }
    if valid_count < required {
        return Err(FitError::InvalidDomain {
            reason: "insufficient positive coordinates for logarithmic polynomial fit",
        });
    }

    let u_mean = sum_tx / (valid_count as f64);
    let u_scale = (u_mean - min_tx).abs().max((max_tx - u_mean).abs());
    if !u_mean.is_finite() || !u_scale.is_finite() || u_scale <= 1e-12 {
        return Err(FitError::SingularMatrix);
    }

    let centered = solve_centered_poly_system(points, degree, (log_x, log_y), (u_mean, u_scale))?;
    let coeffs = uncenter_polynomial_coeffs(u_mean, u_scale, &centered, required);
    let mean_ty = sum_ty / (valid_count as f64);
    let (r_squared, rmse) = compute_poly_metrics(
        points,
        &centered[..required],
        (u_mean, u_scale),
        (log_x, log_y),
        mean_ty,
        valid_count,
    );

    Ok(CurveFit::new(
        fit_model,
        coeffs,
        required,
        r_squared,
        rmse,
        degree,
        valid_count,
        log_x,
        log_y,
        log_x || log_y,
    )
    .with_centered_poly(u_mean, u_scale, centered))
}

fn scan_poly_domain<P: PointSeries + ?Sized>(
    points: &P,
    log_x: bool,
    log_y: bool,
) -> (usize, usize, f64, f64, f64, f64) {
    let mut finite_count = 0usize;
    let mut valid_count = 0usize;
    let mut sum_tx = 0.0_f64;
    let mut sum_ty = 0.0_f64;
    let mut min_tx = f64::INFINITY;
    let mut max_tx = f64::NEG_INFINITY;

    for idx in 0..points.len() {
        let p = points.point_at(idx);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        finite_count += 1;
        if (log_x && p[0] <= 0.0) || (log_y && p[1] <= 0.0) {
            continue;
        }
        let tx = if log_x {
            ComplexField::log10(p[0])
        } else {
            p[0]
        };
        let ty = if log_y {
            ComplexField::log10(p[1])
        } else {
            p[1]
        };
        valid_count += 1;
        sum_tx += tx;
        sum_ty += ty;
        min_tx = min_tx.min(tx);
        max_tx = max_tx.max(tx);
    }
    (finite_count, valid_count, sum_tx, sum_ty, min_tx, max_tx)
}

struct StackPolyQr {
    r: [[f64; 5]; 5],
    qty: [f64; 5],
    ncols: usize,
    nrows: usize,
}

impl StackPolyQr {
    fn new(ncols: usize) -> Result<Self, FitError> {
        if ncols == 0 || ncols > 5 {
            return Err(FitError::SingularMatrix);
        }
        Ok(Self {
            r: [[0.0; 5]; 5],
            qty: [0.0; 5],
            ncols,
            nrows: 0,
        })
    }

    fn push_row(&mut self, row: &[f64], rhs: f64) -> Result<(), FitError> {
        if row.len() != self.ncols || !rhs.is_finite() || row.iter().any(|v| !v.is_finite()) {
            return Err(FitError::SingularMatrix);
        }
        let mut work = [0.0_f64; 5];
        work[..self.ncols].copy_from_slice(row);
        let mut y = rhs;
        for j in 0..self.ncols {
            let xj = work[j];
            if xj == 0.0 {
                continue;
            }
            let rjj = self.r[j][j];
            let hyp = RealField::hypot(rjj, xj);
            let c = rjj / hyp;
            let s = xj / hyp;
            self.r[j][j] = hyp;
            for (r_jk, work_k) in self.r[j][(j + 1)..self.ncols]
                .iter_mut()
                .zip(work[(j + 1)..self.ncols].iter_mut())
            {
                let old_r = *r_jk;
                let old_w = *work_k;
                *r_jk = c * old_r + s * old_w;
                *work_k = -s * old_r + c * old_w;
            }
            let qj = self.qty[j];
            self.qty[j] = c * qj + s * y;
            y = -s * qj + c * y;
        }
        self.nrows += 1;
        Ok(())
    }

    fn solve(&self, rel_tol: f64) -> Result<[f64; 5], FitError> {
        if self.nrows < self.ncols {
            return Err(FitError::SingularMatrix);
        }
        let mut max_diag = 0.0_f64;
        for i in 0..self.ncols {
            max_diag = max_diag.max(self.r[i][i].abs());
        }
        if max_diag == 0.0 {
            return Err(FitError::SingularMatrix);
        }
        let thresh = rel_tol.max((self.nrows as f64) * f64::EPSILON) * max_diag;
        let mut sol = [0.0_f64; 5];
        for i in (0..self.ncols).rev() {
            let diag = self.r[i][i];
            if diag.abs() <= thresh {
                return Err(FitError::SingularMatrix);
            }
            let mut sum = self.qty[i];
            for (&rij, &xj) in self.r[i][(i + 1)..self.ncols]
                .iter()
                .zip(sol[(i + 1)..self.ncols].iter())
            {
                sum -= rij * xj;
            }
            sol[i] = sum / diag;
        }
        Ok(sol)
    }
}

fn solve_centered_poly_system<P: PointSeries + ?Sized>(
    points: &P,
    deg: usize,
    axes: (bool, bool),
    center_scale: (f64, f64),
) -> Result<[f64; 5], FitError> {
    let required = deg + 1;
    let (log_x, log_y) = axes;
    let (u_mean, u_scale) = center_scale;
    let mut qr = StackPolyQr::new(required)?;

    for idx in 0..points.len() {
        let p = points.point_at(idx);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        if (log_x && p[0] <= 0.0) || (log_y && p[1] <= 0.0) {
            continue;
        }
        let tx = if log_x {
            ComplexField::log10(p[0])
        } else {
            p[0]
        };
        let ty = if log_y {
            ComplexField::log10(p[1])
        } else {
            p[1]
        };
        let z = (tx - u_mean) / u_scale;
        let mut row = [0.0_f64; 5];
        let mut zp = 1.0_f64;
        for slot in row.iter_mut().take(required) {
            *slot = zp;
            zp *= z;
        }
        qr.push_row(&row[..required], ty)?;
    }

    qr.solve(1e-13)
}

fn uncenter_polynomial_coeffs(
    u_mean: f64,
    u_scale: f64,
    centered: &[f64; 5],
    count: usize,
) -> [f64; 5] {
    let mut out = [0.0_f64; 5];
    if count == 0 {
        return out;
    }
    let deg = count - 1;
    let mut k = count;
    while k > 0 {
        k -= 1;
        let b_k = centered[k];
        let mut j = deg;
        while j >= 1 {
            out[j] = (out[j - 1] - u_mean * out[j]) / u_scale;
            j -= 1;
        }
        out[0] = (-u_mean * out[0]) / u_scale + b_k;
    }
    out
}

fn compute_poly_metrics<P: PointSeries + ?Sized>(
    points: &P,
    centered_coeffs: &[f64],
    center_scale: (f64, f64),
    axes: (bool, bool),
    mean_ty: f64,
    valid_count: usize,
) -> (f64, f64) {
    let (u_mean, u_scale) = center_scale;
    let (log_x, log_y) = axes;
    let mut ss_tot = 0.0_f64;
    let mut ss_res = 0.0_f64;
    for idx in 0..points.len() {
        let p = points.point_at(idx);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        if (log_x && p[0] <= 0.0) || (log_y && p[1] <= 0.0) {
            continue;
        }
        let tx = if log_x {
            ComplexField::log10(p[0])
        } else {
            p[0]
        };
        let ty = if log_y {
            ComplexField::log10(p[1])
        } else {
            p[1]
        };
        let z = (tx - u_mean) / u_scale;
        let pred = eval_polynomial_horner_raw(centered_coeffs, z);
        let diff = ty - pred;
        let tot = ty - mean_ty;
        ss_res += diff * diff;
        ss_tot += tot * tot;
    }
    let r_squared = if ss_tot > 1e-15 {
        (1.0 - ss_res / ss_tot).clamp(-10.0, 1.0)
    } else if ss_res <= 1e-12 {
        1.0
    } else {
        0.0
    };
    let rmse = ComplexField::sqrt(ss_res / (valid_count as f64));
    (r_squared, rmse)
}

/// Evaluates polynomial $c_0 + c_1 x + \dots + c_d x^d$ via Horner's method.
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn eval_polynomial_horner<T: RealField>(coeffs: &[T], x: T) -> T {
    eval_polynomial_horner_raw(coeffs, x)
}

#[inline(always)]
pub(super) fn eval_polynomial_horner_raw<T: RealField>(coeffs: &[T], x: T) -> T {
    let mut acc = T::ZERO;
    let mut k = coeffs.len();
    while k > 0 {
        k -= 1;
        acc = acc * x + coeffs[k];
    }
    acc
}

/// Solves an $n \times n$ ($1 \le n \le 5$) dense linear system $A x = b$ on the stack via Gaussian elimination with partial pivoting.
///
/// Reference: Golub & Van Loan (2013), DOI: 10.56021/9781421407944; [math_commons specification](../../../../../docs/math_specs/math_commons.md).
///
/// # Stability
/// Partial column pivoting bounds element growth for $n \le 5$; pivots with $|p| < 10^{-12}$ fail closed with [`FitError::SingularMatrix`].
///
/// # Domain of Validity
/// Requires $1 \le n \le 5$, finite entries in the leading $n \times n$ block of `mat` and `rhs`, and $\det(A) \neq 0$.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn solve_dense_small(
    mat: &mut [[f64; 5]; 5],
    rhs: &[f64; 5],
    n: usize,
) -> Result<[f64; 5], FitError> {
    solve_dense_n_raw(mat, rhs, n)
}

pub(super) fn solve_dense_small_raw(
    mat: &mut [[f64; 5]; 5],
    rhs: &[f64; 5],
    n: usize,
) -> Result<[f64; 5], FitError> {
    solve_dense_n_raw(mat, rhs, n)
}

pub(super) fn solve_dense_n_raw<const N: usize>(
    mat: &mut [[f64; N]; N],
    rhs: &[f64; N],
    n: usize,
) -> Result<[f64; N], FitError> {
    if n == 0 || n > N || n > 16 {
        return Err(FitError::SingularMatrix);
    }
    let mut b = [0.0_f64; 16];
    let mut max_abs = 0.0_f64;
    for r in 0..n {
        if !rhs[r].is_finite() {
            return Err(FitError::SingularMatrix);
        }
        b[r] = rhs[r];
        for &v in &mat[r][..n] {
            if !v.is_finite() {
                return Err(FitError::SingularMatrix);
            }
            max_abs = max_abs.max(v.abs());
        }
    }
    let eff_tol = 1e-12_f64.max((n as f64) * f64::EPSILON * max_abs);
    for k in 0..n {
        let mut piv_row = k;
        let mut piv_abs = mat[k][k].abs();
        for (r, row) in mat.iter().enumerate().take(n).skip(k + 1) {
            let cand = row[k].abs();
            if cand > piv_abs {
                piv_abs = cand;
                piv_row = r;
            }
        }
        if piv_abs <= eff_tol || piv_abs == 0.0 {
            return Err(FitError::SingularMatrix);
        }
        if piv_row != k {
            mat.swap(k, piv_row);
            b.swap(k, piv_row);
        }
        let (piv_part, elim_part) = mat[..n].split_at_mut(k + 1);
        let piv_slice = &piv_part[k];
        let piv = piv_slice[k];
        for (row_offset, row_i) in elim_part.iter_mut().enumerate() {
            let mult = row_i[k] / piv;
            b[k + 1 + row_offset] -= mult * b[k];
            for (rij, &ukj) in row_i[(k + 1)..n]
                .iter_mut()
                .zip(piv_slice[(k + 1)..n].iter())
            {
                *rij -= mult * ukj;
            }
        }
    }
    let mut x = [0.0_f64; N];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= mat[i][j] * x[j];
        }
        x[i] = sum / mat[i][i];
    }
    Ok(x)
}

pub(crate) fn symmetrize_upper<const N: usize>(mat: &mut [[f64; N]; N], n: usize) {
    for row in 0..n {
        let (head, tail) = mat[..n].split_at_mut(row + 1);
        let src = &head[row][row + 1..n];
        for (dst_row, &val) in tail.iter_mut().zip(src.iter()) {
            dst_row[row] = val;
        }
    }
}

#[cfg(feature = "alloc")]
pub use super::format::format_polynomial_formula;
