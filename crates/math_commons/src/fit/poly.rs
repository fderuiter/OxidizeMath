//! Stack-allocated centered polynomial ordinary least-squares regression (degrees 1–4).
// [cite:math_commons]

use super::{CurveFit, FitModel};
use crate::error::FitError;
use crate::series::PointSeries;

/// Fits a polynomial of `degree` (`1..=4`) in linear coordinates with zero heap allocations.
#[verified_engine::verified]
pub fn fit_polynomial<P: PointSeries + ?Sized>(
    points: &P,
    degree: usize,
) -> Result<CurveFit, FitError> {
    fit_polynomial_scaled(points, degree, false, false)
}

/// Fits a polynomial of `degree` (`1..=4`) with optional $\log_{10}$ axis transforms.
#[verified_engine::verified]
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
        let tx = if log_x { p[0].log10() } else { p[0] };
        let ty = if log_y { p[1].log10() } else { p[1] };
        valid_count += 1;
        sum_tx += tx;
        sum_ty += ty;
        min_tx = min_tx.min(tx);
        max_tx = max_tx.max(tx);
    }
    (finite_count, valid_count, sum_tx, sum_ty, min_tx, max_tx)
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
    let mut qr = engine_linalg::StreamingQr::<f64, 5>::new(required)
        .map_err(|_| FitError::SingularMatrix)?;

    for idx in 0..points.len() {
        let p = points.point_at(idx);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        if (log_x && p[0] <= 0.0) || (log_y && p[1] <= 0.0) {
            continue;
        }
        let tx = if log_x { p[0].log10() } else { p[0] };
        let ty = if log_y { p[1].log10() } else { p[1] };
        let z = (tx - u_mean) / u_scale;
        let mut row = [0.0_f64; 5];
        let mut zp = 1.0_f64;
        for slot in row.iter_mut().take(required) {
            *slot = zp;
            zp *= z;
        }
        qr.push_row(&row[..required], ty)
            .map_err(|_| FitError::SingularMatrix)?;
    }

    qr.solve(1e-13).map_err(|_| FitError::SingularMatrix)
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
        let tx = if log_x { p[0].log10() } else { p[0] };
        let ty = if log_y { p[1].log10() } else { p[1] };
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
    let rmse = (ss_res / (valid_count as f64)).sqrt();
    (r_squared, rmse)
}

/// Evaluates polynomial $c_0 + c_1 x + \dots + c_d x^d$ via Horner's method.
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn eval_polynomial_horner(coeffs: &[f64], x: f64) -> f64 {
    eval_polynomial_horner_raw(coeffs, x)
}

#[inline(always)]
pub(super) fn eval_polynomial_horner_raw(coeffs: &[f64], x: f64) -> f64 {
    let mut acc = 0.0_f64;
    let mut k = coeffs.len();
    while k > 0 {
        k -= 1;
        acc = acc * x + coeffs[k];
    }
    acc
}

/// Solves an $n \times n$ ($n \le 5$) dense linear system on the stack via partial pivoting.
#[verified_engine::verified]
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
    let mut flat = [0.0_f64; 256];
    for (r, row) in mat.iter().enumerate().take(n) {
        let dst_start = r * n;
        flat[dst_start..(dst_start + n)].copy_from_slice(&row[..n]);
    }
    let mut pivots = [0_usize; 16];
    engine_linalg::lu_factor_in_place(&mut flat[..(n * n)], n, &mut pivots[..n], 1e-12)
        .map_err(|_| FitError::SingularMatrix)?;
    let mut x = [0.0_f64; N];
    engine_linalg::lu_solve_factored(&flat[..(n * n)], n, &pivots[..n], &rhs[..n], &mut x[..n])
        .map_err(|_| FitError::SingularMatrix)?;
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

pub use super::format::format_polynomial_formula;
