//! Stack-allocated polynomial ordinary least-squares regression (degrees 1–4).
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
    let (sum_x_pow, rhs, finite_count, valid_count, sum_ty) =
        accumulate_poly_moments(points, degree, log_x, log_y);

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

    let mut mat = [[0.0_f64; 5]; 5];
    for r in 0..required {
        for c in 0..required {
            mat[r][c] = sum_x_pow[r + c];
        }
    }
    let coeffs = solve_dense_small(&mut mat, &rhs, required)?;
    let mean_ty = sum_ty / (valid_count as f64);
    let (r_squared, rmse) = compute_poly_metrics(
        points,
        &coeffs[..required],
        log_x,
        log_y,
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
    ))
}

#[verified_engine::verified]
fn accumulate_poly_moments<P: PointSeries + ?Sized>(
    points: &P,
    deg: usize,
    log_x: bool,
    log_y: bool,
) -> ([f64; 9], [f64; 5], usize, usize, f64) {
    let mut sum_x_pow = [0.0_f64; 9];
    let mut rhs = [0.0_f64; 5];
    let mut finite_count = 0usize;
    let mut valid_count = 0usize;
    let mut sum_ty = 0.0_f64;

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
        sum_ty += ty;

        let mut xp = 1.0_f64;
        for (k, slot) in sum_x_pow.iter_mut().enumerate().take(2 * deg + 1) {
            *slot += xp;
            if k <= deg {
                rhs[k] += xp * ty;
            }
            xp *= tx;
        }
    }
    (sum_x_pow, rhs, finite_count, valid_count, sum_ty)
}

#[verified_engine::verified]
fn compute_poly_metrics<P: PointSeries + ?Sized>(
    points: &P,
    coeffs: &[f64],
    log_x: bool,
    log_y: bool,
    mean_ty: f64,
    valid_count: usize,
) -> (f64, f64) {
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
        let pred = eval_polynomial_horner(coeffs, tx);
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
    if n == 0 || n > 5 {
        return Err(FitError::SingularMatrix);
    }
    let mut b = *rhs;
    for col in 0..n {
        let pivot_row = select_pivot_row(mat, col, n);
        if !mat[pivot_row][col].is_finite() || mat[pivot_row][col].abs() < 1e-12 {
            return Err(FitError::SingularMatrix);
        }
        if pivot_row != col {
            mat.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }
        eliminate_below(mat, &mut b, col, n);
    }
    back_substitute(mat, &b, n)
}

#[verified_engine::verified]
fn select_pivot_row(mat: &[[f64; 5]; 5], col: usize, n: usize) -> usize {
    let mut pivot_row = col;
    let mut pivot_val = mat[col][col].abs();
    for (row, mat_row) in mat.iter().enumerate().take(n).skip(col + 1) {
        let cand = mat_row[col].abs();
        if cand > pivot_val {
            pivot_val = cand;
            pivot_row = row;
        }
    }
    pivot_row
}

#[verified_engine::verified]
fn eliminate_below(mat: &mut [[f64; 5]; 5], b: &mut [f64; 5], col: usize, n: usize) {
    let diag = mat[col][col];
    for row in (col + 1)..n {
        let factor = mat[row][col] / diag;
        mat[row][col] = 0.0;
        for k in (col + 1)..n {
            mat[row][k] -= factor * mat[col][k];
        }
        b[row] -= factor * b[col];
    }
}

#[verified_engine::verified]
fn back_substitute(mat: &[[f64; 5]; 5], b: &[f64; 5], n: usize) -> Result<[f64; 5], FitError> {
    let mut x = [0.0_f64; 5];
    let mut row = n;
    while row > 0 {
        row -= 1;
        let mut sum = b[row];
        for k in (row + 1)..n {
            sum -= mat[row][k] * x[k];
        }
        let val = sum / mat[row][row];
        if !val.is_finite() {
            return Err(FitError::SingularMatrix);
        }
        x[row] = val;
    }
    Ok(x)
}

/// Formats a human-readable polynomial or power-law formula string.
#[must_use]
pub fn format_polynomial_formula(coeffs: &[f64], log_x: bool, log_y: bool) -> String {
    if log_x && log_y && coeffs.len() == 2 {
        let a = 10.0_f64.powf(coeffs[0]);
        return format!("y = {a:.4}·x^({:.4})", coeffs[1]);
    }
    let x_sym = if log_x { "log10(x)" } else { "x" };
    let y_sym = if log_y { "log10(y)" } else { "y" };
    let mut out = format!("{y_sym} = ");
    for (idx, &c) in coeffs.iter().enumerate() {
        if idx == 0 {
            out.push_str(&format!("{c:.4}"));
        } else {
            let sign = if c >= 0.0 { " + " } else { " - " };
            let abs_c = c.abs();
            if idx == 1 {
                out.push_str(&format!("{sign}{abs_c:.4}·{x_sym}"));
            } else {
                out.push_str(&format!("{sign}{abs_c:.4}·{x_sym}^{idx}"));
            }
        }
    }
    out
}
