//! Stack-allocated centered polynomial ordinary least-squares regression (degrees 1–4).
// [cite:math_commons]

use super::micro_solvers::{StreamingGivensQr, solve_givens_dense_small};
use super::{CurveFit, FitModel};
use crate::error::FitError;
use crate::series::PointSeries;
use math_scalars::RealField;

/// Fits a polynomial of `degree` (`1..=4`) in linear coordinates with zero heap allocations.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn fit_polynomial<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
    degree: usize,
) -> Result<CurveFit<T>, FitError> {
    fit_polynomial_scaled(points, degree, false, false)
}

/// Fits a polynomial of `degree` (`1..=4`) with optional $\log_{10}$ axis transforms.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn fit_polynomial_scaled<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
    degree: usize,
    log_x: bool,
    log_y: bool,
) -> Result<CurveFit<T>, FitError> {
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

    let inv_valid = T::ONE / T::from_usize(valid_count);
    let u_mean = sum_tx * inv_valid;
    let u_scale = (u_mean - min_tx).abs().max((max_tx - u_mean).abs());
    if !u_mean.is_finite() || !u_scale.is_finite() || u_scale <= T::from_f64(1e-12) {
        return Err(FitError::SingularMatrix);
    }

    let centered = solve_centered_poly_system(points, degree, (log_x, log_y), (u_mean, u_scale))?;
    let coeffs = uncenter_polynomial_coeffs(u_mean, u_scale, &centered, required);
    let mean_ty = sum_ty * inv_valid;
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

fn scan_poly_domain<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
    log_x: bool,
    log_y: bool,
) -> (usize, usize, T, T, T, T) {
    let mut finite_count = 0usize;
    let mut valid_count = 0usize;
    let mut sum_tx = T::ZERO;
    let mut sum_ty = T::ZERO;
    let mut min_tx = T::INFINITY;
    let mut max_tx = T::NEG_INFINITY;

    for idx in 0..points.len() {
        let p = points.point_at(idx);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        finite_count += 1;
        if (log_x && p[0] <= T::ZERO) || (log_y && p[1] <= T::ZERO) {
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

fn solve_centered_poly_system<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
    deg: usize,
    axes: (bool, bool),
    center_scale: (T, T),
) -> Result<[T; 5], FitError> {
    let required = deg + 1;
    let (log_x, log_y) = axes;
    let (u_mean, u_scale) = center_scale;
    let mut qr = StreamingGivensQr::<T>::new(required)?;

    for idx in 0..points.len() {
        let p = points.point_at(idx);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        if (log_x && p[0] <= T::ZERO) || (log_y && p[1] <= T::ZERO) {
            continue;
        }
        let tx = if log_x { p[0].log10() } else { p[0] };
        let ty = if log_y { p[1].log10() } else { p[1] };
        let z = (tx - u_mean) / u_scale;
        let mut row = [T::ZERO; 5];
        let mut zp = T::ONE;
        for slot in row.iter_mut().take(required) {
            *slot = zp;
            zp *= z;
        }
        qr.push_row(&row[..required], ty)?;
    }

    qr.solve::<5>(T::from_f64(1e-13))
}

fn uncenter_polynomial_coeffs<T: RealField>(
    u_mean: T,
    u_scale: T,
    centered: &[T; 5],
    count: usize,
) -> [T; 5] {
    let mut out = [T::ZERO; 5];
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

fn compute_poly_metrics<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
    centered_coeffs: &[T],
    center_scale: (T, T),
    axes: (bool, bool),
    mean_ty: T,
    valid_count: usize,
) -> (T, T) {
    let (u_mean, u_scale) = center_scale;
    let (log_x, log_y) = axes;
    let mut ss_tot = T::ZERO;
    let mut ss_res = T::ZERO;
    for idx in 0..points.len() {
        let p = points.point_at(idx);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        if (log_x && p[0] <= T::ZERO) || (log_y && p[1] <= T::ZERO) {
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
    let r_squared = if ss_tot > T::from_f64(1e-15) {
        (T::ONE - ss_res / ss_tot).clamp(T::from_f64(-10.0), T::ONE)
    } else if ss_res <= T::from_f64(1e-12) {
        T::ONE
    } else {
        T::ZERO
    };
    let rmse = (ss_res / T::from_usize(valid_count)).sqrt();
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

/// Solves an $n \times n$ ($1 \le n \le 5$) dense linear system $A x = b$ on the stack via orthogonal Givens QR rotations.
///
/// Reference: Golub & Van Loan (2013), DOI: 10.56021/9781421407944; [math_commons specification](../../../../docs/math_specs/math_commons.md).
///
/// # Stability
/// Coupled Givens plane rotations orthogonalize the leading $n \times n$ block on stack arrays; near-singular diagonal entries fail closed with [`FitError::SingularMatrix`].
///
/// # Domain of Validity
/// Requires $1 \le n \le 5$, finite entries in the leading $n \times n$ block of `mat` and `rhs`, and $\det(A) \neq 0$.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn solve_dense_small<T: RealField>(
    mat: &mut [[T; 5]; 5],
    rhs: &[T; 5],
    n: usize,
) -> Result<[T; 5], FitError> {
    solve_givens_dense_small(mat, rhs, n)
}

pub(super) fn solve_dense_small_raw<T: RealField>(
    mat: &mut [[T; 5]; 5],
    rhs: &[T; 5],
    n: usize,
) -> Result<[T; 5], FitError> {
    solve_givens_dense_small(mat, rhs, n)
}

pub(crate) fn symmetrize_upper<T: Copy, const N: usize>(mat: &mut [[T; N]; N], n: usize) {
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
