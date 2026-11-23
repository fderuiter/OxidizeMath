//! Linear and polynomial Vandermonde least-squares solvers (degrees 1 to 4).

use crate::{
    fit::{FitResult, FitType},
    format::format_fit_param,
};

pub(crate) fn eval_scaled_polynomial(coeffs: &[f64], x: f64, log_x: bool, log_y: bool) -> f64 {
    let u = if log_x {
        if !x.is_finite() || x <= 0.0 {
            return f64::NAN;
        }
        x.log10()
    } else {
        x
    };
    let mut poly = 0.0_f64;
    for &coeff in coeffs.iter().rev() {
        poly = poly.mul_add(u, coeff);
    }
    let out = if log_y {
        10.0_f64
            .powf(poly.clamp(-300.0, 300.0))
            .clamp(1e-300, 1e300)
    } else {
        poly
    };
    if out.is_finite() { out } else { f64::NAN }
}

pub(crate) fn eval_centered_polynomial(
    u_mean: f64,
    u_scale: f64,
    centered_coeffs: &[f64],
    x: f64,
    log_x: bool,
    log_y: bool,
) -> f64 {
    let u = if log_x {
        if !x.is_finite() || x <= 0.0 {
            return f64::NAN;
        }
        x.log10()
    } else {
        x
    };
    let z = (u - u_mean) / u_scale;
    let mut poly = 0.0_f64;
    for &coeff in centered_coeffs.iter().rev() {
        poly = poly.mul_add(z, coeff);
    }
    let out = if log_y {
        10.0_f64
            .powf(poly.clamp(-300.0, 300.0))
            .clamp(1e-300, 1e300)
    } else {
        poly
    };
    if out.is_finite() { out } else { f64::NAN }
}

pub(crate) fn fit_polynomial_scaled(
    points: &[[f64; 2]],
    target_degree: usize,
    dataset_name: &str,
    log_x: bool,
    log_y: bool,
) -> Option<FitResult> {
    let raw_valid: Vec<[f64; 2]> = points
        .iter()
        .copied()
        .filter(|p| {
            p[0].is_finite() && p[1].is_finite() && (!log_x || p[0] > 0.0) && (!log_y || p[1] > 0.0)
        })
        .collect();
    if raw_valid.len() < 2 {
        return None;
    }
    let transformed: Vec<[f64; 2]> = raw_valid
        .iter()
        .map(|p| {
            [
                if log_x { p[0].log10() } else { p[0] },
                if log_y { p[1].log10() } else { p[1] },
            ]
        })
        .collect();

    let degree = target_degree.clamp(1, 4).min(transformed.len() - 1);
    let (u_mean, u_scale, centered_coeffs) = solve_polynomial_normal_eqs(&transformed, degree)?;
    let coeffs = uncenter_polynomial_coeffs(u_mean, u_scale, &centered_coeffs);
    let formula_u_scale = transformed.iter().fold(1.0_f64, |m, p| m.max(p[0].abs()));
    let r_squared = compute_transformed_r2(&transformed, u_mean, u_scale, &centered_coeffs);
    let (rmse, residuals) =
        compute_data_space_residuals(&raw_valid, u_mean, u_scale, &centered_coeffs, log_x, log_y);
    let fit_type = if degree == 1 {
        FitType::Linear
    } else {
        FitType::Polynomial(degree)
    };
    Some(FitResult {
        fit_type,
        formula: format_scaled_polynomial_formula(&coeffs, formula_u_scale, log_x, log_y),
        coefficients: coeffs,
        r_squared,
        rmse,
        residuals,
        degree,
        dataset_name: dataset_name.to_owned(),
        log_x,
        log_y,
        log_mode: log_x || log_y,
        custom_evaluator: None,
        poly_centered: Some((u_mean, u_scale, centered_coeffs)),
    })
}

fn solve_polynomial_normal_eqs(points: &[[f64; 2]], degree: usize) -> Option<(f64, f64, Vec<f64>)> {
    let n = points.len() as f64;
    let u_mean = points.iter().map(|p| p[0]).sum::<f64>() / n;
    let u_scale = points
        .iter()
        .map(|p| (p[0] - u_mean).abs())
        .fold(0.0_f64, f64::max);
    if !u_mean.is_finite() || !u_scale.is_finite() || u_scale <= 1e-12 {
        return None;
    }
    let m = degree + 1;
    let mut sum_z_pow = [0.0_f64; 9];
    let mut rhs = [0.0_f64; 6];
    for p in points {
        let z = (p[0] - u_mean) / u_scale;
        let mut cur = 1.0;
        for (k, slot) in sum_z_pow.iter_mut().enumerate().take(2 * degree + 1) {
            *slot += cur;
            if k < m {
                rhs[k] += p[1] * cur;
            }
            cur *= z;
        }
    }
    let mut mat = [[0.0_f64; 6]; 6];
    for (r, row) in mat.iter_mut().enumerate().take(m) {
        row[..m].copy_from_slice(&sum_z_pow[r..(r + m)]);
    }
    let centered = solve_dense_small(&mut mat, &mut rhs, m)?;
    Some((u_mean, u_scale, centered))
}

fn uncenter_polynomial_coeffs(u_mean: f64, u_scale: f64, centered: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; centered.len()];
    let deg = centered.len().saturating_sub(1);
    for &b_k in centered.iter().rev() {
        for j in (1..=deg).rev() {
            out[j] = (out[j - 1] - u_mean * out[j]) / u_scale;
        }
        if !out.is_empty() {
            out[0] = (-u_mean * out[0]) / u_scale + b_k;
        }
    }
    out
}

/// Gaussian elimination with partial pivoting for small systems ($1 \le n \le 6$).
pub(crate) fn solve_dense_small(
    a: &mut [[f64; 6]; 6],
    b: &mut [f64; 6],
    n: usize,
) -> Option<Vec<f64>> {
    if n == 0 || n > 6 {
        return None;
    }
    for i in 0..n {
        let (mut max_row, mut max_val) = (i, a[i][i].abs());
        for (r, row) in a.iter().enumerate().take(n).skip(i + 1) {
            let val = row[i].abs();
            if val.is_finite() && val > max_val {
                max_val = val;
                max_row = r;
            }
        }
        if !max_val.is_finite() || max_val < 1e-12 {
            return None;
        }
        if max_row != i {
            a.swap(i, max_row);
            b.swap(i, max_row);
        }
        let pivot = a[i][i];
        if !pivot.is_finite() {
            return None;
        }
        for val in a[i].iter_mut().take(n).skip(i) {
            *val /= pivot;
        }
        b[i] /= pivot;
        let pivot_row = a[i];
        let pivot_b = b[i];
        for r in (i + 1)..n {
            let factor = a[r][i];
            if factor.abs() > 1e-15 {
                for (col_val, &piv_val) in a[r][i..n].iter_mut().zip(&pivot_row[i..n]) {
                    *col_val -= factor * piv_val;
                }
                b[r] -= factor * pivot_b;
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut sum = b[r];
        for (coeff, &sol) in a[r][(r + 1)..n].iter().zip(&x[(r + 1)..n]) {
            sum -= coeff * sol;
        }
        if !sum.is_finite() {
            return None;
        }
        x[r] = sum;
    }
    Some(x)
}

fn compute_transformed_r2(
    valid_points: &[[f64; 2]],
    u_mean: f64,
    u_scale: f64,
    centered_coeffs: &[f64],
) -> f64 {
    let n = valid_points.len() as f64;
    let y_mean = valid_points.iter().map(|p| p[1]).sum::<f64>() / n;
    let (mut ss_tot, mut ss_res) = (0.0, 0.0);
    for p in valid_points {
        let z = (p[0] - u_mean) / u_scale;
        let mut y_pred = 0.0_f64;
        for &c in centered_coeffs.iter().rev() {
            y_pred = y_pred.mul_add(z, c);
        }
        ss_tot += (p[1] - y_mean).powi(2);
        ss_res += (p[1] - y_pred).powi(2);
    }
    if ss_tot < 1e-12 {
        if ss_res < 1e-12 { 1.0 } else { 0.0 }
    } else {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    }
}

fn compute_data_space_residuals(
    raw_valid: &[[f64; 2]],
    u_mean: f64,
    u_scale: f64,
    centered_coeffs: &[f64],
    log_x: bool,
    log_y: bool,
) -> (f64, Vec<[f64; 2]>) {
    let mut residuals = Vec::with_capacity(raw_valid.len());
    let mut sq_sum = 0.0;
    for p in raw_valid {
        let y_hat = eval_centered_polynomial(u_mean, u_scale, centered_coeffs, p[0], log_x, log_y);
        let res = p[1] - y_hat;
        if res.is_finite() {
            residuals.push([p[0], res]);
            sq_sum += res * res;
        }
    }
    let rmse = if residuals.is_empty() {
        0.0
    } else {
        (sq_sum / residuals.len() as f64).sqrt()
    };
    (rmse, residuals)
}

fn format_scaled_polynomial_formula(
    coeffs: &[f64],
    u_scale: f64,
    log_x: bool,
    log_y: bool,
) -> String {
    let lhs = if log_y { "log10(y)" } else { "y" };
    let var = if log_x { "log10(x)" } else { "x" };
    let degree = coeffs.len().saturating_sub(1);
    if degree == 0 {
        return format!(
            "{lhs} = {}",
            format_fit_param(coeffs.first().copied().unwrap_or(0.0))
        );
    }
    let max_contrib = coeffs
        .iter()
        .enumerate()
        .map(|(p, &c)| c.abs() * u_scale.powi(p as i32))
        .fold(0.0_f64, f64::max);
    let mut terms = Vec::new();
    for power in (0..=degree).rev() {
        let coeff = coeffs[power];
        let contrib = coeff.abs() * u_scale.powi(power as i32);
        if max_contrib > 0.0 && contrib <= 1e-12 * max_contrib && terms.len() < degree {
            continue;
        }
        if max_contrib == 0.0 && power > 0 {
            continue;
        }
        terms.push(format_poly_term(coeff, power, var, log_x, terms.is_empty()));
    }
    if terms.is_empty() {
        format!("{lhs} = 0")
    } else {
        format!("{lhs} = {}", terms.join(""))
    }
}

fn format_poly_term(coeff: f64, power: usize, var: &str, log_x: bool, is_first: bool) -> String {
    let superscripts = ["", "", "²", "³", "⁴"];
    let sign_str = match (is_first, coeff < 0.0) {
        (true, true) => "-",
        (true, false) => "",
        (false, true) => " - ",
        (false, false) => " + ",
    };
    let abs_c = coeff.abs();
    let mag_str = format_fit_param(abs_c);
    match power {
        0 => format!("{sign_str}{mag_str}"),
        1 if (abs_c - 1.0).abs() < 1e-6 => format!("{sign_str}{var}"),
        1 => format!("{sign_str}{mag_str}{var}"),
        p => {
            let sup = superscripts.get(p).copied().unwrap_or("");
            let base = if log_x {
                format!("({var}){sup}")
            } else {
                format!("{var}{sup}")
            };
            if (abs_c - 1.0).abs() < 1e-6 {
                format!("{sign_str}{base}")
            } else {
                format!("{sign_str}{mag_str}{base}")
            }
        }
    }
}
