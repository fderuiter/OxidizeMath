//! Linear and polynomial Vandermonde least-squares solvers (degrees 1 to 4).

use crate::fit::{FitResult, FitType};

pub(crate) fn eval_scaled_polynomial(coeffs: &[f64], x: f64, log_x: bool, log_y: bool) -> f64 {
    let u = if log_x {
        if !x.is_finite() || x <= 0.0 {
            return f64::NAN;
        }
        x.log10()
    } else {
        x
    };
    let mut poly = 0.0;
    let mut u_pow = 1.0;
    for &coeff in coeffs {
        poly += coeff * u_pow;
        u_pow *= u;
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
    let coeffs = solve_polynomial_normal_eqs(&transformed, degree)?;
    let r_squared = compute_transformed_r2(&transformed, &coeffs);
    let (rmse, residuals) = compute_data_space_residuals(&raw_valid, &coeffs, log_x, log_y);
    let fit_type = if degree == 1 {
        FitType::Linear
    } else {
        FitType::Polynomial(degree)
    };
    Some(FitResult {
        fit_type,
        coefficients: coeffs.clone(),
        r_squared,
        rmse,
        residuals: residuals.into(),
        degree,
        formula: format_scaled_polynomial_formula(&coeffs, log_x, log_y),
        dataset_name: dataset_name.to_owned(),
        log_x,
        log_y,
        log_mode: log_x || log_y,
    })
}

fn solve_polynomial_normal_eqs(points: &[[f64; 2]], degree: usize) -> Option<Vec<f64>> {
    let m = degree + 1;
    let mut sum_x_pow = [0.0_f64; 9];
    let mut rhs = [0.0_f64; 5];
    for p in points {
        let mut cur = 1.0;
        for (k, slot) in sum_x_pow.iter_mut().enumerate().take(2 * degree + 1) {
            *slot += cur;
            if k < m {
                rhs[k] += p[1] * cur;
            }
            cur *= p[0];
        }
    }
    let mut mat = [[0.0_f64; 5]; 5];
    for (r, row) in mat.iter_mut().enumerate().take(m) {
        row[..m].copy_from_slice(&sum_x_pow[r..(r + m)]);
    }
    solve_dense_small(&mut mat, &mut rhs, m)
}

/// Gaussian elimination with partial pivoting for small systems ($1 \le n \le 5$).
pub(crate) fn solve_dense_small(
    a: &mut [[f64; 5]; 5],
    b: &mut [f64; 5],
    n: usize,
) -> Option<Vec<f64>> {
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

fn compute_transformed_r2(valid_points: &[[f64; 2]], coeffs: &[f64]) -> f64 {
    let n = valid_points.len() as f64;
    let y_mean = valid_points.iter().map(|p| p[1]).sum::<f64>() / n;
    let (mut ss_tot, mut ss_res) = (0.0, 0.0);
    for p in valid_points {
        let mut y_pred = 0.0;
        let mut x_pow = 1.0;
        for &c in coeffs {
            y_pred += c * x_pow;
            x_pow *= p[0];
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
    coeffs: &[f64],
    log_x: bool,
    log_y: bool,
) -> (f64, Vec<[f64; 2]>) {
    let mut residuals = Vec::with_capacity(raw_valid.len());
    let mut sq_sum = 0.0;
    for p in raw_valid {
        let y_hat = eval_scaled_polynomial(coeffs, p[0], log_x, log_y);
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

fn format_scaled_polynomial_formula(coeffs: &[f64], log_x: bool, log_y: bool) -> String {
    let lhs = if log_y { "log10(y)" } else { "y" };
    let var = if log_x { "log10(x)" } else { "x" };
    let degree = coeffs.len().saturating_sub(1);
    if degree == 0 {
        return format!("{lhs} = {:.4}", coeffs.first().copied().unwrap_or(0.0));
    }
    let mut terms = Vec::new();
    for power in (0..=degree).rev() {
        let coeff = coeffs[power];
        if coeff.abs() < 1e-9 && terms.len() < degree {
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
    match power {
        0 => format!("{sign_str}{abs_c:.4}"),
        1 if (abs_c - 1.0).abs() < 1e-6 => format!("{sign_str}{var}"),
        1 => format!("{sign_str}{abs_c:.4}{var}"),
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
                format!("{sign_str}{abs_c:.4}{base}")
            }
        }
    }
}
