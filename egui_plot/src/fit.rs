//! Curve fitting and regression analysis for discrete 2D plot datasets.

use std::fmt;

/// Supported curve fitting models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FitType {
    /// Linear regression ($y = a x + b$)
    Linear,
    /// Polynomial regression of specified degree ($1 \le \text{degree} \le 4$)
    Polynomial(usize),
}

impl FitType {
    /// Returns the degree of the polynomial model.
    pub fn degree(&self) -> usize {
        match self {
            FitType::Linear => 1,
            FitType::Polynomial(deg) => (*deg).clamp(1, 4),
        }
    }

    /// User-friendly label for the fit type.
    pub fn label(&self) -> &'static str {
        match self {
            FitType::Linear => "Linear (Degree 1)",
            FitType::Polynomial(1) => "Polynomial (Degree 1)",
            FitType::Polynomial(2) => "Polynomial (Degree 2)",
            FitType::Polynomial(3) => "Polynomial (Degree 3)",
            FitType::Polynomial(4) => "Polynomial (Degree 4)",
            FitType::Polynomial(_) => "Polynomial",
        }
    }
}

impl fmt::Display for FitType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Result of a fitted curve regression model.
#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    /// Polynomial coefficients $[c_0, c_1, \dots, c_d]$ where $f(u) = \sum_{k=0}^d c_k u^k$.
    pub coefficients: Vec<f64>,
    /// Coefficient of determination ($R^2 \in [0, 1]$).
    pub r_squared: f64,
    /// Model degree.
    pub degree: usize,
    /// Mathematical equation string representation (e.g. "y = 2.5000x + 1.2000").
    pub formula: String,
    /// Name of the fitted dataset.
    pub dataset_name: String,
    /// Whether the independent variable was transformed via $\log_{10}(x)$.
    pub log_x: bool,
    /// Whether the dependent variable was transformed via $\log_{10}(y)$.
    pub log_y: bool,
}

impl FitResult {
    /// Evaluates the fitted function at data-space coordinate `x`.
    pub fn eval(&self, x: f64) -> f64 {
        let u = if self.log_x {
            if !x.is_finite() || x <= 0.0 {
                return f64::NAN;
            }
            x.log10()
        } else {
            x
        };
        let mut poly = 0.0;
        let mut u_pow = 1.0;
        for &coeff in &self.coefficients {
            poly += coeff * u_pow;
            u_pow *= u;
        }
        if self.log_y {
            10.0_f64
                .powf(poly.clamp(-300.0, 300.0))
                .clamp(1e-300, 1e300)
        } else {
            poly
        }
    }
}

/// Computes least-squares linear or polynomial regression for a set of data points.
///
/// Returns `None` if fewer than 2 valid points exist or if matrix system is singular.
pub fn fit_dataset(
    points: &[[f64; 2]],
    target_degree: usize,
    dataset_name: &str,
) -> Option<FitResult> {
    fit_dataset_scaled(points, target_degree, dataset_name, false, false)
}

/// Computes least-squares regression in linear, semilog, or log-log coordinates.
///
/// Non-positive points on logarithmic axes are filtered out prior to fitting.
pub fn fit_dataset_scaled(
    points: &[[f64; 2]],
    target_degree: usize,
    dataset_name: &str,
    log_x: bool,
    log_y: bool,
) -> Option<FitResult> {
    let valid_points: Vec<[f64; 2]> = points
        .iter()
        .copied()
        .filter(|p| {
            p[0].is_finite() && p[1].is_finite() && (!log_x || p[0] > 0.0) && (!log_y || p[1] > 0.0)
        })
        .map(|p| {
            [
                if log_x { p[0].log10() } else { p[0] },
                if log_y { p[1].log10() } else { p[1] },
            ]
        })
        .collect();

    let n = valid_points.len();
    if n < 2 {
        return None;
    }

    let degree = target_degree.clamp(1, 4).min(n - 1);
    let m = degree + 1;

    let mut sum_x_pow = vec![0.0; 2 * degree + 1];
    for p in &valid_points {
        let x = p[0];
        let mut cur = 1.0;
        for val in &mut sum_x_pow {
            *val += cur;
            cur *= x;
        }
    }

    let mut mat = vec![vec![0.0; m]; m];
    for (r, mat_row) in mat.iter_mut().enumerate().take(m) {
        mat_row[..m].copy_from_slice(&sum_x_pow[r..(m + r)]);
    }

    let mut rhs = vec![0.0; m];
    for p in &valid_points {
        let x = p[0];
        let y = p[1];
        let mut cur = 1.0;
        for val in &mut rhs {
            *val += y * cur;
            cur *= x;
        }
    }

    let coeffs = solve_linear_system(&mut mat, &mut rhs)?;
    let r_squared = compute_r_squared(&valid_points, &coeffs);
    let formula = format_scaled_polynomial_formula(&coeffs, log_x, log_y);

    Some(FitResult {
        coefficients: coeffs,
        r_squared,
        degree,
        formula,
        dataset_name: dataset_name.to_owned(),
        log_x,
        log_y,
    })
}

fn compute_r_squared(valid_points: &[[f64; 2]], coeffs: &[f64]) -> f64 {
    let n = valid_points.len();
    let y_mean = valid_points.iter().map(|p| p[1]).sum::<f64>() / (n as f64);
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;

    for p in valid_points {
        let x = p[0];
        let y = p[1];

        let mut y_pred = 0.0;
        let mut x_pow = 1.0;
        for &c in coeffs {
            y_pred += c * x_pow;
            x_pow *= x;
        }

        ss_tot += (y - y_mean).powi(2);
        ss_res += (y - y_pred).powi(2);
    }

    if ss_tot < 1e-12 {
        if ss_res < 1e-12 { 1.0 } else { 0.0 }
    } else {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    }
}

/// Solves a linear system $A x = b$ using Gaussian elimination with partial pivoting.
/// Returns `None` if matrix $A$ is singular or ill-conditioned.
#[allow(clippy::needless_range_loop)]
fn solve_linear_system(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = b.len();

    for i in 0..n {
        // Pivot selection
        let mut max_row = i;
        let mut max_val = a[i][i].abs();
        for r in (i + 1)..n {
            let val = a[r][i].abs();
            if val > max_val {
                max_val = val;
                max_row = r;
            }
        }

        if max_val < 1e-12 {
            return None; // Matrix is singular or near-singular
        }

        if max_row != i {
            a.swap(i, max_row);
            b.swap(i, max_row);
        }

        let pivot = a[i][i];
        for c in i..n {
            a[i][c] /= pivot;
        }
        b[i] /= pivot;

        for r in (i + 1)..n {
            let factor = a[r][i];
            if factor.abs() > 1e-15 {
                for c in i..n {
                    a[r][c] -= factor * a[i][c];
                }
                b[r] -= factor * b[i];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut sum = b[r];
        for c in (r + 1)..n {
            sum -= a[r][c] * x[c];
        }
        x[r] = sum;
    }

    Some(x)
}

/// Formats polynomial coefficients into a mathematical formula string.
/// Coefficients order: $[c_0, c_1, \dots, c_d]$ for $f(u) = c_0 + c_1 u + \dots + c_d u^d$.
fn format_scaled_polynomial_formula(coeffs: &[f64], log_x: bool, log_y: bool) -> String {
    let lhs = if log_y { "log10(y)" } else { "y" };
    let var = if log_x { "log10(x)" } else { "x" };
    let degree = coeffs.len().saturating_sub(1);
    if degree == 0 {
        return format!("{lhs} = {:.4}", coeffs.first().copied().unwrap_or(0.0));
    }

    let mut terms = Vec::new();
    let superscripts = ["", "", "²", "³", "⁴"];

    for power in (0..=degree).rev() {
        let coeff = coeffs[power];
        if coeff.abs() < 1e-9 && terms.len() < degree {
            continue;
        }

        let is_first = terms.is_empty();
        let abs_coeff = coeff.abs();

        let sign_str = if is_first {
            if coeff < 0.0 { "-" } else { "" }
        } else if coeff < 0.0 {
            " - "
        } else {
            " + "
        };

        let term_str = match power {
            0 => format!("{sign_str}{abs_coeff:.4}"),
            1 => {
                if (abs_coeff - 1.0).abs() < 1e-6 {
                    format!("{sign_str}{var}")
                } else {
                    format!("{sign_str}{abs_coeff:.4}{var}")
                }
            }
            p => {
                let sup = if p < superscripts.len() {
                    superscripts[p]
                } else {
                    ""
                };
                let base_var = if log_x {
                    format!("({var}){sup}")
                } else {
                    format!("{var}{sup}")
                };
                if (abs_coeff - 1.0).abs() < 1e-6 {
                    format!("{sign_str}{base_var}")
                } else {
                    format!("{sign_str}{abs_coeff:.4}{base_var}")
                }
            }
        };

        terms.push(term_str);
    }

    if terms.is_empty() {
        format!("{lhs} = 0")
    } else {
        format!("{lhs} = {}", terms.join(""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_fit_exact() {
        // y = 2x + 1
        let points = vec![[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]];
        let res = fit_dataset(&points, 1, "TestLine").expect("Fit failed");
        assert_eq!(res.degree, 1);
        assert!((res.coefficients[0] - 1.0).abs() < 1e-6);
        assert!((res.coefficients[1] - 2.0).abs() < 1e-6);
        assert!((res.r_squared - 1.0).abs() < 1e-6);
        assert!(res.formula.contains("2.0000x + 1.0000") || res.formula.contains("2x + 1.0000"));
    }

    #[test]
    fn test_quadratic_fit_exact() {
        // y = x^2 - 3x + 2
        let points = vec![[0.0, 2.0], [1.0, 0.0], [2.0, 0.0], [3.0, 2.0], [4.0, 6.0]];
        let res = fit_dataset(&points, 2, "TestQuad").expect("Fit failed");
        assert_eq!(res.degree, 2);
        assert!((res.coefficients[0] - 2.0).abs() < 1e-6);
        assert!((res.coefficients[1] - (-3.0)).abs() < 1e-6);
        assert!((res.coefficients[2] - 1.0).abs() < 1e-6);
        assert!((res.r_squared - 1.0).abs() < 1e-6);
        assert!(res.formula.contains("x² - 3.0000x + 2.0000"));
    }

    #[test]
    fn test_cubic_and_quartic_fits() {
        // y = x^3 - 2x^2 + x - 5
        let points: Vec<[f64; 2]> = (0..10)
            .map(|i| {
                let x = i as f64 * 0.5;
                let y = x.powi(3) - 2.0 * x.powi(2) + x - 5.0;
                [x, y]
            })
            .collect();

        let res = fit_dataset(&points, 3, "TestCubic").expect("Fit failed");
        assert_eq!(res.degree, 3);
        assert!((res.r_squared - 1.0).abs() < 1e-5);

        let res_q = fit_dataset(&points, 4, "TestQuartic").expect("Fit failed");
        assert_eq!(res_q.degree, 4);
        assert!((res_q.r_squared - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_insufficient_points() {
        let points = vec![[1.0, 2.0]];
        assert!(fit_dataset(&points, 1, "SinglePoint").is_none());
    }

    #[test]
    fn test_fit_type_labels() {
        assert_eq!(FitType::Linear.label(), "Linear (Degree 1)");
        assert_eq!(FitType::Polynomial(2).label(), "Polynomial (Degree 2)");
        assert_eq!(FitType::Polynomial(3).label(), "Polynomial (Degree 3)");
        assert_eq!(FitType::Polynomial(4).label(), "Polynomial (Degree 4)");
        assert_eq!(FitType::Linear.degree(), 1);
        assert_eq!(FitType::Polynomial(2).degree(), 2);
    }

    #[test]
    fn test_constant_y_dataset() {
        let points = vec![[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]];
        let res = fit_dataset(&points, 1, "ConstantY").expect("Fit failed");
        assert!((res.eval(0.5) - 5.0).abs() < 1e-6);
        assert_eq!(res.r_squared, 1.0);
    }

    #[test]
    fn test_benchmark_10k_points_under_16ms() {
        use std::time::Instant;

        let points: Vec<[f64; 2]> = (0..10_000)
            .map(|i| {
                let x = i as f64 * 0.001;
                let y = 0.5 * x * x - 2.1 * x + 3.4;
                [x, y]
            })
            .collect();

        let start = Instant::now();
        let res = fit_dataset(&points, 4, "Bench10k").expect("Fit failed");
        let elapsed = start.elapsed();

        assert!((res.r_squared - 1.0).abs() < 1e-4);

        #[cfg(not(debug_assertions))]
        let max_ms = 16;
        #[cfg(debug_assertions)]
        let max_ms = 100;

        assert!(
            elapsed.as_millis() < max_ms,
            "Fit took {:?} which exceeds {}ms",
            elapsed,
            max_ms
        );
    }

    #[test]
    fn test_plot_ui_rendering_with_fit() {
        egui::__run_test_ui(|ui| {
            let pts = vec![[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]];
            let points = crate::Points::new("Sample Data", crate::PlotPoints::new(pts));

            crate::Plot::new("fit_test_plot")
                .view_aspect(2.0)
                .show(ui, |plot_ui| {
                    plot_ui.points(points);
                });
        });

        // Generate screenshot asset for PR verification
        let sample_pts: Vec<[f64; 2]> = (0..50)
            .map(|i| {
                let x = i as f64 * 0.2;
                let y = 0.8 * x * x - 1.5 * x + 2.0;
                [x, y]
            })
            .collect();
        let fit_res = fit_dataset(&sample_pts, 2, "Biological Growth Points").unwrap();
        let curve_pts: Vec<[f64; 2]> = (0..200)
            .map(|i| {
                let x = i as f64 * 0.05;
                [x, fit_res.eval(x)]
            })
            .collect();

        let datasets = vec![
            ("Discrete Sample Points".to_string(), sample_pts),
            (format!("Fitted Overlay: {}", fit_res.formula), curve_pts),
        ];

        let options = crate::export::ExportOptions {
            width: 800,
            height: 450,
            scale: 1.0,
            line_thickness: 2.5,
            background_opacity: 1.0,
            background_color: egui::Color32::from_rgb(20, 24, 30),
            aspect_ratio: crate::export::AspectRatioOption::SixteenNine,
            ..Default::default()
        };

        if let Ok(png_bytes) = crate::export::export_custom_png(&datasets, &options) {
            let _ = std::fs::write("/tmp/regression_fit_overlay.png", png_bytes);
        }
    }

    #[test]
    fn test_fit_dataset_scaled_power_law_log_log() {
        let pts = vec![
            [0.0, 0.0],
            [-5.0, 10.0],
            [1.0, 2.5],
            [2.0, 20.0],
            [4.0, 160.0],
            [10.0, 2500.0],
            [100.0, 2_500_000.0],
        ];
        let res =
            super::fit_dataset_scaled(&pts, 1, "PowerLaw", true, true).expect("Log-log fit failed");
        assert!(res.log_x && res.log_y);
        assert!((res.r_squared - 1.0).abs() < 1e-9);
        assert!(res.formula.contains("log10(y)") && res.formula.contains("log10(x)"));
        assert!((res.eval(4.0) - 160.0).abs() < 1e-6);
        assert!((res.eval(10.0) - 2500.0).abs() < 1e-5);
        assert!(res.eval(0.0).is_nan());
    }

    #[test]
    fn test_fit_dataset_scaled_exponential_semi_log() {
        let pts = vec![
            [0.0, 10.0],
            [2.0, 1.0],
            [4.0, 0.1],
            [6.0, 0.01],
            [8.0, 0.001],
        ];
        let res = super::fit_dataset_scaled(&pts, 1, "ExpDecay", false, true)
            .expect("Semi-log fit failed");
        assert!(!res.log_x && res.log_y);
        assert!((res.r_squared - 1.0).abs() < 1e-9);
        assert!(res.formula.contains("log10(y)"));
        assert!((res.eval(2.0) - 1.0).abs() < 1e-6);
        assert!((res.eval(6.0) - 0.01).abs() < 1e-8);
    }
}
