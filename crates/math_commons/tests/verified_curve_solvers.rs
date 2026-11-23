//! Integration test verifying zero heap allocations across promoted `math_commons`
//! curve-fitting and critical-point solvers under `VerifiedAllocator::lock_allocations()`.

use math_commons::error::FitError;
use math_commons::fit::{
    FitModel, fit_model, fit_polynomial, fit_polynomial_scaled, levenberg_marquardt_numeric,
    sum_squared_errors_numeric,
};
use math_commons::poi::{
    LocalExtremum, closest_threshold_crossing, detect_local_extrema_into, intersect_polylines_into,
    parabolic_vertex, peak_extremum, threshold_crossings_into, y_at_x_into,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_curve_fitting_and_poi_solvers_zero_heap_allocations() {
    let mut xs = [0.0_f64; 64];
    let mut ys_sin = [0.0_f64; 64];
    let mut poly_pts = [[0.0_f64; 2]; 32];
    let mut exp_pts = [(0.0_f64, 0.0_f64); 24];
    let mut logistic_pts = [[0.0_f64; 2]; 30];

    for i in 0..64 {
        let x = 0.1 + (i as f64) * 0.12;
        xs[i] = x;
        ys_sin[i] = 2.4 * (1.3 * x + 0.25).sin() + 0.5;
    }
    for (i, slot) in poly_pts.iter_mut().enumerate() {
        let x = 0.2 + (i as f64) * 0.2;
        *slot = [
            x,
            1.0 - 0.5 * x + 0.3 * x * x - 0.04 * x.powi(3) + 0.005 * x.powi(4),
        ];
    }
    for (i, slot) in exp_pts.iter_mut().enumerate() {
        let x = 0.1 + (i as f64) * 0.15;
        *slot = (x, 3.0 * (-0.35 * x).exp());
    }
    for (i, slot) in logistic_pts.iter_mut().enumerate() {
        let x = (i as f64) * 0.4;
        *slot = [x, 10.0 / (1.0 + (-0.9 * (x - 5.0)).exp())];
    }

    VerifiedEngine::run_verified(|| {
        verify_zero_alloc_fits(&xs, &ys_sin, &poly_pts, &exp_pts, &logistic_pts);
        verify_zero_alloc_poi(&xs, &ys_sin, &poly_pts, &logistic_pts);
    });
}

fn verify_zero_alloc_fits(
    xs: &[f64],
    ys_sin: &[f64],
    poly_pts: &[[f64; 2]],
    exp_pts: &[(f64, f64)],
    logistic_pts: &[[f64; 2]],
) {
    for deg in 1..=4 {
        let fit = fit_polynomial(poly_pts, deg).expect("poly fit");
        assert!(fit.eval(1.5).is_finite());
    }
    let mut shifted_quartic = [[0.0_f64; 2]; 21];
    for (i, slot) in shifted_quartic.iter_mut().enumerate() {
        let u = -1.0 + (i as f64) * 0.1;
        let x = 1_000_000.0 + u;
        *slot = [
            x,
            1.5 - 0.8 * u + 2.2 * u * u - 0.35 * u.powi(3) + 0.12 * u.powi(4),
        ];
    }
    let sq_fit = fit_polynomial(&shifted_quartic, 4).expect("centered quartic");
    assert!((sq_fit.eval(1_000_000.0) - 1.5).abs() < 1e-6);
    let log_poly = fit_polynomial_scaled(exp_pts, 1, false, true).expect("semilog poly");
    assert!(log_poly.r_squared > 0.999);
    assert_eq!(
        fit_polynomial(poly_pts, 5),
        Err(FitError::UnsupportedDegree { degree: 5 })
    );

    let exp_fit = fit_model(exp_pts, FitModel::Exponential, false).expect("exp fit");
    assert!((exp_fit.coefficients()[0] - 3.0).abs() < 1e-5);
    let pow_fit = fit_model(exp_pts, FitModel::PowerLaw, true).expect("power log-mode");
    assert!(pow_fit.eval(1.0).is_finite());
    let log_fit = fit_model(exp_pts, FitModel::Logarithmic, false).expect("logarithmic");
    assert!(log_fit.eval(1.0).is_finite());
    let l_fit = fit_model(logistic_pts, FitModel::Logistic, false).expect("logistic");
    assert!((l_fit.coefficients()[0] - 10.0).abs() < 1e-4);

    let paired = (xs, ys_sin);
    let s_fit = fit_model(&paired, FitModel::Sinusoidal, false).expect("sinusoidal");
    let mut res_buf = [[0.0_f64; 2]; 16];
    let n_res = s_fit.residuals_into(&paired, &mut res_buf);
    assert_eq!(n_res, 16);

    let mut wave6 = [[0.0_f64; 2]; 41];
    for (i, slot) in wave6.iter_mut().enumerate() {
        let x = (i as f64) * 0.2;
        let y = 0.8 + 0.25 * x + 2.1 * (-0.5 * x).exp() + 0.7 * (1.4 * x).cos();
        *slot = [x, y];
    }
    let eval_fn = |x: f64, p: &[f64]| -> Option<f64> {
        let y = p[0]
            + p[1] * x
            + p[2] * (-p[3] * x).clamp(-700.0, 700.0).exp()
            + p[4] * (p[5] * x).cos();
        y.is_finite().then_some(y)
    };
    let seeds = [0.5, 0.2, 1.8, 0.4, 0.5, 1.3];
    let p6 = levenberg_marquardt_numeric(&wave6, &seeds, eval_fn).expect("6-param zero-alloc");
    let sse6 = sum_squared_errors_numeric(&wave6, &p6, eval_fn).expect("sse6");
    assert!(sse6 < 1e-8);
}

fn verify_zero_alloc_poi(
    xs: &[f64],
    ys_sin: &[f64],
    poly_pts: &[[f64; 2]],
    logistic_pts: &[[f64; 2]],
) {
    let paired = (xs, ys_sin);
    let mut x_hits = [0.0_f64; 16];
    let n_cross = threshold_crossings_into(&paired, 0.5, &mut x_hits);
    assert!(n_cross >= 2);
    assert!(closest_threshold_crossing(&paired, 0.5, 2.5).is_some());

    let mut y_hits = [0.0_f64; 4];
    assert_eq!(y_at_x_into(&paired, 2.0, &mut y_hits), 1);

    let mut extrema = [LocalExtremum {
        point: [0.0, 0.0],
        is_max: true,
    }; 16];
    let n_ext = detect_local_extrema_into(&paired, &mut extrema);
    assert!(n_ext >= 2);
    assert!(peak_extremum(&paired).is_some());

    let v = parabolic_vertex([0.0, 0.0], [1.0, 2.0], [2.0, 0.0]);
    assert!((v[0] - 1.0).abs() < 1e-12);

    let mut intersections = [[0.0_f64; 2]; 16];
    let n_int = intersect_polylines_into(poly_pts, logistic_pts, [0.0, 10.0], &mut intersections);
    assert!(n_int >= 1);
}
