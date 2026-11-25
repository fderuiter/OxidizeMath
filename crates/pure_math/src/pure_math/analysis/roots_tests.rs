use super::*;
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_newton_raphson_sqrt() -> Result<(), AnalysisError> {
    let solver = NewtonRaphson::default();
    let root = solver.find_root_with_derivative(|x| x * x - 2.0, |x| 2.0 * x, 1.5)?;
    assert!((root - std::f64::consts::SQRT_2).abs() < math_commons::registry::TOLERANCE_FAST);
    Ok(())
}

#[test]
#[verified_engine::verified]
fn test_newton_raphson_numerical() -> Result<(), AnalysisError> {
    let solver = NewtonRaphson::default();
    let root = solver.find_root(|x| x * x - 2.0, 1.0, 2.0)?;
    assert!((root - std::f64::consts::SQRT_2).abs() < 1e-5);
    Ok(())
}

#[test]
#[verified_engine::verified]
fn test_bisection_square_root() -> Result<(), AnalysisError> {
    let solver = Bisection::default();
    let root = solver.find_root(|x| x * x - 2.0, 1.0, 2.0)?;
    assert!((root - std::f64::consts::SQRT_2).abs() < math_commons::registry::TOLERANCE_FAST);
    Ok(())
}

#[test]
#[verified_engine::verified]
fn test_bisection_linear() -> Result<(), AnalysisError> {
    let solver = Bisection::default();
    let root = solver.find_root(|x| 2.0 * x - 4.0, 0.0, 5.0)?;
    assert!((root - 2.0).abs() < math_commons::registry::TOLERANCE_FAST);
    Ok(())
}

#[test]
#[verified_engine::verified]
fn test_not_bracketed() {
    let solver = Bisection::default();
    let result = solver.find_root(|x| x * x + 1.0, -2.0, 2.0);
    assert!(matches!(
        result,
        Err(AnalysisError::RootNotBracketed { .. })
    ));

    let brent = Brent::default();
    let brent_res = brent.find_root(|x| x * x + 1.0, -2.0, 2.0);
    assert!(matches!(
        brent_res,
        Err(AnalysisError::RootNotBracketed { .. })
    ));
}

#[test]
#[verified_engine::verified]
fn test_brent_convergence_and_fallbacks() -> Result<(), AnalysisError> {
    let solver = Brent::default();

    let sqrt2 = solver.find_root(|x| x * x - 2.0, 1.0, 2.0)?;
    assert!((sqrt2 - std::f64::consts::SQRT_2).abs() < math_commons::registry::TOLERANCE_STANDARD);

    let dottie = solver.find_root(|x| x.cos() - x, 0.0, 1.0)?;
    assert!((dottie - 0.739_085_133_215_160_7).abs() < math_commons::registry::TOLERANCE_STANDARD);

    // Reversed bounds and flat triple root (forces bisection fallback)
    let reversed = solver.find_root(|x| x * x - 2.0, 2.0, 1.0)?;
    assert!(
        (reversed - std::f64::consts::SQRT_2).abs() < math_commons::registry::TOLERANCE_STANDARD
    );

    let cubic = solver.find_root(|x| (x - 0.3).powi(3), -1.0, 2.0)?;
    assert!((cubic - 0.3).abs() < math_commons::registry::TOLERANCE_STANDARD);

    // Exact endpoint hit
    let exact_min = solver.find_root(|x| x - 1.0, 1.0, 3.0)?;
    assert!((exact_min - 1.0).abs() < math_commons::registry::TOLERANCE_HIGH);
    Ok(())
}

#[test]
#[verified_engine::verified]
fn test_all_root_finders_zero_allocation_on_convergence_and_errors() {
    let ((r_bis, r_newt, r_brent, err_bis, err_newt, err_brent, err_bounds), _) =
        VerifiedEngine::run_verified(|| {
            let bis = Bisection::default();
            let newt = NewtonRaphson::default();
            let brent = Brent::new(200, 1e-11);

            let ok_bis = bis.find_root(|x| x * x - 2.0, 1.0, 2.0);
            let ok_newt = newt.find_root_with_derivative(|x| x * x - 2.0, |x| 2.0 * x, 1.5);
            let ok_brent = brent.find_root(|x| x.exp() - 3.0, 0.0, 2.0);

            let bad_bis = bis.find_root(|x| x * x + 1.0, -1.0, 1.0);
            let bad_newt = newt.find_root_with_derivative(|x| x * x + 1.0, |_| 0.0, 0.0);
            let bad_brent = brent.find_root(|x| x * x + 1.0, -1.0, 1.0);
            let bad_bounds = brent.find_root(|x| x, f64::NAN, 1.0);

            (
                ok_bis, ok_newt, ok_brent, bad_bis, bad_newt, bad_brent, bad_bounds,
            )
        });

    assert!((r_bis.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
    assert!((r_newt.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
    assert!((r_brent.unwrap() - 3.0_f64.ln()).abs() < math_commons::registry::TOLERANCE_STANDARD);
    assert!(matches!(
        err_bis,
        Err(AnalysisError::RootNotBracketed { .. })
    ));
    assert!(matches!(
        err_newt,
        Err(AnalysisError::DerivativeTooSmall { .. })
    ));
    assert!(matches!(
        err_brent,
        Err(AnalysisError::RootNotBracketed { .. })
    ));
    assert!(matches!(
        err_bounds,
        Err(AnalysisError::InvalidBracket { .. })
    ));
}

#[test]
#[verified_engine::verified]
fn test_newton_raphson_ad_and_scaled_converge_at_3e9_and_interval_newton_4_ulps() {
    use math_scalars::{Dual, Interval, Scalar};

    let ((r_ad, r_fd, iv_sqrt2, kr_sqrt2, kr_dottie), _) = VerifiedEngine::run_verified(|| {
        let newt = NewtonRaphson::new(64, 1e-12);
        let target = 3.0e9_f64;
        let r_ad = newt
            .find_root_ad(
                |x: Dual<f64>| (x - Dual::constant(target)) * (x + Dual::constant(2.0)),
                3.4e9,
            )
            .expect("AD Newton at 3.0e9");
        let r_fd = newt
            .find_root(|x| (x - target) * (x + 2.0), 2.5e9, 4.3e9)
            .expect("Scaled FD Newton at 3.0e9");

        let inewt = IntervalNewton::default();
        let iv_sqrt2 = inewt
            .enclose_root_ad(
                |x: Dual<Interval<f64>>| x * x - Dual::constant(Interval::point(2.0)),
                1.0,
                2.0,
            )
            .expect("IntervalNewton sqrt(2)");

        let kraw = Krawczyk::default();
        let kr_sqrt2 = kraw
            .enclose_root_ad(
                |x: Dual<Interval<f64>>| x * x - Dual::constant(Interval::point(2.0)),
                1.0,
                2.0,
            )
            .expect("Krawczyk sqrt(2)");
        let kr_dottie = kraw
            .enclose_root_ad(|x: Dual<Interval<f64>>| x.cos() - x, 0.0, 1.0)
            .expect("Krawczyk Dottie");

        (r_ad, r_fd, iv_sqrt2, kr_sqrt2, kr_dottie)
    });

    assert!(((r_ad - 3.0e9) / 3.0e9).abs() < 1e-14, "r_ad = {r_ad}");
    assert!(((r_fd - 3.0e9) / 3.0e9).abs() < 1e-12, "r_fd = {r_fd}");

    let exact_sqrt2 = std::f64::consts::SQRT_2;
    assert!(iv_sqrt2.contains(exact_sqrt2));
    assert!(
        iv_sqrt2.ulp_width() <= 4,
        "IntervalNewton sqrt(2) ulp_width = {}",
        iv_sqrt2.ulp_width()
    );

    assert!(kr_sqrt2.contains(exact_sqrt2));
    assert!(
        kr_sqrt2.ulp_width() <= 4,
        "Krawczyk sqrt(2) ulp_width = {}",
        kr_sqrt2.ulp_width()
    );

    let exact_dottie = 0.739_085_133_215_160_7_f64;
    assert!(kr_dottie.contains(exact_dottie));
    assert!(
        kr_dottie.ulp_width() <= 4,
        "Krawczyk dottie ulp_width = {}",
        kr_dottie.ulp_width()
    );
}
