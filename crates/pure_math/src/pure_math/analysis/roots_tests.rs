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
fn test_newton_at_3e9_and_interval_newton_krawczyk_sqrt2_enclosures() -> Result<(), AnalysisError> {
    use math_scalars::{Dual, Interval, Scalar};

    let target = 3.0e9_f64;
    let solver = NewtonRaphson::new(64, 1e-12);
    let ((r_fd, r_ad, iv_newt, iv_kraw, ball_newt, ball_kraw), _metrics) =
        VerifiedEngine::run_verified(|| {
            let fd = solver.find_root(|x| (x - target) * (x + 1.0e8), 2.5e9, 3.8e9);
            let ad = solver.find_root_ad(
                |x: Dual<f64>| (x - Dual::from_f64(target)) * (x + Dual::from_f64(1.0e8)),
                3.15e9,
            );

            let bracket = Interval::new(1.0_f64, 2.0_f64);
            let f_iv = |x: Dual<Interval<f64>>| x * x - Dual::constant(Interval::point(2.0_f64));
            let inewt = IntervalNewton::default().enclose_root(f_iv, bracket);
            let kraw = Krawczyk::default().enclose_root(f_iv, bracket);
            let bnewt = IntervalNewton::default().enclose_root_ball(f_iv, bracket);
            let bkraw = Krawczyk::default().enclose_root_ball(f_iv, bracket);
            (fd, ad, inewt, kraw, bnewt, bkraw)
        });

    let root_fd = r_fd?;
    let root_ad = r_ad?;
    assert!(
        (root_fd - target).abs() / target < 1e-13,
        "Newton FD failed at 3.0e9: got {root_fd}"
    );
    assert!(
        (root_ad - target).abs() / target < 1e-15,
        "Newton AD failed at 3.0e9: got {root_ad}"
    );

    let enc_n = iv_newt?;
    let enc_k = iv_kraw?;
    let ball_n = ball_newt?;
    let ball_k = ball_kraw?;
    let sqrt2 = std::f64::consts::SQRT_2;

    assert!(enc_n.contains(sqrt2) && (enc_n * enc_n).contains(2.0));
    assert!(enc_k.contains(sqrt2) && (enc_k * enc_k).contains(2.0));
    assert!(
        enc_n.width_ulps() <= 4,
        "IntervalNewton width {} > 4 ULPs",
        enc_n.width_ulps()
    );
    assert!(
        enc_k.width_ulps() <= 4,
        "Krawczyk width {} > 4 ULPs",
        enc_k.width_ulps()
    );
    assert!(ball_n.contains(sqrt2) && ball_n.to_interval().width_ulps() <= 4);
    assert!(ball_k.contains(sqrt2) && ball_k.to_interval().width_ulps() <= 4);
    Ok(())
}
