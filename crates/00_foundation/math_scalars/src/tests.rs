//! Unit tests for `math_scalars` executed on both host and `wasm32-unknown-unknown` `--lib` targets.

use crate::{
    Ball, Dual, DualVec, HyperDual, Interval, Scalar, certify_reference_roots,
    interval_newton_enclose, krawczyk_enclose,
};
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_lib_interval_newton_sqrt2_and_reference_roots_within_4_ulps() {
    let ((iv_sqrt2, kr_dottie, summary), _) = VerifiedEngine::run_verified(|| {
        let iv_sqrt2 = interval_newton_enclose(
            |x: Dual<Interval<f64>>| x * x - Dual::constant(Interval::point(2.0)),
            Interval::new(1.0, 2.0),
            32,
        )
        .expect("IntervalNewton sqrt(2) enclosure");
        let kr_dottie = krawczyk_enclose(
            |x: Dual<Interval<f64>>| x.cos() - x,
            Ball::new(0.5, 0.5),
            32,
        )
        .expect("Krawczyk Dottie enclosure");
        let summary = certify_reference_roots();
        (iv_sqrt2, kr_dottie, summary)
    });

    assert!(iv_sqrt2.certified_unique);
    assert!(iv_sqrt2.enclosure.contains(core::f64::consts::SQRT_2));
    assert!(
        iv_sqrt2.enclosure.ulp_width() <= 4,
        "ulp_width = {}",
        iv_sqrt2.enclosure.ulp_width()
    );

    assert!(kr_dottie.certified_unique);
    assert!(kr_dottie.enclosure.contains(0.739_085_133_215_160_7));
    assert!(
        kr_dottie.enclosure.ulp_width() <= 4,
        "ulp_width = {}",
        kr_dottie.enclosure.ulp_width()
    );

    assert!(summary.all_contained);
    assert!(summary.max_interval_ulp_width <= 4);
    assert!(summary.max_ball_ulp_width <= 4);
    assert_eq!(Interval::<f64>::PI.ulp_width(), 1);
    assert_eq!(Interval::<f32>::PI.ulp_width(), 1);
    assert!(Ball::<f64>::TAU.ulp_width() <= 2);
    assert!(Ball::<f32>::TAU.ulp_width() <= 2);
}

#[test]
fn test_lib_forward_ad_and_ball_enclosure_zero_alloc() {
    let ((d_val, d_eps, dv_eps, hd_eps12, ball_sin), _) = VerifiedEngine::run_verified(|| {
        let d = (Dual::variable(0.7_f64).sin() * Dual::variable(0.7_f64).exp() + Dual::ONE).ln();
        let seed = DualVec::<f64, 3>::jacobian_seed([1.0, 2.0, -0.5]);
        let dv = (seed[0] * seed[1] + seed[2].exp()).cos();
        let u = HyperDual::bivariate_u(0.4_f64);
        let v = HyperDual::bivariate_v(1.1_f64);
        let hd = (u * v).sin();
        let b = Ball::new(0.5_f64, 1e-12).sin();
        (d.val, d.eps, dv.eps, hd.eps12, b)
    });

    assert!(d_val.is_finite() && d_eps.is_finite());
    assert!(dv_eps.iter().all(|g| g.is_finite()));
    let expected_mixed = (0.4_f64 * 1.1_f64).cos() - 0.4_f64 * 1.1_f64 * (0.4_f64 * 1.1_f64).sin();
    assert!((hd_eps12 - expected_mixed).abs() < 1e-15);
    assert!(ball_sin.contains(0.5_f64.sin()));
}
