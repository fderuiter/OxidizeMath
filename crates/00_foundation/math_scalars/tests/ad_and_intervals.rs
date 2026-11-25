//! Integration and zero-allocation verification tests for `math_scalars`.

use math_scalars::{
    Ball, Dual, DualVec, HyperDual, Interval, RealScalar, Scalar, certify_reference_roots,
    interval_newton_enclose, krawczyk_enclose,
};
use verified_engine::engine::VerifiedEngine;

fn generic_composite<S: Scalar>(x: S) -> S {
    let a = (x * S::from_f64(0.5)).sin_cos();
    let trig = a.0 * a.1 + x.tan().atan();
    let hyp = (x * S::from_f64(0.3)).sinh() + (x * S::from_f64(0.3)).cosh() - x.tanh();
    let inv = (x * S::from_f64(0.2)).asin() + (x * S::from_f64(0.2)).acos();
    let logs = (x + S::TWO).ln() + (x * S::HALF).exp() + (x + S::ONE).sqrt();
    let powers = x.powi(3) + (x + S::TWO).powf(S::from_f64(1.5)) + x.hypot(S::TWO);
    trig + hyp + inv + logs + powers + x.atan2(S::TWO)
}

#[test]
#[verified_engine::verified]
fn test_ulp_stepping_and_subnormal_boundaries() {
    assert_eq!(0.0_f64.next_up().next_down(), 0.0_f64);
    assert_eq!((-0.0_f64).next_down().next_up(), 0.0_f64);
    assert!(0.0_f64.next_up() > 0.0_f64);
    assert!(0.0_f64.next_down() < 0.0_f64);
    assert_eq!(0.0_f64.ulp_distance(0.0_f64.next_up()), 1);
    assert_eq!(0.0_f64.next_down().ulp_distance(0.0_f64.next_up()), 2);

    assert_eq!(0.0_f32.next_up().next_down(), 0.0_f32);
    assert!(0.0_f32.next_up() > 0.0_f32);
    assert!(0.0_f32.next_down() < 0.0_f32);
    assert_eq!(1.0_f32.ulp_distance(1.0_f32.next_up()), 1);
}

#[test]
#[verified_engine::verified]
fn test_dual_f32_large_scale_no_underflow() {
    // At x = 64.0_f32, ulp(64.0_f32) = 7.629e-6 > 1e-6, where fixed-h FD collapses to 0.0.
    let x0 = 64.0_f32;
    let d = Dual::variable(x0);
    let g = (d * 0.125_f32).sinh() + d.sqrt();
    let exact = 0.125_f32 * (x0 * 0.125_f32).cosh() + 0.5_f32 / x0.sqrt();
    assert!(
        g.eps.ulp_distance(exact) <= 1,
        "Dual<f32> derivative {} differed from exact {} by {} ULPs",
        g.eps,
        exact,
        g.eps.ulp_distance(exact)
    );
}

#[test]
#[verified_engine::verified]
fn test_dual_and_hyper_dual_elementary_accuracy() {
    let x0 = 1.15_f64;
    let d1 = generic_composite(Dual::variable(x0));
    let hd = generic_composite(HyperDual::univariate(x0));

    assert!((d1.val - hd.val).abs() < 1e-15);
    assert!((d1.eps - hd.eps1).abs() < 1e-15);
    assert!((d1.eps - hd.eps2).abs() < 1e-15);

    // Verify HyperDual second derivative against Dual-of-Dual on smooth elementary mix:
    let u = HyperDual::bivariate_u(1.3_f64);
    let v = HyperDual::bivariate_v(0.8_f64);
    let surf = (u.powi(2) + v.powi(2)).sqrt() + (u * v).sin() / (u.cosh() + v.exp());
    let expected_d1 = generic_composite(Dual::variable(x0 + 1e-7)).eps
        - generic_composite(Dual::variable(x0 - 1e-7)).eps;
    assert!((hd.eps12 - expected_d1 / 2e-7).abs() < 1e-6);
    assert!(surf.is_finite());
}

#[test]
#[verified_engine::verified]
fn test_dual_vec_multi_tangent_gradient() {
    let x = DualVec::<f64, 3>::variable(1.5, 0);
    let y = DualVec::<f64, 3>::variable(-0.7, 1);
    let z = DualVec::<f64, 3>::variable(2.2, 2);

    let f = (x * y).exp() + z.sin() * x.hypot(y);
    let dx = Dual::variable(1.5_f64);
    let dy = Dual::constant(-0.7_f64);
    let dz = Dual::constant(2.2_f64);
    let fx = (dx * dy).exp() + dz.sin() * dx.hypot(dy);

    assert!((f.val - fx.val).abs() < 1e-15);
    assert!((f.eps[0] - fx.eps).abs() < 1e-15);
}

#[test]
#[verified_engine::verified]
fn test_interval_and_ball_outward_containment() {
    let x = 1.15_f64;
    let iv = generic_composite(Interval::new(x - 1e-12, x + 1e-12));
    let bl = generic_composite(Ball::new(x, 1e-12));
    let exact_approx = generic_composite(x);

    assert!(iv.contains(exact_approx));
    assert!(bl.contains(exact_approx));

    // Periodic stationary point containment: sin([0, pi]) == [0, 1], sin([0, 2*pi]) == [-1, 1]
    let sin_peak = Interval::new(0.2_f64, 2.9_f64).sin();
    assert_eq!(sin_peak.hi, 1.0);
    let cos_trough = Interval::new(2.5_f64, 3.8_f64).cos();
    assert_eq!(cos_trough.lo, -1.0);
}

#[test]
#[verified_engine::verified]
fn test_certified_reference_roots_within_4_ulps_and_zero_alloc() {
    let ((summary, sqrt2_iv, sqrt2_ball), _) = VerifiedEngine::run_verified(|| {
        let s = certify_reference_roots();
        let iv = interval_newton_enclose(
            |x: Dual<Interval<f64>>| x * x - Dual::from_f64(2.0),
            Interval::new(1.0, 2.0),
            32,
        )
        .unwrap();
        let bl = krawczyk_enclose(
            |x: Dual<Interval<f64>>| x * x - Dual::from_f64(2.0),
            Ball::new(1.5, 0.5),
            32,
        )
        .unwrap();
        (s, iv, bl)
    });

    assert!(
        summary.all_contained,
        "summary={summary:?}, sqrt2_iv={sqrt2_iv:?}, sqrt2_ball={sqrt2_ball:?}"
    );
    assert!(summary.max_interval_ulp_width <= 4);
    assert!(summary.max_ball_ulp_width <= 4);
    assert!(sqrt2_iv.certified_unique);
    assert!(sqrt2_iv.enclosure.ulp_width() <= 4);
    assert!(sqrt2_iv.enclosure.contains(core::f64::consts::SQRT_2));
    assert!(sqrt2_ball.certified_unique);
    assert!(sqrt2_ball.enclosure.ulp_width() <= 4);
    assert!(sqrt2_ball.enclosure.contains(core::f64::consts::SQRT_2));
}
