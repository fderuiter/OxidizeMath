// [cite:math_scalars]
//! Comprehensive unit tests for forward-mode AD, IEEE-754 directed rounding, `Interval`, and `Ball`.

use verified_engine::engine::VerifiedEngine;

use crate::{
    Ball, Dual, DualVec, HyperDual, Interval, RealScalar, Scalar, next_down_f32, next_down_f64,
    next_up_f32, next_up_f64, ulp_distance_f32, ulp_distance_f64,
};

fn composite_wave<S: Scalar>(x: S, y: S) -> S {
    (x * y).sin() + (x.sqr() + S::ONE).ln() + (y * S::HALF).exp() / (x.cosh() + S::TWO)
}

#[test]
fn test_ieee754_next_up_down_and_subnormals() {
    let sub_f64 = f64::from_bits(1);
    assert_eq!(next_up_f64(0.0), sub_f64);
    assert_eq!(next_up_f64(-0.0), sub_f64);
    assert_eq!(next_down_f64(0.0), -sub_f64);
    assert_eq!(next_down_f64(-0.0), -sub_f64);
    assert_eq!(next_up_f64(-sub_f64), 0.0);
    assert_eq!(next_up_f64(f64::MAX), f64::INFINITY);
    assert_eq!(next_down_f64(f64::INFINITY), f64::MAX);
    assert_eq!(next_up_f64(f64::NEG_INFINITY), f64::MIN);
    assert!(next_up_f64(f64::NAN).is_nan());
    assert_eq!(ulp_distance_f64(-0.0, 0.0), 0);
    assert_eq!(ulp_distance_f64(-sub_f64, sub_f64), 2);

    let sub_f32 = f32::from_bits(1);
    assert_eq!(next_up_f32(-0.0), sub_f32);
    assert_eq!(next_down_f32(0.0), -sub_f32);
    assert_eq!(ulp_distance_f32(1.0_f32, next_up_f32(1.0_f32)), 1);
}

#[test]
fn test_dual_and_dual_vec_zero_allocation_derivatives() {
    let ((d_x, d_y, dv), _metrics) = VerifiedEngine::run_verified(|| {
        let x0 = 0.7_f64;
        let y0 = -0.4_f64;
        let fx = composite_wave(Dual::variable(x0), Dual::constant(y0));
        let fy = composite_wave(Dual::constant(x0), Dual::variable(y0));
        let fxy = composite_wave(DualVec::<f64, 2>::variable(x0, 0), DualVec::variable(y0, 1));
        (fx, fy, fxy)
    });

    let (x, y) = (0.7_f64, -0.4_f64);
    let exact_dx = y * (x * y).cos() + 2.0 * x / (x * x + 1.0)
        - (0.5 * y).exp() * x.sinh() / (x.cosh() + 2.0).powi(2);
    let exact_dy = x * (x * y).cos() + 0.5 * (0.5 * y).exp() / (x.cosh() + 2.0);

    assert!((d_x.eps - exact_dx).abs() < 1e-15);
    assert!((d_y.eps - exact_dy).abs() < 1e-15);
    assert!((dv.eps[0] - exact_dx).abs() < 1e-15);
    assert!((dv.eps[1] - exact_dy).abs() < 1e-15);
}

#[test]
fn test_hyper_dual_pure_and_mixed_partials_without_cancellation() {
    let (u, v) = (1.1_f64, 0.6_f64);
    // Pure second derivative d^2/du^2 of exp(sin(u)) at tiny & regular scales
    let hd_pure = HyperDual::second_order(u).sin().exp();
    let s = u.sin();
    let c = u.cos();
    let exact_d2 = s.exp() * (c * c - s);
    assert!((hd_pure.eps12 - exact_d2).abs() < 1e-15);

    // Mixed partial d^2/dudv of atan2(v, u) + hypot(u, v)
    let hu = HyperDual::var1(u);
    let hv = HyperDual::var2(v);
    let f_uv = hv.atan2(hu) + hu.hypot(hv);
    let f_vu =
        HyperDual::var1(v).atan2(HyperDual::var2(u)) + HyperDual::var2(u).hypot(HyperDual::var1(v));
    assert_eq!(f_uv.eps12, f_vu.eps12);

    let r2 = u * u + v * v;
    let exact_atan2_uv = (v * v - u * u) / (r2 * r2);
    let exact_hypot_uv = -(u * v) / r2.powf(1.5);
    assert!((f_uv.eps12 - (exact_atan2_uv + exact_hypot_uv)).abs() < 1e-15);
}

#[test]
fn test_dual_f32_large_coordinate_no_underflow() {
    let x = 64.0_f32;
    let g = (S::ONE + Dual::variable(x).sqr()).sqrt();
    type S = Dual<f32>;
    let exact = x / (1.0_f32 + x * x).sqrt();
    assert!(g.eps.ulp_distance(exact) <= 1);
}

#[test]
fn test_interval_outward_arithmetic_and_non_monotone_trig() {
    let one = Interval::point(1.0_f64);
    let three = Interval::point(3.0_f64);
    let third = one / three;
    assert!(third.lo < 1.0_f64 / 3.0_f64 || third.hi > 1.0_f64 / 3.0_f64);
    assert_eq!(third.width_ulps(), 1);
    assert!((third * three).contains(1.0));

    let straddle_zero = Interval::new(-2.0_f64, 3.0_f64);
    assert_eq!(straddle_zero.sqr(), Interval::new(0.0, 9.0));
    assert_eq!(straddle_zero.powi(2), Interval::new(0.0, 9.0));
    assert_eq!(straddle_zero.powi(3), Interval::new(-8.0, 27.0));

    // Straddling pi/2 must clamp sin upper bound to +1.0
    let around_peak = Interval::new(1.2_f64, 2.0_f64);
    let sin_peak = around_peak.sin();
    assert_eq!(sin_peak.hi, 1.0);
    assert!(sin_peak.contains(1.2_f64.sin()));

    // Straddling pi must clamp cos lower bound to -1.0
    let around_pi = Interval::new(3.0_f64, 3.3_f64);
    let cos_trough = around_pi.cos();
    assert_eq!(cos_trough.lo, -1.0);
    assert!(cos_trough.contains(3.0_f64.cos()));
}

#[test]
fn test_interval_newton_and_ball_enclose_sqrt2_within_4_ulps() {
    let f = |x: Dual<Interval<f64>>| x.sqr() - Dual::from_f64(2.0);
    let mut x = Interval::new(1.0_f64, 2.0_f64);
    for _ in 0..8 {
        let m = x.midpoint();
        let fm = Interval::point(m).sqr() - Interval::point(2.0);
        let df = f(Dual::variable(x)).eps;
        let step = Interval::point(m) - fm / df;
        if let Some(next) = x.intersect(step) {
            x = next;
        }
    }
    let sqrt2 = core::f64::consts::SQRT_2;
    assert!(x.contains(sqrt2));
    assert!(x.width_ulps() <= 2);

    let b = Ball::from_interval(x);
    assert!(b.contains(sqrt2));
    assert!(b.width_ulps() <= 4);

    // Transcendental point enclosures within <= 4 ULPs for both Interval<f64> and Ball<f64>
    let sin_half_ref = 0.479_425_538_604_203_f64;
    let exp_half_ref = 1.648_721_270_700_128_2_f64;
    let ln_two_ref = core::f64::consts::LN_2;

    let iv_sqrt2 = Interval::point(2.0_f64).sqrt();
    let iv_sin = Interval::point(0.5_f64).sin();
    let iv_exp = Interval::point(0.5_f64).exp();
    let iv_ln = Interval::point(2.0_f64).ln();
    assert!(iv_sqrt2.contains(sqrt2) && iv_sqrt2.width_ulps() <= 4);
    assert!(iv_sin.contains(sin_half_ref) && iv_sin.width_ulps() <= 4);
    assert!(iv_exp.contains(exp_half_ref) && iv_exp.width_ulps() <= 4);
    assert!(iv_ln.contains(ln_two_ref) && iv_ln.width_ulps() <= 4);

    let ball_sqrt2 = Ball::exact(2.0_f64).sqrt();
    let ball_sin = Ball::exact(0.5_f64).sin();
    let ball_exp = Ball::exact(0.5_f64).exp();
    let ball_ln = Ball::exact(2.0_f64).ln();
    assert!(ball_sqrt2.contains(sqrt2) && ball_sqrt2.width_ulps() <= 4);
    assert!(ball_sin.contains(sin_half_ref) && ball_sin.width_ulps() <= 4);
    assert!(ball_exp.contains(exp_half_ref) && ball_exp.width_ulps() <= 4);
    assert!(ball_ln.contains(ln_two_ref) && ball_ln.width_ulps() <= 4);
}
