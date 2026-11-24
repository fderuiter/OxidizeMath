// [cite:math_scalars]
//! Verification tests for `Dual`, `DualVec`, `HyperDual`, `Interval`, `Ball`, and `RealScalar`
//! across all elementary `libm` operations in `f64` and `f32`.

use math_scalars::{Ball, Dual, DualVec, HyperDual, Interval, RealScalar, Scalar};
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_dual_elementary_derivatives_and_zero_alloc() {
    let x = 0.4_f64;
    let z = 1.7_f64;
    let ((), _metrics) = VerifiedEngine::run_verified(|| {
        assert!((Dual::variable(x).tan().eps - (1.0 + x.tan().powi(2))).abs() < 1e-14);
        assert!((Dual::variable(x).asin().eps - (1.0 - x * x).sqrt().recip()).abs() < 1e-14);
        assert!((Dual::variable(x).acos().eps + (1.0 - x * x).sqrt().recip()).abs() < 1e-14);
        assert!((Dual::variable(x).atan().eps - (1.0 + x * x).recip()).abs() < 1e-14);
        assert!((Dual::variable(x).sinh().eps - x.cosh()).abs() < 1e-14);
        assert!((Dual::variable(x).cosh().eps - x.sinh()).abs() < 1e-14);
        assert!((Dual::variable(x).tanh().eps - (1.0 - x.tanh().powi(2))).abs() < 1e-14);
        assert!((Dual::variable(x).asinh().eps - (1.0 + x * x).sqrt().recip()).abs() < 1e-14);
        assert!((Dual::variable(z).acosh().eps - (z * z - 1.0).sqrt().recip()).abs() < 1e-14);
        assert!((Dual::variable(x).atanh().eps - (1.0 - x * x).recip()).abs() < 1e-14);
        assert!((Dual::variable(z).log2().eps - core::f64::consts::LOG2_E / z).abs() < 1e-14);
        assert!((Dual::variable(z).log10().eps - core::f64::consts::LOG10_E / z).abs() < 1e-14);
        assert!((Dual::variable(z).recip().eps + 1.0 / (z * z)).abs() < 1e-14);
        assert!((Dual::variable(z).powi(3).val - z.powi(3)).abs() < 1e-14);
        assert!((Dual::variable(z).powi(3).eps - 3.0 * z * z).abs() < 1e-14);
        assert!((Dual::variable(z).powi(-2).eps + 2.0 * z.powi(-3)).abs() < 1e-14);
        assert!(
            (Dual::variable(z).powf(Dual::from_f64(2.5)).eps - 2.5 * z.powf(1.5)).abs() < 1e-14
        );
        let zz = Dual::variable(z).powf(Dual::variable(z));
        assert!((zz.eps - z.powf(z) * (z.ln() + 1.0)).abs() < 1e-13);
        let neg_abs = Dual::variable(-z).abs();
        assert_eq!(neg_abs.val, z);
        assert_eq!(neg_abs.eps, -1.0);
        assert_eq!(Dual::variable(0.0_f64).abs().eps, 0.0);
        let (ds, dc) = Dual::variable(x).sin_cos();
        assert!((ds.eps - x.cos()).abs() < 1e-15);
        assert!((dc.eps + x.sin()).abs() < 1e-15);
    });
}

#[test]
#[verified_engine::verified]
fn test_dual_vec_all_elementary_lanes() {
    let u = DualVec::<f64, 2>::variable(0.35, 0);
    let v = DualVec::<f64, 2>::variable(1.65, 1);
    let (x, z) = (0.35_f64, 1.65_f64);

    assert!((u.tan().eps[0] - (1.0 + x.tan().powi(2))).abs() < 1e-14);
    assert_eq!(u.tan().eps[1], 0.0);
    assert!((u.asin().eps[0] - (1.0 - x * x).sqrt().recip()).abs() < 1e-14);
    assert!((u.acos().eps[0] + (1.0 - x * x).sqrt().recip()).abs() < 1e-14);
    assert!((u.atan().eps[0] - (1.0 + x * x).recip()).abs() < 1e-14);
    assert!((u.sinh().eps[0] - x.cosh()).abs() < 1e-14);
    assert!((u.cosh().eps[0] - x.sinh()).abs() < 1e-14);
    assert!((u.tanh().eps[0] - (1.0 - x.tanh().powi(2))).abs() < 1e-14);
    assert!((u.asinh().eps[0] - (1.0 + x * x).sqrt().recip()).abs() < 1e-14);
    assert!((v.acosh().eps[1] - (z * z - 1.0).sqrt().recip()).abs() < 1e-14);
    assert!((u.atanh().eps[0] - (1.0 - x * x).recip()).abs() < 1e-14);
    assert!((v.log2().eps[1] - core::f64::consts::LOG2_E / z).abs() < 1e-14);
    assert!((v.log10().eps[1] - core::f64::consts::LOG10_E / z).abs() < 1e-14);
    assert!((v.sqrt().eps[1] - 0.5 / z.sqrt()).abs() < 1e-14);
    assert!((v.recip().eps[1] + 1.0 / (z * z)).abs() < 1e-14);

    let p_var = v.powf(u);
    assert!((p_var.eps[0] - z.powf(x) * z.ln()).abs() < 1e-14);
    assert!((p_var.eps[1] - x * z.powf(x - 1.0)).abs() < 1e-14);

    let hyp = u.hypot(v);
    let r = x.hypot(z);
    assert!((hyp.eps[0] - x / r).abs() < 1e-14);
    assert!((hyp.eps[1] - z / r).abs() < 1e-14);

    let ang = u.atan2(v);
    let r2 = x * x + z * z;
    assert!((ang.eps[0] - z / r2).abs() < 1e-14);
    assert!((ang.eps[1] + x / r2).abs() < 1e-14);
    assert_eq!((-v).abs().eps[1], 1.0);
    assert_eq!(DualVec::<f64, 2>::variable(-z, 1).abs().eps[1], -1.0);
    assert!(u.is_finite_val() && v.is_finite_val());
}

#[test]
#[verified_engine::verified]
fn test_hyper_dual_second_order_elementary_identities() {
    let x = 0.42_f64;
    let z = 1.85_f64;
    let hx = HyperDual::second_order(x);
    let hz = HyperDual::second_order(z);

    let sec2 = 1.0 + x.tan().powi(2);
    assert!((hx.tan().eps12 - 2.0 * x.tan() * sec2).abs() < 1e-13);
    assert!((hx.asin().eps12 - x / (1.0 - x * x).powf(1.5)).abs() < 1e-13);
    assert!((hx.acos().eps12 + x / (1.0 - x * x).powf(1.5)).abs() < 1e-13);
    assert!((hx.atan().eps12 + 2.0 * x / (1.0 + x * x).powi(2)).abs() < 1e-13);
    assert!((hx.sinh().eps12 - x.sinh()).abs() < 1e-14);
    assert!((hx.cosh().eps12 - x.cosh()).abs() < 1e-14);
    let th = x.tanh();
    assert!((hx.tanh().eps12 + 2.0 * th * (1.0 - th * th)).abs() < 1e-14);
    assert!((hx.asinh().eps12 + x / (1.0 + x * x).powf(1.5)).abs() < 1e-14);
    assert!((hz.acosh().eps12 + z / (z * z - 1.0).powf(1.5)).abs() < 1e-13);
    assert!((hx.atanh().eps12 - 2.0 * x / (1.0 - x * x).powi(2)).abs() < 1e-13);
    assert!((hz.ln().eps12 + 1.0 / (z * z)).abs() < 1e-14);
    assert!((hz.log2().eps12 + core::f64::consts::LOG2_E / (z * z)).abs() < 1e-14);
    assert!((hz.log10().eps12 + core::f64::consts::LOG10_E / (z * z)).abs() < 1e-14);
    assert!((hz.sqrt().eps12 + 0.25 / z.powf(1.5)).abs() < 1e-14);
    assert!((hz.recip().eps12 - 2.0 / z.powi(3)).abs() < 1e-14);
    assert!((hz.powf(HyperDual::from_f64(2.5)).eps12 - 3.75 * z.sqrt()).abs() < 1e-13);
    let zz = hz.powf(hz);
    let exact_zz2 = z.powf(z) * ((z.ln() + 1.0).powi(2) + 1.0 / z);
    assert!((zz.eps12 - exact_zz2).abs() < 1e-12);
    let (hs, hc) = hx.sin_cos();
    assert!((hs.eps12 + x.sin()).abs() < 1e-14);
    assert!((hc.eps12 + x.cos()).abs() < 1e-14);
}

#[test]
#[verified_engine::verified]
fn test_interval_and_ball_enclosures_across_f64_and_f32() {
    let x_iv = Interval::point(0.35_f64);
    let z_iv = Interval::point(1.65_f64);
    assert!(x_iv.tan().contains(0.35_f64.tan()) && x_iv.tan().width_ulps() <= 4);
    assert!(x_iv.asin().contains(0.35_f64.asin()) && x_iv.asin().width_ulps() <= 4);
    assert!(x_iv.acos().contains(0.35_f64.acos()) && x_iv.acos().width_ulps() <= 4);
    assert!(x_iv.atan().contains(0.35_f64.atan()) && x_iv.atan().width_ulps() <= 4);
    assert!(x_iv.sinh().contains(0.35_f64.sinh()) && x_iv.sinh().width_ulps() <= 4);
    assert!(x_iv.cosh().contains(0.35_f64.cosh()) && x_iv.cosh().width_ulps() <= 4);
    assert!(x_iv.tanh().contains(0.35_f64.tanh()) && x_iv.tanh().width_ulps() <= 4);
    assert!(x_iv.asinh().contains(0.35_f64.asinh()) && x_iv.asinh().width_ulps() <= 4);
    assert!(z_iv.acosh().contains(1.65_f64.acosh()) && z_iv.acosh().width_ulps() <= 4);
    assert!(x_iv.atanh().contains(0.35_f64.atanh()) && x_iv.atanh().width_ulps() <= 4);
    assert!(z_iv.log2().contains(1.65_f64.log2()) && z_iv.log2().width_ulps() <= 4);
    assert!(z_iv.log10().contains(1.65_f64.log10()) && z_iv.log10().width_ulps() <= 4);
    assert!(z_iv.powf(x_iv).contains(1.65_f64.powf(0.35_f64)));
    assert!(x_iv.hypot(z_iv).contains(0.35_f64.hypot(1.65_f64)));
    assert!(x_iv.atan2(z_iv).contains(0.35_f64.atan2(1.65_f64)));

    let a = Interval::new(1.0_f64, 3.0_f64);
    let b = Interval::new(2.0_f64, 4.0_f64);
    assert_eq!(a.hull(b), Interval::new(1.0, 4.0));
    assert_eq!(a.intersect(b), Some(Interval::new(2.0, 3.0)));
    assert!(Interval::new(1.0, 4.0).contains_interval(a));
    assert!(Interval::new(3.0_f64, 1.0_f64).is_nan());

    let ba = Ball::new(2.0_f64, 0.25_f64);
    let bb = Ball::new(0.5_f64, 0.125_f64);
    assert!((ba + bb).contains(2.3));
    assert!((ba - bb).contains(1.4));
    assert!((ba * bb).contains(2.25 * 0.625));
    assert!((ba / bb).contains(2.25 / 0.375));

    let iv32 = Interval::point(2.0_f32).sqrt();
    let b32 = Ball::exact(2.0_f32).sqrt();
    assert!(iv32.contains(core::f32::consts::SQRT_2) && iv32.width_ulps() <= 4);
    assert!(b32.contains(core::f32::consts::SQRT_2) && b32.width_ulps() <= 4);
}

#[test]
#[verified_engine::verified]
fn test_real_scalar_rounding_and_sign_helpers() {
    let v64 = -2.75_f64;
    assert_eq!(RealScalar::floor(v64), -3.0);
    assert_eq!(RealScalar::ceil(v64), -2.0);
    assert_eq!(RealScalar::round(v64), -3.0);
    assert_eq!(RealScalar::trunc(v64), -2.0);
    assert_eq!(RealScalar::copysign(3.0_f64, v64), -3.0);
    assert!(RealScalar::is_sign_negative(-0.0_f64));
    assert!(!RealScalar::is_infinite(v64));
    assert_eq!(RealScalar::step_up(1.0_f64, 3).ulp_distance(1.0_f64), 3);
    assert_eq!(RealScalar::step_down(1.0_f64, 3).ulp_distance(1.0_f64), 3);

    let v32 = 2.25_f32;
    assert_eq!(RealScalar::floor(v32), 2.0);
    assert_eq!(RealScalar::ceil(v32), 3.0);
    assert_eq!(RealScalar::round(v32), 2.0);
    assert_eq!(RealScalar::trunc(v32), 2.0);
    assert_eq!(RealScalar::copysign(-4.0_f32, v32), 4.0);
    assert_eq!(RealScalar::step_up(1.0_f32, 4).ulp_distance(1.0_f32), 4);
    assert_eq!(RealScalar::step_down(1.0_f32, 4).ulp_distance(1.0_f32), 4);
}
