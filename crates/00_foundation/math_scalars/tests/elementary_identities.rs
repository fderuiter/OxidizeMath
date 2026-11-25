//! Comprehensive differential, algebraic, and outward-containment identity suite for `math_scalars`.

use math_scalars::{
    Ball, DirectedRounding, Dual, DualVec, HyperDual, Interval, RealScalar, Scalar, next_down_f32,
    next_down_f64, next_up_f32, next_up_f64, ulp_distance_f32, ulp_distance_f64,
};

#[test]
fn test_dual_trig_and_inverse_identities() {
    let x = Dual::variable(0.45_f64);
    let (s, c) = x.sin_cos();
    assert!((s.val - x.sin().val).abs() < 1e-15);
    assert!((c.val - x.cos().val).abs() < 1e-15);
    assert!((s.eps - c.val).abs() < 1e-15);
    assert!((c.eps + s.val).abs() < 1e-15);

    let pyth = s * s + c * c;
    assert!((pyth.val - 1.0).abs() < 1e-15);
    assert!(pyth.eps.abs() < 1e-15);

    let t = x.tan();
    assert!((t.val - s.val / c.val).abs() < 1e-15);
    assert!((t.eps - 1.0 / (c.val * c.val)).abs() < 1e-14);

    let sum_inv = x.asin() + x.acos();
    assert!((sum_inv.val - core::f64::consts::FRAC_PI_2).abs() < 1e-15);
    assert!(sum_inv.eps.abs() < 1e-15);
    assert!((x.atan().eps - 1.0 / (1.0 + 0.45 * 0.45)).abs() < 1e-15);

    let y = Dual::constant(1.25_f64);
    let at2 = x.atan2(y);
    let ratio_at = (x / y).atan();
    assert!((at2.val - ratio_at.val).abs() < 1e-15);
    assert!((at2.eps - ratio_at.eps).abs() < 1e-15);
}

#[test]
fn test_dual_hyperbolic_exp_pow_and_f32_identities() {
    let x = Dual::variable(0.45_f64);
    let sh = x.sinh();
    let ch = x.cosh();
    let hyp_id = ch * ch - sh * sh;
    assert!((hyp_id.val - 1.0).abs() < 1e-15);
    assert!(hyp_id.eps.abs() < 1e-15);
    assert!((x.tanh().eps - 1.0 / (ch.val * ch.val)).abs() < 1e-15);

    let exp_ln = (x + Dual::ONE).ln().exp();
    assert!((exp_ln.val - 1.45).abs() < 1e-15);
    assert!((exp_ln.eps - 1.0).abs() < 1e-15);

    let p3 = x.powi(3);
    assert!((p3.val - 0.45_f64.powi(3)).abs() < 1e-15);
    assert!((p3.eps - 3.0 * 0.45 * 0.45).abs() < 1e-15);

    let pm2 = x.powi(-2);
    assert!((pm2.val - 0.45_f64.powi(-2)).abs() < 1e-14);
    assert!((pm2.eps + 2.0 * 0.45_f64.powi(-3)).abs() < 1e-13);

    let pf = (x + Dual::ONE).powf(Dual::from_f64(2.5));
    assert!((pf.val - 1.45_f64.powf(2.5)).abs() < 1e-14);
    assert!((pf.eps - 2.5 * 1.45_f64.powf(1.5)).abs() < 1e-14);

    let hyp = x.hypot(Dual::constant(0.6_f64));
    assert!((hyp.val - 0.75).abs() < 1e-15);
    assert!((hyp.eps - 0.45 / 0.75).abs() < 1e-15);

    let xf = Dual::variable(0.5_f32);
    assert!((xf.sqrt().eps - 0.5 / 0.5_f32.sqrt()).abs() < 1e-6);
    assert_eq!((-xf).abs().val, 0.5_f32);
    assert_eq!((-xf).signum().val, -1.0_f32);
    assert!((xf.recip().val - 2.0_f32).abs() < 1e-6);
}

#[test]
fn test_dual_vec_and_hyper_dual_identities() {
    let seed = DualVec::<f64, 3>::jacobian_seed([0.3, 0.4, 1.2]);
    let r = seed[0].hypot(seed[1]);
    assert!((r.val - 0.5).abs() < 1e-15);
    assert!((r.eps[0] - 0.6).abs() < 1e-15);
    assert!((r.eps[1] - 0.8).abs() < 1e-15);
    assert_eq!(r.eps[2], 0.0);

    let prod = (seed[0] + DualVec::ONE).powf(seed[2]);
    assert!((prod.val - 1.3_f64.powf(1.2)).abs() < 1e-15);
    assert!((prod.eps[0] - 1.2 * 1.3_f64.powf(0.2)).abs() < 1e-14);
    assert!((prod.eps[2] - 1.3_f64.powf(1.2) * 1.3_f64.ln()).abs() < 1e-14);

    let mut acc = seed[0];
    acc += seed[1];
    acc -= DualVec::constant(0.1);
    acc *= seed[2];
    acc /= DualVec::TWO;
    assert!((acc.val - 0.36).abs() < 1e-15);
    assert!((acc.eps[0] - 0.6).abs() < 1e-15);
    assert!((acc.eps[1] - 0.6).abs() < 1e-15);
    assert!((acc.eps[2] - 0.3).abs() < 1e-15);

    let u = HyperDual::bivariate_u(0.35_f64);
    let v = HyperDual::bivariate_v(0.55_f64);
    let f_uv = (u.sin() * v.cosh()) / (u * u + v * v + HyperDual::ONE).sqrt();
    let f_vu = (HyperDual::bivariate_v(0.35_f64).sin() * HyperDual::bivariate_u(0.55_f64).cosh())
        / (HyperDual::bivariate_v(0.35_f64).powi(2)
            + HyperDual::bivariate_u(0.55_f64).powi(2)
            + HyperDual::ONE)
            .sqrt();
    assert!((f_uv.val - f_vu.val).abs() < 1e-15);
    assert!((f_uv.eps1 - f_vu.eps2).abs() < 1e-15);
    assert!((f_uv.eps2 - f_vu.eps1).abs() < 1e-15);
    assert!((f_uv.eps12 - f_vu.eps12).abs() < 1e-15);

    let inv_check = u.asin() + u.acos();
    assert!((inv_check.val - core::f64::consts::FRAC_PI_2).abs() < 1e-15);
    assert!(inv_check.eps1.abs() < 1e-15);
    assert!(inv_check.eps12.abs() < 1e-15);

    let th = (u + v).tanh();
    let sech2 = 1.0 / (0.9_f64.cosh() * 0.9_f64.cosh());
    assert!((th.eps1 - sech2).abs() < 1e-15);
    assert!((th.eps2 - sech2).abs() < 1e-15);
    assert!((th.eps12 + 2.0 * 0.9_f64.tanh() * sech2).abs() < 1e-14);
}

#[test]
fn test_interval_arithmetic_and_trig_containment() {
    let iv = Interval::new(0.2_f64, 0.7_f64);
    let iv2 = Interval::new(1.1_f64, 1.6_f64);
    for k in 0..=8 {
        let t = f64::from(k) / 8.0;
        let x = ((1.0 - t) * iv.lo + t * iv.hi).clamp(iv.lo, iv.hi);
        let y = (t * iv2.lo + (1.0 - t) * iv2.hi).clamp(iv2.lo, iv2.hi);
        assert!(iv.contains(x) && iv2.contains(y));
        assert!((iv + iv2).contains(x + y));
        assert!((iv - iv2).contains(x - y));
        assert!((iv * iv2).contains(x * y));
        assert!((iv / iv2).contains(x / y));
        assert!(iv.sin().contains(x.sin()));
        assert!(iv.cos().contains(x.cos()));
        assert!(iv.tan().contains(x.tan()));
        assert!(iv.asin().contains(x.asin()));
        assert!(iv.acos().contains(x.acos()));
        assert!(iv.atan().contains(x.atan()));
        assert!(iv.atan2(iv2).contains(x.atan2(y)));
    }
}

#[test]
fn test_interval_exp_hyperbolic_and_power_containment() {
    let iv = Interval::new(0.2_f64, 0.7_f64);
    let iv2 = Interval::new(1.1_f64, 1.6_f64);
    for k in 0..=8 {
        let t = f64::from(k) / 8.0;
        let x = ((1.0 - t) * iv.lo + t * iv.hi).clamp(iv.lo, iv.hi);
        let y = (t * iv2.lo + (1.0 - t) * iv2.hi).clamp(iv2.lo, iv2.hi);
        assert!(iv.sqrt().contains(x.sqrt()));
        assert!(iv.exp().contains(x.exp()));
        assert!(iv.ln().contains(x.ln()));
        assert!(iv.sinh().contains(x.sinh()));
        assert!(iv.cosh().contains(x.cosh()));
        assert!(iv.tanh().contains(x.tanh()));
        assert!(iv.hypot(iv2).contains(x.hypot(y)));
        assert!(iv2.powf(iv).contains(y.powf(x)));
        assert!(iv.powi(2).contains(x * x));
        assert!(iv.powi(3).contains(x * x * x));
    }
}

#[test]
fn test_interval_straddle_and_ball_containment() {
    let straddle = Interval::new(-0.6_f64, 0.4_f64);
    let sq = straddle.powi(2);
    assert_eq!(sq.lo, 0.0);
    assert!(sq.contains(0.36));
    assert!(straddle.abs().contains(0.6) && straddle.abs().contains(0.0));
    assert!(straddle.cosh().contains(1.0));

    let full_trig = Interval::new(-4.0_f64, 4.0_f64);
    assert_eq!((full_trig.sin().lo, full_trig.sin().hi), (-1.0, 1.0));
    assert_eq!((full_trig.cos().lo, full_trig.cos().hi), (-1.0, 1.0));

    let b1 = Ball::new(0.45_f64, 1e-4);
    let b2 = Ball::new(1.35_f64, 2e-4);
    let (x0, y0) = (0.45007_f64, 1.34985_f64);
    assert!((b1 + b2).contains(x0 + y0) && (b1 - b2).contains(x0 - y0));
    assert!((b1 * b2).contains(x0 * y0) && (b1 / b2).contains(x0 / y0));
    assert!(b1.sin().contains(x0.sin()) && b1.cos().contains(x0.cos()));
    assert!(b1.exp().contains(x0.exp()) && b2.ln().contains(y0.ln()));
    assert!(b2.sqrt().contains(y0.sqrt()));
}

#[test]
fn test_ulp_stepping_invariants() {
    assert!(next_up_f64(0.0) > 0.0 && next_down_f64(0.0) < 0.0);
    assert_eq!(ulp_distance_f64(-0.0, 0.0), 0);
    assert_eq!(ulp_distance_f64(1.0, next_up_f64(1.0)), 1);
    assert_eq!(ulp_distance_f64(1.0, next_down_f64(1.0)), 1);

    assert!(next_up_f32(0.0) > 0.0 && next_down_f32(0.0) < 0.0);
    assert_eq!(ulp_distance_f32(-0.0, 0.0), 0);
    assert_eq!(ulp_distance_f32(1.0, next_up_f32(1.0)), 1);
    assert_eq!(ulp_distance_f32(1.0, next_down_f32(1.0)), 1);
}

#[test]
fn test_directed_rounding_and_real_scalar_rounding() {
    let (a, tiny) = (1.0_f64, 1e-30_f64);
    assert!(f64::add_up(a, tiny) > a && f64::sub_down(a, tiny) < a);
    assert_eq!(f64::add_down(a, tiny), a);
    assert_eq!(f64::sub_up(a, tiny), a);
    assert_eq!(
        ulp_distance_f64(f64::div_down(1.0, 3.0), f64::div_up(1.0, 3.0)),
        1
    );
    assert_eq!(ulp_distance_f64(f64::sqrt_down(2.0), f64::sqrt_up(2.0)), 1);

    assert!(f32::add_up(1.0_f32, 1e-20_f32) > 1.0_f32);
    assert_eq!(f32::add_down(1.0_f32, 1e-20_f32), 1.0_f32);
    assert_eq!(
        ulp_distance_f32(f32::div_down(1.0, 3.0), f32::div_up(1.0, 3.0)),
        1
    );
    assert_eq!(ulp_distance_f32(f32::sqrt_down(2.0), f32::sqrt_up(2.0)), 1);

    assert_eq!((f64::min(2.0, 3.0), f64::max(2.0, 3.0)), (2.0, 3.0));
    assert_eq!(
        (RealScalar::floor(2.7_f64), RealScalar::ceil(2.3_f64)),
        (2.0, 3.0)
    );
    assert_eq!(
        (RealScalar::floor(-2.3_f32), RealScalar::ceil(-2.7_f32)),
        (-3.0, -2.0)
    );
}
