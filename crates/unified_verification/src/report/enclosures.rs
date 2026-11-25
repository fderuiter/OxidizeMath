//! Deterministic certified interval and midpoint-radius ball enclosure verification.

use math_scalars::{Ball, Dual, Interval, Scalar};

pub const MAX_CERTIFIED_ROOT_ULPS: u64 = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CertifiedEnclosureSummary {
    pub max_width: f64,
    pub root_max_ulp_bound: u64,
    pub verified: bool,
}

fn contract_interval_newton_sqrt2() -> Option<Interval<f64>> {
    let mut x = Interval::new(1.0_f64, 2.0_f64);
    for _ in 0..32 {
        let m = x.midpoint();
        let m_iv = Interval::point(m);
        let fm = m_iv * m_iv - Interval::point(2.0);
        let df_x = Interval::point(2.0) * x;
        if df_x.contains(0.0) {
            return None;
        }
        let next = x.intersect(m_iv - (fm / df_x))?;
        if next == x || next.width_ulps() <= 2 {
            return Some(next);
        }
        x = next;
    }
    Some(x)
}

fn contract_krawczyk_sqrt2() -> Option<Interval<f64>> {
    let mut x = Interval::new(1.0_f64, 2.0_f64);
    for _ in 0..32 {
        let m = x.midpoint();
        let m_iv = Interval::point(m);
        let jet_m = {
            let dm = Dual::variable(m_iv);
            dm * dm - Dual::constant(Interval::point(2.0))
        };
        let c = jet_m.eps.midpoint();
        if !c.is_finite() || c.abs() < 1e-14 {
            return None;
        }
        let y_iv = Interval::point(1.0 / c);
        let df_x = Interval::point(2.0) * x;
        let r_iv = Interval::ONE - (y_iv * df_x);
        let kraw = m_iv - (y_iv * jet_m.val) + r_iv * (x - m_iv);
        let next = x.intersect(kraw)?;
        if next == x || next.width_ulps() <= 2 {
            return Some(next);
        }
        x = next;
    }
    Some(x)
}

fn verify_elementary_enclosures() -> (bool, f64) {
    let sqrt2 = core::f64::consts::SQRT_2;
    let iv_sqrt2 = Interval::point(2.0_f64).sqrt();
    let iv_pi = Interval::<f64>::pi_enclosure();
    let iv_sin = Interval::point(0.5_f64).sin();
    let iv_exp = Interval::point(0.5_f64).exp();
    let ball_sin = Ball::exact(0.5_f64).sin();
    let ball_exp = Ball::exact(0.5_f64).exp();

    let sin_half = 0.479_425_538_604_203_f64;
    let exp_half = 1.648_721_270_700_128_2_f64;

    let contains = iv_sqrt2.contains(sqrt2)
        && iv_pi.contains(core::f64::consts::PI)
        && iv_pi.sin().contains(0.0)
        && iv_sin.contains(sin_half)
        && iv_exp.contains(exp_half)
        && ball_sin.contains(sin_half)
        && ball_exp.contains(exp_half);

    let tight = iv_sqrt2.width_ulps() <= MAX_CERTIFIED_ROOT_ULPS
        && iv_pi.width_ulps() <= MAX_CERTIFIED_ROOT_ULPS
        && iv_sin.width_ulps() <= MAX_CERTIFIED_ROOT_ULPS
        && iv_exp.width_ulps() <= MAX_CERTIFIED_ROOT_ULPS
        && ball_sin.to_interval().width_ulps() <= MAX_CERTIFIED_ROOT_ULPS
        && ball_exp.to_interval().width_ulps() <= MAX_CERTIFIED_ROOT_ULPS;

    (contains && tight, iv_exp.width())
}

pub fn evaluate_certified_enclosures() -> CertifiedEnclosureSummary {
    let sqrt2 = core::f64::consts::SQRT_2;
    let failed = CertifiedEnclosureSummary {
        max_width: f64::INFINITY,
        root_max_ulp_bound: u64::MAX,
        verified: false,
    };
    let Some(iv_newton) = contract_interval_newton_sqrt2() else {
        return failed;
    };
    let Some(iv_krawczyk) = contract_krawczyk_sqrt2() else {
        return failed;
    };
    let ball_newton = Ball::from_interval(iv_newton);
    let ball_krawczyk = Ball::from_interval(iv_krawczyk);
    let (elem_ok, exp_width) = verify_elementary_enclosures();

    let roots_ok = iv_newton.contains(sqrt2)
        && (iv_newton * iv_newton).contains(2.0)
        && iv_krawczyk.contains(sqrt2)
        && (iv_krawczyk * iv_krawczyk).contains(2.0)
        && ball_newton.contains(sqrt2)
        && ball_krawczyk.contains(sqrt2);

    let root_max_ulp_bound = iv_newton
        .width_ulps()
        .max(iv_krawczyk.width_ulps())
        .max(ball_newton.to_interval().width_ulps())
        .max(ball_krawczyk.to_interval().width_ulps());

    let max_width = iv_newton
        .width()
        .max(iv_krawczyk.width())
        .max(ball_newton.to_interval().width())
        .max(ball_krawczyk.to_interval().width())
        .max(exp_width);

    let verified = roots_ok
        && elem_ok
        && root_max_ulp_bound <= MAX_CERTIFIED_ROOT_ULPS
        && max_width.is_finite();

    CertifiedEnclosureSummary {
        max_width,
        root_max_ulp_bound,
        verified,
    }
}
