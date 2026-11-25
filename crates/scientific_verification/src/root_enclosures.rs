//! Tier 5 interval-Newton and Krawczyk root-enclosure certification (`<= 4` ULPs) via `math_scalars`.

use math_scalars::{
    Ball, Dual, Interval, RealScalar, Scalar, certify_reference_roots, interval_newton_enclose,
    krawczyk_enclose,
};
use std::collections::BTreeMap;

/// Maximum certified enclosure width in ULPs permitted for Tier 5 reference root solvers.
const MAX_CERTIFIED_ROOT_ULP_WIDTH: u64 = 4;

pub(super) fn record_certified_root_enclosures(
    bounds: &mut BTreeMap<String, [f64; 2]>,
) -> Result<(), String> {
    let summary = certify_reference_roots();
    if !summary.all_contained {
        return Err("math_scalars::certify_reference_roots failed root containment".to_string());
    }
    if summary.max_interval_ulp_width > MAX_CERTIFIED_ROOT_ULP_WIDTH {
        return Err(format!(
            "Certified interval root max ULP width {} exceeds ceiling {}",
            summary.max_interval_ulp_width, MAX_CERTIFIED_ROOT_ULP_WIDTH
        ));
    }
    if summary.max_ball_ulp_width > MAX_CERTIFIED_ROOT_ULP_WIDTH {
        return Err(format!(
            "Certified ball root max ULP width {} exceeds ceiling {}",
            summary.max_ball_ulp_width, MAX_CERTIFIED_ROOT_ULP_WIDTH
        ));
    }

    record_sqrt2_enclosures(bounds)?;
    record_ln3_enclosures(bounds)?;
    record_dottie_enclosures(bounds)?;
    Ok(())
}

fn record_sqrt2_enclosures(bounds: &mut BTreeMap<String, [f64; 2]>) -> Result<(), String> {
    let exact = core::f64::consts::SQRT_2;
    let f = |x: Dual<Interval<f64>>| x * x - Dual::from_f64(2.0);
    let iv = interval_newton_enclose(f, Interval::new(1.0_f64, 2.0_f64), 24)
        .ok_or_else(|| "interval_newton_enclose failed on x^2 - 2".to_string())?;
    let ball = krawczyk_enclose(f, Ball::new(1.5_f64, 0.5_f64), 24)
        .ok_or_else(|| "krawczyk_enclose failed on x^2 - 2".to_string())?;
    insert_verified_pair(
        bounds,
        "sqrt2",
        iv.enclosure,
        ball.enclosure,
        iv.certified_unique && ball.certified_unique,
        exact,
    )
}

fn record_ln3_enclosures(bounds: &mut BTreeMap<String, [f64; 2]>) -> Result<(), String> {
    let exact = 3.0_f64.ln();
    let f = |x: Dual<Interval<f64>>| x.exp() - Dual::from_f64(3.0);
    let iv = interval_newton_enclose(f, Interval::new(0.5_f64, 1.5_f64), 24)
        .ok_or_else(|| "interval_newton_enclose failed on exp(x) - 3".to_string())?;
    let ball = krawczyk_enclose(f, Ball::new(1.0_f64, 0.5_f64), 24)
        .ok_or_else(|| "krawczyk_enclose failed on exp(x) - 3".to_string())?;
    insert_verified_pair(
        bounds,
        "ln3",
        iv.enclosure,
        ball.enclosure,
        iv.certified_unique && ball.certified_unique,
        exact,
    )
}

fn record_dottie_enclosures(bounds: &mut BTreeMap<String, [f64; 2]>) -> Result<(), String> {
    const DOTTIE_REF: f64 = 0.739_085_133_215_160_7;
    let f = |x: Dual<Interval<f64>>| x.cos() - x;
    let iv = interval_newton_enclose(f, Interval::new(0.5_f64, 1.0_f64), 24)
        .ok_or_else(|| "interval_newton_enclose failed on cos(x) - x".to_string())?;
    let ball = krawczyk_enclose(f, Ball::new(0.75_f64, 0.25_f64), 24)
        .ok_or_else(|| "krawczyk_enclose failed on cos(x) - x".to_string())?;
    insert_verified_pair(
        bounds,
        "dottie",
        iv.enclosure,
        ball.enclosure,
        iv.certified_unique && ball.certified_unique,
        DOTTIE_REF,
    )
}

fn insert_verified_pair(
    bounds: &mut BTreeMap<String, [f64; 2]>,
    name: &str,
    iv: Interval<f64>,
    ball: Ball<f64>,
    certified_unique: bool,
    exact: f64,
) -> Result<(), String> {
    if !certified_unique {
        return Err(format!("Certified {name} enclosures failed uniqueness"));
    }
    let ball_iv = ball.to_interval();
    if !iv.contains(exact) || !ball.contains(exact) {
        return Err(format!(
            "Certified {name} enclosures failed to contain {exact}"
        ));
    }
    if iv.lo.ulp_distance(iv.hi) > MAX_CERTIFIED_ROOT_ULP_WIDTH
        || ball_iv.lo.ulp_distance(ball_iv.hi) > MAX_CERTIFIED_ROOT_ULP_WIDTH
    {
        return Err(format!(
            "Certified {name} enclosure exceeded {MAX_CERTIFIED_ROOT_ULP_WIDTH} ULPs"
        ));
    }
    bounds.insert(format!("root_{name}_interval"), [iv.lo, iv.hi]);
    bounds.insert(format!("root_{name}_ball"), [ball_iv.lo, ball_iv.hi]);
    Ok(())
}
