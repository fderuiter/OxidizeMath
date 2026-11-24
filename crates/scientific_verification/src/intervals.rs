#![cfg_attr(any(), verified(opt_out = "verification_harness"))]

use math_commons::primitives::{Interval, NonNegativeFloat, PositiveFloat, UnitInterval};
use pure_math::pure_math::special_functions::bessel::bessel_j;
use std::collections::BTreeMap;

/// Summary of Tier 5 IEEE-754 outward-rounded interval enclosures for certified mathematical kernels.
#[derive(Clone, Debug)]
pub struct IntervalCertification {
    /// Maximum enclosure width `hi - lo` across all certified kernels.
    pub max_width: f64,
    /// Deterministic map of certified kernel names to `[lo, hi]` enclosures.
    pub bounds: BTreeMap<String, [f64; 2]>,
}

fn enclosure_bessel_j0(x: Interval) -> Result<Interval, String> {
    // 14-term alternating series enclosure + rigorous remainder bound |(x/2)^(28) / (14!)^2|
    let half = x * Interval::point(0.5).map_err(|e| format!("{e:?}"))?;
    let q = half.sqr();
    let mut sum = Interval::point(1.0).map_err(|e| format!("{e:?}"))?;
    let mut term = sum;
    for k in 1..=14u32 {
        let denom = Interval::point((k as f64) * (k as f64)).map_err(|e| format!("{e:?}"))?;
        term = -(term * q).checked_div(denom).expect("non-zero k^2");
        sum = sum + term;
    }
    let rem_bound = term.abs().hi();
    Interval::new(sum.lo() - rem_bound, sum.hi() + rem_bound).map_err(|e| format!("{e:?}"))
}

fn record_special_and_elementary_enclosures(
    bounds: &mut BTreeMap<String, [f64; 2]>,
) -> Result<(), String> {
    let one = Interval::point(1.0).map_err(|e| format!("{e:?}"))?;
    let j0_1_iv = enclosure_bessel_j0(one)?;
    let j0_1_ref = bessel_j(0.0, 1.0);
    if !j0_1_iv.contains(j0_1_ref) {
        return Err(format!(
            "Interval {j0_1_iv:?} failed to enclose J_0(1)={j0_1_ref}"
        ));
    }
    bounds.insert("bessel_j0_at_1".to_string(), [j0_1_iv.lo(), j0_1_iv.hi()]);

    let theta = Interval::point(0.7).map_err(|e| format!("{e:?}"))?;
    let pythag = theta.sin().sqr() + theta.cos().sqr();
    if !pythag.contains(1.0) {
        return Err(format!(
            "Pythagorean interval {pythag:?} failed to enclose 1.0"
        ));
    }
    bounds.insert(
        "pythagorean_unit_at_0_7".to_string(),
        [pythag.lo(), pythag.hi()],
    );

    let pos = PositiveFloat::new(2.5).map_err(|e| format!("{e:?}"))?;
    let exp_ln = pos
        .to_interval()
        .interval()
        .ln()
        .map_err(|e| format!("{e:?}"))?
        .exp();
    if !exp_ln.contains(2.5) {
        return Err(format!(
            "exp(ln(2.5)) interval {exp_ln:?} failed to enclose 2.5"
        ));
    }
    bounds.insert(
        "exp_ln_roundtrip_2_5".to_string(),
        [exp_ln.lo(), exp_ln.hi()],
    );
    Ok(())
}

fn record_domain_conservation_enclosures(
    bounds: &mut BTreeMap<String, [f64; 2]>,
) -> Result<(), String> {
    let one = Interval::point(1.0).map_err(|e| format!("{e:?}"))?;
    let u = UnitInterval::new(0.5)
        .map_err(|e| format!("{e:?}"))?
        .to_interval();
    let four = Interval::point(4.0).map_err(|e| format!("{e:?}"))?;
    let logistic = four * u.interval() * (one - u.interval());
    if !logistic.contains(1.0) {
        return Err(format!(
            "Logistic interval {logistic:?} failed to enclose 1.0"
        ));
    }
    bounds.insert(
        "logistic_peak_at_0_5".to_string(),
        [logistic.lo(), logistic.hi()],
    );

    let s0 = NonNegativeFloat::new(990.0)
        .map_err(|e| format!("{e:?}"))?
        .to_interval();
    let i0 = NonNegativeFloat::new(10.0)
        .map_err(|e| format!("{e:?}"))?
        .to_interval();
    let r0 = NonNegativeFloat::new(0.0)
        .map_err(|e| format!("{e:?}"))?
        .to_interval();
    let n_inv = one
        .checked_div(s0.interval() + i0.interval() + r0.interval())
        .map_err(|e| format!("{e:?}"))?;
    let dt = Interval::point(0.05).map_err(|e| format!("{e:?}"))?;
    let beta = Interval::point(0.3).map_err(|e| format!("{e:?}"))?;
    let gamma = Interval::point(0.1).map_err(|e| format!("{e:?}"))?;
    let inf_flux = beta * s0.interval() * i0.interval() * n_inv * dt;
    let rec_flux = gamma * i0.interval() * dt;
    let total_sir = (s0.interval() - inf_flux)
        + (i0.interval() + inf_flux - rec_flux)
        + (r0.interval() + rec_flux);
    if !total_sir.contains(1000.0) {
        return Err(format!(
            "SIR interval {total_sir:?} failed to enclose 1000.0"
        ));
    }
    bounds.insert(
        "sir_total_population_step".to_string(),
        [total_sir.lo(), total_sir.hi()],
    );
    Ok(())
}

/// Computes and verifies outward-rounded IEEE-754 interval enclosures across core kernels,
/// ensuring every exact/high-precision reference lies strictly inside `[lo, hi]`.
pub fn compute_certified_interval_enclosures() -> Result<IntervalCertification, String> {
    let mut bounds = BTreeMap::new();
    record_special_and_elementary_enclosures(&mut bounds)?;
    record_domain_conservation_enclosures(&mut bounds)?;

    let max_width = bounds
        .values()
        .map(|&[lo, hi]| hi - lo)
        .fold(0.0_f64, f64::max);
    if max_width > 1e-11 {
        return Err(format!(
            "Max interval width {max_width:.3e} exceeded 1e-11 budget"
        ));
    }
    Ok(IntervalCertification { max_width, bounds })
}
