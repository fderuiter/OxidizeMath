//! # Hurwitz-Kronecker Class Number
//!
//! This module implements the Hurwitz-Kronecker class number and related formulas.

use super::class_number::class_number;

/// Calculates the weighted class number h_w(d) as defined in the paper.
/// Note: The paper uses a convention where h_w(d) = 0 for d.rem_euclid(4) in {2, 3}.
#[verified_engine::verified]
pub fn weighted_class_number(d: i64) -> f64 {
    if d >= 0 {
        return 0.0;
    }
    if d == -3 {
        return 1.0 / 3.0;
    }
    if d == -4 {
        return 1.0 / 2.0;
    }
    if d.rem_euclid(4) != 0 && d.rem_euclid(4) != 1 {
        return 0.0;
    }

    class_number(d) as f64
}

/// Calculates the Hurwitz-Kronecker class number H(D).
/// H(D) = sum_{f^2 | D} h_w(D / f^2)
#[verified_engine::verified]
pub fn hurwitz_class_number(d: i64) -> f64 {
    if d > 0 {
        return 0.0;
    }
    if d == 0 {
        return -1.0 / 12.0;
    }

    let mut sum = 0.0;
    let limit = (d.abs() as f64).sqrt() as i64;

    for f in 1..=limit {
        if (d % (f * f)) == 0 {
            sum += weighted_class_number(d / (f * f));
        }
    }

    sum
}

/// Verifies the summation formula for a given prime p.
/// sum_{t^2 < p} H(t^2 - p) = (p - 2) / 3
#[verified_engine::verified]
pub fn verify_summation_formula(p: u64) -> bool {
    let p_i64 = p as i64;
    let mut sum = 0.0;
    let limit = ((p - 1) as f64).sqrt() as i64;

    for t in -limit..=limit {
        sum += hurwitz_class_number(t * t - p_i64);
    }

    let expected = (p_i64 - 2) as f64 / 3.0;

    // Compare floating point numbers with a tolerance.
    (sum - expected).abs() < math_commons::registry::TOLERANCE_STANDARD
}
