//! Integration tests for test favoritism security.
use domain_applied::applied::favoritism::{FavoritismInputs, calculate_favoritism_score};

#[test]
#[verified_engine::verified]
fn test_security_division_by_zero_prevention() {
    let mut inputs = FavoritismInputs::default();
    // Set x_0 to zero to trigger division by zero in proximity integral
    inputs.time.x_0 = 0.0;

    let score = calculate_favoritism_score(&inputs);

    // Should return a finite value (clamped), not Infinity or NaN
    assert!(
        score.is_finite(),
        "Score should be finite even when x_0 is 0.0"
    );
    // Since x_0 is small (high proximity), score should be high/positive
    assert!(score > 0.0);
}

#[test]
#[verified_engine::verified]
fn test_security_empty_siblings_prevention() {
    let mut inputs = FavoritismInputs::default();
    // Empty siblings list means denominator integral becomes 0
    inputs.family.sibling_distances = vec![];

    let score = calculate_favoritism_score(&inputs);

    // Should handle empty siblings gracefully (e.g. treat as no competition)
    // and not return Infinity
    assert!(
        score.is_finite(),
        "Score should be finite even with no siblings"
    );
    assert!(score > 0.0);
}

#[test]
#[verified_engine::verified]
fn test_security_log_domain_prevention() {
    let mut inputs = FavoritismInputs::default();
    // Set f_initial to -1.0 or less to trigger log(0) or log(negative)
    inputs.contact.f_initial = -5.0;

    let score = calculate_favoritism_score(&inputs);

    assert!(
        score.is_finite(),
        "Score should be finite even with invalid contact frequency"
    );
}
