//! Integration tests for hurwitz kronecker.
use pure_math::pure_math::number_theory::class_number::class_number;
use pure_math::pure_math::number_theory::hurwitz_kronecker::{
    hurwitz_class_number, verify_summation_formula, weighted_class_number,
};

#[test]
#[verified_engine::verified]
fn test_class_number() {
    assert_eq!(class_number(-3), 1);
    assert_eq!(class_number(-4), 1);
    assert_eq!(class_number(-7), 1);
    assert_eq!(class_number(-8), 1);
    assert_eq!(class_number(-11), 1);
    assert_eq!(class_number(-15), 2);
    assert_eq!(class_number(-19), 1);
    assert_eq!(class_number(-20), 2);
    assert_eq!(class_number(-23), 3);
}

#[test]
#[verified_engine::verified]
fn test_weighted_class_number() {
    assert_eq!(weighted_class_number(-3), 1.0 / 3.0);
    assert_eq!(weighted_class_number(-4), 1.0 / 2.0);
    assert_eq!(weighted_class_number(-7), 1.0);
    assert_eq!(weighted_class_number(-12), 1.0); // h(-12) = 1
    assert_eq!(weighted_class_number(-15), 2.0);
}

#[test]
#[verified_engine::verified]
fn test_hurwitz_class_number() {
    // H(-1) = h_w(-1) = 0
    assert_eq!(hurwitz_class_number(-1), 0.0);
    // H(-3) = h_w(-3) = 1/3
    assert!(
        (hurwitz_class_number(-3) - 1.0 / 3.0).abs() < math_commons::registry::TOLERANCE_STANDARD
    );
    // H(-4) = h_w(-4) + h_w(-1) = 1/2 + 0 = 1/2
    assert!(
        (hurwitz_class_number(-4) - 1.0 / 2.0).abs() < math_commons::registry::TOLERANCE_STANDARD
    );
    // H(-7) = h_w(-7) = 1
    assert!((hurwitz_class_number(-7) - 1.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
    // H(-12) = h_w(-12) + h_w(-3) = 1 + 1/3 = 4/3
    assert!(
        (hurwitz_class_number(-12) - 4.0 / 3.0).abs() < math_commons::registry::TOLERANCE_STANDARD
    );
}

#[test]
#[verified_engine::verified]
fn test_summation_formula_example() {
    assert!(verify_summation_formula(5));
    assert!(verify_summation_formula(7));
    assert!(verify_summation_formula(11));
}

#[test]
#[verified_engine::verified]
fn test_summation_formula_large_scale() {
    let primes = pure_math::pure_math::number_theory::primes::primes_up_to(100);
    for p in primes {
        if p > 3 {
            assert!(verify_summation_formula(p), "Formula failed for p = {}", p);
        }
    }
}
