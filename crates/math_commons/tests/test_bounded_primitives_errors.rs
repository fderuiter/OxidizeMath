//! Verifies typed `MathError` bounds diagnostics and zero-allocation failure paths
//! on `UnitInterval`, `PositiveFloat`, and `NonNegativeFloat`.

use diagnostics::Diagnostic;
use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;
use math_commons::primitives::{NonNegativeFloat, PositiveFloat, UnitInterval};
use verified_engine::engine::VerifiedEngine;

#[test]
fn unit_interval_bounds_methods_and_operators_return_math_error() {
    assert_eq!(
        UnitInterval::new(-0.1),
        Err(MathError::OutOfBounds {
            value: -0.1,
            min: 0.0,
            max: 1.0,
        })
    );
    assert!(matches!(
        UnitInterval::new(f64::NAN),
        Err(MathError::OutOfBounds { .. })
    ));

    let a = UnitInterval::new(0.5).expect("0.5 in [0, 1]");
    assert_eq!(
        a * 3.0,
        Err(MathError::OutOfBounds {
            value: 1.5,
            min: 0.0,
            max: 1.0,
        })
    );
    assert!(matches!(3.0 * a, Err(MathError::OutOfBounds { .. })));
    assert!(matches!(a / 0.1, Err(MathError::OutOfBounds { .. })));
    assert!(matches!(a.ln(), Err(MathError::OutOfBounds { .. })));
    assert!(matches!(a.exp(), Err(MathError::OutOfBounds { .. })));
    assert!(matches!(a.powf(-2.0), Err(MathError::OutOfBounds { .. })));
    assert!(a.sqrt().is_ok());
    assert!(a.abs().is_ok());
}

#[test]
fn positive_and_non_negative_float_return_typed_bounds_errors() {
    assert_eq!(
        PositiveFloat::new(0.0),
        Err(MathError::StrictlyOutOfBounds {
            value: 0.0,
            min: 0.0,
            max: f64::INFINITY,
        })
    );
    let p = PositiveFloat::new(1.0).expect("1.0 > 0");
    assert_eq!(
        p * -2.0,
        Err(MathError::StrictlyOutOfBounds {
            value: -2.0,
            min: 0.0,
            max: f64::INFINITY,
        })
    );
    assert!(matches!(
        0.0 * p,
        Err(MathError::StrictlyOutOfBounds { .. })
    ));
    assert!(matches!(
        p / -1.0,
        Err(MathError::StrictlyOutOfBounds { .. })
    ));
    assert!(matches!(p.ln(), Err(MathError::StrictlyOutOfBounds { .. })));

    assert_eq!(
        NonNegativeFloat::new(-0.5),
        Err(MathError::OutOfBounds {
            value: -0.5,
            min: 0.0,
            max: f64::INFINITY,
        })
    );
    let n = NonNegativeFloat::new(2.0).expect("2.0 >= 0");
    assert!(matches!(n * -1.0, Err(MathError::OutOfBounds { .. })));
    assert!(matches!(n / 0.0, Err(MathError::OutOfBounds { .. })));
}

#[test]
fn dimension_mismatch_display_omits_debug_wrappers_and_failures_do_not_allocate() {
    let err = MathError::DimensionMismatch {
        expected: Dimension(4),
        actual: Dimension(2),
    };
    assert_eq!(err.to_string(), "Dimension mismatch: expected 4, actual 2");
    assert_eq!(
        err.metadata().get("expected_dim").map(String::as_str),
        Some("4")
    );

    let (ok, _) = VerifiedEngine::run_verified(|| {
        let e1 = UnitInterval::new(2.0).unwrap_err();
        let e2 = PositiveFloat::new(0.0).unwrap_err();
        let e3 = NonNegativeFloat::new(-1.0).unwrap_err();
        matches!(e1, MathError::OutOfBounds { .. })
            && matches!(e2, MathError::StrictlyOutOfBounds { .. })
            && matches!(e3, MathError::OutOfBounds { .. })
    });
    assert!(ok);
}
