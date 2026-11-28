//! Verifies typed `MathError` bounds diagnostics and zero-allocation failure paths
//! on `UnitInterval`, `PositiveFloat`, and `NonNegativeFloat`.

use diagnostics::{Diagnostic, DiagnosticVisitor};
use math_commons::error::MathError;
use math_commons::math_kernel::types::{Dimension, ScalarWorkspace};
use math_commons::primitives::{NonNegativeFloat, PositiveFloat, UnitInterval};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

struct CountingVisitor(usize);

impl DiagnosticVisitor for CountingVisitor {
    fn record_str(&mut self, _key: &str, _value: &str) {
        self.0 += 1;
    }

    fn record_display(&mut self, _key: &str, _value: &dyn core::fmt::Display) {
        self.0 += 1;
    }
}

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

#[test]
fn solver_workspace_try_take_slice_returns_typed_error_without_allocating() {
    let mut backing = [0.0_f64; 6];
    let (ok, _) = VerifiedEngine::run_verified(|| {
        let mut ws = ScalarWorkspace::new(&mut backing);
        let first = ws.try_take_slice(4).expect("first slice fits");
        first[0] = 3.5;
        let err = ws
            .try_take_slice(4)
            .expect_err("overflow slice must return WorkspaceTooSmall");
        assert_eq!(
            err,
            MathError::WorkspaceTooSmall {
                required: 4,
                available: 2,
            }
        );
        let mut v = CountingVisitor(0);
        err.visit_fields(&mut v);
        assert_eq!(v.0, 5);

        let tail = ws
            .try_take_slice(2)
            .expect("remaining capacity preserved after failed try_take_slice");
        tail[1] = 1.5;
        first[0] + tail[1] == 5.0
    });
    assert!(ok);

    let ws_err = MathError::WorkspaceTooSmall {
        required: 8,
        available: 3,
    };
    assert_eq!(
        ws_err.to_string(),
        "Solver workspace too small: required 8 elements, available 3"
    );
    assert_eq!(
        ws_err.metadata().get("required").map(String::as_str),
        Some("8")
    );
    assert_eq!(
        ws_err.metadata().get("available").map(String::as_str),
        Some("3")
    );
}
