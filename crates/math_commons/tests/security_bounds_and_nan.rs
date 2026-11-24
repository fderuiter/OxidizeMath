//! Tier 5 security tests for `math_commons`: `CapacityOverflow` diagnostics,
//! `NaN`/infinity rejection, subnormal stability, and zero-allocation error paths.

use diagnostics::Diagnostic;
use math_commons::error::MathError;
use math_commons::primitives::{NonNegativeFloat, PositiveFloat, UnitInterval};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static TEST_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn capacity_overflow_and_bounds_errors_are_zero_alloc_inside_run_verified() {
    let ((ovf, ui_nan, ui_inf, pf_zero, pf_neg_inf, nnf_nan), metrics) =
        VerifiedEngine::run_verified(|| {
            let ovf = usize::MAX
                .checked_mul(2)
                .ok_or(MathError::CapacityOverflow {
                    context: "security_bounds_and_nan usize::MAX * 2",
                });
            let ui_nan = UnitInterval::new(f64::NAN);
            let ui_inf = UnitInterval::new(f64::INFINITY);
            let pf_zero = PositiveFloat::new(0.0);
            let pf_neg_inf = PositiveFloat::new(f64::NEG_INFINITY);
            let nnf_nan = NonNegativeFloat::new(f64::NAN);
            (ovf, ui_nan, ui_inf, pf_zero, pf_neg_inf, nnf_nan)
        });

    assert_eq!(metrics.allocations, 0);
    assert_eq!(metrics.allocated_bytes, 0);

    let err = ovf.expect_err("usize::MAX * 2 must overflow");
    assert_eq!(
        err,
        MathError::CapacityOverflow {
            context: "security_bounds_and_nan usize::MAX * 2",
        }
    );
    assert!(err.to_string().contains("Capacity overflow"));
    assert_eq!(
        err.metadata().get("context").map(String::as_str),
        Some("security_bounds_and_nan usize::MAX * 2")
    );

    assert!(ui_nan.is_err());
    assert!(ui_inf.is_err());
    assert!(pf_zero.is_err());
    assert!(pf_neg_inf.is_err());
    assert!(nnf_nan.is_err());
}

#[test]
fn subnormal_floats_remain_finite_and_allocation_free() {
    let subnormal = f64::MIN_POSITIVE * 0.5;
    assert!(subnormal > 0.0 && subnormal < f64::MIN_POSITIVE);

    let ((ui_root, pf_scaled, nnf_val), metrics) = VerifiedEngine::run_verified(|| {
        let ui = UnitInterval::new(subnormal).expect("subnormal lies in [0, 1]");
        let pf = PositiveFloat::new(subnormal).expect("subnormal > 0");
        let nnf = NonNegativeFloat::new(subnormal).expect("subnormal >= 0");
        let ui_root = *ui.sqrt().expect("sqrt of subnormal in [0, 1]");
        let pf_scaled = *(pf * 4.0).expect("positive scaled subnormal");
        (ui_root, pf_scaled, *nnf)
    });

    assert_eq!(metrics.allocations, 0);
    assert_eq!(metrics.allocated_bytes, 0);
    assert!(ui_root.is_finite() && ui_root > 0.0);
    assert!(pf_scaled.is_finite() && pf_scaled > subnormal);
    assert_eq!(nnf_val, subnormal);
}
