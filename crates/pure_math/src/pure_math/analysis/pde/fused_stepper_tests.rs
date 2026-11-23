#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_stencil_operators_precomputation() {
    let ops = StencilOperators::new_2d(StepSize(0.5), StepSize(0.25));
    assert!((ops.inv_dx - 2.0).abs() < 1e-12);
    assert!((ops.inv_two_dx - 1.0).abs() < 1e-12);
    assert!((ops.inv_dx_sq - 4.0).abs() < 1e-12);
    assert!((ops.inv_dy_sq - 16.0).abs() < 1e-12);
    assert!((ops.inv_dz_sq - 1.0).abs() < 1e-12);

    let ops_3d =
        StencilOperators::try_new_3d(StepSize(0.5), StepSize(0.25), StepSize(0.2)).unwrap();
    assert!((ops_3d.inv_dz_sq - 25.0).abs() < 1e-12);
    assert!(FusedStencilStepper::try_new_3d(StepSize(1.0), StepSize(1.0), StepSize(0.0)).is_err());

    let ops_1d = StencilOperators::new(StepSize(2.0));
    assert!((ops_1d.inv_dx - 0.5).abs() < 1e-12);
    assert!((ops_1d.inv_two_dx - 0.25).abs() < 1e-12);
    assert!((ops_1d.inv_dx_sq - 0.25).abs() < 1e-12);
    assert!((ops_1d.inv_dy_sq - 1.0).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "Step size dx must be positive and non-zero")]
fn test_stencil_operators_invalid_dx() {
    StencilOperators::new(StepSize(0.0));
}

#[test]
#[should_panic(expected = "Step size dy must be positive and non-zero")]
fn test_stencil_operators_invalid_dy() {
    StencilOperators::new_2d(StepSize(1.0), StepSize(-0.5));
}

#[test]
fn test_stencil_operator_evaluations() {
    let dx = 0.2;
    let dy = 0.4;
    let ops = StencilOperators::new_2d(StepSize(dx), StepSize(dy));

    let cd1 = ops.central_diff_1st(1.0, 3.0);
    let cd1_expected = (3.0 - 1.0) / (2.0 * dx);
    assert!((cd1 - cd1_expected).abs() < 1e-12);

    let cd2 = ops.central_diff_2nd(1.0, 4.0, 9.0);
    let cd2_expected = (9.0 - 2.0 * 4.0 + 1.0) / (dx * dx);
    assert!((cd2 - cd2_expected).abs() < 1e-12);

    let cd2_2d = ops.central_diff_2nd_2d(5.0, 2.0, 8.0, 3.0, 7.0);
    let d2x_expected = (8.0 - 2.0 * 5.0 + 2.0) / (dx * dx);
    let d2y_expected = (7.0 - 2.0 * 5.0 + 3.0) / (dy * dy);
    assert!((cd2_2d - (d2x_expected + d2y_expected)).abs() < 1e-12);

    let flux_pos = ops.upwind_flux(2.0, 1.0, 3.0, 5.0);
    let flux_pos_expected = (3.0 * 2.0 - 1.0 * 2.0) / dx;
    assert!((flux_pos - flux_pos_expected).abs() < 1e-12);

    let flux_neg = ops.upwind_flux(-2.0, 1.0, 3.0, 5.0);
    let flux_neg_expected = (5.0 * (-2.0) - 3.0 * (-2.0)) / dx;
    assert!((flux_neg - flux_neg_expected).abs() < 1e-12);
}
