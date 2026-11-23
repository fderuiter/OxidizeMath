#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_stencil_operators_precomputation() {
    let ops = StencilOperators::new_2d(StepSize(0.5), StepSize(0.25));
    assert!((ops.inv_dx - 2.0).abs() < 1e-12);
    assert!((ops.inv_two_dx - 1.0).abs() < 1e-12);
    assert!((ops.inv_dx_sq - 4.0).abs() < 1e-12);
    assert!((ops.inv_dy_sq - 16.0).abs() < 1e-12);

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

#[test]
fn test_step_2d_coupled_periodic_wraps_opposite_edges_zero_alloc() {
    let (w, h) = (6, 6);
    let stepper = FusedStencilStepper::new_2d(StepSize(1.0), StepSize(1.0));
    let mut u = vec![0.0; w * h];
    let mut v = vec![0.0; w * h];
    u[flatten_2d_index(0, 0, w)] = 1.0;
    v[flatten_2d_index(w - 1, h - 1, w)] = 2.0;

    let mut next_u = vec![0.0; w * h];
    let mut next_v = vec![0.0; w * h];

    let _ = verified_engine::engine::VerifiedEngine::run_verified(|| {
        stepper.step_2d_coupled_periodic(
            (w, h),
            [u.as_slice(), v.as_slice()],
            [next_u.as_mut_slice(), next_v.as_mut_slice()],
            0.1,
            1.0,
            |_idx, curr, left, right, up, down, ops| {
                [
                    ops.central_diff_2nd_2d(curr[0], left[0], right[0], up[0], down[0]),
                    ops.central_diff_2nd_2d(curr[1], left[1], right[1], up[1], down[1]),
                ]
            },
        );
    });

    assert!((next_u[flatten_2d_index(0, 0, w)] - 0.6).abs() < 1e-12);
    assert!((next_u[flatten_2d_index(w - 1, 0, w)] - 0.1).abs() < 1e-12);
    assert!((next_u[flatten_2d_index(0, h - 1, w)] - 0.1).abs() < 1e-12);
    assert!((next_v[flatten_2d_index(0, h - 1, w)] - 0.2).abs() < 1e-12);
    assert!((next_v[flatten_2d_index(w - 1, 0, w)] - 0.2).abs() < 1e-12);
    let sum_u: f64 = next_u.iter().sum();
    assert!((sum_u - 1.0).abs() < 1e-12);
}
