#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_stencil_operators_precomputation() {
    let ops = StencilOperators::new_2d(StepSize(0.5), StepSize(0.25));
    assert!((ops.inv_dx - 2.0).abs() < 1e-12);
    assert!((ops.inv_dy - 4.0).abs() < 1e-12);
    assert!((ops.inv_two_dx - 1.0).abs() < 1e-12);
    assert!((ops.inv_two_dy - 2.0).abs() < 1e-12);
    assert!((ops.inv_dx_sq - 4.0).abs() < 1e-12);
    assert!((ops.inv_dy_sq - 16.0).abs() < 1e-12);

    let ops_1d = StencilOperators::new(StepSize(2.0));
    assert!((ops_1d.inv_dx - 0.5).abs() < 1e-12);
    assert!((ops_1d.inv_dy - 1.0).abs() < 1e-12);
    assert!((ops_1d.inv_two_dx - 0.25).abs() < 1e-12);
    assert!((ops_1d.inv_two_dy - 0.5).abs() < 1e-12);
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
    assert!((cd1 - (3.0 - 1.0) / (2.0 * dx)).abs() < 1e-12);

    let cd1_y = ops.central_diff_1st_y(2.0, 6.0);
    assert!((cd1_y - (6.0 - 2.0) / (2.0 * dy)).abs() < 1e-12);

    let cd2 = ops.central_diff_2nd(1.0, 4.0, 9.0);
    assert!((cd2 - (9.0 - 2.0 * 4.0 + 1.0) / (dx * dx)).abs() < 1e-12);

    let cd2_2d = ops.central_diff_2nd_2d(5.0, 2.0, 8.0, 3.0, 7.0);
    let d2x_expected = (8.0 - 2.0 * 5.0 + 2.0) / (dx * dx);
    let d2y_expected = (7.0 - 2.0 * 5.0 + 3.0) / (dy * dy);
    assert!((cd2_2d - (d2x_expected + d2y_expected)).abs() < 1e-12);

    let flux_pos = ops.upwind_flux(2.0, 1.0, 3.0, 5.0);
    assert!((flux_pos - (3.0 * 2.0 - 1.0 * 2.0) / dx).abs() < 1e-12);

    let flux_neg = ops.upwind_flux(-2.0, 1.0, 3.0, 5.0);
    assert!((flux_neg - (5.0 * (-2.0) - 3.0 * (-2.0)) / dx).abs() < 1e-12);

    let flux_2d = ops.upwind_flux_2d(1.5, -0.5, 4.0, 1.0, 6.0, 2.0, 8.0);
    let expected_2d = (4.0 * 1.5 - 1.0 * 1.5) / dx + (8.0 * (-0.5) - 4.0 * (-0.5)) / dy;
    assert!((flux_2d - expected_2d).abs() < 1e-12);

    let div_2d = ops.upwind_divergence_2d([1.5, 1.5, -0.5, -0.5], 4.0, 1.0, 6.0, 2.0, 8.0);
    assert!((div_2d - expected_2d).abs() < 1e-12);
}

#[test]
fn test_reflect_2d_neighbors_and_zero_normal_obstacle_faces() {
    let width = 4;
    let height = 4;
    let mut obstacles = vec![false; width * height];
    let obs_idx = flatten_2d_index(2, 1, width);
    obstacles[obs_idx] = true;

    let (idx, idx_l, idx_r, idx_u, idx_d) = reflect_2d_neighbors(1, 1, width, height, &obstacles);
    assert_eq!(idx, flatten_2d_index(1, 1, width));
    assert_eq!(idx_l, flatten_2d_index(0, 1, width));
    assert_eq!(
        idx_r, idx,
        "Obstacle at (2,1) must reflect East neighbor back to idx"
    );
    assert_eq!(idx_u, flatten_2d_index(1, 0, width));
    assert_eq!(idx_d, flatten_2d_index(1, 2, width));

    let vx = vec![1.0; width * height];
    let vy = vec![-0.5; width * height];
    let face_vel = advection_face_velocities(idx, idx_l, idx_r, idx_u, idx_d, &vx, &vy);
    assert!((face_vel[0] - 1.0).abs() < 1e-12);
    assert_eq!(
        face_vel[1], 0.0,
        "Reflected East obstacle face must have zero normal flux"
    );
    assert!((face_vel[2] + 0.5).abs() < 1e-12);
    assert!((face_vel[3] + 0.5).abs() < 1e-12);
}
