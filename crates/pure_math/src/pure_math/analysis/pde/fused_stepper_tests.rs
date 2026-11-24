#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use math_commons::math_kernel::types::{StepSize, flatten_2d_index};

#[test]
fn test_stencil_operators_precomputation() {
    let ops = StencilOperators::new_2d(StepSize(0.5), StepSize(0.25));
    assert!((ops.inv_dx - 2.0).abs() < 1e-12);
    assert!((ops.inv_dy - 4.0).abs() < 1e-12);
    assert!((ops.inv_two_dx - 1.0).abs() < 1e-12);
    assert!((ops.inv_two_dy - 2.0).abs() < 1e-12);
    assert!((ops.inv_dx_sq - 4.0).abs() < 1e-12);
    assert!((ops.inv_dy_sq - 16.0).abs() < 1e-12);
    assert!((ops.inv_dz_sq - 1.0).abs() < 1e-12);

    let ops_3d =
        StencilOperators::try_new_3d(StepSize(0.5), StepSize(0.25), StepSize(0.2)).unwrap();
    assert!((ops_3d.inv_dz_sq - 25.0).abs() < 1e-12);
    assert!(FusedStencilStepper::try_new_3d(StepSize(1.0), StepSize(1.0), StepSize(0.0)).is_err());

    let ops_1d = StencilOperators::new(StepSize(2.0));
    assert!((ops_1d.inv_dx - 0.5).abs() < 1e-12);
    assert!((ops_1d.inv_dy - 1.0).abs() < 1e-12);
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
fn test_upwind_flux_2d_and_divergence_equivalence_on_uniform_flow() {
    let dx = 0.5;
    let dy = 0.25;
    let ops = StencilOperators::new_2d(StepSize(dx), StepSize(dy));

    let curr = 4.0;
    let left = 1.5;
    let right = 7.0;
    let up = 2.0;
    let down = 9.0;

    for &(vx, vy) in &[
        (1.2, 0.8),
        (1.2, -0.8),
        (-1.2, 0.8),
        (-1.2, -0.8),
        (0.0, 0.0),
    ] {
        let dir_upwind = ops.upwind_flux_2d(vx, vy, curr, left, right, up, down);
        let div_upwind = ops.upwind_divergence_2d(curr, [left, right, up, down], [vx, vx, vy, vy]);
        assert!(
            (dir_upwind - div_upwind).abs() < 1e-12,
            "Mismatch for (vx={vx}, vy={vy}): {dir_upwind} vs {div_upwind}"
        );
    }
}

fn fluid_sum(vals: &[f64], solid: &[bool]) -> f64 {
    vals.iter()
        .zip(solid.iter())
        .filter(|&(_, &is_obs)| !is_obs)
        .map(|(&val, _)| val)
        .sum()
}

fn seed_masked_pulse(width: usize, height: usize) -> (Vec<bool>, Vec<f64>, Vec<f64>) {
    let total = width * height;
    let mut solid = vec![false; total];
    for y in 3..7 {
        for x in 5..7 {
            solid[flatten_2d_index(x, y, width)] = true;
        }
    }
    let mut u = vec![0.0; total];
    let mut v = vec![0.0; total];
    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            if solid[idx] {
                u[idx] = 42.0;
                v[idx] = -7.0;
            } else {
                let (dx, dy) = (x as f64 - 2.5, y as f64 - 4.5);
                u[idx] = (-0.35 * (dx * dx + dy * dy)).exp();
                v[idx] = 1.0 - 0.5 * u[idx];
            }
        }
    }
    (solid, u, v)
}

fn advect_diffuse_rhs(
    curr: [f64; 2],
    nb: StencilNeighbors2D<2>,
    ops: &StencilOperators,
    vel: (f64, f64),
    diff: f64,
) -> [f64; 2] {
    let (vx, vy) = vel;
    let face_vel = [
        if nb.face_open[0] { vx } else { 0.0 },
        if nb.face_open[1] { vx } else { 0.0 },
        if nb.face_open[2] { vy } else { 0.0 },
        if nb.face_open[3] { vy } else { 0.0 },
    ];
    let mut rhs = [0.0; 2];
    for s in 0..2 {
        let lap = ops.central_diff_2nd_2d(curr[s], nb.left[s], nb.right[s], nb.up[s], nb.down[s]);
        let adv = ops.upwind_divergence_2d(
            curr[s],
            [nb.left[s], nb.right[s], nb.up[s], nb.down[s]],
            face_vel,
        );
        rhs[s] = diff * lap - adv;
    }
    rhs
}

#[test]
fn test_masked_neumann_upwind_advection_and_zero_flux_obstacle_conservation() {
    let (width, height) = (12_usize, 10_usize);
    let stepper = FusedStencilStepper::new_2d(StepSize(1.0), StepSize(1.0));
    let (solid, mut u, mut v) = seed_masked_pulse(width, height);

    let initial_fluid_u = fluid_sum(&u, &solid);
    let initial_fluid_v = fluid_sum(&v, &solid);
    let mut next_u = vec![0.0; width * height];
    let mut next_v = vec![0.0; width * height];

    for _ in 0..80 {
        stepper.step_2d_coupled_masked_neumann(
            (width, height),
            &solid,
            [&u, &v],
            [&mut next_u, &mut next_v],
            0.2,
            1.0,
            |_idx, curr, nb, ops| advect_diffuse_rhs(curr, nb, ops, (0.35, -0.15), 0.08),
        );
        std::mem::swap(&mut u, &mut next_u);
        std::mem::swap(&mut v, &mut next_v);
    }

    assert!((fluid_sum(&u, &solid) - initial_fluid_u).abs() < 1e-11);
    assert!((fluid_sum(&v, &solid) - initial_fluid_v).abs() < 1e-11);
    for (idx, &is_obs) in solid.iter().enumerate() {
        if is_obs {
            assert_eq!((u[idx], v[idx]), (42.0, -7.0));
        } else {
            assert!(u[idx] >= -1e-14);
        }
    }
}
