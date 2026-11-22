use super::*;
use crate::biology::diffusion::fd2d::boundary::{BoundaryStrategy, PeriodicBoundary};
use math_commons::math_kernel::types::{Dimension, StepSize, flatten_2d_index};

#[test]
#[verified_engine::verified]
fn test_laplacian_uniform() {
    let width = 10;
    let height = 10;
    let diff = FiniteDifference2D::new(
        Dimension(width),
        Dimension(height),
        StepSize(1.0),
        StepSize(1.0),
    );

    let n = width * height;
    let u = vec![1.0; n];
    let v = vec![2.0; n];
    let mut out_u = vec![0.0; n];
    let mut out_v = vec![0.0; n];

    diff.apply(
        [u.as_slice(), v.as_slice()],
        [out_u.as_mut_slice(), out_v.as_mut_slice()],
        [1.0, 1.0],
    );

    for val in out_u {
        assert_eq!(val, 0.0);
    }
    for val in out_v {
        assert_eq!(val, 0.0);
    }
}

#[test]
#[verified_engine::verified]
fn test_laplacian_parabolic() {
    let width = 5;
    let height = 5;
    let diff = FiniteDifference2D::new(
        Dimension(width),
        Dimension(height),
        StepSize(1.0),
        StepSize(1.0),
    );

    let n = width * height;
    let mut u = vec![0.0; n];
    let v = vec![0.0; n];
    let mut out_u = vec![0.0; n];
    let mut out_v = vec![0.0; n];

    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            u[idx] = (x as f64).powi(2) + (y as f64).powi(2);
        }
    }

    diff.apply(
        [u.as_slice(), v.as_slice()],
        [out_u.as_mut_slice(), out_v.as_mut_slice()],
        [1.0, 1.0],
    );

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = flatten_2d_index(x, y, width);
            assert!(
                (out_u[idx] - 4.0).abs() < math_commons::registry::TOLERANCE_HIGH,
                "Failed at ({}, {}): {}",
                x,
                y,
                out_u[idx]
            );
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_map_diffusion_equivalence() {
    let width = 5;
    let height = 5;
    let diff = FiniteDifference2D::new(
        Dimension(width),
        Dimension(height),
        StepSize(1.0),
        StepSize(1.0),
    );

    let n = width * height;
    let mut u = vec![0.0; n];
    let mut v = vec![0.0; n];
    for i in 0..n {
        u[i] = (i as f64) * 0.1;
        v[i] = (n - i) as f64 * 0.1;
    }

    let mut out_u_1 = vec![0.0; n];
    let mut out_v_1 = vec![0.0; n];
    let mut out_u_2 = vec![0.0; n];
    let mut out_v_2 = vec![0.0; n];

    let dt = 0.01;
    let d_u = 0.5;
    let d_v = 0.1;

    diff.apply(
        [u.as_slice(), v.as_slice()],
        [out_u_1.as_mut_slice(), out_v_1.as_mut_slice()],
        [d_u, d_v],
    );
    for i in 0..n {
        out_u_1[i] = u[i] + dt * (out_u_1[i] + 1.0);
        out_v_1[i] = v[i] + dt * (out_v_1[i] + 2.0);
    }

    diff.map_diffusion(
        [u.as_slice(), v.as_slice()],
        [d_u, d_v],
        |i, vals, diffs| {
            out_u_2[i] = vals[0] + dt * (diffs[0] + 1.0);
            out_v_2[i] = vals[1] + dt * (diffs[1] + 2.0);
        },
    );

    for i in 0..n {
        assert!((out_u_1[i] - out_u_2[i]).abs() < math_commons::registry::TOLERANCE_HIGH);
        assert!((out_v_1[i] - out_v_2[i]).abs() < math_commons::registry::TOLERANCE_HIGH);
    }
}

#[test]
#[verified_engine::verified]
fn test_diffusion_model_2d() {
    use crate::biology::reaction_diffusion::{ChemicalState, DiffusionModel};

    let width = 5;
    let height = 5;
    let diff = FiniteDifference2D::new(
        Dimension(width),
        Dimension(height),
        StepSize(1.0),
        StepSize(1.0),
    );

    let n = width * height;
    let mut state = ChemicalState::new(2, n);
    let mut out = ChemicalState::new(2, n);

    let center_idx = flatten_2d_index(2, 2, width);
    state.species_mut(0)[center_idx] = 1.0;

    let coeffs = [0.1, 0.2];
    DiffusionModel::apply(&diff, &state, &mut out, &coeffs);

    let expected_center = -4.0 * coeffs[0];
    let val = out.species(0)[center_idx];
    assert!((val - expected_center).abs() < math_commons::registry::TOLERANCE_HIGH);

    let neighbor_idx = flatten_2d_index(3, 2, width);
    let expected_neighbor = coeffs[0];
    let val_neighbor = out.species(0)[neighbor_idx];
    assert!((val_neighbor - expected_neighbor).abs() < math_commons::registry::TOLERANCE_HIGH);
    assert_eq!(out.species(1)[center_idx], 0.0);
}

#[test]
#[verified_engine::verified]
fn test_periodic_boundary_wraps_opposite_edges_and_conserves_mass() {
    let width = 8;
    let height = 8;
    let diff = FiniteDifference2D::new_periodic(
        Dimension(width),
        Dimension(height),
        StepSize(1.0),
        StepSize(1.0),
    );
    assert!(diff.is_periodic());

    let (l, r, u_idx, d_idx) = PeriodicBoundary.neighbors(0, 0, &diff.geometry);
    assert_eq!(*l, flatten_2d_index(width - 1, 0, width));
    assert_eq!(*r, flatten_2d_index(1, 0, width));
    assert_eq!(*u_idx, flatten_2d_index(0, height - 1, width));
    assert_eq!(*d_idx, flatten_2d_index(0, 1, width));

    let n = width * height;
    let mut u = vec![0.1; n];
    let mut v = vec![0.2; n];
    u[flatten_2d_index(0, 0, width)] = 2.5;
    v[flatten_2d_index(width - 1, height - 1, width)] = 1.7;

    let mut out_u = vec![0.0; n];
    let mut out_v = vec![0.0; n];
    diff.apply(
        [u.as_slice(), v.as_slice()],
        [out_u.as_mut_slice(), out_v.as_mut_slice()],
        [0.8, 0.4],
    );

    // Sum of discrete periodic Laplacian over a closed torus must equal zero
    let laplacian_sum_u: f64 = out_u.iter().sum();
    let laplacian_sum_v: f64 = out_v.iter().sum();
    assert!(laplacian_sum_u.abs() < 1e-12);
    assert!(laplacian_sum_v.abs() < 1e-12);
    assert!(
        (out_u[flatten_2d_index(width - 1, 0, width)] - out_u[flatten_2d_index(1, 0, width)]).abs()
            < 1e-12
    );
}
