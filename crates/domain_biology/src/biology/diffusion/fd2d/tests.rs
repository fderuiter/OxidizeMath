use super::*;
use crate::biology::morphogenesis::{
    DiffusionCoeff, GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
use crate::biology::reaction_diffusion::ChemicalState;
use math_commons::math_kernel::types::{Dimension, StepSize, flatten_2d_index};

#[test]
#[verified_engine::verified]
fn test_laplacian_uniform() {
    let (width, height) = (10, 10);
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
    assert!(out_u.iter().all(|&val| val == 0.0));
    assert!(out_v.iter().all(|&val| val == 0.0));
}

#[test]
#[verified_engine::verified]
fn test_laplacian_parabolic() {
    let (width, height) = (5, 5);
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
            u[flatten_2d_index(x, y, width)] = (x as f64).powi(2) + (y as f64).powi(2);
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
            assert!((out_u[idx] - 4.0).abs() < math_commons::registry::TOLERANCE_HIGH);
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_map_diffusion_equivalence() {
    let (width, height) = (5, 5);
    let diff = FiniteDifference2D::new(
        Dimension(width),
        Dimension(height),
        StepSize(1.0),
        StepSize(1.0),
    );
    let n = width * height;
    let u: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
    let v: Vec<f64> = (0..n).map(|i| ((n - i) as f64) * 0.1).collect();
    let mut out_u_1 = vec![0.0; n];
    let mut out_v_1 = vec![0.0; n];
    let mut out_u_2 = vec![0.0; n];
    let mut out_v_2 = vec![0.0; n];
    let (dt, d_u, d_v) = (0.01, 0.5, 0.1);

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
    use crate::biology::reaction_diffusion::DiffusionModel;

    let (width, height) = (5, 5);
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
    assert!(
        (out.species(0)[center_idx] - (-4.0 * coeffs[0])).abs()
            < math_commons::registry::TOLERANCE_HIGH
    );
    let neighbor_idx = flatten_2d_index(3, 2, width);
    assert!(
        (out.species(0)[neighbor_idx] - coeffs[0]).abs() < math_commons::registry::TOLERANCE_HIGH
    );
    assert_eq!(out.species(1)[center_idx], 0.0);
}

#[test]
#[verified_engine::verified]
fn test_periodic_laplacian_symmetry_and_zero_mass() {
    let (w, h) = (8, 8);
    let diff =
        FiniteDifference2D::new_periodic(Dimension(w), Dimension(h), StepSize(1.0), StepSize(1.0));
    let n = w * h;
    let mut u_corner = vec![0.0; n];
    let mut u_interior = vec![0.0; n];
    let v = vec![0.0; n];
    u_corner[flatten_2d_index(0, 0, w)] = 1.0;
    u_interior[flatten_2d_index(4, 4, w)] = 1.0;

    let mut out_corner = vec![0.0; n];
    let mut out_interior = vec![0.0; n];
    let mut scratch_v = vec![0.0; n];

    diff.apply(
        [u_corner.as_slice(), v.as_slice()],
        [out_corner.as_mut_slice(), scratch_v.as_mut_slice()],
        [0.5, 0.5],
    );
    diff.apply(
        [u_interior.as_slice(), v.as_slice()],
        [out_interior.as_mut_slice(), scratch_v.as_mut_slice()],
        [0.5, 0.5],
    );

    assert!((out_corner.iter().sum::<f64>()).abs() < 1e-13);
    for dy in 0..h {
        for dx in 0..w {
            let c_val = out_corner[flatten_2d_index(dx, dy, w)];
            let i_val = out_interior[flatten_2d_index((4 + dx) % w, (4 + dy) % h, w)];
            assert!((c_val - i_val).abs() < 1e-13);
        }
    }
}

#[test]
fn test_periodic_schnakenberg_and_gray_scott_seam_wrap_zero_alloc() {
    let (w, h) = (12, 12);
    let fd_periodic =
        FiniteDifference2D::new_periodic(Dimension(w), Dimension(h), StepSize(1.0), StepSize(1.0))
            .with_mask();
    let mut sys_gs = TuringSystem::new_with_kinetics(
        Dimension(w * h),
        DiffusionCoeff(0.16),
        DiffusionCoeff(0.08),
        GrayScottKinetics::mitosis(),
        fd_periodic.clone(),
    );
    sys_gs.u_mut().fill(1.0);
    sys_gs.v_mut().fill(0.0);
    sys_gs.v_mut()[flatten_2d_index(0, 0, w)] = 0.8;

    let mut sys_sch = TuringSystem::new_with_kinetics(
        Dimension(w * h),
        DiffusionCoeff(1.0),
        DiffusionCoeff(20.0),
        SchnakenbergKinetics::new(0.1, 0.9),
        fd_periodic,
    );
    sys_sch.u_mut().fill(1.0);
    sys_sch.v_mut().fill(0.9);
    sys_sch.u_mut()[flatten_2d_index(0, 0, w)] = 1.5;

    let _ = verified_engine::engine::VerifiedEngine::run_verified(|| {
        sys_gs.step(0.5);
        sys_sch.step(0.005);
    });

    assert!(
        (sys_gs.v()[flatten_2d_index(1, 0, w)] - sys_gs.v()[flatten_2d_index(w - 1, 0, w)]).abs()
            < 1e-13
    );
    assert!(
        (sys_gs.v()[flatten_2d_index(0, 1, w)] - sys_gs.v()[flatten_2d_index(0, h - 1, w)]).abs()
            < 1e-13
    );
    assert!(
        (sys_sch.u()[flatten_2d_index(1, 0, w)] - sys_sch.u()[flatten_2d_index(w - 1, 0, w)]).abs()
            < 1e-13
    );
    assert!(
        (sys_sch.u()[flatten_2d_index(0, 1, w)] - sys_sch.u()[flatten_2d_index(0, h - 1, w)]).abs()
            < 1e-13
    );
}
// [cite:essay]
