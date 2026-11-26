//! Integration tests for 2D conservative upwind advection in `FiniteDifference2D`,
//! `MaskedFiniteDifference2D`, and `TuringSystem::step_advected`.

use domain_biology::biology::diffusion::{FiniteDifference2D, ReactionCellRole, SpatialDiffusion};
use domain_biology::biology::morphogenesis::{
    DiffusionCoeff, GrayScottKinetics, ReactionKinetics, SchnakenbergKinetics,
    SpatialSchnakenbergKinetics, TuringSystem,
};
use math_commons::math_kernel::types::{Dimension, StepSize, flatten_2d_index};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

struct ZeroKinetics;

impl ReactionKinetics<2> for ZeroKinetics {
    #[inline(always)]
    fn reaction(&self, _concentrations: [f64; 2]) -> [f64; 2] {
        [0.0, 0.0]
    }
}

fn sample_initial_fields(w: usize, h: usize) -> (Vec<f64>, Vec<f64>) {
    let n = w * h;
    let mut u0 = vec![0.0; n];
    let mut v0 = vec![0.0; n];
    for y in 0..h {
        for x in 0..w {
            let idx = flatten_2d_index(x, y, w);
            u0[idx] = 0.8 + 0.03 * (x as f64) + 0.01 * (y as f64);
            v0[idx] = 0.7 + 0.02 * ((w - x) as f64);
        }
    }
    (u0, v0)
}

#[test]
fn test_zero_velocity_reproduces_baseline_fd2d_within_1e12() {
    let (w, h) = (18, 14);
    let n = w * h;
    let base = FiniteDifference2D::new(Dimension(w), Dimension(h), StepSize(1.0), StepSize(1.0));
    let kinetics = SchnakenbergKinetics::new(0.12, 0.88);
    let (u0, v0) = sample_initial_fields(w, h);
    let zero_vel = vec![0.0; n];

    let (mut u_ref, mut v_ref) = (vec![0.0; n], vec![0.0; n]);
    let (mut u_adv, mut v_adv) = (vec![0.0; n], vec![0.0; n]);
    SpatialDiffusion::<2>::step_fused(
        &base,
        [&u0, &v0],
        [&mut u_ref, &mut v_ref],
        0.01,
        [1.0, 20.0],
        &kinetics,
    );
    base.step_advected_fused(
        [&u0, &v0],
        [&mut u_adv, &mut v_adv],
        &zero_vel,
        &zero_vel,
        0.01,
        [1.0, 20.0],
        &kinetics,
    );
    for i in 0..n {
        assert!((u_adv[i] - u_ref[i]).abs() < 1e-12);
        assert!((v_adv[i] - v_ref[i]).abs() < 1e-12);
    }
}

#[test]
fn test_zero_velocity_reproduces_masked_fd2d_within_1e12() {
    let (w, h) = (18, 14);
    let n = w * h;
    let base = FiniteDifference2D::new(Dimension(w), Dimension(h), StepSize(1.0), StepSize(1.0));
    let mut masked = base.with_mask();
    masked.mask.stamp_microfluidic_maze();
    let kinetics = SchnakenbergKinetics::new(0.12, 0.88);
    let (u0, v0) = sample_initial_fields(w, h);
    let zero_vel = vec![0.0; n];

    let (mut u_ref, mut v_ref) = (vec![0.0; n], vec![0.0; n]);
    let (mut u_adv, mut v_adv) = (vec![0.0; n], vec![0.0; n]);
    SpatialDiffusion::<2>::step_fused(
        &masked,
        [&u0, &v0],
        [&mut u_ref, &mut v_ref],
        0.01,
        [1.0, 20.0],
        &kinetics,
    );
    masked.step_advected_fused(
        [&u0, &v0],
        [&mut u_adv, &mut v_adv],
        &zero_vel,
        &zero_vel,
        0.01,
        [1.0, 20.0],
        &kinetics,
    );
    for i in 0..n {
        assert!((u_adv[i] - u_ref[i]).abs() < 1e-12);
        assert!((v_adv[i] - v_ref[i]).abs() < 1e-12);
    }
}

#[test]
fn test_uniform_ux_gaussian_drift_conserves_mass() {
    let (w, h) = (36, 18);
    let n = w * h;
    let diff = FiniteDifference2D::new(Dimension(w), Dimension(h), StepSize(1.0), StepSize(1.0));
    let mut sys = TuringSystem::new_with_kinetics(
        Dimension(n),
        DiffusionCoeff(0.02),
        DiffusionCoeff(0.01),
        ZeroKinetics,
        diff,
    );
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f64 - 9.0, y as f64 - 9.0);
            sys.u_mut()[flatten_2d_index(x, y, w)] = (-0.18 * (dx * dx + dy * dy)).exp();
        }
    }
    let (ux, uy) = (vec![0.14; n], vec![0.0; n]);
    let mass0: f64 = sys.u().iter().sum();
    let u0_slice = sys.u();
    let cx0: f64 = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x as f64) * u0_slice[flatten_2d_index(x, y, w)]))
        .sum::<f64>()
        / mass0;
    for _ in 0..35 {
        sys.step_advected(1.0, &ux, &uy);
    }
    let mass1: f64 = sys.u().iter().sum();
    let u1_slice = sys.u();
    let cx1: f64 = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x as f64) * u1_slice[flatten_2d_index(x, y, w)]))
        .sum::<f64>()
        / mass1;
    assert!((mass1 - mass0).abs() < 1e-10);
    assert!(cx1 > cx0 + 4.0);
}

#[test]
fn test_sealed_wall_blocks_advective_flux_with_zero_allocations() {
    let (w, h) = (36, 18);
    let n = w * h;
    let diff = FiniteDifference2D::new(Dimension(w), Dimension(h), StepSize(1.0), StepSize(1.0));
    let mut walled = diff.with_mask();
    let wall_x = w / 2;
    for y in 0..h {
        walled.mask.set(wall_x, y, ReactionCellRole::ZeroFluxWall);
    }
    let mut sys = TuringSystem::new_with_kinetics(
        Dimension(n),
        DiffusionCoeff(0.04),
        DiffusionCoeff(0.02),
        GrayScottKinetics::mitosis(),
        walled,
    );
    for y in 0..h {
        for x in 0..wall_x {
            sys.u_mut()[flatten_2d_index(x, y, w)] = 0.8;
            sys.v_mut()[flatten_2d_index(x, y, w)] = 0.3;
        }
    }
    let (ux, uy) = (vec![0.14; n], vec![0.0; n]);
    sys.step_advected(0.5, &ux, &uy);
    let ((), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..50 {
            sys.step_advected(0.5, &ux, &uy);
        }
    });
    assert!(metrics.function_calls > 0);
    for y in 0..h {
        assert_eq!(sys.u()[flatten_2d_index(wall_x, y, w)], 0.0);
        for x in (wall_x + 1)..w {
            assert!(sys.v()[flatten_2d_index(x, y, w)].abs() < 1e-12);
        }
    }
    assert!(sys.u().iter().all(|&c| c.is_finite() && c >= 0.0));
    assert!(sys.v().iter().all(|&c| c.is_finite() && c >= 0.0));
}

#[test]
fn test_advected_spatial_schnakenberg_mask_with_zero_allocations() {
    let (w, h) = (20, 16);
    let n = w * h;
    let mut kin = SpatialSchnakenbergKinetics::new(0.10, 0.90, n);
    for y in 0..h {
        for x in 0..(w / 2) {
            kin.set_cell_rates(flatten_2d_index(x, y, w), 0.16, 0.84);
        }
    }
    let diff = FiniteDifference2D::new(Dimension(w), Dimension(h), StepSize(1.0), StepSize(1.0))
        .with_mask();
    let mut sys = TuringSystem::new_with_kinetics(
        Dimension(n),
        DiffusionCoeff(0.2),
        DiffusionCoeff(2.0),
        kin,
        diff,
    );
    sys.u_mut().fill(1.0);
    sys.v_mut().fill(0.9);
    let (ux, uy) = (vec![0.08; n], vec![0.02; n]);
    sys.step_advected(0.02, &ux, &uy);
    let ((), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..10 {
            sys.step_advected(0.02, &ux, &uy);
        }
    });
    assert!(metrics.function_calls > 0);
    let left_u = sys.u()[flatten_2d_index(3, h / 2, w)];
    let right_u = sys.u()[flatten_2d_index(w - 4, h / 2, w)];
    assert!((left_u - right_u).abs() > 1e-4);
}
