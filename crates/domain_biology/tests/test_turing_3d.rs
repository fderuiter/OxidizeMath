#![allow(missing_docs)]
use domain_biology::biology::diffusion::FiniteDifference3D;
use domain_biology::biology::morphogenesis::{
    DiffusionCoeff, ReactionKinetics, SchnakenbergKinetics, TuringSystem,
};
use math_commons::math_kernel::types::{Dimension, StepSize, flatten_3d_index};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

struct ZeroKinetics;

impl ReactionKinetics<2> for ZeroKinetics {
    #[verified_engine::verified]
    fn reaction(&self, _state: [f64; 2]) -> [f64; 2] {
        [0.0, 0.0]
    }
}

#[test]
#[verified_engine::verified]
fn test_finite_difference_3d_neumann_mass_conservation() {
    let dim = 10;
    let diff = FiniteDifference3D::isotropic(
        Dimension(dim),
        Dimension(dim),
        Dimension(dim),
        StepSize(1.0),
    );
    let mut sys = TuringSystem::new_with_kinetics(
        Dimension(diff.grid_len()),
        DiffusionCoeff(0.2),
        DiffusionCoeff(0.8),
        ZeroKinetics,
        diff,
    );
    let center = flatten_3d_index(4, 5, 3, dim, dim);
    sys.u_mut()[center] = 12.0;
    sys.v_mut()[0] = 6.0;
    let u0: f64 = sys.u().iter().sum();
    let v0: f64 = sys.v().iter().sum();

    for _ in 0..10 {
        sys.step(0.05);
    }

    let u1: f64 = sys.u().iter().sum();
    let v1: f64 = sys.v().iter().sum();
    assert!((u0 - u1).abs() < 1e-10);
    assert!((v0 - v1).abs() < 1e-10);
}

#[test]
fn test_turing_system_3d_schnakenberg_zero_allocation_hot_path() {
    let dim = 18;
    let kinetics = SchnakenbergKinetics::new(0.1, 0.9);
    let mut sys = TuringSystem::new_3d(
        (Dimension(dim), Dimension(dim), Dimension(dim)),
        DiffusionCoeff(1.0),
        DiffusionCoeff(10.0),
        StepSize(1.0),
        kinetics,
    );
    let u_ss = 1.0;
    let v_ss = 0.9;
    sys.u_mut().fill(u_ss);
    sys.v_mut().fill(v_ss);
    sys.u_mut()[flatten_3d_index(9, 9, 9, dim, dim)] = u_ss + 0.35;

    let mut out_voxels = vec![0.0_f32; dim * dim * dim];

    VerifiedEngine::run_verified(|| {
        for _ in 0..4 {
            sys.step(0.005);
            sys.write_activator_f32_into(&mut out_voxels);
        }
    });

    assert!(out_voxels.iter().all(|v| v.is_finite() && *v > 0.0));
}
