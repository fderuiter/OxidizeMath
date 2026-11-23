#![allow(missing_docs)]
use approx::assert_relative_eq;
use domain_physics::physics::quantum::{
    HydrogenicSuperposition3D, OrbitalQuantumNumbers, QuantumFieldObservable,
    hydrogenic_orbital_cartesian, hydrogenic_radial_wavefunction,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_hydrogenic_radial_normalization_and_nodes() {
    // Verify radial integral int_0^inf |R_{nl}(r)|^2 r^2 dr = 1 for 1s, 2s, 2p, 3s, 3d
    for &(n, l) in &[(1, 0), (2, 0), (2, 1), (3, 0), (3, 2)] {
        let r_max = 45.0;
        let steps = 4500;
        let dr = r_max / (steps as f64);
        let mut norm = 0.0;
        for i in 0..steps {
            let r = (i as f64 + 0.5) * dr;
            let val = hydrogenic_radial_wavefunction(n, l, r, 1.0);
            norm += val * val * r * r * dr;
        }
        assert_relative_eq!(norm, 1.0, epsilon = 1e-3);
    }

    // 2s radial node at r = 2 * a0
    let r20_node = hydrogenic_radial_wavefunction(2, 0, 2.0, 1.0);
    assert!(r20_node.abs() < 1e-12, "2s must vanish at r = 2 a0");
    assert!(
        hydrogenic_radial_wavefunction(2, 0, 1.0, 1.0)
            * hydrogenic_radial_wavefunction(2, 0, 3.0, 1.0)
            < 0.0,
        "2s must change sign across r = 2 a0"
    );

    // 3s has two radial nodes at r = (9 +- 3 sqrt(3)) / 2 * a0
    let n1 = 0.5 * (9.0 - 3.0 * 3.0_f64.sqrt());
    let n2 = 0.5 * (9.0 + 3.0 * 3.0_f64.sqrt());
    assert!(hydrogenic_radial_wavefunction(3, 0, n1, 1.0).abs() < 1e-11);
    assert!(hydrogenic_radial_wavefunction(3, 0, n2, 1.0).abs() < 1e-11);
}

#[test]
#[verified_engine::verified]
fn test_3dz2_nodal_cone_and_1s_2pz_dipole_oscillation() {
    let dz2 = OrbitalQuantumNumbers::new(3, 2, 0).expect("valid 3dz2");
    // Nodal cone 3 cos^2(theta) - 1 = 0 => z^2 = (x^2 + y^2) / 2
    let x = 2.0_f64.sqrt();
    let z = 1.0;
    let on_cone = hydrogenic_orbital_cartesian(dz2, x, 0.0, z, 1.0);
    assert!(on_cone.abs() < 1e-12, "3dz2 must vanish on nodal cone");
    let polar = hydrogenic_orbital_cartesian(dz2, 0.0, 0.0, 3.0, 1.0);
    let equatorial = hydrogenic_orbital_cartesian(dz2, 3.0, 0.0, 0.0, 1.0);
    assert!(
        polar * equatorial < 0.0,
        "3dz2 polar and equatorial lobes have opposite phase"
    );

    // 1s + 2pz dipole oscillation shifts <z> sign after half Bohr period T/2 = pi / (E2 - E1)
    let s1 = OrbitalQuantumNumbers::new(1, 0, 0).unwrap();
    let p2z = OrbitalQuantumNumbers::new(2, 1, 0).unwrap();
    let delta_e = -0.5 / 4.0 - (-0.5); // 0.375 Ha
    let half_period = std::f64::consts::PI / delta_e;
    let steps = 16;
    let dt = half_period / (steps as f64);
    let dim = 20;
    let mut engine = HydrogenicSuperposition3D::new(s1, p2z, dim, 24, 6.0, dt);
    engine.set_superposition(0.5, 0.0);
    let mut grid = vec![0.0_f32; dim * dim * dim];
    engine.write_field_into(QuantumFieldObservable::ProbabilityDensity, &mut grid);
    let z_init = z_dipole_moment(&grid, dim, 6.0);
    for _ in 0..steps {
        engine.step_in_place();
    }
    engine.write_field_into(QuantumFieldObservable::ProbabilityDensity, &mut grid);
    let z_half = z_dipole_moment(&grid, dim, 6.0);
    assert!(
        z_init > 0.0 && z_half < 0.0,
        "1s+2pz dipole <z> must invert sign: {z_init} -> {z_half}"
    );
}

fn z_dipole_moment(density: &[f32], dim: usize, half_extent: f64) -> f64 {
    let step = (2.0 * half_extent) / ((dim - 1) as f64);
    let mut num = 0.0;
    let mut den = 0.0;
    for z in 0..dim {
        let wz = -half_extent + (z as f64) * step;
        for y in 0..dim {
            for x in 0..dim {
                let p = density[(z * dim + y) * dim + x] as f64;
                num += wz * p;
                den += p;
            }
        }
    }
    num / den.max(1e-12)
}

#[test]
fn test_hydrogenic_superposition_zero_allocation_hot_path() {
    let s1 = OrbitalQuantumNumbers::new(1, 0, 0).unwrap();
    let p2z = OrbitalQuantumNumbers::new(2, 1, 0).unwrap();
    let dim = 24;
    let mut engine = HydrogenicSuperposition3D::new(s1, p2z, dim, 24, 7.0, 0.25);
    engine.set_superposition(0.5, 0.0);
    let mut voxels = vec![0.0_f32; dim * dim * dim];

    VerifiedEngine::run_verified(|| {
        for _ in 0..4 {
            engine.step_in_place();
            engine.write_field_into(QuantumFieldObservable::SignedRealWavefunction, &mut voxels);
            engine.write_field_into(QuantumFieldObservable::ProbabilityDensity, &mut voxels);
        }
    });

    assert!(voxels.iter().all(|v| v.is_finite()));
    assert!(voxels.iter().any(|&v| v > 0.0));
}
