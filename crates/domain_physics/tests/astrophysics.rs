#![allow(missing_docs)]
use domain_physics::physics::astrophysics::galaxies::{
    GalaxyModel, GeneralIrregular, Magnitude, Mpc, SolarMassLog, TypeCode10, TypeCode95To99,
};
use domain_physics::physics::astrophysics::{
    DwarfMorphology, GalacticDiskSystem, HaloProfile, MIN_GALACTOCENTRIC_RADIUS_KPC,
    OrbitParticleKind, OrbitPopulationConfig, VelocityDecomposition,
    baryons_from_absolute_magnitude, baryons_from_distance, decompose_rotation_curve,
    planar_acceleration, sample_rotation_curve_into,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static TEST_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

const F64_TOLERANCE: f64 = math_commons::registry::TOLERANCE_STANDARD;

#[test]
#[verified_engine::verified]
fn test_calculate_log_mass_from_distance() {
    // Test case for GeneralIrregular
    let distance = Mpc(10.0); // Mpc
    let expected = 0.0230 * distance.as_f64() + 0.7840;
    let result = GeneralIrregular.log_mass_from_distance(distance).as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode10
    let expected = 0.0250 * distance.as_f64() + 7.6860;
    let result = TypeCode10.log_mass_from_distance(distance).as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode95To99
    let expected = 0.0504 * distance.as_f64() + 7.5715;
    let result = TypeCode95To99.log_mass_from_distance(distance).as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);
}

#[test]
#[verified_engine::verified]
fn test_calculate_apparent_magnitude_from_distance() {
    let distance = Mpc(10.0); // Mpc

    // Test case for GeneralIrregular
    let expected = 0.0206 * distance.as_f64() + 16.0010;
    let result = GeneralIrregular
        .apparent_magnitude_from_distance(distance)
        .unwrap()
        .as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode10
    let expected = 0.0140 * distance.as_f64() + 16.575;
    let result = TypeCode10
        .apparent_magnitude_from_distance(distance)
        .unwrap()
        .as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode95To99 (should be None)
    let result = TypeCode95To99.apparent_magnitude_from_distance(distance);
    assert!(result.is_none());
}

#[test]
#[verified_engine::verified]
fn test_calculate_log_mass_from_absolute_magnitude() {
    let absolute_magnitude_v = Magnitude(-15.0);

    // Test case for GeneralIrregular
    let expected = -0.6670 * absolute_magnitude_v.as_f64() - 1.4975;
    let result = GeneralIrregular
        .log_mass_from_absolute_magnitude(absolute_magnitude_v)
        .as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode10
    let result = TypeCode10
        .log_mass_from_absolute_magnitude(absolute_magnitude_v)
        .as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode95To99
    let expected = -0.3837 * absolute_magnitude_v.as_f64() - 2.2864;
    let result = TypeCode95To99
        .log_mass_from_absolute_magnitude(absolute_magnitude_v)
        .as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);
}

#[test]
#[verified_engine::verified]
fn test_calculate_redshift_from_log_mass() {
    let log_mass_solar = SolarMassLog(10.0);

    // Test case for GeneralIrregular
    let expected = 0.0094 * log_mass_solar.as_f64() - 0.7270;
    let result = GeneralIrregular
        .redshift_from_log_mass(log_mass_solar)
        .as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode10
    let expected = 0.00093 * log_mass_solar.as_f64() - 0.0716;
    let result = TypeCode10.redshift_from_log_mass(log_mass_solar).as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);

    // Test case for TypeCode95To99
    let expected = 0.0031 * log_mass_solar.as_f64() - 0.0223;
    let result = TypeCode95To99
        .redshift_from_log_mass(log_mass_solar)
        .as_f64();
    assert!((result - expected).abs() < F64_TOLERANCE);
}

#[test]
#[verified_engine::verified]
fn test_baryonic_keplerian_outer_falloff_and_quadrature() {
    let baryons = baryons_from_distance(&TypeCode10, Mpc(12.0), 0.6, 0.75);
    assert!(baryons.scaling.apparent_magnitude_b.is_some());
    assert!(
        (baryons.disk_mass_solar + baryons.gas_mass_solar
            - baryons.scaling.total_baryonic_mass_solar)
            .abs()
            < 1.0e-3
    );

    let r_inner_outer = decompose_rotation_curve(15.0, &baryons, &HaloProfile::BaryonicOnly);
    let r_far_outer = decompose_rotation_curve(60.0, &baryons, &HaloProfile::BaryonicOnly);
    assert_eq!(r_inner_outer.v_halo_kms, 0.0);
    assert!(r_far_outer.v_circ_kms < r_inner_outer.v_circ_kms);

    // At r >> R_g, Keplerian speed scales as r^{-1/2}, so v(15) / v(60) -> sqrt(60 / 15) = 2.0
    let ratio = r_inner_outer.v_circ_kms / r_far_outer.v_circ_kms;
    assert!((ratio - 2.0).abs() < 0.05);

    let unestablished = baryons_from_distance(&TypeCode95To99, Mpc(12.0), 0.5, 0.8);
    assert!(unestablished.scaling.apparent_magnitude_b.is_none());
    let from_mag = baryons_from_absolute_magnitude(&GeneralIrregular, Magnitude(-16.0), 0.5, 0.8);
    assert!(from_mag.scaling.total_baryonic_mass_solar > 1.0e8);
}

#[test]
#[verified_engine::verified]
fn test_nfw_cusp_versus_burkert_core_and_center_clamping() {
    let baryons = DwarfMorphology::TypeCode10.baryonic_config_from_distance(Mpc(10.0), 0.6, 0.8);
    let nfw = HaloProfile::NfwCusp {
        virial_mass_solar: 1.5e10,
        scale_radius_kpc: 2.5,
        concentration: 12.0,
    };
    let burkert = HaloProfile::BurkertCore {
        virial_mass_solar: 1.5e10,
        core_radius_kpc: 2.5,
        concentration: 12.0,
    };

    // Zero and negative radii clamp safely to MIN_GALACTOCENTRIC_RADIUS_KPC without NaN/Inf
    let at_zero = decompose_rotation_curve(0.0, &baryons, &nfw);
    assert_eq!(at_zero.radius_kpc, MIN_GALACTOCENTRIC_RADIUS_KPC);
    assert!(at_zero.v_circ_kms.is_finite());
    let (ax0, ay0) = planar_acceleration(0.0, 0.0, &baryons, &nfw);
    assert_eq!((ax0, ay0), (0.0, 0.0));

    // Inner core vs cusp: NFW rho ~ r^-1 cusp yields higher inner halo speed than Burkert constant-density core
    let inner_nfw = decompose_rotation_curve(0.25, &baryons, &nfw);
    let inner_burkert = decompose_rotation_curve(0.25, &baryons, &burkert);
    assert!(inner_nfw.v_halo_kms > inner_burkert.v_halo_kms);
    assert!(inner_nfw.v_circ_kms > inner_burkert.v_circ_kms);

    // Outer halo flattens rotation curve well above pure baryonic Keplerian falloff
    let outer_bary = decompose_rotation_curve(10.0, &baryons, &HaloProfile::BaryonicOnly);
    let outer_nfw = decompose_rotation_curve(10.0, &baryons, &nfw);
    let outer_burkert = decompose_rotation_curve(10.0, &baryons, &burkert);
    assert!(outer_nfw.v_circ_kms > 2.0 * outer_bary.v_circ_kms);
    assert!(outer_burkert.v_circ_kms > 2.0 * outer_bary.v_circ_kms);
}

#[test]
#[verified_engine::verified]
fn test_symplectic_leapfrog_conserves_invariants_with_zero_allocations() {
    let halo = HaloProfile::BurkertCore {
        virial_mass_solar: 1.2e10,
        core_radius_kpc: 2.2,
        concentration: 11.0,
    };
    let mut system = GalacticDiskSystem::from_morphology(
        DwarfMorphology::TypeCode95To99,
        Mpc(14.0),
        0.65,
        0.85,
        halo,
        OrbitPopulationConfig {
            disk_tracers: 14,
            tidal_tracers: 6,
            trail_capacity: 48,
            tidal_tangential_ratio: 0.6,
        },
    );
    assert_eq!(system.particles().len(), 20);
    assert!(
        system
            .particles()
            .iter()
            .any(|p| p.kind == OrbitParticleKind::DiskTracer)
    );
    assert!(
        system
            .particles()
            .iter()
            .any(|p| p.kind == OrbitParticleKind::TidalStream)
    );

    let mut curve_buf = [VelocityDecomposition::default(); 16];
    let mut trail_buf = [[0.0_f64; 2]; 32];

    let (trail_written, metrics) = VerifiedEngine::run_verified(|| {
        sample_rotation_curve_into(0.1, 10.0, &system.baryons, &system.halo, &mut curve_buf);
        system.step_n_leapfrog(1.5, 80);
        system.copy_particle_trail(0, &mut trail_buf)
    });

    assert_eq!(trail_written, 32);
    assert!(metrics.function_calls > 80);
    assert!(metrics.arithmetic_ops > 500);
    assert!((system.elapsed_myr() - 120.0).abs() < 1.0e-9);
    assert!(system.relative_energy_drift() < 5.0e-4);
    assert!(system.relative_angular_momentum_drift() < 1.0e-11);
}
