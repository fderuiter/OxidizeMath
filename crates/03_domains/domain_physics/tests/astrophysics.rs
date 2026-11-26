//! Integration tests for astrophysics.
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

#[test]
#[verified_engine::verified]
fn test_schwarzschild_and_halo_lensing_jacobians_shadow_and_caustics_zero_alloc() {
    use domain_physics::physics::astrophysics::{
        ThinLensSample, clamp_einstein_separation, halo_lens_deflection_and_jacobian,
        halo_thin_lens_sample, sample_halo_lensing_into, schwarzschild_thin_lens_images,
        tangential_critical_radius_kpc,
    };
    use domain_physics::physics::high_energy::general_relativity::SchwarzschildBlackHole;
    use nalgebra::Vector2;

    let bh = SchwarzschildBlackHole::new(1.0e8 * math_commons::constants::SOLAR_MASS)
        .expect("valid black hole");
    let nfw = HaloProfile::NfwCusp {
        virial_mass_solar: 4.0e11,
        scale_radius_kpc: 3.5,
        concentration: 14.0,
    };
    let burkert = HaloProfile::BurkertCore {
        virial_mass_solar: 4.0e11,
        core_radius_kpc: 3.5,
        concentration: 14.0,
    };
    let mut nfw_buf = [ThinLensSample::default(); 24];
    let mut burk_buf = [ThinLensSample::default(); 24];

    let ((ring, occluded, r_crit_nfw, r_crit_burk), _metrics) =
        VerifiedEngine::run_verified(|| {
            let on_axis = schwarzschild_thin_lens_images(&bh, Vector2::new(0.0, 0.0), 1.0, 0.25)
                .expect("ring");
            let inside_shadow =
                schwarzschild_thin_lens_images(&bh, Vector2::new(0.05, 0.0), 0.30, 0.30)
                    .expect("shadow");
            sample_halo_lensing_into(0.05, 8.0, None, &nfw, 2.5e7, &mut nfw_buf);
            sample_halo_lensing_into(0.05, 8.0, None, &burkert, 2.5e7, &mut burk_buf);
            let rc_n = tangential_critical_radius_kpc(None, &nfw, 2.5e7, 15.0);
            let rc_b = tangential_critical_radius_kpc(None, &burkert, 2.5e7, 15.0);
            (on_axis, inside_shadow, rc_n, rc_b)
        });

    // Singularity clamp right on the Einstein ring: |theta - theta_E| >= 1e-3 * theta_E
    let clamped = clamp_einstein_separation(1.0, 1.0);
    assert!((clamped - 1.0).abs() >= 1.0e-3 - 1.0e-12);
    assert!(ring.primary.is_some() && ring.secondary.is_some());
    assert!((ring.shadow_radius - 0.375).abs() < 1.0e-12);
    // Secondary image inside 1.5 * R_s = 0.45 is masked out by the photon-sphere shadow
    assert!(occluded.secondary.is_none());

    // Inner NFW cusp yields stronger inner convergence and shear than Burkert flat core
    assert!(nfw_buf[0].convergence > burk_buf[0].convergence);
    assert!(nfw_buf[0].shear > burk_buf[0].shear);
    let burk_center = halo_thin_lens_sample(1.0e-3, None, &burkert, 2.5e7);
    assert!(burk_center.shear < 1.0e-6);

    // Verify exact differential relation d(R^2 * mean_kappa)/dR == 2 * R * kappa for Burkert
    let r_mid = 2.4;
    let h = 1.0e-4;
    let s_minus = halo_thin_lens_sample(r_mid - h, None, &burkert, 2.5e7);
    let s_plus = halo_thin_lens_sample(r_mid + h, None, &burkert, 2.5e7);
    let s_zero = halo_thin_lens_sample(r_mid, None, &burkert, 2.5e7);
    let deriv = ((r_mid + h).powi(2) * s_plus.mean_convergence
        - (r_mid - h).powi(2) * s_minus.mean_convergence)
        / (2.0 * h);
    assert!((deriv - 2.0 * r_mid * s_zero.convergence).abs() < 1.0e-5);

    let (_defl, jac, _sample) =
        halo_lens_deflection_and_jacobian(Vector2::new(1.5, -0.8), None, &nfw, 2.5e7);
    assert!(jac.iter().all(|v| v.is_finite()));
    assert!(r_crit_nfw.is_some() && r_crit_burk.is_some());
}

#[test]
#[verified_engine::verified]
fn test_paczynski_wiita_isco_capture_rosetta_christoffel_and_beaming_zero_alloc() {
    use domain_physics::physics::astrophysics::{
        CentralBlackHoleConfig, evaluate_relativistic_beaming,
        validate_christoffel_radial_connection,
    };
    use math_commons::constants::C;

    let halo = HaloProfile::NfwCusp {
        virial_mass_solar: 1.0e10,
        scale_radius_kpc: 2.5,
        concentration: 12.0,
    };
    let pop = OrbitPopulationConfig {
        disk_tracers: 4,
        tidal_tracers: 4,
        trail_capacity: 32,
        tidal_tangential_ratio: 0.18,
    };
    let mut newt_sys = GalacticDiskSystem::from_morphology(
        DwarfMorphology::TypeCode10,
        Mpc(10.0),
        0.5,
        0.6,
        halo,
        pop,
    );
    let mut rel_sys = newt_sys.clone();
    let bh_cfg = CentralBlackHoleConfig {
        enabled: true,
        mass_solar: 2.5e9,
        effective_rs_kpc: 0.22,
        relativistic_beta_scale: 0.004,
        magnetic_energy_density_j_m3: 2.0,
    };
    rel_sys.set_central_black_hole(bh_cfg);
    rel_sys.reseed_orbits(4, 4, 0.18);
    newt_sys.reseed_orbits(4, 4, 0.18);

    let bh_metric = bh_cfg.to_schwarzschild_bh().expect("valid metric");

    let ((conn, beam_horizon, beam_approach, beam_recede), _metrics) =
        VerifiedEngine::run_verified(|| {
            newt_sys.step_n_leapfrog(0.4, 90);
            rel_sys.step_n_leapfrog(0.4, 90);
            let conn = validate_christoffel_radial_connection(&bh_metric, 6.0)
                .expect("christoffel connection");
            // Singularity clamps: r <= R_s (0.5 * R_s) and v >= 0.999 * C (1.5 * C) stay finite!
            let b_hor = evaluate_relativistic_beaming(&bh_metric, 0.5, 1.5 * C, 1.0, 2.0)
                .expect("clamped beaming");
            let b_app = evaluate_relativistic_beaming(&bh_metric, 6.0, 0.45 * C, 0.9, 2.0)
                .expect("approaching");
            let b_rec = evaluate_relativistic_beaming(&bh_metric, 6.0, 0.45 * C, -0.9, 2.0)
                .expect("receding");
            (conn, b_hor, b_app, b_rec)
        });

    assert!(rel_sys.captured_particle_count() >= 1);
    assert_eq!(newt_sys.captured_particle_count(), 0);
    // Surviving disk tracers exhibit relativistic phase/apsidal shift vs Newtonian potential
    let p_newt = newt_sys.particles()[3];
    let p_rel = rel_sys.particles()[3];
    let phase_diff = (p_rel.y_kpc.atan2(p_rel.x_kpc) - p_newt.y_kpc.atan2(p_newt.x_kpc)).abs();
    assert!(phase_diff > 1.0e-2);

    // Verify exact ratio between Christoffel \Gamma^r_{00} and Paczynski-Wiita at r = 6 R_s:
    // \Gamma^r_{00} * (r / (r - R_s))^3 == R_s / (2 (r - R_s)^2)
    let ratio_cubed = (6.0_f64 / 5.0_f64).powi(3);
    assert!((conn.0 * ratio_cubed - conn.1).abs() / conn.1 < 1.0e-5);

    assert!(beam_horizon.radiance_weight.is_finite());
    assert!(beam_horizon.beta < 0.999);
    assert!(beam_approach.radiance_weight > beam_recede.radiance_weight);
    assert!(beam_approach.synchrotron_power_watts > 0.0);
}
