//! Unit tests for Schwarzschild, equatorial Kerr, and Morris-Thorne wormhole geodesics.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, PI};

use approx::assert_relative_eq;
use math_commons::constants::SOLAR_MASS;
use nalgebra::SVector;
use pure_math::pure_math::analysis::ode::{ArrayState, RungeKutta4};
use pure_math::pure_math::tensor::{
    GeodesicState, GeodesicStepStatus, GeodesicSystem, StaticMetric, static_christoffel_symbols,
};
use verified_engine::allocator::{lock_allocations, unlock_allocations};
use verified_engine::metrics::{disable_verification, enable_verification};

use super::*;
use crate::physics::high_energy::{C, G};

#[test]
#[verified_engine::verified]
fn test_schwarzschild() {
    let mass = SOLAR_MASS;
    let bh = SchwarzschildBlackHole::new(mass).expect("Failed to create BH");

    let rs = bh.schwarzschild_radius().expect("Failed to calc Rs");
    let expected_rs = 2.0 * G * mass / (C * C);
    assert_relative_eq!(rs, expected_rs);

    let isco = bh.isco().expect("Failed to calc ISCO");
    assert_relative_eq!(isco, 3.0 * rs);

    // Time dilation at r = 2 Rs.
    // factor = sqrt(1 - 1/2) = sqrt(0.5) approx 0.707
    let factor = bh
        .gravitational_time_dilation(2.0 * rs)
        .expect("Failed to calc time dilation");
    assert_relative_eq!(factor, 0.5f64.sqrt());
}

#[test]
#[verified_engine::verified]
fn test_errors() {
    assert!(SchwarzschildBlackHole::new(-1.0).is_err());

    let bh = SchwarzschildBlackHole::new(SOLAR_MASS).unwrap();
    let rs = bh.schwarzschild_radius().unwrap();
    assert!(bh.gravitational_time_dilation(0.5 * rs).is_err());
}

#[test]
#[verified_engine::verified]
fn test_helpers_flamm_potential_and_combined_clock_rate() {
    let bh = SchwarzschildBlackHole::from_schwarzschild_radius(2.0).unwrap();
    let rs = bh.schwarzschild_radius().unwrap();
    assert_relative_eq!(rs, 2.0, epsilon = 1e-12);
    assert_relative_eq!(bh.photon_sphere_radius().unwrap(), 3.0, epsilon = 1e-12);
    assert_relative_eq!(bh.flamm_paraboloid_z(rs).unwrap(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(
        bh.flamm_paraboloid_z(3.0 * rs).unwrap(),
        4.0 * 2.0_f64.sqrt(),
        epsilon = 1e-12
    );

    // Circular orbit at ISCO r = 3 Rs has local shell speed v/c = 0.5 => dtau/dt = sqrt(2/3)*sqrt(3/4) = sqrt(1/2)
    let rate = combined_clock_rate(&bh, 3.0 * rs, 0.5 * C).unwrap();
    assert_relative_eq!(rate, 0.5_f64.sqrt(), epsilon = 1e-12);
    assert!(bh.flamm_paraboloid_z(0.9 * rs).is_err());
    assert!(effective_potential(rs, 0.9 * rs, 4.0, GeodesicKind::Null).is_err());
}

#[test]
#[verified_engine::verified]
fn test_schwarzschild_and_kerr_christoffel_match_numerical_within_1e_6() {
    let bh = SchwarzschildBlackHole::from_schwarzschild_radius(2.0).unwrap();
    for (r, theta) in [(2.6, FRAC_PI_2), (4.5, FRAC_PI_3), (9.0, 1.95)] {
        let pt = SVector::<f64, 4>::new(1.2, r, theta, 0.7);
        let exact = bh.christoffel_at(&pt).unwrap();
        let approx = static_christoffel_symbols(&bh, &pt).unwrap();
        for mu in 0..4 {
            for a in 0..4 {
                for b in 0..4 {
                    assert!((exact[mu][(a, b)] - approx[mu][(a, b)]).abs() < 1e-6);
                }
            }
        }
    }

    let kerr = KerrEquatorialMetric::new(2.0, 0.75).unwrap();
    for r in [2.3, 4.0, 8.5] {
        let pt = SVector::<f64, 4>::new(0.5, r, FRAC_PI_2, 1.1);
        let prod = kerr.inverse_metric_at(&pt).unwrap() * kerr.metric_at(&pt).unwrap();
        assert!((prod - nalgebra::SMatrix::<f64, 4, 4>::identity()).norm() < 1e-12);
        let exact = kerr.christoffel_at(&pt).unwrap();
        let approx = static_christoffel_symbols(&kerr, &pt).unwrap();
        for mu in 0..4 {
            for a in 0..4 {
                for b in 0..4 {
                    assert!((exact[mu][(a, b)] - approx[mu][(a, b)]).abs() < 1e-6);
                }
            }
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_wormhole_christoffel_matches_numerical_and_transits_throat() {
    let wh = MorrisThorneWormhole::new(1.5).unwrap();
    for l in [-3.2, 0.0, 2.4] {
        let pt = SVector::<f64, 3>::new(0.0, l, 0.8);
        let exact = wh.christoffel_at(&pt).unwrap();
        let approx = static_christoffel_symbols(&wh, &pt).unwrap();
        for mu in 0..3 {
            for a in 0..3 {
                for b in 0..3 {
                    assert!((exact[mu][(a, b)] - approx[mu][(a, b)]).abs() < 1e-6);
                }
            }
        }
    }

    // Null ray with impact parameter b = 0.8 < b_0 = 1.5 crosses from +l across l=0 into -l
    let l0 = 4.0;
    let b = 0.8;
    let r0_sq = wh.throat_radius * wh.throat_radius + l0 * l0;
    let dphi = b / r0_sq;
    let dl = -(1.0 - b * b / r0_sq).sqrt();
    let sys = GeodesicSystem::new(wh);
    let mut state = ArrayState::<6>::from_pos_vel(
        &SVector::<f64, 3>::new(0.0, l0, 0.0),
        &SVector::<f64, 3>::new(1.0, dl, dphi),
    );
    let mut solver = RungeKutta4::new(&state);
    for step in 0..900 {
        let status = sys.step_rk4(&mut solver, f64::from(step) * 0.01, &mut state, 0.01);
        assert!(matches!(status, GeodesicStepStatus::Advanced { .. }));
    }
    assert!(
        state.0[1] < -3.5,
        "wormhole geodesic must emerge at negative l"
    );
    assert!(wh.embedding_z(l0).unwrap() > 0.0 && wh.embedding_z(state.0[1]).unwrap() < 0.0);
    assert!(sys.metric_norm(&state).unwrap().abs() < 1e-8);
}

#[test]
#[verified_engine::verified]
fn test_schwarzschild_1000_rk4_steps_zero_heap_alloc_and_photon_sphere() {
    let bh = SchwarzschildBlackHole::from_schwarzschild_radius(2.0).unwrap();
    let rs = bh.schwarzschild_radius().unwrap();
    let r_ph = 1.5 * rs;
    let b_crit = 1.5 * 3.0_f64.sqrt() * rs;
    let sys = GeodesicSystem::new(bh);
    let mut state = ArrayState::<8>::from_pos_vel(
        &SVector::<f64, 4>::new(0.0, r_ph, FRAC_PI_2, 0.0),
        &SVector::<f64, 4>::new(1.0, 0.0, 0.0, 1.0 / b_crit),
    );
    let mut solver = RungeKutta4::new(&state);

    enable_verification();
    lock_allocations();
    let mut lambda = 0.0;
    let dt = 0.005;
    for _ in 0..1_000 {
        let status = sys.step_rk4(&mut solver, lambda, &mut state, dt);
        assert!(matches!(status, GeodesicStepStatus::Advanced { .. }));
        lambda += dt;
    }
    unlock_allocations();
    disable_verification();

    let norm = sys.metric_norm(&state).unwrap();
    assert!(norm.abs() < 1e-5, "null norm drifted: {norm}");
    assert!(
        (state.0[1] - r_ph).abs() < 1e-4,
        "photon sphere radius drifted: {}",
        state.0[1]
    );
}

#[test]
#[verified_engine::verified]
fn test_horizon_singularity_halt_and_isco_stability() {
    let bh = SchwarzschildBlackHole::from_schwarzschild_radius(2.0).unwrap();
    let rs = bh.schwarzschild_radius().unwrap();
    let sys = GeodesicSystem::new(bh);

    // Radial infall photon halts cleanly before r <= 1.02 * R_s
    let f0 = 1.0 - rs / 5.0;
    let mut infall = ArrayState::<8>::from_pos_vel(
        &SVector::<f64, 4>::new(0.0, 5.0, FRAC_PI_2, 0.0),
        &SVector::<f64, 4>::new(1.0 / f0, -1.0, 0.0, 0.0),
    );
    let mut solver = RungeKutta4::new(&infall);
    let mut halted = false;
    for step in 0..800 {
        if sys.step_rk4(&mut solver, f64::from(step) * 0.01, &mut infall, 0.01)
            == GeodesicStepStatus::HaltedSingularity
        {
            halted = true;
            break;
        }
    }
    assert!(halted);
    assert!(infall.0[1] > 1.02 * rs);

    // Timelike circular orbit at ISCO r = 3 R_s stays stable and conserves -1 norm
    let r_isco = bh.isco().unwrap();
    let u0 = 2.0_f64.sqrt();
    let u_phi = 1.0 / (3.0 * 3.0_f64.sqrt() * rs);
    let mut isco_state = ArrayState::<8>::from_pos_vel(
        &SVector::<f64, 4>::new(0.0, r_isco, FRAC_PI_2, 0.0),
        &SVector::<f64, 4>::new(u0, 0.0, 0.0, u_phi),
    );
    let mut isco_solver = RungeKutta4::new(&isco_state);
    for step in 0..2_000 {
        let status = sys.step_rk4(
            &mut isco_solver,
            f64::from(step) * 0.02,
            &mut isco_state,
            0.02,
        );
        assert!(matches!(status, GeodesicStepStatus::Advanced { .. }));
    }
    assert!((isco_state.0[1] - r_isco).abs() < 1e-4);
    assert!((sys.metric_norm(&isco_state).unwrap() + 1.0).abs() < 1e-6);
}

#[test]
#[verified_engine::verified]
fn test_weak_field_light_deflection_matches_2_rs_over_b() {
    let rs = 2.0;
    let bh = SchwarzschildBlackHole::from_schwarzschild_radius(rs).unwrap();
    let sys = GeodesicSystem::new(bh);
    let b = 80.0;
    let x0 = -1_200.0_f64;
    let r0 = x0.hypot(b);
    let phi0 = b.atan2(x0);
    let f0 = 1.0 - rs / r0;
    // Conserved impact parameter b = l / e with e = 1 => ut = 1/f0, uphi = -b / r0^2
    let ut = 1.0 / f0;
    let uphi = -b / (r0 * r0);
    let ur = -(1.0 - f0 * (b * b) / (r0 * r0)).sqrt();

    let mut state = ArrayState::<8>::from_pos_vel(
        &SVector::<f64, 4>::new(0.0, r0, FRAC_PI_2, phi0),
        &SVector::<f64, 4>::new(ut, ur, 0.0, uphi),
    );
    let mut solver = RungeKutta4::new(&state);
    for step in 0..6_000 {
        let _ = sys.step_rk4(&mut solver, f64::from(step) * 0.5, &mut state, 0.5);
        if state.0[1] >= r0 && state.0[5] > 0.0 {
            break;
        }
    }
    // Total swept angle |phi_final - phi0| = pi + Delta_phi - 2*arcsin(b/r_end)
    let swept = (phi0 - state.0[3]).abs();
    let end_corr = (b / r0).asin() + (b / state.0[1]).asin();
    let delta_phi = swept + end_corr - PI;
    let expected = 2.0 * rs / b;
    assert_relative_eq!(delta_phi, expected, max_relative = 0.04);
}

#[test]
#[verified_engine::verified]
fn test_64_mixed_geodesic_bundle_zero_alloc_and_killing_drift() {
    let mut bundle = SchwarzschildGeodesicBundle::new(1.0, 64, 128).unwrap();
    bundle.seed_mixed_ensemble(64);
    assert_eq!(bundle.trajectories().len(), 64);

    let (calls, ops) = bundle.step_n_verified(0.01, 200);
    assert!(calls > 0 && ops > 0);
    let drift = bundle.max_killing_drift();
    assert!(drift < 1e-6, "Killing drift exceeded 1e-6: {drift}");
}

#[test]
#[verified_engine::verified]
fn test_photon_ring_deflection_6_51_rad_and_critical_impact_parameter() {
    let bh = SchwarzschildBlackHole::from_schwarzschild_radius(1.0).unwrap();
    assert_relative_eq!(
        bh.critical_impact_parameter().unwrap(),
        1.5 * 3.0_f64.sqrt(),
        epsilon = 1e-12
    );
    assert_relative_eq!(bh.circular_shell_beta(3.0).unwrap(), 0.5, epsilon = 1e-12);

    let mut bundle = SchwarzschildGeodesicBundle::new(1.0, 8, 256).unwrap();
    bundle.seed_photon_lensing_fan(8, 2.60, 17.45, 0.8);
    let _ = bundle.step_n_verified(0.01, 4_000);
    let deflection = bundle
        .probe_photon_deflection_rad()
        .expect("b = 2.60 > b_crit must scatter back to infinity");
    assert_relative_eq!(deflection, 6.51, epsilon = 0.03);
    assert!(bundle.max_killing_drift() < 1e-6);
}

#[test]
#[verified_engine::verified]
fn test_sub_isco_plunge_37_03_proper_time_vs_2_88_newtonian_periapsis() {
    assert_relative_eq!(
        newtonian_periapsis_radius(3.0, 0.99),
        2.8829,
        epsilon = 1e-3
    );

    let mut bundle = SchwarzschildGeodesicBundle::new(1.0, 2, 256).unwrap();
    bundle.seed_timelike_isco_comparison(3.0, 0.99, 0.001);
    let _ = bundle.step_n_verified(0.005, 4_096);
    let _ = bundle.step_n_verified(0.005, 4_096);

    let tr = &bundle.trajectories()[0];
    assert!(
        tr.halted_horizon,
        "sub-ISCO particle must plunge to horizon"
    );
    assert_relative_eq!(tr.affine_elapsed, 37.03, epsilon = 0.5);
    assert_relative_eq!(bundle.newtonian_state().min_radius, 2.883, epsilon = 0.01);
}
