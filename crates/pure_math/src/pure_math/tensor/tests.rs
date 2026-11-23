//! Unit tests for stack-allocated metrics, Christoffel symbols, and RK4 geodesics.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4};

use nalgebra::{DVector, SMatrix, SVector, Vector4};
use verified_engine::allocator::{lock_allocations, unlock_allocations};
use verified_engine::metrics::{disable_verification, enable_verification};

use crate::pure_math::analysis::ode::{ArrayState, RungeKutta4};

use super::{
    GeodesicState, GeodesicStepStatus, GeodesicSystem, LorentzianConnection4D,
    SchwarzschildConnection4D, StaticMetric, StaticRiemannianMetric, TensorError, TwoSphereMetric,
    christoffel_symbols, static_christoffel_symbols,
};

#[test]
#[verified_engine::verified]
fn two_sphere_analytical_and_numerical_christoffel_match_within_1e_6() {
    let sphere = TwoSphereMetric::new(2.5).expect("valid sphere");
    for theta in [0.35, FRAC_PI_4, FRAC_PI_3, 1.2, 2.3] {
        let pt = SVector::<f64, 2>::new(theta, 0.85);
        let analytical = sphere.christoffel_at(&pt).expect("analytical christoffel");
        let numerical = static_christoffel_symbols(&sphere, &pt).expect("numerical christoffel");
        for mu in 0..2 {
            for alpha in 0..2 {
                for beta in 0..2 {
                    let diff = (analytical[mu][(alpha, beta)] - numerical[mu][(alpha, beta)]).abs();
                    assert!(
                        diff < 1e-6,
                        "mismatch at ({mu},{alpha},{beta}) theta={theta}: diff={diff}"
                    );
                }
            }
        }
    }
}

#[test]
#[verified_engine::verified]
fn geodesic_system_1000_rk4_steps_zero_heap_allocations_analytical_and_numerical() {
    let sphere = TwoSphereMetric::new(1.0).expect("valid sphere");
    let sys_analytical = GeodesicSystem::new(sphere);
    let closure_metric = StaticRiemannianMetric::new(|p: &SVector<f64, 2>| {
        let s = p[0].sin();
        Ok(SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, s * s))
    });
    let sys_numerical = GeodesicSystem::new(closure_metric);

    let mut state_a = ArrayState::<4>::from_pos_vel(
        &SVector::<f64, 2>::new(FRAC_PI_3, 0.0),
        &SVector::<f64, 2>::new(0.35, 0.8),
    );
    let mut state_n = state_a;
    let mut solver = RungeKutta4::new(&state_a);
    let initial_norm = sys_analytical.metric_norm(&state_a).expect("initial norm");

    enable_verification();
    lock_allocations();
    let mut lambda = 0.0;
    let dt = 0.002;
    for _ in 0..1_000 {
        let s1 = sys_analytical.step_rk4(&mut solver, lambda, &mut state_a, dt);
        let s2 = sys_numerical.step_rk4(&mut solver, lambda, &mut state_n, dt);
        assert!(matches!(s1, GeodesicStepStatus::Advanced { .. }));
        assert!(matches!(s2, GeodesicStepStatus::Advanced { .. }));
        lambda += dt;
    }
    unlock_allocations();
    disable_verification();

    let final_norm = sys_analytical.metric_norm(&state_a).expect("final norm");
    assert!((final_norm - initial_norm).abs() < 1e-8);
    assert!((state_a.0[0] - state_n.0[0]).abs() < 1e-6);
    assert!((state_a.0[1] - state_n.0[1]).abs() < 1e-6);
}

#[test]
#[verified_engine::verified]
fn two_sphere_pole_singularity_halts_cleanly_without_nan() {
    let sphere = TwoSphereMetric::new(1.0).expect("valid sphere");
    let system = GeodesicSystem::new(sphere);
    let mut state = ArrayState::<4>::from_pos_vel(
        &SVector::<f64, 2>::new(0.05, FRAC_PI_2),
        &SVector::<f64, 2>::new(-1.0, 0.0),
    );
    let mut solver = RungeKutta4::new(&state);

    let mut halted = false;
    for step in 0..200 {
        let status = system.step_rk4(&mut solver, f64::from(step) * 0.001, &mut state, 0.001);
        if status == GeodesicStepStatus::HaltedSingularity {
            halted = true;
            break;
        }
    }
    assert!(
        halted,
        "meridian geodesic must halt before sin(theta) <= 1e-4"
    );
    assert!(state.0.iter().all(|v| v.is_finite()));
    assert!(state.0[0].sin().abs() > 1e-4);
    assert!(matches!(
        TwoSphereMetric::new(0.0),
        Err(TensorError::SingularMetric)
    ));
}

#[test]
#[verified_engine::verified]
fn schwarzschild_connection_4d_matches_christoffel_symbols_at_3_rs_within_1e_10() {
    let rs = 2.0;
    let conn = SchwarzschildConnection4D::new(rs).expect("valid Schwarzschild connection");
    let pt = Vector4::new(0.5, 3.0 * rs, FRAC_PI_2, 1.1);
    let dpt = DVector::from_vec(vec![pt[0], pt[1], pt[2], pt[3]]);

    let closed_form = conn.christoffel_tensors(&pt).expect("closed-form gammas");
    let fd_gammas = christoffel_symbols(&conn, &dpt).expect("finite-difference gammas");

    for mu in 0..4 {
        for alpha in 0..4 {
            for beta in 0..4 {
                let diff = (closed_form[mu][(alpha, beta)] - fd_gammas[mu][(alpha, beta)]).abs();
                assert!(
                    diff < 1e-10,
                    "Gamma^{mu}_{{{alpha},{beta}}} mismatch at r=3 Rs: diff={diff}"
                );
            }
        }
    }
}

#[test]
#[verified_engine::verified]
fn schwarzschild_connection_4d_avoids_horizon_stencil_breakdown_and_zero_allocs() {
    let rs = 1.0;
    let conn = SchwarzschildConnection4D::new(rs).expect("valid Schwarzschild connection");
    // Within h = 1e-5 of r = R_s, central difference steps across r - h <= R_s and fails.
    let near_horizon = Vector4::new(0.0, rs + 5.0e-6, FRAC_PI_2, 0.0);
    let near_dvec = DVector::from_vec(vec![0.0, rs + 5.0e-6, FRAC_PI_2, 0.0]);
    assert!(christoffel_symbols(&conn, &near_dvec).is_err());

    let exact_near = conn
        .christoffel_tensors(&near_horizon)
        .expect("closed-form succeeds near horizon");
    let delta_r = near_horizon[1] - rs;
    let expected_g0_01 = rs / (2.0 * near_horizon[1] * delta_r);
    assert!((exact_near[0][(0, 1)] - expected_g0_01).abs() / expected_g0_01 < 1e-14);

    let coords = Vector4::new(0.0, 3.0 * rs, FRAC_PI_2, 0.0);
    let vel = Vector4::new(2.0_f64.sqrt(), 0.0, 0.0, 1.0 / (3.0 * 3.0_f64.sqrt() * rs));
    enable_verification();
    lock_allocations();
    let mut acc_sum = 0.0;
    for _ in 0..1_000 {
        let acc = conn
            .geodesic_acceleration(&coords, &vel)
            .expect("zero-alloc geodesic acceleration");
        let norm = conn
            .interval_norm(&coords, &vel)
            .expect("zero-alloc interval norm");
        acc_sum += acc[1] + norm;
    }
    unlock_allocations();
    disable_verification();
    assert!(acc_sum.is_finite());
}
