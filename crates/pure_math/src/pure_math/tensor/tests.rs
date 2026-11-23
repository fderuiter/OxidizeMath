//! Unit tests for stack-allocated metrics, Christoffel symbols, and RK4 geodesics.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4};

use nalgebra::{SMatrix, SVector};
use verified_engine::allocator::{lock_allocations, unlock_allocations};
use verified_engine::metrics::{disable_verification, enable_verification};

use crate::pure_math::analysis::ode::{ArrayState, RungeKutta4};

use super::{
    GeodesicState, GeodesicStepStatus, GeodesicSystem, StaticMetric, StaticRiemannianMetric,
    TensorError, TwoSphereMetric, static_christoffel_symbols,
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
