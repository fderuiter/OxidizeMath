//! Integration and zero-allocation verification tests for continuous chaotic systems.

use domain_physics::physics::chaos::double_pendulum::{DoublePendulumBuilder, DoublePendulumState};
use domain_physics::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
use domain_physics::physics::chaos::metrics::{
    double_pendulum_lyapunov, finite_time_lyapunov, lorenz_lyapunov,
};
use domain_physics::physics::chaos::rossler::{RosslerBuilder, RosslerState};
use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{ArrayState, RungeKutta4, Solver};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_lorenz_on_attractor_twin_divergence_and_wolf_lyapunov() {
    let dt = 0.01;
    let system = LorenzBuilder::new()
        .sigma(10.0)
        .rho(28.0)
        .beta(8.0 / 3.0)
        .build(LorenzState::new(1.0, 1.0, 1.0));
    let mut rk4 = RungeKutta4::new(&Vector3::zeros());

    let mut primary = Vector3::new(-4.385295763939413, -2.064285615514713, 25.90419269166981);
    let epsilon = 1e-9;
    let mut shadow = primary + Vector3::new(epsilon, 0.0, 0.0);

    let mut cross_time = None;
    let mut lobe_flip_time = None;
    for step in 1..=3000_usize {
        rk4.step(&system, 0.0, &mut primary, dt);
        rk4.step(&system, 0.0, &mut shadow, dt);
        let t = step as f64 * dt;
        let sep = (shadow - primary).norm();
        if cross_time.is_none() && sep >= 0.1 {
            cross_time = Some(t);
        }
        if lobe_flip_time.is_none() && primary.x * shadow.x < 0.0 && sep > 5.0 {
            lobe_flip_time = Some(t);
        }
    }

    let t_cross = cross_time.expect("1e-9 perturbation must cross 0.1 separation");
    assert!(
        (t_cross - 19.96).abs() <= 0.05,
        "Expected crossing near t=19.96, got {t_cross:.4}"
    );
    let t_flip = lobe_flip_time.expect("twins must split onto opposite butterfly wings");
    assert!(
        (t_flip - 24.61).abs() <= 0.2,
        "Expected lobe split near t=24.61, got {t_flip:.4}"
    );

    let lambda_1 = lorenz_lyapunov(&system, &mut rk4, primary, dt, 1000, 0.5).unwrap();
    assert!(
        (lambda_1 - 0.9066).abs() <= 0.02,
        "Expected Wolf lambda_1 = 0.9066 +- 0.02, got {lambda_1:.5}"
    );
}

#[test]
#[verified_engine::verified]
fn test_rossler_boundedness_and_positive_lyapunov() {
    let mut rossler = RosslerBuilder::new()
        .a(0.2)
        .b(0.2)
        .c(5.7)
        .build(RosslerState::new(0.1, 0.0, 0.0));

    for _ in 0..4000 {
        rossler.step(0.01);
        assert!(rossler.state.vec.norm() < 100.0);
    }

    let mut rk4 = RungeKutta4::new(&Vector3::zeros());
    let lambda = lorenz_lyapunov(&rossler, &mut rk4, rossler.state.vec, 0.01, 400, 1.0).unwrap();
    assert!(
        lambda > 0.04 && lambda < 0.15,
        "Expected positive Rossler Lyapunov exponent (~0.07), got {lambda:.5}"
    );
}

#[test]
#[verified_engine::verified]
fn test_double_pendulum_energy_conservation_and_lyapunov() {
    let initial = DoublePendulumState::new(2.4, 2.4, 0.0, 0.0);
    let mut pendulum = DoublePendulumBuilder::new().build(initial);
    let initial_energy = pendulum.total_energy();

    for _ in 0..2000 {
        pendulum.step(0.005);
    }
    let drift = (pendulum.total_energy() - initial_energy).abs();
    assert!(
        drift < 1e-3,
        "RK4 double pendulum energy drift too high: {drift:.6e}"
    );
    let ([x1, y1], [x2, y2]) = pendulum.bob_positions();
    assert!(x1.is_finite() && y1.is_finite() && x2.is_finite() && y2.is_finite());

    let mut rk4 = RungeKutta4::new(&ArrayState([0.0; 4]));
    let lambda = double_pendulum_lyapunov(&pendulum, &mut rk4, initial.state, 0.005, 160, 0.25)
        .expect("double pendulum Wolf Lyapunov must succeed");
    assert!(
        lambda > 0.2,
        "High-energy double pendulum should be chaotic (lambda > 0.2), got {lambda:.4}"
    );

    let ftle = finite_time_lyapunov(1e-9, 1e-1, 20.0);
    assert!((ftle - (1e8_f64.ln() / 20.0)).abs() < 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_zero_allocation_chaos_stepping_under_verified_allocator() {
    let lorenz = LorenzBuilder::new().build(LorenzState::new(1.0, 1.0, 1.0));
    let mut lorenz_vec = lorenz.state.vec;
    let mut rk4_vec3 = RungeKutta4::new(&lorenz_vec);
    let mut rossler = RosslerBuilder::new().build(RosslerState::new(0.1, 0.0, 0.0));
    let mut pendulum =
        DoublePendulumBuilder::new().build(DoublePendulumState::new(2.4, 2.4, 0.0, 0.0));

    VerifiedEngine::run_verified(|| {
        for _ in 0..64 {
            rk4_vec3.step(&lorenz, 0.0, &mut lorenz_vec, 0.01);
            rossler.step(0.01);
            pendulum.step(0.005);
        }
    });

    assert!(lorenz_vec.norm().is_finite());
    assert!(rossler.state.vec.norm().is_finite());
    assert!(pendulum.total_energy().is_finite());
}
