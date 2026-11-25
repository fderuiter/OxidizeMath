//! Integration tests verifying `LorenzSystem` and `BlochSimulator`:
//! - Honor `IntegrationMethod` identically across inherent `step(dt)` and `TimeStepper::step(&mut self, dt)`.
//! - Advance simulation time `t` consistently across steps.
//! - Execute with zero dynamic heap allocations after initialization under `VerifiedAllocator`.
//! - Support `Debug`, `Clone`, `PartialEq`, `assert_eq!`, and `unwrap_err()`.

use domain_physics::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
use domain_physics::physics::mri::BlochSimulator;
use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{
    Euler, Heun, IntegrationMethod, RungeKutta4, Solver, TimeStepper,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

const METHODS: [IntegrationMethod; 5] = [
    IntegrationMethod::Euler,
    IntegrationMethod::Heun,
    IntegrationMethod::RungeKutta4,
    IntegrationMethod::ImplicitMidpoint,
    IntegrationMethod::GaussLegendre4,
];

#[test]
#[verified_engine::verified]
fn test_lorenz_integration_method_parity_and_derives() {
    let initial = LorenzState::new(10.0, 10.0, 10.0);
    let dt = 0.01;

    for method in METHODS {
        let builder = LorenzBuilder::new().integration_method(method);
        assert_eq!(builder, builder.clone());
        assert!(!format!("{builder:?}").is_empty());

        let mut via_inherent = builder.try_build(initial).unwrap();
        let mut via_trait = via_inherent.clone();
        assert_eq!(via_inherent, via_trait);
        assert!(!format!("{via_inherent:?}").is_empty());

        for _ in 0..8 {
            via_inherent.step(dt);
            <LorenzSystem as TimeStepper<Vector3<f64>>>::step(&mut via_trait, dt);
        }
        assert_eq!(via_inherent, via_trait);
        assert_eq!(via_inherent.state, via_trait.state);
        assert!((via_inherent.time() - 8.0 * dt).abs() < 1e-15);
    }

    let mut euler_sys = LorenzBuilder::new()
        .integration_method(IntegrationMethod::Euler)
        .build(initial);
    let mut heun_sys = LorenzBuilder::new()
        .integration_method(IntegrationMethod::Heun)
        .build(initial);
    let mut rk4_sys = LorenzBuilder::new()
        .integration_method(IntegrationMethod::RungeKutta4)
        .build(initial);
    euler_sys.step(dt);
    heun_sys.step(dt);
    rk4_sys.step(dt);

    assert_ne!(euler_sys.state, heun_sys.state);
    assert_ne!(heun_sys.state, rk4_sys.state);
    assert_ne!(euler_sys.state, rk4_sys.state);

    let err = LorenzBuilder::new()
        .sigma(-1.0)
        .try_build(initial)
        .unwrap_err();
    assert!(!format!("{err:?}").is_empty());
}

#[test]
#[verified_engine::verified]
fn test_bloch_integration_method_parity_and_derives() {
    let m_init = Vector3::new(1.0, 0.0, 0.2);
    let b_field = Vector3::new(0.0, 0.0, 1.0e-6);
    let dt = 1.0e-3;

    for method in METHODS {
        let mut via_inherent = BlochSimulator::try_new(m_init, 1.0, 0.8, 0.1)
            .unwrap()
            .with_integration_method(method);
        via_inherent.set_b_field(b_field);
        let mut via_trait = via_inherent.clone();
        assert_eq!(via_inherent, via_trait);
        assert!(!format!("{via_inherent:?}").is_empty());

        for _ in 0..8 {
            via_inherent.step(dt);
            <BlochSimulator as TimeStepper<Vector3<f64>>>::step(&mut via_trait, dt);
        }
        assert_eq!(via_inherent, via_trait);
        assert_eq!(via_inherent.magnetization, via_trait.magnetization);
        assert!((via_inherent.time() - 8.0 * dt).abs() < 1e-15);
    }

    let make = |m: IntegrationMethod| {
        let mut sim = BlochSimulator::try_new(m_init, 1.0, 0.8, 0.1)
            .unwrap()
            .with_integration_method(m);
        sim.set_b_field(b_field);
        sim.step(dt);
        sim
    };
    let euler_sim = make(IntegrationMethod::Euler);
    let heun_sim = make(IntegrationMethod::Heun);
    let rk4_sim = make(IntegrationMethod::RungeKutta4);

    let mut direct_euler = m_init;
    Euler::new(&m_init).step(&euler_sim, 0.0, &mut direct_euler, dt);
    assert_eq!(euler_sim.magnetization, direct_euler);

    let mut direct_heun = m_init;
    Heun::new(&m_init).step(&heun_sim, 0.0, &mut direct_heun, dt);
    assert_eq!(heun_sim.magnetization, direct_heun);

    let mut direct_rk4 = m_init;
    RungeKutta4::new(&m_init).step(&rk4_sim, 0.0, &mut direct_rk4, dt);
    assert_eq!(rk4_sim.magnetization, direct_rk4);

    assert_ne!(euler_sim.magnetization, heun_sim.magnetization);
    assert_ne!(heun_sim.magnetization, rk4_sim.magnetization);

    let bad = BlochSimulator::try_new(m_init, 1.0, -0.5, 0.1).unwrap_err();
    assert!(!format!("{bad:?}").is_empty());
}

#[test]
#[verified_engine::verified]
fn test_lorenz_and_bloch_zero_heap_allocations_across_methods() {
    let seed = LorenzState::new(1.0, 2.0, 3.0);
    let mut lorenz_models: [LorenzSystem; 5] = std::array::from_fn(|idx| {
        LorenzBuilder::new()
            .integration_method(METHODS[idx])
            .build(seed)
    });
    let mut bloch_models: [BlochSimulator; 5] = std::array::from_fn(|idx| {
        let mut sim = BlochSimulator::new(Vector3::new(1.0, 0.0, 0.0), 1.0)
            .with_integration_method(METHODS[idx]);
        sim.set_relaxation(1.0, 0.2);
        sim.set_b_field(Vector3::new(0.0, 0.0, 1.0e-6));
        sim
    });

    let (_, metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..50 {
            for model in &mut lorenz_models {
                model.step(0.005);
                <LorenzSystem as TimeStepper<Vector3<f64>>>::step(model, 0.005);
            }
            for sim in &mut bloch_models {
                sim.step(1.0e-4);
                <BlochSimulator as TimeStepper<Vector3<f64>>>::step(sim, 1.0e-4);
            }
        }
    });

    assert!(metrics.function_calls > 0);
    for model in &lorenz_models {
        assert!(model.state.vec.iter().all(|v| v.is_finite()));
    }
    for sim in &bloch_models {
        assert!(sim.magnetization.iter().all(|v| v.is_finite()));
    }
}
