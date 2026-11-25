use super::*;
use pure_math::pure_math::analysis::ode::{Euler, RungeKutta4};

#[test]
#[verified_engine::verified]
fn test_builder() {
    let model = SIRModel::builder()
        .n(1000.0)
        .i0(10.0)
        .beta(0.5)
        .gamma(0.1)
        .build();
    assert!(model.is_ok());
    let model = model.unwrap();
    assert_eq!(model.n(), 1000.0);
    assert_eq!(model.beta(), 0.5);
}

#[test]
#[verified_engine::verified]
fn test_builder_missing_param() {
    let model = SIRModel::builder().n(1000.0).build();
    assert!(matches!(
        model,
        Err(EpidemiologyError::MissingParameter { .. })
    ));
}

#[test]
#[verified_engine::verified]
fn test_threshold_theorem() {
    let n = 1000.0;
    let i0 = 10.0;
    // R0 = beta / gamma = 0.5 / 1.0 = 0.5 < 1
    let mut model = SIRModel::new(n, i0, 0.5, 1.0).unwrap();

    let initial_i = model.state().i;
    model.step(0.1);

    assert!(
        model.state().i < initial_i,
        "Infected should decrease when R0 < 1"
    );
}

#[test]
#[verified_engine::verified]
fn test_sir_step_with_rk4() {
    let n = 1000.0;
    let i0 = 10.0;
    let mut model_std = SIRModel::new(n, i0, 0.5, 0.1).unwrap();
    let mut model_with = SIRModel::new(n, i0, 0.5, 0.1).unwrap();

    let dt = 0.1;
    model_std.step(dt);
    let state = *model_with.state();

    // Use external solver
    model_with.step_with(&mut RungeKutta4::new(&state), dt);

    assert_eq!(
        model_std.state(),
        model_with.state(),
        "step and step_with(RK4) should yield identical results"
    );
}

#[test]
#[verified_engine::verified]
fn test_sir_step_with_euler() {
    let n = 1000.0;
    let i0 = 10.0;
    let mut model = SIRModel::new(n, i0, 0.5, 0.1).unwrap();

    // Euler is less accurate but should still run without panic
    let state = *model.state();
    model.step_with(&mut Euler::new(&state), 0.1);

    assert!(model.state().s <= n);
    assert!(model.state().i >= 0.0);
}

#[test]
#[verified_engine::verified]
fn test_independent_dynamics() {
    // Demonstrate usage of SIRDynamics without SIRModel (pure strategy pattern)
    let dynamics = SIRDynamics {
        n: 1000.0,
        beta: 0.5,
        gamma: 0.1,
    };

    // Initial State
    let mut state = SIRState {
        s: 990.0,
        i: 10.0,
        r: 0.0,
    };

    // Use a generic solver directly with the dynamics and the state
    let mut solver = RungeKutta4::new(&state);
    let dt = 0.1;

    // Step forward
    solver.step(&dynamics, 0.0, &mut state, dt);

    // Verify state changed appropriately
    // S should decrease (infection spreads)
    assert!(state.s < 990.0);
    // I should increase (R0 = 5 > 1)
    assert!(state.i > 10.0);
    // R should increase (recovery)
    assert!(state.r > 0.0);

    // Ensure conservation of mass (population size constant)
    let total = state.s + state.i + state.r;
    assert!((total - 1000.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_swapping_solver() {
    let n = 1000.0;
    let i0 = 10.0;
    let model = SIRModel::new(n, i0, 0.5, 0.1).unwrap();

    // Swap from RK4 (default) to Euler
    let state = *model.state();
    let mut model_euler = model.with_solver(Euler::new(&state));

    model_euler.step(0.1);

    // Just check it ran
    assert!(model_euler.state().s <= n);
}
