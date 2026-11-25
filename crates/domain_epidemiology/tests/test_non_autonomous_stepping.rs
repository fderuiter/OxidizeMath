//! Integration tests verifying `SIRModel` and `SEIRModel` advance elapsed simulation time `t`
//! across steps and support `Debug`, `Clone`, `PartialEq`, `assert_eq!`, and `unwrap_err()`.

use domain_epidemiology::epidemiology::compartmental::{SEIRModel, SEIRState, SIRModel, SIRState};
use pure_math::pure_math::analysis::ode::{OdeSystem, Solver, TimeStepper};

#[derive(Debug, Clone, PartialEq, Default)]
struct TimeRecordingSolver {
    recorded_times: Vec<f64>,
}

impl<State: pure_math::pure_math::analysis::ode::VectorOperations> Solver<State>
    for TimeRecordingSolver
{
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        self.recorded_times.push(t);
        let deriv = system.derivative(t, state);
        *state += deriv * dt;
    }
}

#[test]
#[verified_engine::verified]
fn test_sir_model_advances_time_and_supports_derives() {
    let base = SIRModel::builder()
        .n(1000.0)
        .i0(10.0)
        .beta(0.3)
        .gamma(0.1)
        .build()
        .unwrap();
    assert_eq!(base, base.clone());
    assert!(!format!("{base:?}").is_empty());
    assert_eq!(base.time(), 0.0);

    let mut model = base.with_solver(TimeRecordingSolver::default());
    let dt = 0.5;
    for _ in 0..4 {
        model.step(dt);
    }
    assert!((model.time() - 2.0).abs() < 1e-15);

    let mut probe = TimeRecordingSolver::default();
    model.step_with(&mut probe, dt);
    assert_eq!(probe.recorded_times, vec![2.0]);
    assert!((model.time() - 2.5).abs() < 1e-15);

    // Also verify default RungeKutta4 model advances time() across steps
    let mut rk4_sir = SIRModel::new(1000.0, 10.0, 0.3, 0.1).unwrap();
    <SIRModel<_> as TimeStepper<SIRState>>::step(&mut rk4_sir, 0.25);
    <SIRModel<_> as TimeStepper<SIRState>>::step(&mut rk4_sir, 0.25);
    assert!((rk4_sir.time() - 0.5).abs() < 1e-15);

    let err = SIRModel::builder()
        .n(-5.0)
        .i0(1.0)
        .beta(0.3)
        .gamma(0.1)
        .build()
        .unwrap_err();
    assert_eq!(err, err.clone());
    assert!(!format!("{err:?}").is_empty());
}

#[test]
#[verified_engine::verified]
fn test_seir_model_advances_time_and_supports_derives() {
    let base = SEIRModel::builder()
        .n(1000.0)
        .i0(10.0)
        .beta(0.4)
        .sigma(0.2)
        .gamma(0.1)
        .build()
        .unwrap();
    assert_eq!(base, base.clone());
    assert!(!format!("{base:?}").is_empty());
    assert_eq!(base.time(), 0.0);

    let mut model = base.with_solver(TimeRecordingSolver::default());
    let dt = 0.25;
    for _ in 0..4 {
        model.step(dt);
    }
    assert!((model.time() - 1.0).abs() < 1e-15);

    let mut probe = TimeRecordingSolver::default();
    model.step_with(&mut probe, dt);
    assert_eq!(probe.recorded_times, vec![1.0]);
    assert!((model.time() - 1.25).abs() < 1e-15);

    let mut rk4_seir = SEIRModel::new(1000.0, 10.0, 0.4, 0.2, 0.1).unwrap();
    <SEIRModel<_> as TimeStepper<SEIRState>>::step(&mut rk4_seir, 0.5);
    assert!((rk4_seir.time() - 0.5).abs() < 1e-15);

    let err = SEIRModel::builder().n(100.0).build().unwrap_err();
    assert_eq!(err, err.clone());
    assert!(!format!("{err:?}").is_empty());
}
