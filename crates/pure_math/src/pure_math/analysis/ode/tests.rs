use super::*;
use nalgebra::{DVector, Vector2, Vector3};

struct LinearTimeRamp;

impl OdeSystem<ArrayState<1>> for LinearTimeRamp {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, _state: &ArrayState<1>) -> ArrayState<1> {
        ArrayState([t])
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, _state: &ArrayState<1>, out: &mut ArrayState<1>) {
        out.0[0] = t;
    }
}

struct ModulatedDecay {
    k: f64,
}

impl OdeSystem<ArrayState<1>> for ModulatedDecay {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        ArrayState([-self.k * state.0[0] * t.sin()])
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &ArrayState<1>, out: &mut ArrayState<1>) {
        out.0[0] = -self.k * state.0[0] * t.sin();
    }
}

struct DrivenFirstOrder;

impl OdeSystem<ArrayState<1>> for DrivenFirstOrder {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        ArrayState([t.cos() - state.0[0]])
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &ArrayState<1>, out: &mut ArrayState<1>) {
        out.0[0] = t.cos() - state.0[0];
    }
}

struct StiffLinearDecay;

impl OdeSystem<ArrayState<1>> for StiffLinearDecay {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        ArrayState([-1000.0 * state.0[0]])
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, state: &ArrayState<1>, out: &mut ArrayState<1>) {
        out.0[0] = -1000.0 * state.0[0];
    }
}

#[test]
fn ode_model_integrates_linear_time_ramp_to_machine_precision() {
    let s0 = ArrayState([0.0]);
    let mut rk4_model = OdeModel::new(s0, LinearTimeRamp, RungeKutta4::new(&s0));
    for _ in 0..100 {
        rk4_model.step(0.1);
    }
    assert!((rk4_model.current_time() - 10.0).abs() < 1e-12);
    assert!(
        (rk4_model.get_state().0[0] - 50.0).abs() < 1e-12,
        "RK4 expected 50.0, got {}",
        rk4_model.get_state().0[0]
    );

    let mut dp45_model = OdeModel::new(s0, LinearTimeRamp, DormandPrince45::new(&s0));
    for _ in 0..20 {
        dp45_model.step(0.5);
    }
    assert!((dp45_model.current_time() - 10.0).abs() < 1e-12);
    assert!(
        (dp45_model.get_state().0[0] - 50.0).abs() < 1e-12,
        "DormandPrince45 expected 50.0, got {}",
        dp45_model.get_state().0[0]
    );
}

#[test]
fn non_autonomous_modulated_decay_and_driven_system_match_analytical() {
    // y'(t) = -k * y * sin(t), y(0) = 1 => y(t) = exp(k * (cos(t) - 1))
    let k = 1.5;
    let t_end = 4.0;
    let steps = 200;
    let dt = t_end / steps as f64;
    let exact_mod = (k * (t_end.cos() - 1.0)).exp();

    let s0 = ArrayState([1.0]);
    let mut rk4 = OdeModel::new(s0, ModulatedDecay { k }, RungeKutta4::new(&s0));
    let mut dp45 = OdeModel::new(s0, ModulatedDecay { k }, DormandPrince45::new(&s0));
    let mut ros = OdeModel::new(s0, ModulatedDecay { k }, Rosenbrock23::new(&s0));
    for _ in 0..steps {
        rk4.step(dt);
        dp45.step(dt);
        ros.step(dt);
    }
    assert!((rk4.get_state().0[0] - exact_mod).abs() < 1e-7);
    assert!((dp45.get_state().0[0] - exact_mod).abs() < 1e-8);
    assert!((ros.get_state().0[0] - exact_mod).abs() < 1e-5);

    // y'(t) = cos(t) - y, y(0) = 0 => y(t) = 0.5 * (sin(t) + cos(t) - exp(-t))
    let exact_driven = 0.5 * (t_end.sin() + t_end.cos() - (-t_end).exp());
    let z0 = ArrayState([0.0]);
    let mut driven_dp = OdeModel::new(z0, DrivenFirstOrder, DormandPrince45::new(&z0));
    let mut driven_trap = OdeModel::new(z0, DrivenFirstOrder, ImplicitTrapezoidal::new(&z0));
    for _ in 0..steps {
        driven_dp.step(dt);
        driven_trap.step(dt);
    }
    assert!((driven_dp.get_state().0[0] - exact_driven).abs() < 1e-8);
    assert!((driven_trap.get_state().0[0] - exact_driven).abs() < 1e-4);
}

#[test]
fn implicit_and_rosenbrock_solvers_keep_stiff_decay_bounded_and_positive() {
    let s0 = ArrayState([1.0]);
    let dt = 0.05;
    let mut be = OdeModel::new(s0, StiffLinearDecay, BackwardEuler::new(&s0));
    let mut trap = OdeModel::new(s0, StiffLinearDecay, ImplicitTrapezoidal::new(&s0));
    let mut ros = OdeModel::new(s0, StiffLinearDecay, Rosenbrock23::new(&s0));

    for step_idx in 0..20 {
        be.step(dt);
        trap.step(dt);
        ros.step(dt);

        let y_be = be.get_state().0[0];
        let y_trap = trap.get_state().0[0];
        let y_ros = ros.get_state().0[0];

        assert!(
            y_be.is_finite() && y_be > 0.0 && y_be < 1.0,
            "BackwardEuler diverged or lost positivity at step {step_idx}: {y_be}"
        );
        assert!(
            y_trap.is_finite() && y_trap > 0.0 && y_trap < 1.0,
            "ImplicitTrapezoidal diverged or lost positivity at step {step_idx}: {y_trap}"
        );
        assert!(
            y_ros.is_finite() && (0.0..1.0).contains(&y_ros),
            "Rosenbrock23 diverged or lost positivity at step {step_idx}: {y_ros}"
        );
        if step_idx == 0 {
            assert!(
                y_ros > 0.0,
                "Rosenbrock23 should remain positive at t = 0.05, got {y_ros}"
            );
        }
    }
}

#[test]
fn indexed_state_works_across_stack_nalgebra_and_heap_containers() {
    let mut v2 = Vector2::new(1.0, 2.0);
    assert_eq!(IndexedState::len(&v2), 2);
    IndexedState::set(&mut v2, 1, 4.0);
    assert_eq!(IndexedState::get(&v2, 1), 4.0);

    let mut v3 = Vector3::new(1.0, 2.0, 3.0);
    assert_eq!(IndexedState::len(&v3), 3);
    IndexedState::set(&mut v3, 2, 9.0);
    assert_eq!(IndexedState::get(&v3, 2), 9.0);

    let mut dv = DVector::from_vec(vec![3.0, 5.0]);
    assert_eq!(IndexedState::len(&dv), 2);
    IndexedState::set(&mut dv, 0, -2.0);
    assert_eq!(IndexedState::get(&dv, 0), -2.0);

    let mut vs = VecState(vec![7.0, 8.0]);
    assert_eq!(vs.len(), 2);
    vs.set(0, 11.0);
    assert_eq!(vs.get(0), 11.0);
    assert_eq!(IntegrationMethod::ALL.len(), 7);
}
