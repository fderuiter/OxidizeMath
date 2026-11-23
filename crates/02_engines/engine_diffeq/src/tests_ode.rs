#![cfg_attr(any(), verified_engine::verified(opt_out = "tests"))]

use crate::{
    ArrayState, Bdf2, DormandPrince45, Euler, Heun, OdeModel, OdeSystem, Rosenbrock23, RungeKutta4,
    SliceOdeSystem, Solver, SolverWorkspace, StackWorkspace, TimeStepper, step_slice_rk4,
    step_slice_rk45,
};
use verified_engine::engine::VerifiedEngine;

#[derive(Clone, Copy)]
struct LinearTimeRamp;

impl OdeSystem<ArrayState<1>> for LinearTimeRamp {
    fn derivative(&self, t: f64, _state: &ArrayState<1>) -> ArrayState<1> {
        ArrayState([t])
    }

    fn derivative_in_place(&self, t: f64, _state: &ArrayState<1>, out: &mut ArrayState<1>) {
        out.0[0] = t;
    }
}

#[derive(Clone, Copy)]
struct StiffLinearDecay;

impl OdeSystem<ArrayState<1>> for StiffLinearDecay {
    fn derivative(&self, _t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        ArrayState([-1000.0 * state.0[0]])
    }

    fn derivative_in_place(&self, _t: f64, state: &ArrayState<1>, out: &mut ArrayState<1>) {
        out.0[0] = -1000.0 * state.0[0];
    }
}

#[derive(Clone, Copy)]
struct StiffQuadraticDecay;

impl OdeSystem<ArrayState<1>> for StiffQuadraticDecay {
    fn derivative(&self, _t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        let y = state.0[0];
        ArrayState([-5.0 * y * y])
    }

    fn derivative_in_place(&self, _t: f64, state: &ArrayState<1>, out: &mut ArrayState<1>) {
        let y = state.0[0];
        out.0[0] = -5.0 * y * y;
    }
}

#[derive(Clone, Copy)]
struct NonAutonomousSinDecay {
    k: f64,
}

impl OdeSystem<ArrayState<1>> for NonAutonomousSinDecay {
    fn derivative(&self, t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        ArrayState([-self.k * state.0[0] * t.sin()])
    }

    fn derivative_in_place(&self, t: f64, state: &ArrayState<1>, out: &mut ArrayState<1>) {
        out.0[0] = -self.k * state.0[0] * t.sin();
    }
}

#[derive(Clone, Copy)]
struct ForcedLinearCos;

impl OdeSystem<ArrayState<1>> for ForcedLinearCos {
    fn derivative(&self, t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        ArrayState([t.cos() - state.0[0]])
    }

    fn derivative_in_place(&self, t: f64, state: &ArrayState<1>, out: &mut ArrayState<1>) {
        out.0[0] = t.cos() - state.0[0];
    }
}

struct HarmonicSliceOde;

impl SliceOdeSystem for HarmonicSliceOde {
    fn derivative_slice(&self, _t: f64, state: &[f64], out: &mut [f64]) {
        out[0] = state[1];
        out[1] = -state[0];
    }
}

fn integrate_model_to_10<S: Solver<ArrayState<1>>>(solver: S) -> (f64, f64) {
    let init = ArrayState([0.0]);
    let mut model = OdeModel::new(init, LinearTimeRamp, solver);
    let dt = 0.05;
    for _ in 0..200 {
        model.step(dt);
    }
    (model.get_state().0[0], model.get_time())
}

#[test]
fn test_ode_model_integrates_y_prime_eq_t_to_50_within_1e_12() {
    let init = ArrayState([0.0]);
    let (y_heun, t_heun) = integrate_model_to_10(Heun::new(&init));
    let (y_rk4, t_rk4) = integrate_model_to_10(RungeKutta4::new(&init));
    let (y_rk45, t_rk45) = integrate_model_to_10(DormandPrince45::new(&init));
    let (y_ros, t_ros) = integrate_model_to_10(Rosenbrock23::new(&init));
    let (y_bdf2, t_bdf2) = integrate_model_to_10(Bdf2::new(&init));

    for (name, y, t) in [
        ("Heun", y_heun, t_heun),
        ("RungeKutta4", y_rk4, t_rk4),
        ("DormandPrince45", y_rk45, t_rk45),
        ("Rosenbrock23", y_ros, t_ros),
        ("Bdf2", y_bdf2, t_bdf2),
    ] {
        assert!((t - 10.0).abs() < 1e-12, "{name} time = {t}");
        assert!(
            (y - 50.0).abs() < 1e-12,
            "{name} y(10) = {y}, err = {:e}",
            (y - 50.0).abs()
        );
    }

    // Verify TimeStepper::step_with also advances elapsed time accurately
    let mut model_ext = OdeModel::new(init, LinearTimeRamp, Euler::new(&init));
    let mut rk4 = RungeKutta4::new(&init);
    for _ in 0..200 {
        model_ext.step_with(&mut rk4, 0.05);
    }
    assert!((model_ext.get_state().0[0] - 50.0).abs() < 1e-12);
    assert!((model_ext.get_time() - 10.0).abs() < 1e-12);
}

#[test]
fn test_stiff_solvers_integrate_linear_and_quadratic_decay_without_overflow() {
    let dt = 0.05;
    let steps = 200; // T = 10.0

    for use_bdf2 in [false, true] {
        let mut y_lin = ArrayState([1.0]);
        let mut y_quad = ArrayState([1.0]);
        if use_bdf2 {
            let mut s1 = Bdf2::new(&y_lin);
            let mut s2 = Bdf2::new(&y_quad);
            for i in 0..steps {
                let t = (i as f64) * dt;
                s1.step(&StiffLinearDecay, t, &mut y_lin, dt);
                s2.step(&StiffQuadraticDecay, t, &mut y_quad, dt);
                assert!(y_lin.is_finite());
                assert!(y_quad.is_finite());
            }
        } else {
            let mut s1 = Rosenbrock23::new(&y_lin);
            let mut s2 = Rosenbrock23::new(&y_quad);
            for i in 0..steps {
                let t = (i as f64) * dt;
                s1.step(&StiffLinearDecay, t, &mut y_lin, dt);
                s2.step(&StiffQuadraticDecay, t, &mut y_quad, dt);
                assert!(y_lin.is_finite());
                assert!(y_quad.is_finite());
            }
        }
        assert!(
            y_lin.0[0].abs() < 1e-6,
            "stiff linear final = {}",
            y_lin.0[0]
        );
        let exact_quad = 1.0 / (1.0 + 5.0 * 10.0); // 1 / 51
        assert!(
            (y_quad.0[0] - exact_quad).abs() < 5e-3,
            "stiff quadratic final = {}, exact = {exact_quad}",
            y_quad.0[0]
        );
    }
}

#[test]
fn test_non_autonomous_analytical_accuracy_rk4_and_dormand_prince() {
    let k = 1.5;
    let dt = 0.02;
    let steps = 250; // T = 5.0
    let t_end = (steps as f64) * dt;
    // Exact solution for y' = -k * y * sin(t), y(0) = 1: y(t) = exp(k * (cos(t) - 1))
    let exact_sin = (k * (t_end.cos() - 1.0)).exp();
    // Exact solution for y' = cos(t) - y, y(0) = 1: y(t) = 0.5*(sin(t) + cos(t)) + 0.5*exp(-t)
    let exact_cos = 0.5 * (t_end.sin() + t_end.cos()) + 0.5 * (-t_end).exp();

    let mut m_sin_rk4 = OdeModel::new(
        ArrayState([1.0]),
        NonAutonomousSinDecay { k },
        RungeKutta4::new(&ArrayState([1.0])),
    );
    let mut m_sin_dp = OdeModel::new(
        ArrayState([1.0]),
        NonAutonomousSinDecay { k },
        DormandPrince45::new(&ArrayState([1.0])),
    );
    let mut m_cos_rk4 = OdeModel::new(
        ArrayState([1.0]),
        ForcedLinearCos,
        RungeKutta4::new(&ArrayState([1.0])),
    );
    let mut m_cos_dp = OdeModel::new(
        ArrayState([1.0]),
        ForcedLinearCos,
        DormandPrince45::new(&ArrayState([1.0])),
    );

    for _ in 0..steps {
        m_sin_rk4.step(dt);
        m_sin_dp.step(dt);
        m_cos_rk4.step(dt);
        m_cos_dp.step(dt);
    }

    assert!((m_sin_rk4.state.0[0] - exact_sin).abs() < 1e-6);
    assert!((m_sin_dp.state.0[0] - exact_sin).abs() < 1e-7);
    assert!((m_cos_rk4.state.0[0] - exact_cos).abs() < 1e-6);
    assert!((m_cos_dp.state.0[0] - exact_cos).abs() < 1e-7);
}

#[test]
fn test_zero_heap_allocations_under_verified_engine_for_all_ode_solvers() {
    let init = ArrayState([1.0]);
    let mut euler = Euler::new(&init);
    let mut heun = Heun::new(&init);
    let mut rk4 = RungeKutta4::new(&init);
    let mut rk45 = DormandPrince45::new(&init);
    let mut ros = Rosenbrock23::new(&init);
    let mut bdf = Bdf2::new(&init);
    let mut stack_ws = StackWorkspace::<64>::new();
    let mut pivots = [0_usize; 4];

    let (_, _metrics) = VerifiedEngine::run_verified(|| {
        let mut s = init;
        euler.step(&ForcedLinearCos, 0.0, &mut s, 0.05);
        heun.step(&ForcedLinearCos, 0.05, &mut s, 0.05);
        rk4.step(&ForcedLinearCos, 0.10, &mut s, 0.05);
        rk45.step(&ForcedLinearCos, 0.15, &mut s, 0.05);
        ros.step(&StiffLinearDecay, 0.0, &mut s, 0.05);
        bdf.step(&StiffQuadraticDecay, 0.0, &mut s, 0.05);
        bdf.step(&StiffQuadraticDecay, 0.05, &mut s, 0.05);

        let mut ws = stack_ws.as_workspace();
        let _ = ros.step_in_workspace(&StiffLinearDecay, 0.1, &mut s, 0.05, &mut ws, &mut pivots);
        let mut raw_slice = [1.0_f64, 0.0];
        let mut raw_buf = [0.0_f64; 32];
        let mut slice_ws = SolverWorkspace::from_slice(&mut raw_buf);
        let _ = step_slice_rk4(&HarmonicSliceOde, 0.0, &mut raw_slice, 0.05, &mut slice_ws);
        let _ = step_slice_rk45(
            &HarmonicSliceOde,
            0.05,
            &mut raw_slice,
            0.2,
            1e-8,
            1e-10,
            &mut slice_ws,
        );
        assert!(s.is_finite() && raw_slice[0].is_finite());
    });
}
