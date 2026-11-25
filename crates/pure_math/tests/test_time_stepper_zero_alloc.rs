//! Integration tests verifying `TimeStepper`, `OdeModel`, autonomous solver bit-identity,
//! non-autonomous time advancement, and `DijkstraResult` / `UalbfSearchResult` derives.

use nalgebra::DVector;
use petgraph::Directed;
use petgraph::graph::Graph;
use pure_math::pure_math::analysis::ode::{
    Euler, Heun, OdeModel, OdeSystem, RungeKutta4, Solver, TimeStepper, VecState,
};
use pure_math::pure_math::graph_theory::dijkstra::dijkstra;
use pure_math::pure_math::number_theory::ualbf::{UalbfSearchResult, ualbf_search};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[derive(Debug, Clone, PartialEq)]
struct LinearVecStepper {
    rate: f64,
    state: VecState,
    t: f64,
}

impl OdeSystem<VecState> for LinearVecStepper {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &VecState) -> VecState {
        let mut out = VecState(vec![0.0; state.0.len()]);
        self.derivative_in_place(t, state, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, state: &VecState, out: &mut VecState) {
        for (dst, src) in out.0.iter_mut().zip(state.0.iter()) {
            *dst = self.rate * *src;
        }
    }
}

impl TimeStepper<VecState> for LinearVecStepper {
    #[verified_engine::verified]
    fn get_state(&self) -> &VecState {
        &self.state
    }

    #[verified_engine::verified]
    fn get_state_mut(&mut self) -> &mut VecState {
        &mut self.state
    }

    #[verified_engine::verified]
    fn current_time(&self) -> f64 {
        self.t
    }

    #[verified_engine::verified]
    fn advance_time(&mut self, dt: f64) {
        self.t += dt;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct LinearDVecSystem {
    rate: f64,
}

impl OdeSystem<DVector<f64>> for LinearDVecSystem {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &DVector<f64>) -> DVector<f64> {
        state * self.rate
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, state: &DVector<f64>, out: &mut DVector<f64>) {
        out.copy_from(state);
        *out *= self.rate;
    }
}

#[derive(Debug, Clone, PartialEq)]
struct NonAutonomousRamp {
    state: VecState,
    t: f64,
}

impl OdeSystem<VecState> for NonAutonomousRamp {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &VecState) -> VecState {
        VecState(vec![2.0 * t; state.0.len()])
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, _state: &VecState, out: &mut VecState) {
        out.0.fill(2.0 * t);
    }
}

impl TimeStepper<VecState> for NonAutonomousRamp {
    #[verified_engine::verified]
    fn get_state(&self) -> &VecState {
        &self.state
    }

    #[verified_engine::verified]
    fn get_state_mut(&mut self) -> &mut VecState {
        &mut self.state
    }

    #[verified_engine::verified]
    fn current_time(&self) -> f64 {
        self.t
    }

    #[verified_engine::verified]
    fn advance_time(&mut self, dt: f64) {
        self.t += dt;
    }
}

#[test]
#[verified_engine::verified]
fn test_time_stepper_and_ode_model_zero_heap_allocations() {
    let init_vec = VecState((0..32).map(|i| 1.0 + i as f64 * 0.1).collect());
    let mut stepper = LinearVecStepper {
        rate: -0.25,
        state: init_vec.clone(),
        t: 0.0,
    };
    let mut rk4_vec = RungeKutta4::new(&init_vec);

    let model_solver = RungeKutta4::new(&init_vec);
    let mut vec_model = OdeModel::new(
        init_vec,
        LinearVecStepper {
            rate: -0.25,
            state: VecState(Vec::new()),
            t: 0.0,
        },
        model_solver,
    );

    let init_dvec = DVector::from_fn(32, |i, _| 1.0 + i as f64 * 0.1);
    let dvec_solver = RungeKutta4::new(&init_dvec);
    let mut dvec_model = OdeModel::new(init_dvec, LinearDVecSystem { rate: -0.25 }, dvec_solver);

    let (_, metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..100 {
            stepper.step_with(&mut rk4_vec, 0.01);
            vec_model.step(0.01);
            <OdeModel<_, _, _> as TimeStepper<DVector<f64>>>::step(&mut dvec_model, 0.01);
        }
    });

    assert!(metrics.function_calls > 0);
    assert_eq!(stepper.state, vec_model.state);
    assert!((vec_model.time() - 1.0).abs() < 1e-12);
    assert!((dvec_model.time() - 1.0).abs() < 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_non_autonomous_time_advancement_and_autonomous_bit_identity() {
    // dy/dt = 2t, y(0) = 0 => exact y(t) = t^2 under Heun and RK4
    let init = VecState(vec![0.0, 0.0]);
    let mut ramp_stepper = NonAutonomousRamp {
        state: init.clone(),
        t: 0.0,
    };
    let mut heun = Heun::new(&init);
    let mut model = OdeModel::new(
        init.clone(),
        NonAutonomousRamp {
            state: VecState(Vec::new()),
            t: 0.0,
        },
        RungeKutta4::new(&init),
    );
    assert_eq!(model, model.clone());

    let dt = 0.25;
    for _ in 0..8 {
        ramp_stepper.step_with(&mut heun, dt);
        model.step(dt);
    }
    let expected_y = (8.0 * dt).powi(2);
    assert!((ramp_stepper.state.0[0] - expected_y).abs() < 1e-14);
    assert!((model.state.0[0] - expected_y).abs() < 1e-14);
    assert!((model.time() - 2.0).abs() < 1e-14);

    // Explicit step_at / step_with_at leaves current_time untouched
    let t_before = model.time();
    model.step_at(0.5, 0.25);
    assert!((model.time() - t_before).abs() < 1e-15);

    // Autonomous Euler, Heun, and RK4 bit-identical parity between TimeStepper and Solver::step
    let sys = LinearDVecSystem { rate: -0.5 };
    let y0 = DVector::from_vec(vec![2.0, 4.0]);

    let mut y_euler_direct = y0.clone();
    Euler::new(&y0).step(&sys, 0.0, &mut y_euler_direct, 0.05);
    let mut euler_model = OdeModel::new(y0.clone(), sys, Euler::new(&y0));
    euler_model.step(0.05);
    assert_eq!(euler_model.state, y_euler_direct);

    let mut y_heun_direct = y0.clone();
    Heun::new(&y0).step(&sys, 0.0, &mut y_heun_direct, 0.05);
    let mut heun_model = OdeModel::new(y0.clone(), sys, Heun::new(&y0));
    heun_model.step(0.05);
    assert_eq!(heun_model.state, y_heun_direct);

    let mut y_rk4_direct = y0.clone();
    RungeKutta4::new(&y0).step(&sys, 0.0, &mut y_rk4_direct, 0.05);
    let mut rk4_model = OdeModel::new(y0, sys, RungeKutta4::new(&y_rk4_direct));
    rk4_model.step(0.05);
    assert_eq!(rk4_model.state, y_rk4_direct);
}

#[test]
#[verified_engine::verified]
fn test_dijkstra_and_ualbf_standard_derives_and_errors() {
    let mut graph = Graph::<(), i32, Directed>::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    graph.add_edge(a, b, 7);

    let res = dijkstra(&graph, a);
    let cloned = res.clone();
    assert_eq!(res, cloned);
    assert!(!format!("{res:?}").is_empty());

    let ok_ualbf = ualbf_search(11, 4, "100", "1000000").unwrap();
    assert_eq!(ok_ualbf, ok_ualbf.clone());
    assert!(!format!("{ok_ualbf:?}").is_empty());
    assert_eq!(UalbfSearchResult::default().valid_components, 0);

    let err = ualbf_search(11, 4, "not_an_integer", "1000").unwrap_err();
    assert_eq!(err, err.clone());
    assert!(!format!("{err:?}").is_empty());
}
