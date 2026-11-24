//! Zero-allocation runtime verification for extended analysis root, quadrature, and ODE kernels.

use math_commons::registry::{MAX_ITERATIONS, TOLERANCE_FAST};
use nalgebra::DVector;
use pure_math::pure_math::analysis::integration::{
    GK15_MAX_PANELS, GaussKronrod15, GkSubinterval, Integrator, Simpson, Trapezoidal,
};
use pure_math::pure_math::analysis::ode::{
    ArrayState, BackwardEuler, DormandPrince45, ImplicitTrapezoidal, OdeModel, OdeSystem,
    Rosenbrock23, RungeKutta4, TimeStepper, VecState,
};
use pure_math::pure_math::analysis::roots::{
    Bisection, Brent, MultivariateNewton, NewtonRaphson, RootFinder, Secant,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

struct CoupledOscillator;

impl OdeSystem<ArrayState<2>> for CoupledOscillator {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &ArrayState<2>) -> ArrayState<2> {
        ArrayState([state.0[1], -state.0[0] + 0.1 * t.cos()])
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &ArrayState<2>, out: &mut ArrayState<2>) {
        out.0[0] = state.0[1];
        out.0[1] = -state.0[0] + 0.1 * t.cos();
    }
}

struct StiffVecSystem;

impl OdeSystem<VecState> for StiffVecSystem {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &VecState) -> VecState {
        VecState(vec![-1000.0 * state.0[0], -10.0 * state.0[1]])
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, state: &VecState, out: &mut VecState) {
        out.0[0] = -1000.0 * state.0[0];
        out.0[1] = -10.0 * state.0[1];
    }
}

#[test]
#[verified_engine::verified]
fn test_extended_analysis_solvers_zero_alloc_hot_paths() {
    let mv_stack = MultivariateNewton::new(MAX_ITERATIONS, TOLERANCE_FAST);
    let mut mv_dyn = MultivariateNewton::with_dimension(2, MAX_ITERATIONS, TOLERANCE_FAST);
    let mut x_dyn = DVector::from_vec(vec![0.6, 0.8]);

    let s_arr = ArrayState([1.0, 0.0]);
    let mut rk4_model = OdeModel::new(s_arr, CoupledOscillator, RungeKutta4::new(&s_arr));
    let mut dp45_model = OdeModel::new(s_arr, CoupledOscillator, DormandPrince45::new(&s_arr));
    let mut be_model = OdeModel::new(s_arr, CoupledOscillator, BackwardEuler::new(&s_arr));
    let mut trap_model = OdeModel::new(s_arr, CoupledOscillator, ImplicitTrapezoidal::new(&s_arr));
    let mut ros_model = OdeModel::new(s_arr, CoupledOscillator, Rosenbrock23::new(&s_arr));

    let s_vec = VecState(vec![1.0, 1.0]);
    let mut vec_be = OdeModel::new(s_vec.clone(), StiffVecSystem, BackwardEuler::new(&s_vec));
    let mut vec_ros = OdeModel::new(s_vec.clone(), StiffVecSystem, Rosenbrock23::new(&s_vec));

    let mut gk_panels = [GkSubinterval::default(); GK15_MAX_PANELS];

    let ((), metrics) = VerifiedEngine::run_verified(|| {
        let r_bis = Bisection::default()
            .find_root(|x| x * x - 2.0, 1.0, 2.0)
            .expect("bisection root");
        let r_brent = Brent::default()
            .find_root(|x| x.cos() - x, 0.0, 1.0)
            .expect("brent root");
        let r_nr = NewtonRaphson::default()
            .find_root(|x| x * x - 4.0, 1.0, 3.0)
            .expect("newton root");
        let r_sec = Secant::default()
            .find_root(|x| x * x * x - x - 2.0, 1.0, 2.0)
            .expect("secant root");
        assert!((r_bis - std::f64::consts::SQRT_2).abs() < TOLERANCE_FAST);
        assert!((r_brent - 0.739_085_133).abs() < TOLERANCE_FAST);
        assert!((r_nr - 2.0).abs() < TOLERANCE_FAST);
        assert!((r_sec - 1.521_379_7).abs() < TOLERANCE_FAST);

        let r_fixed = mv_stack
            .solve_fixed(
                |x| [x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]],
                [0.6, 0.8],
            )
            .expect("fixed multivariate newton");
        assert!((r_fixed[0] - std::f64::consts::FRAC_1_SQRT_2).abs() < TOLERANCE_FAST);

        mv_dyn
            .solve_dvector_in_place(
                |x, out| {
                    out[0] = x[0] * x[0] + x[1] * x[1] - 1.0;
                    out[1] = x[0] - x[1];
                },
                &mut x_dyn,
            )
            .expect("dynamic multivariate newton");

        let q_trap = Trapezoidal::new(64).integrate(|x| x.sin(), 0.0, std::f64::consts::PI, 1e-6);
        let q_simp = Simpson::new(64).integrate(|x| x.sin(), 0.0, std::f64::consts::PI, 1e-8);
        let (q_gk, panel_n) = GaussKronrod15::new().integrate_panels(
            |x| (3.0 * x).sin().clamp(-0.5, 0.8),
            0.0,
            2.0,
            1e-8,
            &mut gk_panels,
        );
        assert!(q_trap.error > 0.0 && q_simp.error > 0.0 && q_gk.error > 0.0 && panel_n >= 4);

        rk4_model.step(0.05);
        dp45_model.step(0.05);
        be_model.step(0.05);
        trap_model.step(0.05);
        ros_model.step(0.05);
        vec_be.step(0.05);
        vec_ros.step(0.05);
        assert!(vec_be.get_state().0[0] > 0.0 && vec_ros.get_state().0[0] > 0.0);
    });

    assert!(metrics.function_calls > 0);
    assert!(!verified_engine::allocator::MEMORY_LOCK.load(std::sync::atomic::Ordering::SeqCst));
}
