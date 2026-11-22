use super::*;
use crate::pure_math::analysis::ode::{ArrayState, OdeSystem, RungeKutta4, Solver};
use crate::pure_math::analysis::optimization::least_squares::{
    LevenbergMarquardtSolver, LevenbergMarquardtWorkspace, ParameterBounds,
};
use verified_engine::engine::VerifiedEngine;

struct LogisticOde {
    r: f64,
    cap_k: f64,
}

impl OdeSystem<ArrayState<1>> for LogisticOde {
    fn derivative(&self, _t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        let y = state.0[0];
        ArrayState([self.r * y * (1.0 - y / self.cap_k)])
    }
}

struct SirTestOde {
    n: f64,
    beta: f64,
    gamma: f64,
}

impl OdeSystem<ArrayState<3>> for SirTestOde {
    fn derivative(&self, _t: f64, state: &ArrayState<3>) -> ArrayState<3> {
        let s = state.0[0];
        let i = state.0[1];
        let inf = self.beta * s * i / self.n;
        let rec = self.gamma * i;
        ArrayState([-inf, inf - rec, rec])
    }
}

#[test]
#[verified_engine::verified]
fn test_rk4_shooting_recovers_logistic_growth_zero_alloc() {
    let true_r = 0.85_f64;
    let true_k = 120.0_f64;
    let true_y0 = 8.0_f64;

    let times: [f64; 11] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
    let mut y_obs = [0.0_f64; 11];
    for (idx, &t) in times.iter().enumerate() {
        let ratio = (true_k - true_y0) / true_y0;
        y_obs[idx] = true_k / (1.0 + ratio * (-true_r * t).exp());
    }

    let example = ArrayState([true_y0]);
    let mut shooting = Rk4ShootingObjective::new(
        LogisticOde {
            r: 0.4,
            cap_k: 80.0,
        },
        &example,
        &times,
        &y_obs,
        0.01,
        |sys, p, _t0, st| {
            if p[1] <= 1e-6 {
                return false;
            }
            sys.r = p[0];
            sys.cap_k = p[1];
            st.0[0] = p[2];
            true
        },
        |st| st.0[0],
        |st| st.0[0].is_finite(),
    )
    .expect("valid shooting objective");

    let bounds = [
        ParameterBounds::new(0.05, 5.0).unwrap(),
        ParameterBounds::new(10.0, 500.0).unwrap(),
        ParameterBounds::new(0.5, 80.0).unwrap(),
    ];
    let initial = [0.45, 85.0, 12.0];
    let mut ws = LevenbergMarquardtWorkspace::new(16, 4).unwrap();
    let solver = LevenbergMarquardtSolver::default();

    let (res, _) = VerifiedEngine::run_verified(|| {
        solver.solve_in_place(&mut shooting, &initial, &bounds, &mut ws)
    });
    let summary = ws.to_summary(&res.expect("logistic shooting converged"));

    assert!(((summary.parameters[0] - true_r) / true_r).abs() < 1e-3);
    assert!(((summary.parameters[1] - true_k) / true_k).abs() < 1e-3);
    assert!(((summary.parameters[2] - true_y0) / true_y0).abs() < 1e-3);
    assert!(summary.rmse < 1e-4);
    assert!((summary.r_squared - 1.0).abs() < 1e-7);
}

fn generate_synthetic_sir_series(
    pop_n: f64,
    beta: f64,
    gamma: f64,
    i0: f64,
) -> ([f64; 21], [f64; 21]) {
    let mut times = [0.0_f64; 21];
    let mut infected_obs = [0.0_f64; 21];
    let truth_sys = SirTestOde {
        n: pop_n,
        beta,
        gamma,
    };
    let mut rk4 = RungeKutta4::new(&ArrayState([0.0; 3]));
    let mut state = ArrayState([pop_n - i0, i0, 0.0]);
    infected_obs[0] = state.0[1];
    for step_idx in 1..21 {
        let t_prev = ((step_idx - 1) * 2) as f64;
        times[step_idx] = (step_idx * 2) as f64;
        let dt = 2.0 / 100.0;
        for s in 0..100 {
            rk4.step(&truth_sys, t_prev + (s as f64) * dt, &mut state, dt);
        }
        infected_obs[step_idx] = state.0[1];
    }
    (times, infected_obs)
}

#[test]
#[verified_engine::verified]
fn test_rk4_shooting_recovers_sir_outbreak_parameters_zero_alloc() {
    let pop_n = 1_000.0_f64;
    let (true_beta, true_gamma, true_i0) = (0.42_f64, 0.14_f64, 5.0_f64);
    let (times, infected_obs) =
        generate_synthetic_sir_series(pop_n, true_beta, true_gamma, true_i0);

    let example = ArrayState([pop_n - 10.0, 10.0, 0.0]);
    let mut shooting = Rk4ShootingObjective::new(
        SirTestOde {
            n: pop_n,
            beta: 0.25,
            gamma: 0.1,
        },
        &example,
        &times,
        &infected_obs,
        0.02,
        move |sys, p, _t0, st| {
            let (beta, gamma, i0) = (p[0], p[1], p[2]);
            if beta <= 0.0 || gamma <= 0.0 || i0 <= 0.0 || i0 >= pop_n {
                return false;
            }
            sys.beta = beta;
            sys.gamma = gamma;
            st.0 = [pop_n - i0, i0, 0.0];
            true
        },
        |st| st.0[1],
        |st| st.0.iter().all(|v| v.is_finite()),
    )
    .expect("valid SIR shooting objective");

    let bounds = [
        ParameterBounds::new(0.02, 3.0).unwrap(),
        ParameterBounds::new(0.01, 1.5).unwrap(),
        ParameterBounds::new(0.5, 200.0).unwrap(),
    ];
    let mut ws = LevenbergMarquardtWorkspace::new(32, 4).unwrap();
    let solver = LevenbergMarquardtSolver::default();

    let (res, _) = VerifiedEngine::run_verified(|| {
        solver.solve_in_place(&mut shooting, &[0.28, 0.20, 12.0], &bounds, &mut ws)
    });
    let summary = ws.to_summary(&res.expect("SIR shooting converged"));

    assert!(((summary.parameters[0] - true_beta) / true_beta).abs() < 1e-3);
    assert!(((summary.parameters[1] - true_gamma) / true_gamma).abs() < 1e-3);
    assert!(((summary.parameters[2] - true_i0) / true_i0).abs() < 1e-3);
    assert!(summary.rmse < 1e-3);
}
