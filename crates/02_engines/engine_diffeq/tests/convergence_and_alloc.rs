// [cite:engine_diffeq]
//! Convergence-order, Itô/Stratonovich calculus, BAOAB equilibrium, and zero-allocation verification.

use engine_diffeq::{
    BaoabLangevinSolver, BaoabParams, CorrelatedWienerIncrement, EulerMaruyamaSolver,
    LangevinSystem, MilsteinSolver, OwnedWorkspace, SdeSystem, Sriw1Solver, StochasticCalculus,
    ssa_sample_event,
};
use math_commons::math_kernel::types::SolverWorkspace;
use math_rng::OxidizeRng;
use verified_engine::engine::VerifiedEngine;

/// Nonlinear hyperbolic-sine SDE whose exact solution is $X_t = \sinh(\operatorname{arsinh}(X_0) + \mu t + \sigma W_t)$.
///
/// - Stratonovich: $dX_t = \mu \sqrt{1 + X_t^2}\,dt + \sigma \sqrt{1 + X_t^2} \circ dW_t$.
/// - Itô: $dX_t = \left(\mu \sqrt{1 + X_t^2} + \frac{1}{2}\sigma^2 X_t\right) dt + \sigma \sqrt{1 + X_t^2}\,dW_t$.
struct NonlinearSinhSde {
    mu: f64,
    sigma: f64,
    calculus: StochasticCalculus,
}

impl SdeSystem<f64> for NonlinearSinhSde {
    fn state_dim(&self) -> usize {
        1
    }
    fn wiener_dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out_drift: &mut [f64]) {
        let base = self.mu * (1.0 + x[0] * x[0]).sqrt();
        out_drift[0] = match self.calculus {
            StochasticCalculus::Ito => base + 0.5 * self.sigma * self.sigma * x[0],
            StochasticCalculus::Stratonovich => base,
        };
    }
    fn diffusion(&self, _t: f64, x: &[f64], out_diffusion: &mut [f64]) {
        out_diffusion[0] = self.sigma * (1.0 + x[0] * x[0]).sqrt();
    }
}

fn exact_sinh_solution(x0: f64, mu: f64, sigma: f64, t: f64, w_t: f64) -> f64 {
    (x0.asinh() + mu * t + sigma * w_t).sinh()
}

fn linear_regression_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let num: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - mx) * (y - my))
        .sum();
    let den: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    num / den
}

#[allow(clippy::too_many_arguments)]
fn integrate_coarsened_path(
    sys: &NonlinearSinhSde,
    em: &EulerMaruyamaSolver,
    milstein: &MilsteinSolver,
    sriw1: &Sriw1Solver,
    fine_dw: &[f64],
    fine_dz: &[f64],
    dt_fine: f64,
    ratio: usize,
    x0: f64,
) -> (f64, f64, f64) {
    let steps = fine_dw.len() / ratio;
    let dt = dt_fine * (ratio as f64);
    let mut x_em = [x0];
    let mut x_mil = [x0];
    let mut x_srk = [x0];
    let mut scratch = [0.0_f64; 64];

    for s in 0..steps {
        let mut dw_c = 0.0;
        let mut dz_c = 0.0;
        let mut w_local = 0.0;
        for sub in 0..ratio {
            let idx = s * ratio + sub;
            dz_c += w_local * dt_fine + fine_dz[idx];
            w_local += fine_dw[idx];
            dw_c += fine_dw[idx];
        }
        let t = (s as f64) * dt;
        let mut ws = SolverWorkspace::new(&mut scratch);
        em.step_with_increments(sys, t, dt, &[dw_c], &mut x_em, &mut ws);
        milstein.step_with_increments(sys, t, dt, &[dw_c], &mut x_mil, &mut ws);
        sriw1.step_with_correlated_increments(sys, t, dt, &[dw_c], &[dz_c], &mut x_srk, &mut ws);
    }
    (x_em[0], x_mil[0], x_srk[0])
}

#[test]
fn test_strong_convergence_orders_em_milstein_sriw1() {
    let (x0, mu, sigma, n_fine, n_paths) = (0.4_f64, 0.35_f64, 0.65_f64, 512_usize, 240_usize);
    let dt_fine = 1.0 / (n_fine as f64);
    let ratios = [16_usize, 8, 4, 2];
    let sys_ito = NonlinearSinhSde {
        mu,
        sigma,
        calculus: StochasticCalculus::Ito,
    };
    let em = EulerMaruyamaSolver::new(StochasticCalculus::Ito);
    let milstein = MilsteinSolver::new(StochasticCalculus::Ito);
    let sriw1 = Sriw1Solver::new(StochasticCalculus::Ito);

    let mut fine_dw = vec![0.0_f64; n_fine];
    let mut fine_dz = vec![0.0_f64; n_fine];
    let mut rng = OxidizeRng::new(0xA11C_E5ED_2026_0001);
    let mut sq_err_em = [0.0_f64; 4];
    let mut sq_err_mil = [0.0_f64; 4];
    let mut sq_err_srk = [0.0_f64; 4];

    for _ in 0..n_paths {
        let mut w_total = 0.0;
        for k in 0..n_fine {
            let inc = CorrelatedWienerIncrement::sample(dt_fine, &mut rng);
            fine_dw[k] = inc.dw;
            fine_dz[k] = inc.dz;
            w_total += inc.dw;
        }
        let x_exact = exact_sinh_solution(x0, mu, sigma, 1.0, w_total);
        for (r_idx, &ratio) in ratios.iter().enumerate() {
            let (xe, xm, xs) = integrate_coarsened_path(
                &sys_ito, &em, &milstein, &sriw1, &fine_dw, &fine_dz, dt_fine, ratio, x0,
            );
            sq_err_em[r_idx] += (xe - x_exact).powi(2);
            sq_err_mil[r_idx] += (xm - x_exact).powi(2);
            sq_err_srk[r_idx] += (xs - x_exact).powi(2);
        }
    }

    let log_dts: Vec<f64> = ratios
        .iter()
        .map(|&r| (dt_fine * (r as f64)).ln())
        .collect();
    let rms_logs = |errs: &[f64; 4]| -> Vec<f64> {
        errs.iter()
            .map(|&s| (s / (n_paths as f64)).sqrt().ln())
            .collect()
    };
    let order_em = linear_regression_slope(&log_dts, &rms_logs(&sq_err_em));
    let order_mil = linear_regression_slope(&log_dts, &rms_logs(&sq_err_mil));
    let order_srk = linear_regression_slope(&log_dts, &rms_logs(&sq_err_srk));

    assert!(
        (0.38..=0.72).contains(&order_em),
        "Euler-Maruyama strong order {order_em} outside expected ~0.5 range"
    );
    assert!(
        (0.88..=1.18).contains(&order_mil),
        "Milstein strong order {order_mil} outside expected ~1.0 range"
    );
    assert!(
        order_srk >= 1.35,
        "SRIW1 strong order {order_srk} below expected ~1.5 threshold"
    );
}

#[test]
fn test_stratonovich_solvers_match_exact_chain_rule_solution() {
    let x0 = 0.5_f64;
    let mu = 0.25_f64;
    let sigma = 0.6_f64;
    let steps = 256_usize;
    let dt = 1.0 / (steps as f64);
    let sys_strat = NonlinearSinhSde {
        mu,
        sigma,
        calculus: StochasticCalculus::Stratonovich,
    };

    let heun = EulerMaruyamaSolver::new(StochasticCalculus::Stratonovich);
    let mil_strat = MilsteinSolver::new(StochasticCalculus::Stratonovich);
    let srk_strat = Sriw1Solver::new(StochasticCalculus::Stratonovich);

    let mut rng = OxidizeRng::new(0x57AA_7000_2026_0002);
    let mut scratch = [0.0_f64; 64];
    let mut x_heun = [x0];
    let mut x_mil = [x0];
    let mut x_srk = [x0];
    let mut w_total = 0.0;

    for s in 0..steps {
        let inc = CorrelatedWienerIncrement::sample(dt, &mut rng);
        w_total += inc.dw;
        let t = (s as f64) * dt;
        let mut ws = SolverWorkspace::new(&mut scratch);
        heun.step_with_increments(&sys_strat, t, dt, &[inc.dw], &mut x_heun, &mut ws);
        mil_strat.step_with_increments(&sys_strat, t, dt, &[inc.dw], &mut x_mil, &mut ws);
        srk_strat.step_with_correlated_increments(
            &sys_strat,
            t,
            dt,
            &[inc.dw],
            &[inc.dz],
            &mut x_srk,
            &mut ws,
        );
    }

    let exact = exact_sinh_solution(x0, mu, sigma, 1.0, w_total);
    assert!((x_heun[0] - exact).abs() < 2e-2, "Heun error too large");
    assert!(
        (x_mil[0] - exact).abs() < 1.5e-2,
        "Milstein Strat error too large"
    );
    assert!(
        (x_srk[0] - exact).abs() < 2e-3,
        "SRIW1 Strat error too large"
    );
}

struct HarmonicPotential2D {
    k: f64,
}

impl LangevinSystem<f64> for HarmonicPotential2D {
    fn dim(&self) -> usize {
        2
    }
    fn grad_potential(&self, q: &[f64], out_grad: &mut [f64]) {
        out_grad[0] = self.k * q[0];
        out_grad[1] = self.k * q[1];
    }
}

#[test]
fn test_baoab_deterministic_second_order_and_gibbs_boltzmann_variance() {
    let sys = HarmonicPotential2D { k: 4.0 };
    let mut scratch = [0.0_f64; 8];
    let mut rng = OxidizeRng::new(0xBA0A_B000_1234_5678);

    // 1. Deterministic limit (friction = 0, temperature = 0): global 2nd-order convergence on cos(2t)
    let exact_q = 1.0_f64.cos();
    let mut errs = [0.0_f64; 3];
    let dts = [0.05_f64, 0.025, 0.0125];
    for (idx, &dt) in dts.iter().enumerate() {
        let solver = BaoabLangevinSolver::new(BaoabParams {
            mass: 1.0,
            friction: 0.0,
            temperature: 0.0,
            dt,
        });
        let mut q = [1.0_f64, 0.0];
        let mut p = [0.0_f64, 0.0];
        let n_steps = (0.5 / dt).round() as usize;
        let mut ws = SolverWorkspace::new(&mut scratch);
        for _ in 0..n_steps {
            solver.step(&sys, &mut q, &mut p, &mut ws, &mut rng);
        }
        errs[idx] = (q[0] - exact_q).abs();
    }
    let log_dts: Vec<f64> = dts.iter().map(|d| d.ln()).collect();
    let log_errs: Vec<f64> = errs.iter().map(|e| e.ln()).collect();
    let order = linear_regression_slope(&log_dts, &log_errs);
    assert!(order > 1.95, "BAOAB deterministic order {order} < 1.95");

    // 2. Stochastic thermal equilibrium: E[q_x^2] = k_B T / k = 1.5 / 4.0 = 0.375
    let solver = BaoabLangevinSolver::new(BaoabParams {
        mass: 1.0,
        friction: 2.0,
        temperature: 1.5,
        dt: 0.05,
    });
    let mut q = [0.5_f64, -0.5];
    let mut p = [0.0_f64, 0.0];
    let mut ws = SolverWorkspace::new(&mut scratch);
    for _ in 0..500 {
        solver.step(&sys, &mut q, &mut p, &mut ws, &mut rng);
    }
    let samples = 6000_usize;
    let mut sum_q2 = 0.0;
    let mut sum_p2 = 0.0;
    for _ in 0..samples {
        solver.step(&sys, &mut q, &mut p, &mut ws, &mut rng);
        sum_q2 += 0.5 * (q[0] * q[0] + q[1] * q[1]);
        sum_p2 += 0.5 * (p[0] * p[0] + p[1] * p[1]);
    }
    let mean_q2 = sum_q2 / (samples as f64);
    let mean_p2 = sum_p2 / (samples as f64);
    assert!(
        (mean_q2 - 0.375).abs() < 0.035,
        "BAOAB configurational variance {mean_q2} != 0.375"
    );
    assert!(
        (mean_p2 - 1.5).abs() < 0.09,
        "BAOAB kinetic variance {mean_p2} != 1.5"
    );
}

#[test]
fn test_all_engine_diffeq_solvers_zero_heap_allocations_under_verified_lock() {
    let sys = NonlinearSinhSde {
        mu: 0.2,
        sigma: 0.4,
        calculus: StochasticCalculus::Stratonovich,
    };
    let harm = HarmonicPotential2D { k: 2.0 };
    let em = EulerMaruyamaSolver::new(StochasticCalculus::Stratonovich);
    let mil = MilsteinSolver::new(StochasticCalculus::Ito).with_levy_truncation(4);
    let srk = Sriw1Solver::new(StochasticCalculus::Stratonovich);
    let baoab = BaoabLangevinSolver::new(BaoabParams {
        mass: 1.0,
        friction: 1.0,
        temperature: 1.0,
        dt: 0.01,
    });

    let mut owned = OwnedWorkspace::<f64>::with_capacity(64);
    let mut state = [0.25_f64];
    let mut q = [1.0_f64, -0.5];
    let mut p = [0.1_f64, 0.2];
    let propensities = [0.5_f64, 1.5, 2.0];
    let mut rng = OxidizeRng::new(0x0A11_0C00_2026_9999);

    let ((), receipt) = VerifiedEngine::run_verified(|| {
        let mut ws = owned.as_workspace();
        for k in 0..16 {
            let t = (k as f64) * 0.01;
            em.step(&sys, t, 0.01, &mut state, &mut ws, &mut rng);
            mil.step(&sys, t, 0.01, &mut state, &mut ws, &mut rng);
            srk.step(&sys, t, 0.01, &mut state, &mut ws, &mut rng);
            baoab.step(&harm, &mut q, &mut p, &mut ws, &mut rng);
            let jump = ssa_sample_event(&propensities, &mut rng);
            assert!(jump.is_some());
        }
    });
    assert!(state[0].is_finite() && q[0].is_finite() && p[0].is_finite());
    assert!(receipt.function_calls > 0);
}
