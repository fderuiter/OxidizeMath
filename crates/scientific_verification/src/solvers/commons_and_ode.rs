#![cfg_attr(any(), verified(opt_out = "verification_harness"))]

use crate::mms::{SolverCertificate, ols_convergence_order};
use math_commons::fit::{WeightedOlsAccum, solve_dense_small};
use nalgebra::Vector2;
use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::analysis::momentum_dynamics::{
    HamiltonianLeapfrog2D, LangevinBaoab2D, PhasePoint2D,
};
use pure_math::pure_math::analysis::ode::{
    ArrayState, Euler, GaussLegendre4, HamiltonianSystem, Heun, OdeModel, OdeSystem, RungeKutta4,
    SeparableHamiltonian, Solver, StormerVerlet, TimeStepper, Yoshida4,
};

struct NonAutonomousMmsOde;

impl OdeSystem<ArrayState<1>> for NonAutonomousMmsOde {
    // Exact solution y*(t) = exp(-t) * (1 + sin(t)), so dy/dt + y = exp(-t) * cos(t)
    fn derivative(&self, t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        ArrayState([-state.0[0] + (-t).exp() * t.cos()])
    }
}

#[derive(Clone, Copy)]
struct QuarticHamiltonian1D;

impl OdeSystem<ArrayState<2>> for QuarticHamiltonian1D {
    fn derivative(&self, _t: f64, state: &ArrayState<2>) -> ArrayState<2> {
        let q = state.0[0];
        let p = state.0[1];
        ArrayState([p, -(q + q * q * q)])
    }
}

impl HamiltonianSystem<ArrayState<2>> for QuarticHamiltonian1D {
    fn hamiltonian(&self, state: &ArrayState<2>) -> f64 {
        let q = state.0[0];
        let p = state.0[1];
        0.5 * p * p + 0.5 * q * q + 0.25 * q.powi(4)
    }
}

impl SeparableHamiltonian<ArrayState<2>> for QuarticHamiltonian1D {
    type Coord = ArrayState<1>;

    fn split_canonical(&self, state: &ArrayState<2>, q: &mut Self::Coord, p: &mut Self::Coord) {
        q.0[0] = state.0[0];
        p.0[0] = state.0[1];
    }
    fn join_canonical(&self, q: &Self::Coord, p: &Self::Coord, state: &mut ArrayState<2>) {
        state.0[0] = q.0[0];
        state.0[1] = p.0[0];
    }
    fn kinetic_energy(&self, p: &Self::Coord) -> f64 {
        0.5 * p.0[0] * p.0[0]
    }
    fn potential_energy(&self, q: &Self::Coord) -> f64 {
        let x = q.0[0];
        0.5 * x * x + 0.25 * x.powi(4)
    }
    fn grad_kinetic_in_place(&self, p: &Self::Coord, dq_dt: &mut Self::Coord) {
        dq_dt.0[0] = p.0[0];
    }
    fn neg_grad_potential_in_place(&self, q: &Self::Coord, dp_dt: &mut Self::Coord) {
        let x = q.0[0];
        dp_dt.0[0] = -(x + x * x * x);
    }
}

fn certify_algebraic_fit_solvers() -> Result<Vec<SolverCertificate>, String> {
    let x_true = [1.25, -0.75, 2.5, -1.5, 0.5];
    let raw_a = [
        [4.0, 1.0, 0.2, 0.0, 0.1],
        [1.0, 5.0, 0.5, -0.2, 0.0],
        [0.2, 0.5, 6.0, 0.4, -0.1],
        [0.0, -0.2, 0.4, 4.5, 0.3],
        [0.1, 0.0, -0.1, 0.3, 3.5],
    ];
    let mut b = [0.0; 5];
    for r in 0..5 {
        for c in 0..5 {
            b[r] += raw_a[r][c] * x_true[c];
        }
    }
    let mut mat = raw_a;
    let x_num = solve_dense_small(&mut mat, &b, 5).map_err(|e| format!("{e:?}"))?;
    let err_dense = (0..5)
        .map(|i| (x_num[i] - x_true[i]).abs())
        .fold(0.0_f64, f64::max);

    let mut ols = WeightedOlsAccum::default();
    for i in 0..16 {
        let u = -1.5 + 0.25 * (i as f64);
        let v = -0.85 + 2.35 * u;
        let w = 1.0 + 0.2 * (i as f64);
        ols.push(u, v, w);
    }
    let (intercept, slope) = ols.solve().map_err(|e| format!("{e:?}"))?;
    let err_ols = (intercept - (-0.85)).abs().max((slope - 2.35).abs());

    Ok(vec![
        SolverCertificate::evaluate(
            "solve_dense_small",
            "math_commons",
            0.0,
            1.0,
            err_dense,
            1e-12,
        ),
        SolverCertificate::evaluate("WeightedOlsAccum", "math_commons", 0.0, 1.0, err_ols, 1e-12),
    ])
}

fn pm_cert(
    name: &'static str,
    min_p: f64,
    hs: &[f64],
    errs: &[f64],
    max_err: f64,
) -> SolverCertificate {
    SolverCertificate::evaluate(
        name,
        "pure_math",
        min_p,
        ols_convergence_order(hs, errs),
        errs[2],
        max_err,
    )
}

fn certify_scalar_ode_solvers() -> Vec<SolverCertificate> {
    let sys = NonAutonomousMmsOde;
    let exact = (-1.0_f64).exp() * (1.0 + 1.0_f64.sin());
    let mut hs = [0.0; 3];
    let (mut e_eu, mut e_he, mut e_rk, mut e_om) = ([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]);
    for (k, &n) in [20usize, 40, 80].iter().enumerate() {
        let dt = 1.0 / n as f64;
        hs[k] = dt;
        let y0 = ArrayState([1.0]);
        let (mut y_eu, mut y_he, mut y_rk) = (y0, y0, y0);
        let (mut eu, mut he, mut rk) = (Euler::new(&y0), Heun::new(&y0), RungeKutta4::new(&y0));
        let mut model = OdeModel::new(y0, NonAutonomousMmsOde, RungeKutta4::new(&y0));
        for i in 0..n {
            let t = i as f64 * dt;
            eu.step(&sys, t, &mut y_eu, dt);
            he.step(&sys, t, &mut y_he, dt);
            rk.step(&sys, t, &mut y_rk, dt);
            model.step(dt);
        }
        e_eu[k] = (y_eu.0[0] - exact).abs();
        e_he[k] = (y_he.0[0] - exact).abs();
        e_rk[k] = (y_rk.0[0] - exact).abs();
        e_om[k] = (model.get_state().0[0] - exact).abs();
    }
    vec![
        pm_cert("Euler", 0.95, &hs, &e_eu, 1e-2),
        pm_cert("Heun", 1.95, &hs, &e_he, 2e-4),
        pm_cert("RungeKutta4", 3.95, &hs, &e_rk, 1e-7),
        pm_cert("OdeModel", 3.95, &hs, &e_om, 1e-7),
    ]
}

fn certify_symplectic_1d_solvers() -> Vec<SolverCertificate> {
    let sys = QuarticHamiltonian1D;
    let y0 = ArrayState([1.05, 0.15]);
    let mut exact = y0;
    let mut ref_gl = GaussLegendre4::new(&y0);
    for s in 0..2_000usize {
        ref_gl.step(&sys, s as f64 * 0.0005, &mut exact, 0.0005);
    }
    let mut hs = [0.0; 3];
    let (mut e_sv, mut e_yo, mut e_gl) = ([0.0; 3], [0.0; 3], [0.0; 3]);
    for (k, &n) in [20usize, 40, 80].iter().enumerate() {
        let dt = 1.0 / n as f64;
        hs[k] = dt;
        let (mut s_sv, mut s_yo, mut s_gl) = (y0, y0, y0);
        let mut sv = StormerVerlet::new(&ArrayState([0.0]));
        let mut yo = Yoshida4::new(&ArrayState([0.0]));
        let mut gl = GaussLegendre4::new(&y0);
        for i in 0..n {
            sv.step_separable(&sys, &mut s_sv, dt);
            yo.step_separable(&sys, &mut s_yo, dt);
            gl.step(&sys, i as f64 * dt, &mut s_gl, dt);
        }
        e_sv[k] = s_sv.max_abs_diff(&exact);
        e_yo[k] = s_yo.max_abs_diff(&exact);
        e_gl[k] = s_gl.max_abs_diff(&exact);
    }
    vec![
        pm_cert("StormerVerlet", 1.95, &hs, &e_sv, 5e-4),
        pm_cert("Yoshida4", 3.90, &hs, &e_yo, 1e-6),
        pm_cert("GaussLegendre4", 3.95, &hs, &e_gl, 1e-7),
    ]
}

fn certify_momentum_2d_solvers() -> Result<Vec<SolverCertificate>, String> {
    let t_end = 0.8_f64;
    let q_exact = Vector2::new(t_end.cos(), 0.5 * t_end.sin());
    let dts = [0.04_f64, 0.02, 0.01];
    let (mut e_lf, mut e_ba) = ([0.0; 3], [0.0; 3]);
    let lf = HamiltonianLeapfrog2D::new(1.0, 0.0);
    let baoab = LangevinBaoab2D::new(1.0, 0.0, 0.0);
    let mut rng = OxidizeRng::from_seed(77);
    for (k, &dt) in dts.iter().enumerate() {
        let steps = (t_end / dt).round() as usize;
        let mut st_lf = PhasePoint2D::new(Vector2::new(1.0, 0.0), Vector2::new(0.0, 0.5));
        let mut st_ba = st_lf;
        for _ in 0..steps {
            lf.step_with_kinetic(&mut st_lf, &mut |q| q, &mut |p| p, dt)
                .map_err(|e| format!("{e:?}"))?;
            baoab
                .step_with_kinetic(&mut st_ba, &mut |q| q, &mut |p| p, &mut rng, dt)
                .map_err(|e| format!("{e:?}"))?;
        }
        e_lf[k] = (st_lf.q - q_exact).norm();
        e_ba[k] = (st_ba.q - q_exact).norm();
    }
    Ok(vec![
        pm_cert("HamiltonianLeapfrog2D", 1.95, &dts, &e_lf, 2e-4),
        pm_cert("LangevinBaoab2D", 1.95, &dts, &e_ba, 2e-4),
    ])
}

/// Certifies all 11 linear-algebra, ODE, symplectic, and phase-space solvers in `math_commons` and `pure_math`.
pub fn certify_commons_and_ode() -> Result<Vec<SolverCertificate>, String> {
    let mut out = certify_algebraic_fit_solvers()?;
    out.extend(certify_scalar_ode_solvers());
    out.extend(certify_symplectic_1d_solvers());
    out.extend(certify_momentum_2d_solvers()?);
    Ok(out)
}
