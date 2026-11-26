use crate::mms::{SolverCertificate, ols_convergence_order};
use domain_biology::biology::diffusion::FiniteDifference1D;
use domain_biology::biology::morphogenesis::{
    DiffusionCoeff, FusedEulerSolver, ReactionKinetics, StandardSolverAdapter, TuringDynamics,
    TuringSolverStrategy, TuringState, TuringSystem,
};
use domain_biology::biology::reaction_diffusion::{ReactionDiffusionSystem, ReactionModel};
use domain_epidemiology::epidemiology::compartmental::{SEIRModel, SEIRState, SIRModel, SIRState};
use math_commons::math_kernel::types::{Dimension, StepSize};
use pure_math::pure_math::analysis::ode::RungeKutta4;
use verified_engine::allocator::{lock_allocations, unlock_allocations};
use verified_engine::metrics::{disable_verification, enable_verification};

struct TimeForcedLinearKinetics;

impl ReactionKinetics<2> for TimeForcedLinearKinetics {
    fn reaction(&self, state: [f64; 2]) -> [f64; 2] {
        [-state[0], -state[1]]
    }
    // Exact spatially-uniform ODE u(t) = exp(sin(t)), v(t) = exp(-t):
    // du/dt = cos(t) * u, dv/dt = -v
    fn reaction_at_time(&self, t: f64, _node_idx: usize, state: [f64; 2]) -> [f64; 2] {
        [t.cos() * state[0], -state[1]]
    }
    fn is_time_dependent(&self) -> bool {
        true
    }
}

struct ExponentialDecayReaction;

impl ReactionModel for ExponentialDecayReaction {
    fn reaction(&self, concs: &[f64], rates: &mut [f64]) {
        rates[0] = -0.5 * concs[0];
    }
}

fn certify_sir() -> Result<SolverCertificate, String> {
    let (n_pop, beta0, gamma0) = (1000.0_f64, 0.3_f64, 0.1_f64);
    let s_exact = |t: f64| 800.0 * (-0.2 * t).exp();
    let i_exact = |t: f64| 200.0 * (-0.1 * t).exp();
    let sir_forcing = |t: f64, _st: &SIRState| {
        let (s, i) = (s_exact(t), i_exact(t));
        let (ds_star, di_star) = (-0.2 * s, -0.1 * i);
        let (inf, rec) = (beta0 * s * i / n_pop, gamma0 * i);
        SIRState {
            s: ds_star + inf,
            i: di_star - (inf - rec),
            r: -(ds_star + di_star) - rec,
        }
    };
    let hs = [0.5 / 10.0, 0.5 / 20.0, 0.5 / 40.0];
    let mut err_sir = [0.0; 3];
    for (k, &n) in [10usize, 20, 40].iter().enumerate() {
        let mut sir = SIRModel::builder()
            .n(n_pop)
            .i0(200.0)
            .beta(beta0)
            .gamma(gamma0)
            .build()
            .map_err(|e| format!("{e:?}"))?;
        enable_verification();
        lock_allocations();
        for _ in 0..n {
            sir.step_forced(0.5 / n as f64, sir_forcing);
        }
        unlock_allocations();
        disable_verification();
        let st = sir.state();
        err_sir[k] = (st.s - s_exact(0.5)).abs().max((st.i - i_exact(0.5)).abs());
    }
    Ok(SolverCertificate::evaluate(
        "SIRModel",
        "domain_epidemiology",
        3.90,
        ols_convergence_order(&hs, &err_sir),
        err_sir[2],
        1e-6,
    ))
}

fn certify_seir() -> Result<SolverCertificate, String> {
    let (n_pop, beta0, gamma0, sigma0) = (1000.0_f64, 0.3_f64, 0.1_f64, 0.2_f64);
    let s_seir = |t: f64| 850.0 * (-0.18 * t).exp();
    let e_seir = |t: f64| 40.0 * (1.0 - (-0.5 * t).exp());
    let i_seir = |t: f64| 150.0 * (-0.1 * t).exp();
    let seir_forcing = |t: f64, _st: &SEIRState| {
        let (s, e, i) = (s_seir(t), e_seir(t), i_seir(t));
        let (ds, de, di) = (-0.18 * s, 20.0 * (-0.5 * t).exp(), -0.1 * i);
        let (inf, inc, rec) = (beta0 * s * i / n_pop, sigma0 * e, gamma0 * i);
        SEIRState {
            s: ds + inf,
            e: de - (inf - inc),
            i: di - (inc - rec),
            r: -(ds + de + di) - rec,
        }
    };
    let hs = [0.5 / 10.0, 0.5 / 20.0, 0.5 / 40.0];
    let mut err_seir = [0.0; 3];
    for (k, &n) in [10usize, 20, 40].iter().enumerate() {
        let mut seir = SEIRModel::builder()
            .n(n_pop)
            .i0(150.0)
            .beta(beta0)
            .sigma(sigma0)
            .gamma(gamma0)
            .build()
            .map_err(|e| format!("{e:?}"))?;
        enable_verification();
        lock_allocations();
        for _ in 0..n {
            seir.step_forced(0.5 / n as f64, seir_forcing);
        }
        unlock_allocations();
        disable_verification();
        let st = seir.state();
        err_seir[k] = (st.s - s_seir(0.5))
            .abs()
            .max((st.e - e_seir(0.5)).abs())
            .max((st.i - i_seir(0.5)).abs());
    }
    Ok(SolverCertificate::evaluate(
        "SEIRModel",
        "domain_epidemiology",
        3.90,
        ols_convergence_order(&hs, &err_seir),
        err_seir[2],
        1e-6,
    ))
}

fn certify_sir_and_seir() -> Result<Vec<SolverCertificate>, String> {
    Ok(vec![certify_sir()?, certify_seir()?])
}

fn certify_turing_adapters() -> Vec<SolverCertificate> {
    let u_exact_end = (0.6_f64.sin()).exp();
    let hs_m = [0.6 / 12.0, 0.6 / 24.0, 0.6 / 48.0];
    let (mut err_ssa, mut err_fe) = ([0.0; 3], [0.0; 3]);
    for (k, &n) in [12usize, 24, 48].iter().enumerate() {
        let dt = 0.6 / n as f64;
        let diff_1d = FiniteDifference1D::new(StepSize(1.0));
        let kinetics = TimeForcedLinearKinetics;
        let dynamics = TuringDynamics {
            kinetics: &kinetics,
            diffusion: &diff_1d,
            diffusion_coeffs: [0.0, 0.0],
        };

        let (mut s_ssa, mut next_ssa) = (TuringState::<2>::new(8), TuringState::<2>::new(8));
        s_ssa.u_mut().fill(1.0);
        s_ssa.v_mut().fill(1.0);
        let mut rk_adapt = StandardSolverAdapter::new(RungeKutta4::new(&s_ssa));
        enable_verification();
        lock_allocations();
        for step in 0..n {
            rk_adapt.step_at(step as f64 * dt, &s_ssa, &mut next_ssa, &dynamics, dt);
            std::mem::swap(&mut s_ssa, &mut next_ssa);
        }
        unlock_allocations();
        disable_verification();
        err_ssa[k] = (s_ssa.u()[2] - u_exact_end).abs();

        let (mut s_fe, mut next_fe) = (TuringState::<2>::new(8), TuringState::<2>::new(8));
        s_fe.u_mut().fill(1.0);
        s_fe.v_mut().fill(1.0);
        let mut fe = FusedEulerSolver::new();
        for step in 0..n {
            fe.step_at(step as f64 * dt, &s_fe, &mut next_fe, &dynamics, dt);
            std::mem::swap(&mut s_fe, &mut next_fe);
        }
        err_fe[k] = (s_fe.u()[2] - u_exact_end).abs();
    }
    vec![
        SolverCertificate::evaluate(
            "FusedEulerSolver",
            "domain_biology",
            0.95,
            ols_convergence_order(&hs_m, &err_fe),
            err_fe[2],
            2e-2,
        ),
        SolverCertificate::evaluate(
            "StandardSolverAdapter",
            "domain_biology",
            3.90,
            ols_convergence_order(&hs_m, &err_ssa),
            err_ssa[2],
            1e-7,
        ),
    ]
}

fn certify_reaction_diffusion_systems() -> Result<Vec<SolverCertificate>, String> {
    let (u_exact_end, rds_exact_end) = ((0.6_f64.sin()).exp(), (-0.5 * 0.6_f64).exp());
    let hs_m = [0.6 / 12.0, 0.6 / 24.0, 0.6 / 48.0];
    let (mut err_ts, mut err_rds) = ([0.0; 3], [0.0; 3]);
    for (k, &n) in [12usize, 24, 48].iter().enumerate() {
        let dt = 0.6 / n as f64;
        let mut ts = TuringSystem::new_with_kinetics(
            Dimension(8),
            DiffusionCoeff(0.0),
            DiffusionCoeff(0.0),
            TimeForcedLinearKinetics,
            FiniteDifference1D::new(StepSize(1.0)),
        );
        ts.u_mut().fill(1.0);
        ts.v_mut().fill(1.0);
        for _ in 0..n {
            ts.step(dt);
        }
        err_ts[k] = (ts.u()[2] - u_exact_end).abs();

        let mut rds = ReactionDiffusionSystem::builder()
            .num_species(1)
            .grid_size(8)
            .reaction(ExponentialDecayReaction)
            .diffusion(FiniteDifference1D::new(StepSize(1.0)))
            .diffusion_coeffs(vec![0.0])
            .build()
            .map_err(|e| format!("{e:?}"))?;
        rds.state.species_mut(0).fill(1.0);
        for _ in 0..n {
            rds.step(dt);
        }
        err_rds[k] = (rds.state.species(0)[2] - rds_exact_end).abs();
    }
    Ok(vec![
        SolverCertificate::evaluate(
            "TuringSystem",
            "domain_biology",
            0.95,
            ols_convergence_order(&hs_m, &err_ts),
            err_ts[2],
            2e-2,
        ),
        SolverCertificate::evaluate(
            "ReactionDiffusionSystem",
            "domain_biology",
            0.95,
            ols_convergence_order(&hs_m, &err_rds),
            err_rds[2],
            5e-3,
        ),
    ])
}

/// Certifies all 15 solvers across `domain_epidemiology`, `domain_biology`, `domain_applied`, `domain_ai`, and `domain_climate`.
pub fn certify_bio_epi_applied_ai() -> Result<Vec<SolverCertificate>, String> {
    let mut out = certify_sir_and_seir()?;
    out.extend(certify_turing_adapters());
    out.extend(certify_reaction_diffusion_systems()?);
    out.extend(super::neuro_applied_ai::certify_neuro_lenia_applied_ai_climate()?);
    Ok(out)
}
