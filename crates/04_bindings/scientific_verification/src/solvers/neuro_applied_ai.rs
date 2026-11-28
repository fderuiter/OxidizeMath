use crate::mms::{SolverCertificate, ols_convergence_order};
use domain_ai::ai::reinforcement_learning::{BellmanSolution, SculptedGridMdp};
use domain_applied::applied::cannibalism::{
    CannibalismKernel, JuvenileAdultParams, JuvenileAdultSolver, McKendrickSolver,
};
use domain_applied::applied::game_theory::evolutionary::ReplicatorDynamics;
use domain_applied::applied::game_theory::mean_field::{
    FixedPointSolver, MFGConfigBuilder, MFGSolver,
};
use domain_biology::biology::lenia::{LeniaEngine, LeniaSpecies};
use domain_biology::biology::neuroscience::{
    HodgkinHuxleyModel, HodgkinHuxleyParameters, HodgkinHuxleyState, KuramotoNetwork,
};
use domain_climate::climate::autoencoder::ConvLayer;
use nalgebra::{DMatrix, DVector};
use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::analysis::ode::{RungeKutta4, Solver};
use pure_math::pure_math::analysis::optimization::SGD;
use std::num::NonZeroUsize;

fn certify_neuro() -> Vec<SolverCertificate> {
    let hh = HodgkinHuxleyModel::new(HodgkinHuxleyParameters::default(), 10.0);
    let hh0 = HodgkinHuxleyState {
        v: -65.0,
        n: 0.3177,
        m: 0.0529,
        h: 0.5961,
    };
    let eval_hh = |n: usize| -> f64 {
        let dt = 0.2 / n as f64;
        let (mut st, mut rk) = (hh0, RungeKutta4::new(&hh0));
        for i in 0..n {
            rk.step(&hh, i as f64 * dt, &mut st, dt);
        }
        st.v
    };
    let hh_ref = eval_hh(640);
    let hs_hh = [0.2 / 10.0, 0.2 / 20.0, 0.2 / 40.0];
    let err_hh = [10usize, 20, 40].map(|n| (eval_hh(n) - hh_ref).abs());

    let eval_kur = |n: usize| -> f64 {
        let mut net = KuramotoNetwork::new(6, 1.2);
        net.set_frequency_spread(0.5);
        for i in 0..6 {
            net.set_phase(i, 0.2 + 0.05 * i as f64);
        }
        for _ in 0..n {
            net.step(0.2 / n as f64);
        }
        net.phases()[0]
    };
    let kur_ref = eval_kur(640);
    let err_kur = [10usize, 20, 40].map(|n| (eval_kur(n) - kur_ref).abs());

    vec![
        SolverCertificate::evaluate(
            "HodgkinHuxleyModel",
            "domain_biology",
            3.85,
            ols_convergence_order(&hs_hh, &err_hh),
            err_hh[2],
            1e-5,
        ),
        SolverCertificate::evaluate(
            "KuramotoNetwork",
            "domain_biology",
            0.90,
            ols_convergence_order(&hs_hh, &err_kur),
            err_kur[2],
            1e-2,
        ),
    ]
}

fn certify_lenia() -> SolverCertificate {
    let eval_lenia = |n: usize| -> f64 {
        let mut rng = OxidizeRng::from_seed(1234);
        let mut eng = LeniaEngine::from_species(32, 32, LeniaSpecies::Orbium, 0.0, &mut rng);
        eng.set_dt(0.04 / n as f64);
        for _ in 0..n {
            eng.step();
        }
        eng.telemetry().total_mass
    };
    let len_ref = eval_lenia(128);
    let hs_len = [0.04 / 4.0, 0.04 / 8.0, 0.04 / 16.0];
    let err_len = [4usize, 8, 16].map(|n| (eval_lenia(n) - len_ref).abs());
    SolverCertificate::evaluate(
        "LeniaEngine",
        "domain_biology",
        0.85,
        ols_convergence_order(&hs_len, &err_len),
        err_len[2],
        5e-2,
    )
}

fn certify_mckendrick() -> (SolverCertificate, [f64; 3]) {
    let (mut hs_c, mut err_mck) = ([0.0; 3], [0.0; 3]);
    let mu = 0.25_f64;
    let init_rho = |a: f64| {
        if (0.5..3.5).contains(&a) {
            ((a - 0.5) * std::f64::consts::PI / 3.0).sin().powi(4)
        } else {
            0.0
        }
    };
    let kernel_mck = CannibalismKernel::new(mu, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0);
    for (k, &n) in [101usize, 201, 401].iter().enumerate() {
        let mut mck = McKendrickSolver::new(n, 6.0, kernel_mck);
        mck.set_initial_density(init_rho);
        let da = mck.da();
        hs_c[k] = da;
        let steps = (0.5 / (0.5 * da)).round() as usize;
        for _ in 0..steps {
            mck.step_upwind_with_boundary(0.5 / steps as f64, 0.0);
        }
        let decay = (-mu * 0.5).exp();
        let l2_sq: f64 = mck
            .ages()
            .iter()
            .zip(mck.density().iter())
            .map(|(&a, &un)| {
                let exact = if a >= 0.5 {
                    decay * init_rho(a - 0.5)
                } else {
                    0.0
                };
                (un - exact).powi(2) * da
            })
            .sum();
        err_mck[k] = l2_sq.sqrt();
    }
    let cert = SolverCertificate::evaluate(
        "McKendrickSolver",
        "domain_applied",
        0.85,
        ols_convergence_order(&hs_c, &err_mck),
        err_mck[2],
        5e-2,
    );
    (cert, hs_c)
}

fn certify_juvenile_adult(hs_c: &[f64; 3]) -> SolverCertificate {
    let ja_params = JuvenileAdultParams::default();
    let run_ja = |n_ages: usize| -> (f64, f64) {
        let mut ja = JuvenileAdultSolver::new(n_ages, 12.0, ja_params);
        ja.set_initial_state(12.0, |a| {
            6.0 * (std::f64::consts::PI * a / ja_params.maturation_age).sin()
        });
        let steps = (0.5 / (0.5 * ja.da())).round() as usize;
        for _ in 0..steps {
            ja.step_upwind(0.5 / steps as f64);
        }
        (ja.adults(), ja.total_juveniles())
    };
    let (a_ref, j_ref) = run_ja(961);
    let ja_norm = a_ref.hypot(j_ref).max(1.0);
    let err_ja = [61usize, 121, 241].map(|n_ages| {
        let (a_n, j_n) = run_ja(n_ages);
        (a_n - a_ref).hypot(j_n - j_ref) / ja_norm
    });
    SolverCertificate::evaluate(
        "JuvenileAdultSolver",
        "domain_applied",
        0.85,
        ols_convergence_order(hs_c, &err_ja),
        err_ja[2],
        5e-2,
    )
}

fn certify_games_and_mdp() -> Result<Vec<SolverCertificate>, String> {
    let rep = ReplicatorDynamics::new(DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 0.0]))
        .map_err(|e| format!("{e:?}"))?;
    let x_init = DVector::from_vec(vec![0.25, 0.75]);
    let exact_rep = 0.25 * 1.0_f64.exp() / (0.75 + 0.25 * 1.0_f64.exp());
    let hs_rep = [1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0];
    let err_rep = [8usize, 16, 32].map(|n| {
        let traj = rep.simulate(x_init.clone(), 1.0, 1.0 / n as f64);
        (traj.last().unwrap().1[0] - exact_rep).abs()
    });

    let cfg = MFGConfigBuilder::new()
        .viscosity(0.1)
        .time_horizon(0.2)
        .grid_points(NonZeroUsize::new(21).unwrap())
        .time_steps(NonZeroUsize::new(20).unwrap())
        .space_bounds(-1.0, 1.0)
        .map_err(|e| format!("{e:?}"))?
        .build()
        .map_err(|e| format!("{e:?}"))?;
    let (u_mat, m_mat) =
        FixedPointSolver::new(8).solve(&cfg, &|p, _m| 0.5 * p.0 * p.0, &|p, _m| p.0 * p.0, &|p| {
            (-4.0 * p.0 * p.0).exp()
        });
    let err_mfg = ((m_mat.column(20).sum() - m_mat.column(0).sum()) * 0.1)
        .abs()
        .max(u_mat[(10, 20)].abs());
    let sol = BellmanSolution::solve(&SculptedGridMdp::new(4, 4, 0.9, 0.0), 1e-12, 300);
    Ok(vec![
        SolverCertificate::evaluate(
            "ReplicatorDynamics",
            "domain_applied",
            3.90,
            ols_convergence_order(&hs_rep, &err_rep),
            err_rep[2],
            1e-7,
        ),
        SolverCertificate::evaluate(
            "FixedPointSolver",
            "domain_applied",
            0.0,
            1.0,
            err_mfg,
            5e-2,
        ),
        SolverCertificate::evaluate(
            "BellmanSolution",
            "domain_ai",
            0.0,
            1.0,
            sol.max_delta,
            1e-11,
        ),
    ])
}

fn certify_cera_conv() -> Result<SolverCertificate, String> {
    let mut layer = ConvLayer {
        kernel: DMatrix::from_element(1, 1, 0.0_f32),
        bias: DVector::from_element(1, 0.0_f32),
    };
    let mut sgd = SGD::new(0.01_f32);
    let x_in = DMatrix::from_column_slice(4, 1, &[-1.0_f32, -0.5, 0.5, 1.0]);
    let y_true = DMatrix::from_column_slice(4, 1, &[0.5_f32, 1.5, 3.5, 4.5]);
    let mut step_losses = [0.0_f64; 3];
    for loss_slot in &mut step_losses {
        for _ in 0..200 {
            let pred =
                &x_in * layer.kernel.transpose() + DMatrix::from_element(4, 1, layer.bias[0]);
            layer
                .backward_update(&x_in, &((&pred - &y_true) * 2.0_f32), &mut sgd, 0)
                .map_err(|e| format!("{e:?}"))?;
        }
        *loss_slot = ((layer.kernel[(0, 0)] - 2.0)
            .abs()
            .max((layer.bias[0] - 2.5).abs()) as f64)
            .max(1e-12);
    }
    Ok(SolverCertificate::evaluate(
        "CeraConvOptimizer",
        "domain_climate",
        1.50,
        ols_convergence_order(&[1.0, 0.5, 0.25], &step_losses),
        step_losses[2],
        1e-2,
    ))
}

pub(super) fn certify_neuro_lenia_applied_ai_climate() -> Result<Vec<SolverCertificate>, String> {
    let mut out = certify_neuro();
    out.push(certify_lenia());
    let (mck, hs_c) = certify_mckendrick();
    out.push(mck);
    out.push(certify_juvenile_adult(&hs_c));
    out.extend(certify_games_and_mdp()?);
    out.push(certify_cera_conv()?);
    Ok(out)
}
