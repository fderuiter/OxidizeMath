//! Zero-allocation runtime verification tests for initialized `Adam`, `SGD`,
//! proximal `ISTA`/`FISTA`, `SimplexSolver`, `ActiveSetQpSolver`,
//! `ProjectedGradientSolver`, `LbfgsBSolver`, `AugmentedLagrangianSolver`,
//! `NelderMead`, `BrentMinimizer`, and `ParetoWorkspace` under `VerifiedAllocator`.

use approx::assert_relative_eq;
use nalgebra::{DMatrix, DVector};
use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::analysis::evolution::EvolutionEngine;
use pure_math::pure_math::analysis::optimization::{
    ActiveSetQpSolver, Adam, AdamState, AugmentedLagrangianSolver, AugmentedLagrangianWorkspace,
    BoxBounds, BrentMinimizer, L1RegularizedLeastSquares, LbfgsBSolver, LbfgsBWorkspace,
    LinearProgram, LpStatus, NelderMead, NelderMeadWorkspace, OneDimMinimizer, Optimizer,
    ParamType, ParetoWorkspace, ProjectedGradientSolver, ProjectedGradientWorkspace,
    ProximalConfig, ProximalWorkspace, QpStatus, QpWorkspace, QuadraticProgram, SGD, SimplexSolver,
    SimplexWorkspace,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

struct PreallocSuite {
    lasso: L1RegularizedLeastSquares,
    lasso_a: DMatrix<f64>,
    lasso_y: DVector<f64>,
    x_ista: DVector<f64>,
    x_fista: DVector<f64>,
    prox_ws: ProximalWorkspace,
    lp: LinearProgram,
    lp_ws: SimplexWorkspace,
    qp: QuadraticProgram,
    x_qp: DVector<f64>,
    qp_ws: QpWorkspace,
    bounds: BoxBounds,
    x_pg: DVector<f64>,
    pg_ws: ProjectedGradientWorkspace,
    x_lbfgs: DVector<f64>,
    lbfgs_ws: LbfgsBWorkspace,
    x_al: DVector<f64>,
    al_ws: AugmentedLagrangianWorkspace,
    x_nm: DVector<f64>,
    nm_ws: NelderMeadWorkspace,
    pareto_mat: DMatrix<f64>,
    pareto_ws: ParetoWorkspace,
}

fn build_prealloc_suite() -> PreallocSuite {
    let lasso_a = DMatrix::identity(3, 3);
    let lasso_y = DVector::from_vec(vec![2.0, -0.1, -1.5]);
    let lp = LinearProgram::maximize(
        DVector::from_vec(vec![3.0, 2.0]),
        DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 2.0, 1.0]),
        DVector::from_vec(vec![4.0, 5.0]),
    )
    .expect("lp");
    let qp = QuadraticProgram::new(
        DMatrix::identity(2, 2),
        DVector::from_vec(vec![-2.0, -2.0]),
        DMatrix::zeros(0, 2),
        DVector::zeros(0),
        DMatrix::from_row_slice(3, 2, &[1.0, 1.0, -1.0, 0.0, 0.0, -1.0]),
        DVector::from_vec(vec![2.0, 0.0, 0.0]),
    )
    .expect("qp");
    let pareto_mat = DMatrix::from_row_slice(4, 2, &[1.0, 4.0, 2.0, 2.0, 3.0, 3.0, 4.0, 1.0]);
    PreallocSuite {
        lasso: L1RegularizedLeastSquares::new(0.25),
        lasso_a,
        lasso_y,
        x_ista: DVector::zeros(3),
        x_fista: DVector::zeros(3),
        prox_ws: ProximalWorkspace::new(3, 3),
        lp,
        lp_ws: SimplexWorkspace::new(2, 2, 32),
        qp,
        x_qp: DVector::from_vec(vec![0.2, 0.2]),
        qp_ws: QpWorkspace::new(2, 0, 3, 64),
        bounds: BoxBounds::uniform(2, 0.0, 1.0).expect("bounds"),
        x_pg: DVector::from_vec(vec![0.5, 0.5]),
        pg_ws: ProjectedGradientWorkspace::new(2, 128),
        x_lbfgs: DVector::from_vec(vec![0.2, 0.8]),
        lbfgs_ws: LbfgsBWorkspace::new(2, 6, 128),
        x_al: DVector::from_vec(vec![0.4, 0.4]),
        al_ws: AugmentedLagrangianWorkspace::new(2, 0, 1, 80),
        x_nm: DVector::from_vec(vec![0.0, 0.0]),
        nm_ws: NelderMeadWorkspace::new(2, 200),
        pareto_mat,
        pareto_ws: ParetoWorkspace::new(4),
    }
}

fn run_convex_solvers(s: &mut PreallocSuite) -> (LpStatus, QpStatus) {
    let cfg = ProximalConfig::default();
    s.lasso
        .solve_ista_in_place(&s.lasso_a, &s.lasso_y, &mut s.x_ista, &cfg, &mut s.prox_ws)
        .expect("ista");
    s.lasso
        .solve_fista_in_place(&s.lasso_a, &s.lasso_y, &mut s.x_fista, &cfg, &mut s.prox_ws)
        .expect("fista");
    let lp_st = SimplexSolver::default()
        .solve_in_place(&s.lp, &mut s.lp_ws)
        .expect("lp");
    let qp_st = ActiveSetQpSolver::default()
        .solve_in_place(&s.qp, &mut s.x_qp, &mut s.qp_ws)
        .expect("qp");
    (lp_st, qp_st)
}

fn run_nonlinear_and_pareto_solvers(s: &mut PreallocSuite) -> (f64, usize) {
    let quad_fg = |x: &DVector<f64>, g: &mut DVector<f64>| {
        g[0] = 2.0 * (x[0] - 1.4);
        g[1] = 2.0 * (x[1] - 0.3);
        (x[0] - 1.4).powi(2) + (x[1] - 0.3).powi(2)
    };
    ProjectedGradientSolver::default()
        .solve_in_place(quad_fg, &s.bounds, &mut s.x_pg, &mut s.pg_ws)
        .expect("pg");
    LbfgsBSolver::default()
        .solve_in_place(quad_fg, &s.bounds, &mut s.x_lbfgs, &mut s.lbfgs_ws)
        .expect("lbfgsb");
    AugmentedLagrangianSolver::default()
        .solve_in_place(
            |x: &DVector<f64>,
             gf: &mut DVector<f64>,
             _h: &mut DVector<f64>,
             _jh: &mut DMatrix<f64>,
             g: &mut DVector<f64>,
             jg: &mut DMatrix<f64>| {
                gf[0] = 2.0 * (x[0] - 1.0);
                gf[1] = 2.0 * (x[1] - 1.0);
                g[0] = x[0] + x[1] - 1.0;
                jg[(0, 0)] = 1.0;
                jg[(0, 1)] = 1.0;
                (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2)
            },
            &s.bounds,
            &mut s.x_al,
            &mut s.al_ws,
        )
        .expect("al");
    NelderMead::default()
        .minimize_in_place(
            |v| (v[0] - 0.7).powi(2) + (v[1] + 0.4).powi(2),
            &mut s.x_nm,
            &mut s.nm_ws,
        )
        .expect("nm");
    let (brent_x, _) = BrentMinimizer::default()
        .minimize(|x| (x - 1.25).powi(2), -1.0, 3.0)
        .expect("brent");
    let front_len = s
        .pareto_ws
        .extract_indices_in_place(&s.pareto_mat, 1e-9)
        .len();
    (brent_x, front_len)
}

fn run_prealloc_solvers(s: &mut PreallocSuite) -> (LpStatus, QpStatus, f64, usize) {
    let (lp_st, qp_st) = run_convex_solvers(s);
    let (brent_x, front_len) = run_nonlinear_and_pareto_solvers(s);
    (lp_st, qp_st, brent_x, front_len)
}

#[test]
#[verified_engine::verified]
fn test_initialized_optimizers_and_solvers_zero_heap_allocations() {
    let mut rng = OxidizeRng::new(0x0A11_0C00);
    let mut adam = Adam::<f64, (usize, ParamType)>::new(0.05).expect("valid Adam");
    let mut sgd = SGD::new(0.25_f64);

    let mut adam_mat = DMatrix::from_element(3, 4, 1.0_f64);
    let adam_mat_grad = DMatrix::from_element(3, 4, 0.2_f64);
    let mut adam_vec = DVector::from_element(5, -0.5_f64);
    let adam_vec_grad = DVector::from_element(5, -0.1_f64);
    let _ = adam.get_state((0, ParamType::Weight), (3, 4));
    let _ = adam.get_state((0, ParamType::Bias), (5, 1));

    let mut evo_adam_mat = DMatrix::from_element(2, 3, 0.75_f64);
    let mut evo_adam_mat_state = AdamState {
        m: DMatrix::zeros(2, 3),
        v: DMatrix::zeros(2, 3),
        t: 0,
    };
    let mut evo_adam_mat_aux = (
        DMatrix::from_element(2, 3, 0.15_f64),
        &mut evo_adam_mat_state,
    );
    let mut evo_adam_vec = DVector::from_element(4, 0.4_f64);
    let mut evo_adam_vec_state = AdamState {
        m: DMatrix::zeros(4, 1),
        v: DMatrix::zeros(4, 1),
        t: 0,
    };
    let mut evo_adam_vec_aux = (DVector::from_element(4, 0.05_f64), &mut evo_adam_vec_state);

    let mut sgd_mat = DMatrix::from_element(3, 4, 2.0_f64);
    let sgd_mat_grad = DMatrix::from_element(3, 4, 0.5_f64);
    let mut sgd_vec = DVector::from_element(5, 1.5_f64);
    let sgd_vec_grad = DVector::from_element(5, 0.4_f64);
    let mut evo_sgd_mat = DMatrix::from_element(2, 2, 1.0_f64);
    let mut evo_sgd_mat_grad = DMatrix::from_element(2, 2, 0.2_f64);
    let mut evo_sgd_vec = DVector::from_element(3, 1.0_f64);
    let mut evo_sgd_vec_grad = DVector::from_element(3, 0.2_f64);
    let mut suite = build_prealloc_suite();

    // Keep all VerifiedEngine::run_verified checks in a single test to avoid MEMORY_LOCK contention.
    let ((), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..16 {
            adam.update_matrix((0, ParamType::Weight), &mut adam_mat, &adam_mat_grad)
                .expect("adam matrix step");
            adam.update_vector((0, ParamType::Bias), &mut adam_vec, &adam_vec_grad)
                .expect("adam vector step");
            adam.step(&mut evo_adam_mat, &mut evo_adam_mat_aux, &mut rng, 0.02)
                .expect("adam evo matrix step");
            adam.step(&mut evo_adam_vec, &mut evo_adam_vec_aux, &mut rng, 0.02)
                .expect("adam evo vector step");

            Optimizer::<f64, usize>::update_matrix(&mut sgd, 0, &mut sgd_mat, &sgd_mat_grad)
                .expect("sgd matrix step");
            Optimizer::<f64, usize>::update_vector(&mut sgd, 1, &mut sgd_vec, &sgd_vec_grad)
                .expect("sgd vector step");
            sgd.step(&mut evo_sgd_mat, &mut evo_sgd_mat_grad, &mut rng, 0.05)
                .expect("sgd evo matrix step");
            sgd.step(&mut evo_sgd_vec, &mut evo_sgd_vec_grad, &mut rng, 0.05)
                .expect("sgd evo vector step");
        }
    });
    assert_eq!(metrics.memory_loads, 0);

    let ((lp_st, qp_st, brent_x, front_len), solver_metrics) =
        VerifiedEngine::run_verified(|| run_prealloc_solvers(&mut suite));
    assert!(solver_metrics.function_calls >= 9);
    assert!(adam_mat[(0, 0)] < 1.0 && adam_vec[0] > -0.5);
    assert!(evo_adam_mat[(0, 0)] < 0.75 && evo_adam_vec[0] < 0.4);
    assert_relative_eq!(
        sgd_mat[(0, 0)],
        2.0 - 16.0 * 0.25 * 0.5,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
    assert_relative_eq!(
        sgd_vec[0],
        1.5 - 16.0 * 0.25 * 0.4,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
    assert_relative_eq!(
        evo_sgd_mat[(0, 0)],
        1.0 - 16.0 * 0.05 * 0.2,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
    assert_relative_eq!(
        evo_sgd_vec[0],
        1.0 - 16.0 * 0.05 * 0.2,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
    assert_eq!(lp_st, LpStatus::Optimal);
    assert_eq!(qp_st, QpStatus::Optimal);
    assert!((suite.x_fista[0] - 1.75).abs() < 1e-5 && suite.x_fista[1].abs() < 1e-9);
    assert!((suite.x_qp[0] - 1.0).abs() < 1e-6 && (suite.x_qp[1] - 1.0).abs() < 1e-6);
    assert!((suite.x_lbfgs[0] - 1.0).abs() < 1e-6 && (suite.x_lbfgs[1] - 0.3).abs() < 1e-6);
    assert!((suite.x_al[0] - 0.5).abs() < 1e-4 && (suite.x_al[1] - 0.5).abs() < 1e-4);
    assert!((suite.x_nm[0] - 0.7).abs() < 1e-4 && (suite.x_nm[1] + 0.4).abs() < 1e-4);
    assert!((brent_x - 1.25).abs() < 1e-5);
    assert_eq!(front_len, 3);
}
