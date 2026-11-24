//! Strict `VerifiedAllocator` zero-allocation verification across `normal_form`, `cooperative`,
//! and `decision_theory` hot-path solvers.

use domain_applied::applied::decision_theory::{
    DecisionEdge, DecisionNode, DecisionTreeView, DecisionTreeWorkspace, RiskAttitude,
    ThompsonBandit, Ucb1Bandit, VoiWorkspace, backward_induction, evpi, evsi,
    simulate_bernoulli_thompson, simulate_bernoulli_ucb1,
};
use domain_applied::applied::game_theory::cooperative::{
    is_convex, is_in_core, is_superadditive, shapley_values,
};
use domain_applied::applied::game_theory::normal_form::{
    BimatrixGameView, DominanceMode, DominanceWorkspace, LemkeHowsonWorkspace,
    MAX_SUPPORT_ENUM_DIM, MinimaxWorkspace, SupportEquilibrium, evaluate_bimatrix_profile,
    iterated_dominance, lemke_howson_equilibrium, solve_zero_sum_minimax, support_enumeration,
};
use oxidize_core::rng::OxidizeRng;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static TEST_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_game_and_cooperative_solvers_zero_heap_allocations() {
    let a = [2.0, 0.0, 0.0, 1.0];
    let b = [1.0, 0.0, 0.0, 2.0];
    let zs = [0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0];
    let coal = [0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 6.0];

    let mut dom_ws = DominanceWorkspace::new(2, 2);
    let mut mm_ws = MinimaxWorkspace::new(3, 3);
    let mut lh_ws = LemkeHowsonWorkspace::new(2, 2);
    let mut r_alive = [false; 2];
    let mut c_alive = [false; 2];
    let mut p2 = [0.0; 2];
    let mut q2 = [0.0; 2];
    let mut p3 = [0.0; 3];
    let mut q3 = [0.0; 3];
    let mut eqs = [SupportEquilibrium {
        rows: 0,
        cols: 0,
        row_strategy: [0.0; MAX_SUPPORT_ENUM_DIM],
        col_strategy: [0.0; MAX_SUPPORT_ENUM_DIM],
        row_payoff: 0.0,
        col_payoff: 0.0,
    }; 4];
    let mut phi = [0.0; 3];

    let ((eq_count, regret, in_core), metrics) = VerifiedEngine::run_verified(|| {
        let game = BimatrixGameView::new(&a, &b, 2, 2).unwrap();
        let _ = iterated_dominance(
            game,
            DominanceMode::Strict,
            &mut dom_ws,
            &mut r_alive,
            &mut c_alive,
        )
        .unwrap();
        let _ = solve_zero_sum_minimax(&zs, 3, 3, &mut mm_ws, &mut p3, &mut q3).unwrap();
        let count = support_enumeration(game, &mut eqs).unwrap();
        let _ = lemke_howson_equilibrium(game, 0, &mut lh_ws, &mut p2, &mut q2).unwrap();
        let (_, _, reg) = evaluate_bimatrix_profile(game, &p2, &q2).unwrap();
        shapley_values(3, &coal, &mut phi).unwrap();
        let _ = is_superadditive(3, &coal).unwrap();
        let _ = is_convex(3, &coal).unwrap();
        let core = is_in_core(3, &coal, &phi).unwrap();
        (count, reg, core)
    });

    assert!(metrics.function_calls > 0);
    assert_eq!(eq_count, 3);
    assert!(regret < 1e-6);
    assert!(in_core);
}

#[test]
#[verified_engine::verified]
fn test_decision_theory_and_bandits_zero_heap_allocations() {
    let nodes = [
        DecisionNode::Decision {
            edge_start: 0,
            edge_len: 2,
        },
        DecisionNode::Terminal { payoff: 50.0 },
        DecisionNode::Chance {
            edge_start: 2,
            edge_len: 2,
        },
        DecisionNode::Terminal { payoff: 20.0 },
        DecisionNode::Terminal { payoff: 120.0 },
    ];
    let edges = [
        DecisionEdge {
            target: 1,
            probability: 1.0,
        },
        DecisionEdge {
            target: 2,
            probability: 1.0,
        },
        DecisionEdge {
            target: 3,
            probability: 0.4,
        },
        DecisionEdge {
            target: 4,
            probability: 0.6,
        },
    ];
    let voi_payoffs = [100.0, -40.0, 0.0, 0.0];
    let prior = [0.4, 0.6];
    let lik = [0.85, 0.15, 0.15, 0.85];
    let arms = [0.2, 0.8, 0.4];

    let mut tree_ws = DecisionTreeWorkspace::new(5, 4);
    let mut voi_ws = VoiWorkspace::new(2);
    let mut ucb = Ucb1Bandit::new(3, std::f64::consts::SQRT_2).unwrap();
    let mut thompson = ThompsonBandit::new(3).unwrap();
    let mut rng = OxidizeRng::from_seed(12345);
    let mut regret = [0.0; 64];

    let ((ce, evpi_val, evsi_val, r_ucb, r_ts), metrics) = VerifiedEngine::run_verified(|| {
        let tree = DecisionTreeView::new(&nodes, &edges, 0).unwrap();
        let ind =
            backward_induction(tree, RiskAttitude::Cara { alpha: 0.005 }, &mut tree_ws).unwrap();
        let pi = evpi(&voi_payoffs, 2, 2, &prior).unwrap();
        let si = evsi(&voi_payoffs, 2, 2, &prior, &lik, 2, &mut voi_ws).unwrap();
        let ru = simulate_bernoulli_ucb1(&mut ucb, &arms, &mut rng, &mut regret).unwrap();
        let rt = simulate_bernoulli_thompson(&mut thompson, &arms, &mut rng, &mut regret).unwrap();
        (ind.root_certainty_equivalent, pi.evpi, si.evsi, ru, rt)
    });

    assert!(metrics.function_calls > 0);
    assert!(ce > 0.0 && evsi_val > 0.0 && evsi_val <= evpi_val && r_ucb >= 0.0 && r_ts >= 0.0);
}

// [cite:graph_parameters_rust]
