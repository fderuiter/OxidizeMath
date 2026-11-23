use super::*;
use approx::assert_relative_eq;
use math_commons::registry::{TOLERANCE_FAST, TOLERANCE_HIGH};
use oxidize_core::rng::OxidizeRng;

#[test]
#[verified_engine::verified]
fn test_cara_and_crra_certainty_equivalents() {
    let payoffs = [100.0, 400.0];
    let probs = [0.5, 0.5];

    let ce_neutral = RiskAttitude::Neutral
        .certainty_equivalent(&payoffs, &probs)
        .unwrap();
    assert_relative_eq!(ce_neutral, 250.0, epsilon = TOLERANCE_HIGH);

    let cara = RiskAttitude::Cara { alpha: 0.01 };
    let ce_cara = cara.certainty_equivalent(&payoffs, &probs).unwrap();
    assert!(ce_cara < 250.0 && ce_cara > 100.0);
    assert_relative_eq!(
        cara.utility(ce_cara).unwrap(),
        cara.expected_utility(&payoffs, &probs).unwrap(),
        epsilon = TOLERANCE_FAST
    );

    // Log utility (CRRA gamma = 1): CE of 50/50 over (100, 400) is geometric mean sqrt(40000) = 200.
    let crra_log = RiskAttitude::Crra { gamma: 1.0 };
    assert_relative_eq!(
        crra_log.certainty_equivalent(&payoffs, &probs).unwrap(),
        200.0,
        epsilon = TOLERANCE_FAST
    );

    // Square-root utility (CRRA gamma = 0.5): E[sqrt(W)] = 0.5*(10 + 20) = 15 -> CE = 225.
    let crra_half = RiskAttitude::Crra { gamma: 0.5 };
    assert_relative_eq!(
        crra_half.certainty_equivalent(&payoffs, &probs).unwrap(),
        225.0,
        epsilon = TOLERANCE_FAST
    );
    assert!(crra_half.risk_premium(&payoffs, &probs).unwrap() > 0.0);
}

#[test]
#[verified_engine::verified]
fn test_arena_backward_induction_and_cycle_detection() {
    // Safe option pays 210; risky lottery pays 100 or 400 equally (EV = 250, log-CE = 200).
    let nodes = [
        DecisionNode::Decision {
            edge_start: 0,
            edge_len: 2,
        },
        DecisionNode::Terminal { payoff: 210.0 },
        DecisionNode::Chance {
            edge_start: 2,
            edge_len: 2,
        },
        DecisionNode::Terminal { payoff: 100.0 },
        DecisionNode::Terminal { payoff: 400.0 },
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
            probability: 0.5,
        },
        DecisionEdge {
            target: 4,
            probability: 0.5,
        },
    ];
    let tree = DecisionTreeView::new(&nodes, &edges, 0).unwrap();
    let mut ws = DecisionTreeWorkspace::new(5, 4);

    // Risk-neutral chooses branch 1 (risky, EV = 250).
    let neutral_res = backward_induction(tree, RiskAttitude::Neutral, &mut ws).unwrap();
    assert_relative_eq!(
        neutral_res.root_certainty_equivalent,
        250.0,
        epsilon = TOLERANCE_HIGH
    );
    assert_eq!(ws.optimal_branch(0), Some(1));

    // Log-CRRA chooses branch 0 (safe 210 > risky CE 200).
    let risk_averse_res =
        backward_induction(tree, RiskAttitude::Crra { gamma: 1.0 }, &mut ws).unwrap();
    assert_relative_eq!(
        risk_averse_res.root_certainty_equivalent,
        210.0,
        epsilon = TOLERANCE_FAST
    );
    assert_eq!(ws.optimal_branch(0), Some(0));

    // Cycle detection test: 0 -> 1 -> 0.
    let cyc_nodes = [
        DecisionNode::Decision {
            edge_start: 0,
            edge_len: 1,
        },
        DecisionNode::Decision {
            edge_start: 1,
            edge_len: 1,
        },
    ];
    let cyc_edges = [
        DecisionEdge {
            target: 1,
            probability: 1.0,
        },
        DecisionEdge {
            target: 0,
            probability: 1.0,
        },
    ];
    let cyc_tree = DecisionTreeView::new(&cyc_nodes, &cyc_edges, 0).unwrap();
    assert!(matches!(
        backward_induction(cyc_tree, RiskAttitude::Neutral, &mut ws),
        Err(DecisionTheoryError::CyclicTree { .. })
    ));
}

#[test]
#[verified_engine::verified]
fn test_evpi_and_evsi_blackwell_ordering() {
    // Oil drilling problem: Drill (action 0: +100 wet, -40 dry) vs Pass (action 1: 0, 0).
    // Prior: P(wet)=0.4, P(dry)=0.6 -> EMV(Drill) = 16, EMV(Pass) = 0 -> baseline = 16.
    // EV|PI = 0.4*100 + 0.6*0 = 40 -> EVPI = 24.
    let payoffs = [100.0, -40.0, 0.0, 0.0];
    let prior = [0.4, 0.6];
    let pi = evpi(&payoffs, 2, 2, &prior).unwrap();
    assert_relative_eq!(pi.baseline_value, 16.0, epsilon = TOLERANCE_HIGH);
    assert_relative_eq!(pi.evpi, 24.0, epsilon = TOLERANCE_HIGH);

    let mut ws = VoiWorkspace::new(2);
    // Perfect seismic test: EVSI == EVPI == 24.
    let perfect_lik = [1.0, 0.0, 0.0, 1.0];
    let si_perf = evsi(&payoffs, 2, 2, &prior, &perfect_lik, 2, &mut ws).unwrap();
    assert_relative_eq!(si_perf.evsi, pi.evpi, epsilon = TOLERANCE_HIGH);

    // Noisy seismic test (80% sensitivity & specificity): 0 < EVSI < EVPI.
    let noisy_lik = [0.8, 0.2, 0.2, 0.8];
    let si_noisy = evsi(&payoffs, 2, 2, &prior, &noisy_lik, 2, &mut ws).unwrap();
    assert!(si_noisy.evsi > 0.0 && si_noisy.evsi < pi.evpi);

    // Uninformative coin-flip signal: EVSI == 0.
    let coin_lik = [0.5, 0.5, 0.5, 0.5];
    let si_zero = evsi(&payoffs, 2, 2, &prior, &coin_lik, 2, &mut ws).unwrap();
    assert_relative_eq!(si_zero.evsi, 0.0, epsilon = TOLERANCE_HIGH);
}

#[test]
#[verified_engine::verified]
fn test_seeded_ucb1_and_thompson_bandits_determinism_and_sublinear_regret() {
    let arms = [0.15, 0.85, 0.35];
    let mut ucb_a = Ucb1Bandit::new(3, std::f64::consts::SQRT_2).unwrap();
    let mut ucb_b = Ucb1Bandit::new(3, std::f64::consts::SQRT_2).unwrap();
    let mut reg_a = [0.0; 400];
    let mut reg_b = [0.0; 400];

    let mut rng1 = OxidizeRng::from_seed(42);
    let mut rng2 = OxidizeRng::from_seed(42);
    let final_ucb_a = simulate_bernoulli_ucb1(&mut ucb_a, &arms, &mut rng1, &mut reg_a).unwrap();
    let final_ucb_b = simulate_bernoulli_ucb1(&mut ucb_b, &arms, &mut rng2, &mut reg_b).unwrap();
    assert_eq!(final_ucb_a, final_ucb_b);
    assert!(ucb_a.pull_counts()[1] > 300);

    let mut ts_a = ThompsonBandit::new(3).unwrap();
    let mut ts_b = ThompsonBandit::new(3).unwrap();
    let mut rng3 = OxidizeRng::from_seed(99);
    let mut rng4 = OxidizeRng::from_seed(99);
    let final_ts_a = simulate_bernoulli_thompson(&mut ts_a, &arms, &mut rng3, &mut reg_a).unwrap();
    let final_ts_b = simulate_bernoulli_thompson(&mut ts_b, &arms, &mut rng4, &mut reg_b).unwrap();
    assert_eq!(final_ts_a, final_ts_b);
    assert!(ts_a.pull_counts()[1] > 320);
    assert!(final_ts_a < 40.0);
}

// [cite:graph_parameters_rust]
