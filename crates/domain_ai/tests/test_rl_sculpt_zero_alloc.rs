//! Integration and zero-allocation tests for `SculptedGridMdp`, `BellmanSolution`,
//! `step_tabular_q_into`, and `step_softmax_pg_into`.

use domain_ai::ai::reinforcement_learning::{
    ALL_MOVES, BellmanSolution, DenseRlWorkspace, GridCellRole, GridState, GridWorldEnv,
    MarkovDecisionProcess, Move, Policy, SculptedGridMdp, TabularGridPolicy, TabularQAgent,
    compute_softmax_pg_delta, step_softmax_pg_into, step_tabular_q_into,
};
use math_commons::primitives::UnitInterval;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_sculpted_mdp_bounce_walls_wind_probabilities_and_legacy_from() {
    let legacy = GridWorldEnv {
        width: 5,
        height: 5,
        start: GridState { x: 0, y: 0 },
        goal: GridState { x: 4, y: 4 },
        traps: vec![GridState { x: 2, y: 2 }, GridState { x: 3, y: 2 }],
        walls: Vec::new(),
        gamma: 0.9,
    };
    let mut mdp = SculptedGridMdp::from(&legacy);
    assert!(mdp.is_default_layout());
    assert_eq!(
        mdp.step_deterministic(GridState { x: 0, y: 0 }, Move::Left),
        GridState { x: 0, y: 0 }
    );

    mdp.set_cell_role(GridState { x: 1, y: 0 }, GridCellRole::SolidWall);
    mdp.set_cell_role(GridState { x: 0, y: 1 }, GridCellRole::WindRight);
    mdp.wind_slip_prob = 0.35;
    assert!(!mdp.is_default_layout());

    // Moving Right from (0,0) into SolidWall at (1,0) bounces back to (0,0)
    assert_eq!(
        mdp.step_deterministic(GridState { x: 0, y: 0 }, Move::Right),
        GridState { x: 0, y: 0 }
    );

    // Moving Down from WindRight cell (0,1) lands at (0,2) with prob 0.65 and slips right to (1,2) with prob 0.35
    let p_no_slip = mdp.transition_probability(
        &GridState { x: 0, y: 2 },
        &GridState { x: 0, y: 1 },
        &Move::Down,
    );
    let p_slip = mdp.transition_probability(
        &GridState { x: 1, y: 2 },
        &GridState { x: 0, y: 1 },
        &Move::Down,
    );
    assert!((p_no_slip - 0.65).abs() < 1e-12);
    assert!((p_slip - 0.35).abs() < 1e-12);

    // Verify transition probabilities sum to 1.0 for every state and action
    let mut all_states = Vec::new();
    for y in 0..mdp.height {
        for x in 0..mdp.width {
            all_states.push(GridState { x, y });
        }
    }
    for s in &all_states {
        for a in ALL_MOVES {
            let sum_p: f64 = all_states
                .iter()
                .map(|ns| mdp.transition_probability(ns, s, &a))
                .sum();
            assert!(
                (sum_p - 1.0).abs() < 1e-12,
                "Probabilities from {s:?} under {a:?} summed to {sum_p}"
            );
        }
    }
}

#[test]
fn test_bellman_solver_matches_converged_tabular_q() {
    let env = GridWorldEnv {
        width: 4,
        height: 4,
        start: GridState { x: 0, y: 0 },
        goal: GridState { x: 3, y: 3 },
        traps: vec![GridState { x: 1, y: 2 }],
        walls: vec![GridState { x: 2, y: 1 }],
        gamma: 0.9,
    };
    let mdp = SculptedGridMdp::from(&env);
    let bellman = BellmanSolution::solve(&mdp, 1e-9, 250);

    let mut agent = TabularQAgent::new(
        UnitInterval::new(0.5).unwrap(),
        UnitInterval::new(0.9).unwrap(),
        UnitInterval::new(0.5).unwrap(),
        Some(12345),
    );
    for _ep in 0..4_000 {
        let mut s = env.start;
        for _step in 0..40 {
            if env.is_terminal(&s) {
                break;
            }
            let acts = env.actions(&s);
            if let Some(a) = agent.select_action(&s, &acts) {
                let ns = env.step(&s, &a);
                let r = env.reward(&s, &a, &ns);
                let next_acts = env.actions(&ns);
                agent.update(&s, &a, r, &ns, &next_acts);
                s = ns;
            }
        }
    }

    for y in 0..env.height {
        for x in 0..env.width {
            let st = GridState { x, y };
            if env.is_terminal(&st) {
                continue;
            }
            let max_q = ALL_MOVES
                .iter()
                .map(|a| agent.get_q_value(&st, a))
                .fold(f64::NEG_INFINITY, f64::max);
            let v_star = bellman.v_star_at(st);
            assert!(
                (max_q - v_star).abs() < 1e-2,
                "Mismatch at {st:?}: empirical max_q={max_q:.5}, V*={v_star:.5}"
            );
        }
    }
}

#[test]
fn test_bellman_policy_evaluation_and_sculpt_greedy_arrow_update() {
    let mut mdp = SculptedGridMdp::new(4, 4, 0.9, 0.0);
    mdp.set_cell_role(GridState { x: 1, y: 2 }, GridCellRole::Trap);
    let bellman = BellmanSolution::solve(&mdp, 1e-9, 250);

    let greedy_policy = TabularGridPolicy::from_q_table(4, 4, &bellman.q_star);
    assert!(greedy_policy.probability(&GridState { x: 0, y: 0 }, &Move::Right) >= 0.0);
    let mut eval_sol = bellman.clone();
    eval_sol.evaluate_policy(&mdp, &greedy_policy, 1e-9, 250);
    for idx in 0..16 {
        assert!((eval_sol.v_pi[idx] - eval_sol.v_star[idx]).abs() < 1e-6);
    }

    let before_act = bellman.optimal_action_at(GridState { x: 2, y: 1 });
    mdp.set_cell_role(GridState { x: 2, y: 2 }, GridCellRole::SolidWall);
    mdp.set_cell_role(GridState { x: 3, y: 1 }, GridCellRole::Trap);
    let after_bellman = BellmanSolution::solve(&mdp, 1e-9, 250);
    assert_ne!(
        after_bellman.optimal_action_at(GridState { x: 2, y: 1 }),
        Some(Move::Right)
    );
    assert!(before_act.is_some());
}

#[test]
fn test_1000_steps_dense_rl_workspace_zero_heap_allocations() {
    let mut mdp = SculptedGridMdp::default_with_traps(6, 6, 0.92);
    mdp.set_cell_role(GridState { x: 1, y: 1 }, GridCellRole::SolidWall);
    mdp.set_cell_role(GridState { x: 2, y: 1 }, GridCellRole::WindUp);
    mdp.wind_slip_prob = 0.25;

    let mut ws_q = DenseRlWorkspace::new(6, 6, 128, Some(77));
    let mut ws_pg = DenseRlWorkspace::new(6, 6, 128, Some(99));
    let alpha = UnitInterval::new(0.15).unwrap();
    let eps = UnitInterval::new(0.15).unwrap();
    let pg_lr = UnitInterval::new(0.05).unwrap();

    let delta = compute_softmax_pg_delta([0.0; 4], Move::Right, 2.0, pg_lr);
    assert!(delta[3] > 0.0);

    let (finite_ok, _metrics) = VerifiedEngine::run_verified(|| {
        let mut all_finite = true;
        for _ in 0..1_000 {
            let out_q = step_tabular_q_into(&mdp, &mut ws_q, alpha, eps);
            let out_pg = step_softmax_pg_into(&mdp, &mut ws_pg, pg_lr);
            all_finite &= out_q.td_error.is_finite()
                && out_q.state_value.is_finite()
                && out_pg.td_error.is_finite()
                && out_pg.state_value.is_finite();
        }
        all_finite
    });
    assert!(finite_ok);
    assert!(ws_q.episodes_completed > 0);
    assert!(ws_pg.episodes_completed > 0);
}

#[test]
fn test_apply_cells_u8_returns_typed_ai_errors_with_zero_allocations() {
    use domain_ai::error::AIError;

    let mut mdp = SculptedGridMdp::new(3, 2, 0.9, 0.0);
    let short_cells = [0_u8; 5];
    let bad_tag_cells = [0_u8, 1, 2, 99, 0, 0];
    let valid_cells = [0_u8, 1, 2, 3, 4, 5];

    let ((err_len, err_tag, ok_res), _) = VerifiedEngine::run_verified(|| {
        let e1 = mdp.apply_cells_u8(&short_cells);
        let e2 = mdp.apply_cells_u8(&bad_tag_cells);
        let ok = mdp.apply_cells_u8(&valid_cells);
        (e1, e2, ok)
    });

    assert_eq!(
        err_len,
        Err(AIError::CellMaskLengthMismatch {
            expected: 6,
            actual: 5,
        })
    );
    assert_eq!(
        err_tag,
        Err(AIError::InvalidCellRoleTag { index: 3, code: 99 })
    );
    assert_eq!(ok_res, Ok(()));
    let msg = err_len.unwrap_err().to_string();
    assert!(!msg.contains('{') && msg.contains("expected 6 cells, got 5"));
}
