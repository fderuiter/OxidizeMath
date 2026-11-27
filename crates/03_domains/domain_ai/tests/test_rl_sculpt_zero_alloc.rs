//! Integration and zero-allocation tests for `SculptedGridMdp`, `BellmanSolution`,
//! `step_tabular_q_into`, and `step_softmax_pg_into`.

use domain_ai::ai::reinforcement_learning::{
    ALL_MOVES, BellmanSolution, DenseRlWorkspace, GridCellRole, GridState, GridWorldEnv,
    MarkovDecisionProcess, Move, Policy, SculptedGridMdp, TabularGridPolicy, TabularQAgent,
    compute_softmax_pg_delta, flush_reinforce_episode_in_place, move_to_index,
    softmax_action_probs, step_softmax_pg_into, step_tabular_q_into,
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
fn test_reinforce_discounted_batch_equivalence_and_no_q_table_fabrication() {
    let mut ws = DenseRlWorkspace::new(4, 4, 16, Some(42));
    ws.theta_logits[0] = [0.4, -0.2, 0.1, 0.3];
    ws.theta_logits[1] = [-0.1, 0.2, -0.3, 0.5];
    ws.state_values[0] = 0.8;
    ws.state_values[1] = 1.4;

    // Repeated state-action visits within one episode: s=0 (Right), s=0 (Left), s=1 (Right)
    ws.ep_states.push(0);
    ws.ep_actions.push(move_to_index(Move::Right));
    ws.ep_rewards.push(-0.2);
    ws.ep_states.push(0);
    ws.ep_actions.push(move_to_index(Move::Left));
    ws.ep_rewards.push(0.3);
    ws.ep_states.push(1);
    ws.ep_actions.push(move_to_index(Move::Right));
    ws.ep_rewards.push(1.0);
    ws.episode_steps = 3;

    let gamma = 0.9;
    let lr = 0.1;
    let bootstrap = 0.0;
    let g2 = 1.0 + gamma * bootstrap;
    let g1 = 0.3 + gamma * g2;
    let g0 = -0.2 + gamma * g1;
    let adv0 = g0 - 0.8;
    let adv1 = g1 - 0.8;
    let adv2 = g2 - 1.4;

    let probs_s0 = softmax_action_probs(ws.theta_logits[0]);
    let probs_s1 = softmax_action_probs(ws.theta_logits[1]);
    let mut expected_theta_s0 = ws.theta_logits[0];
    let mut expected_theta_s1 = ws.theta_logits[1];
    for (k, (&p0, &p1)) in probs_s0.iter().zip(probs_s1.iter()).enumerate() {
        let ind_right = f64::from(u8::from(k == move_to_index(Move::Right)));
        let ind_left = f64::from(u8::from(k == move_to_index(Move::Left)));
        expected_theta_s0[k] += lr * (adv0 * (ind_right - p0) + gamma * adv1 * (ind_left - p0));
        expected_theta_s1[k] += lr * (gamma * gamma * adv2 * (ind_right - p1));
    }
    let critic_lr = lr * 0.5;
    let expected_v0 = 0.8 + critic_lr * (adv0 + adv1);
    let expected_v1 = 1.4 + critic_lr * adv2;

    let (last_adv, _metrics) =
        VerifiedEngine::run_verified(|| flush_reinforce_episode_in_place(&mut ws, gamma, lr, 0.0));
    assert!((last_adv - adv2).abs() < 1e-12);
    for k in 0..4 {
        assert!((ws.theta_logits[0][k] - expected_theta_s0[k]).abs() < 1e-12);
        assert!((ws.theta_logits[1][k] - expected_theta_s1[k]).abs() < 1e-12);
    }
    assert!((ws.state_values[0] - expected_v0).abs() < 1e-12);
    assert!((ws.state_values[1] - expected_v1).abs() < 1e-12);

    let mdp = SculptedGridMdp::default_with_traps(4, 4, 0.9);
    let mut ws_pg = DenseRlWorkspace::new(4, 4, 64, Some(7));
    let pg_lr = UnitInterval::new(0.05).unwrap();
    for _ in 0..200 {
        let _ = step_softmax_pg_into(&mdp, &mut ws_pg, pg_lr);
    }
    assert!(ws_pg.q_table.iter().all(|q| q.iter().all(|&v| v == 0.0)));
}

#[test]
fn test_reinforce_timeout_truncation_bootstrap_and_20x20_bellman_convergence() {
    // Pardo et al. (2018): r = -0.1, gamma = 0.9, V(s_next) = 3.0, V(s_t) = 1.5 -> A = +1.10
    let mut mdp = SculptedGridMdp::new(4, 4, 0.9, 0.0);
    mdp.set_cell_role(GridState { x: 1, y: 0 }, GridCellRole::SolidWall);
    mdp.set_cell_role(GridState { x: 0, y: 1 }, GridCellRole::SolidWall);

    let mut ws = DenseRlWorkspace::new(4, 4, 4, Some(123));
    ws.state_values[0] = 1.5;
    for _ in 0..4 {
        let out_trunc = step_softmax_pg_into(&mdp, &mut ws, UnitInterval::new(0.1).unwrap());
        assert!(out_trunc.td_error.is_finite());
    }

    let mut ws_distinct = DenseRlWorkspace::new(4, 4, 4, Some(123));
    ws_distinct.ep_states.push(0);
    ws_distinct.ep_actions.push(move_to_index(Move::Right));
    ws_distinct.ep_rewards.push(-0.1);
    ws_distinct.episode_steps = 1;
    ws_distinct.state_values[0] = 1.5;
    ws_distinct.state_values[1] = 3.0;
    let adv_trunc = flush_reinforce_episode_in_place(&mut ws_distinct, 0.9, 0.1, 3.0);
    assert!((adv_trunc - 1.10).abs() < 1e-12);

    // TabularGridPolicy::sample must draw stochastically according to action probabilities
    let mut logits = vec![[0.0; 4]; 16];
    logits[0] = [f64::ln(0.6), f64::ln(0.4), -30.0, -30.0];
    let policy = TabularGridPolicy::from_softmax_logits(4, 4, &logits);
    let mut up_count = 0usize;
    let mut down_count = 0usize;
    for _ in 0..500 {
        match policy.sample(&GridState { x: 0, y: 0 }) {
            Move::Up => up_count += 1,
            Move::Down => down_count += 1,
            other => panic!("Unexpected sampled move {other:?} for near-zero probability arm"),
        }
    }
    assert!((230..=370).contains(&up_count) && down_count > 120);

    // 20x20 grid at gamma = 0.99 must converge Bellman residuals near 0.0
    let large_mdp = SculptedGridMdp::default_with_traps(20, 20, 0.99);
    let sweeps = BellmanSolution::recommended_sweeps(&large_mdp, 1e-9);
    let sol = BellmanSolution::solve(&large_mdp, 1e-9, sweeps);
    let mut max_residual = 0.0_f64;
    for y in 0..20 {
        for x in 0..20 {
            let s = GridState { x, y };
            if large_mdp.is_terminal(&s) || large_mdp.cell_role(s) == GridCellRole::SolidWall {
                continue;
            }
            let best_q = ALL_MOVES
                .iter()
                .map(|&m| sol.q_star_at(s, m))
                .fold(f64::NEG_INFINITY, f64::max);
            max_residual = max_residual.max((sol.v_star_at(s) - best_q).abs());
        }
    }
    assert!(
        max_residual < 1e-6,
        "Expected Bellman residual near 0.0, got {max_residual}"
    );
}
