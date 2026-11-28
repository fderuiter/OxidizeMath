use super::*;
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_howard_bellman_solver_zero_alloc_converges_at_gamma_0_999_under_20_sweeps() {
    let mut mdp = SculptedGridMdp::default_with_traps(6, 6, 0.999);
    mdp.set_cell_role(GridState { x: 1, y: 1 }, GridCellRole::SolidWall);
    mdp.set_cell_role(GridState { x: 3, y: 1 }, GridCellRole::WindDown);
    mdp.wind_slip_prob = 0.20;

    let mut ws_buf = [0.0_f64; bellman_grid_workspace_len(6, 6)];
    let mut sol = BellmanSolution::empty(6, 6);

    let stats = {
        let _guard = crate::VERIFIED_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (s, _) = VerifiedEngine::run_verified(|| {
            let mut ws = ScalarWorkspace::new(&mut ws_buf);
            let st = sol
                .solve_into(&mdp, 1.0e-11, 20, &mut ws)
                .expect("Howard policy iteration must succeed");
            let greedy = TabularGridPolicy::from_q_table(6, 6, &sol.q_star);
            sol.evaluate_policy_with_workspace(&mdp, &greedy, &mut ws)
                .expect("LU policy evaluation must succeed");
            st
        });
        s
    };

    assert!(stats.converged);
    assert!(
        stats.sweeps < 20,
        "Howard policy iteration took {} sweeps at gamma=0.999 (expected < 20)",
        stats.sweeps
    );
    assert!(
        stats.bellman_residual < 1.0e-9,
        "Bellman residual {} exceeded 1e-9 at gamma=0.999",
        stats.bellman_residual
    );
    assert!(sol.max_delta < 1.0e-9);
    for idx in 0..36 {
        assert!((sol.v_pi[idx] - sol.v_star[idx]).abs() < 1.0e-9);
    }
}
