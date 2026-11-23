#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::SimulationRunner;
use crate::url_state::MAX_DECODED_URL_STATE_BYTES;
use math_explorer::ai::reinforcement_learning::{GridCellRole, GridState};

#[test]
fn test_reward_plots_sculpted_bellman_residuals_and_sonification() {
    let mut mdp = RlMazePreset::WindyCliff.build_mdp(5, 5, 0.92);
    mdp.set_cell_role(GridState { x: 2, y: 1 }, GridCellRole::SolidWall);
    let mut runner = RewardPlotsRunner::from_mdp(mdp, RlSolverMode::PolicyGradient);
    runner.process_command(SimCommand::TrainEpisodes(6));
    while !runner.is_complete() {
        runner.step();
    }
    let snap = runner.get_snapshot();
    assert!(snap.sonification.is_audible());
    let data = snap
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<RlSnapshotData>())
        .expect("RlSnapshotData");
    assert_eq!(data.episodes_trained, 6);
    assert_eq!(data.rewards_per_episode.len(), 6);
    assert_eq!(data.bellman_residuals.len(), 6);
    assert!(data.bellman_residuals.iter().all(|r| r.is_finite()));
}

#[test]
fn test_reward_plots_rle_state_roundtrip() {
    let mut tool = RewardPlotsTool {
        training_steps: 250,
        solver_mode: RlSolverMode::PolicyGradient,
        mdp: RlMazePreset::ObstacleCorridor.build_mdp(6, 6, 0.95),
        ..Default::default()
    };
    tool.mdp.wind_slip_prob = 0.25;

    let saved = tool.save_state().expect("save RewardPlotsState");
    assert!(saved.len() < MAX_DECODED_URL_STATE_BYTES);

    let mut restored = RewardPlotsTool::default();
    restored
        .try_load_state(&saved)
        .expect("restore RewardPlotsState");
    assert_eq!(restored.training_steps, 250);
    assert_eq!(restored.solver_mode, RlSolverMode::PolicyGradient);
    assert_eq!(restored.mdp, tool.mdp);
}
