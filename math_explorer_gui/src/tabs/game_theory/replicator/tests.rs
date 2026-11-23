#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use std::collections::BTreeMap;

#[test]
fn test_replicator_step_once_and_reset_playback() {
    let mut tool = ReplicatorDynamicsTool::default();
    assert!(!tool.is_playing);
    assert_eq!(tool.playback_time, 0.0);

    tool.step_once();
    assert!(!tool.is_playing);
    assert!((tool.playback_time - tool.dt).abs() < 1e-9);

    tool.reset_playback();
    assert!(!tool.is_playing);
    assert_eq!(tool.playback_time, 0.0);
}

#[test]
fn test_symbolic_hawk_dove_parameter_sweep_and_transient_error_preservation() {
    let mut tool = ReplicatorDynamicsTool::default();
    tool.load_preset_symbolic_hawk_dove();
    assert!(tool.symbolic_mode);
    assert!((tool.payoff_matrix[(0, 0)] - -1.0).abs() < 1e-12);
    assert!((tool.payoff_matrix[(0, 1)] - 2.0).abs() < 1e-12);
    assert!((tool.payoff_matrix[(1, 0)] - 0.0).abs() < 1e-12);
    assert!((tool.payoff_matrix[(1, 1)] - 1.0).abs() < 1e-12);

    // Increase injury cost c from 4.0 to 8.0 -> Hawk-Hawk payoff becomes (2 - 8)/2 = -3.0,
    // shifting equilibrium Hawk frequency v/c from 0.5 down to 0.25.
    let mut updated_params = BTreeMap::new();
    updated_params.insert("v".to_string(), 2.0);
    updated_params.insert("c".to_string(), 8.0);
    tool.symbolic_sliders.set_params_map(&updated_params);
    tool.evaluate_symbolic_matrix();
    tool.recompute_trajectory_preserving_playback();

    assert!((tool.payoff_matrix[(0, 0)] - -3.0).abs() < 1e-12);
    let final_hawk = tool.trajectory.last().expect("trajectory non-empty").1[0];
    assert!(
        (final_hawk - 0.25).abs() < 0.02,
        "expected Hawk ESS near v/c = 0.25, got {final_hawk}"
    );

    // Transient syntax edit in cell (0, 0) preserves last valid compiled formula and finite trajectory
    tool.symbolic_bars[0][0].set_expression("(v - c) /");
    assert!(!tool.symbolic_bars[0][0].is_valid());
    tool.sync_and_eval_symbolic_payoffs();
    assert!((tool.payoff_matrix[(0, 0)] - -3.0).abs() < 1e-12);

    // Round-trip state serialization preserves symbolic_payoffs and symbolic_params
    tool.symbolic_bars[0][0].set_expression("(v - c) / 2");
    tool.sync_and_eval_symbolic_payoffs();
    let saved = tool.save_state().expect("save_state");
    let mut restored = ReplicatorDynamicsTool::default();
    restored.try_load_state(&saved).expect("try_load_state");
    assert!(restored.symbolic_mode);
    assert_eq!(restored.save_state().as_deref(), Some(saved.as_str()));
}

#[test]
fn test_symbolic_payoffs_couple_with_spatial_workspace_graph_and_roundtrip() {
    use crate::app::data_bus::{SharedGraph, SharedNode};
    use std::sync::Arc;

    let mut tool = ReplicatorDynamicsTool::default();
    tool.load_preset_symbolic_hawk_dove();
    let graph = Arc::new(SharedGraph {
        revision: 7,
        nodes: vec![
            SharedNode {
                id: 0,
                norm_pos: [0.2, 0.3],
            },
            SharedNode {
                id: 1,
                norm_pos: [0.5, 0.7],
            },
            SharedNode {
                id: 2,
                norm_pos: [0.8, 0.3],
            },
        ],
        edges: vec![(0, 1, 1.5), (1, 2, 1.5), (0, 2, 1.0)],
    });
    tool.apply_shared_game_graph(&graph);
    let initial_spatial_end = tool.spatial.trajectory.last().expect("spatial traj").1[0];

    let mut updated_params = BTreeMap::new();
    updated_params.insert("v".to_string(), 2.0);
    updated_params.insert("c".to_string(), 9.0);
    tool.symbolic_sliders.set_params_map(&updated_params);
    tool.evaluate_symbolic_matrix();
    tool.recompute_trajectory_preserving_playback();
    let shifted_spatial_end = tool.spatial.trajectory.last().expect("spatial traj").1[0];
    assert!(
        shifted_spatial_end < initial_spatial_end - 0.03,
        "higher injury cost c must lower spatial Hawk frequency ({initial_spatial_end} -> {shifted_spatial_end})"
    );

    tool.spatial.follow_workspace_graph = false;
    let saved = tool.save_state().expect("save_state");
    let mut restored = ReplicatorDynamicsTool::default();
    restored.try_load_state(&saved).expect("try_load_state");
    assert!(restored.symbolic_mode);
    assert!(!restored.spatial.follow_workspace_graph);
    assert_eq!(
        restored.spatial.trajectory.len(),
        tool.spatial.trajectory.len()
    );
    assert_eq!(restored.save_state().as_deref(), Some(saved.as_str()));
}

#[test]
fn test_prisoners_dilemma_mean_fitness_decline_and_stag_hunt() {
    let mut tool = ReplicatorDynamicsTool::default();
    tool.load_preset_symbolic_hawk_dove();
    assert!(tool.symbolic_mode);

    tool.load_preset_prisoners_dilemma();
    assert!(!tool.symbolic_mode);
    assert_eq!(tool.symbolic_bars.len(), 2);
    let obs = tool.observables();
    let init_fit = obs
        .iter()
        .find(|(k, _)| *k == "initial_mean_fitness")
        .unwrap()
        .1;
    let final_fit = obs
        .iter()
        .find(|(k, _)| *k == "final_mean_fitness")
        .unwrap()
        .1;
    let x1_final = obs.iter().find(|(k, _)| *k == "x1_final").unwrap().1;
    assert!(init_fit > 2.8 && final_fit < 1.05 && x1_final > 0.99);

    tool.load_preset_stag_hunt();
    let low_final = tool
        .observables()
        .into_iter()
        .find(|(k, _)| *k == "x0_final")
        .unwrap()
        .1;
    assert!(low_final < 0.05);

    tool.load_preset_spiral_rps();
    assert!(!tool.symbolic_mode);
    assert_eq!(tool.symbolic_bars.len(), 3);
    tool.symbolic_mode = true;
    tool.sync_and_eval_symbolic_payoffs();
    assert!(tool.symbolic_error.is_none());
}
