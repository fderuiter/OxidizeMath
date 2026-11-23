#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{BrushMode, BrushStroke};
use crate::tabs::ai::rl_sculpt::RlOverlayMode;
use crate::url_state::MAX_DECODED_URL_STATE_BYTES;

#[test]
fn test_grid_world_default_playback_and_rl_hyperparameters() {
    let mut tool = GridWorldTool::default();
    assert!(!tool.is_playing && (1.0..=20.0).contains(&tool.playback_speed));
    assert_eq!(
        (
            tool.learning_rate,
            tool.discount_factor,
            tool.exploration_rate
        ),
        (0.1, 0.9, 0.1)
    );

    tool.learning_rate = 0.35;
    tool.discount_factor = 0.8;
    tool.exploration_rate = 0.25;
    tool.sync_agent_hyperparameters();
    assert_eq!(tool.mdp.gamma, 0.8);

    let saved = tool.save_state().expect("save_state");
    let mut restored = GridWorldTool::default();
    restored.try_load_state(&saved).expect("try_load_state");
    assert_eq!(
        (
            restored.learning_rate,
            restored.discount_factor,
            restored.exploration_rate
        ),
        (0.35, 0.8, 0.25)
    );
    restored.reset_agent();
    assert_eq!(
        (
            restored.learning_rate,
            restored.discount_factor,
            restored.exploration_rate
        ),
        (0.35, 0.8, 0.25)
    );
}

#[test]
fn test_grid_world_step_terminal_reset_and_sonification() {
    let mut tool = GridWorldTool::default();
    let initial_state = tool.current_state;
    tool.step_agent();
    assert_ne!(
        (tool.current_state, tool.steps, tool.episodes),
        (initial_state, 0, 0)
    );
    assert!(tool.latest_audio.is_audible());

    tool.current_state = tool.mdp.goal;
    tool.step_agent();
    assert_eq!(
        (tool.current_state, tool.steps, tool.episodes),
        (tool.mdp.start, 0, 1)
    );
}

#[test]
fn test_grid_world_sculpt_rle_roundtrip_and_bellman_update() {
    let mut tool = GridWorldTool::default();
    tool.mdp.wind_slip_prob = 0.35;
    tool.solver_mode = RlSolverMode::PolicyGradient;
    tool.overlay_mode = RlOverlayMode::BellmanVStar;

    let stroke = BrushStroke {
        start: [1.5, 3.5],
        end: [3.5, 3.5],
        radius: 0.5,
        mode: BrushMode::Solid,
        drag_vector: [2.0, 0.0],
    };
    assert!(apply_rl_brush_stroke(&mut tool.mdp, &stroke));
    tool.bellman_dirty = true;
    tool.ensure_bellman_up_to_date();
    assert_eq!(
        tool.mdp.cell_role(GridState { x: 1, y: 1 }),
        GridCellRole::SolidWall
    );

    let saved = tool.save_state().expect("save sculpted state");
    assert!(saved.len() < MAX_DECODED_URL_STATE_BYTES);

    let mut restored = GridWorldTool::default();
    restored
        .try_load_state(&saved)
        .expect("restore sculpted state");
    assert_eq!(restored.mdp, tool.mdp);
    assert_eq!(restored.solver_mode, RlSolverMode::PolicyGradient);
    assert_eq!(restored.overlay_mode, RlOverlayMode::BellmanVStar);
    assert_eq!(restored.bellman.v_star, tool.bellman.v_star);
}
