#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

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
    assert_eq!(tool.env.gamma, 0.8);

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
fn test_grid_world_step_and_terminal_reset() {
    let mut tool = GridWorldTool::default();
    let initial_state = tool.current_state;
    tool.step_agent();
    assert_ne!(
        (tool.current_state, tool.steps, tool.episodes),
        (initial_state, 0, 0)
    );
    tool.current_state = tool.env.goal;
    tool.step_agent();
    assert_eq!(
        (tool.current_state, tool.steps, tool.episodes),
        (tool.env.start, 0, 1)
    );
}
