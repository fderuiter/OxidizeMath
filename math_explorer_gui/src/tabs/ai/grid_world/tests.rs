#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::RasterBrushStroke;
use crate::url_state::{encode_share_fragment, parse_share_location, URL_JSON_BOUNDS};

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

fn assert_greedy_bellman_reroute(tool: &mut GridWorldTool) {
    tool.learning_rate = 0.4;
    tool.exploration_rate = 0.25;
    tool.sync_agent_hyperparameters();
    for _ in 0..4 {
        tool.train_batch();
    }
    assert!(!tool.env.walls.contains(&tool.current_state));
    tool.exploration_rate = 0.0;
    tool.sync_agent_hyperparameters();
    let mut probe = tool.env.start;
    let mut reached_goal = false;
    for _ in 0..25 {
        let acts = tool.env.actions(&probe);
        let act = tool.agent.select_action(&probe, &acts).unwrap();
        probe = tool.env.step(&probe, &act);
        assert!(
            !tool.env.walls.contains(&probe),
            "Agent entered impassable wall"
        );
        if probe == tool.env.goal {
            reached_goal = true;
            break;
        }
    }
    assert!(
        reached_goal,
        "Bellman Q-learning agent must route around sculpted wall to reach relocated Goal"
    );
}

#[test]
fn test_grid_world_live_sculpting_bellman_rerouting_and_url_roundtrip() {
    let mut tool = GridWorldTool::default();
    let rev0 = tool.state_revision();

    tool.clear_sculpted_obstacles();
    tool.brush_mode = GridBrushMode::Wall;
    tool.apply_brush_stroke(RasterBrushStroke {
        cells: vec![(1, 0), (1, 1), (1, 2), (1, 3)],
        secondary: false,
        via_keyboard: false,
        focus_cell: (1, 3),
    });
    assert_eq!(tool.env.walls.len(), 4);
    assert_ne!(tool.state_revision(), rev0);

    tool.brush_mode = GridBrushMode::Goal;
    tool.apply_brush_stroke(RasterBrushStroke {
        cells: vec![(2, 0)],
        secondary: false,
        via_keyboard: true,
        focus_cell: (2, 0),
    });
    assert_eq!(tool.env.goal, GridState { x: 2, y: 0 });

    tool.brush_mode = GridBrushMode::Wall;
    tool.apply_brush_stroke(RasterBrushStroke {
        cells: vec![(0, 0), (2, 0)],
        secondary: false,
        via_keyboard: false,
        focus_cell: (2, 0),
    });
    assert!(!tool.env.walls.contains(&tool.env.start));
    assert!(!tool.env.walls.contains(&tool.env.goal));

    assert_greedy_bellman_reroute(&mut tool);

    let saved = tool.save_state().expect("GridWorld save_state");
    assert!(saved.contains("layout_rle"));
    URL_JSON_BOUNDS
        .validate_json(&saved)
        .expect("GridWorld sculpted state fits URL_JSON_BOUNDS");
    let fragment = encode_share_fragment("ai", Some("GridWorldTool"), Some(&saved))
        .expect("encode GridWorld URL fragment");
    let decoded_url = parse_share_location(&fragment, "")
        .expect("decode GridWorld URL fragment")
        .expect("present share params");

    let mut restored = GridWorldTool::default();
    restored
        .try_load_state(decoded_url.state_json.as_deref().unwrap())
        .expect("restore sculpted GridWorld");
    assert_eq!(restored.env.walls.len(), 4);
    assert_eq!(restored.env.goal, GridState { x: 2, y: 0 });
}
