#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{BrushMode, BrushStroke, RasterBrushStroke};
use crate::tabs::ai::rl_sculpt::RlOverlayMode;
use crate::url_state::{
    encode_share_fragment, parse_share_location, MAX_DECODED_URL_STATE_BYTES, URL_JSON_BOUNDS,
};

fn count_role(mdp: &SculptedGridMdp, role: GridCellRole) -> usize {
    let mut count = 0;
    for y in 0..mdp.height {
        for x in 0..mdp.width {
            if mdp.cell_role(GridState { x, y }) == role {
                count += 1;
            }
        }
    }
    count
}

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

fn assert_greedy_bellman_reroute(tool: &mut GridWorldTool) {
    tool.learning_rate = 0.4;
    tool.exploration_rate = 0.25;
    tool.sync_agent_hyperparameters();
    for _ in 0..4 {
        tool.train_batch();
    }
    assert_ne!(
        tool.mdp.cell_role(tool.current_state),
        GridCellRole::SolidWall
    );
    tool.exploration_rate = 0.0;
    tool.sync_agent_hyperparameters();
    let mut probe = tool.mdp.start;
    let mut reached_goal = false;
    for _ in 0..25 {
        let act = tool
            .workspace
            .greedy_q_action(probe)
            .expect("greedy action");
        probe = tool.mdp.step_deterministic(probe, act);
        assert_ne!(
            tool.mdp.cell_role(probe),
            GridCellRole::SolidWall,
            "Agent entered impassable wall"
        );
        if probe == tool.mdp.goal {
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
    assert_eq!(count_role(&tool.mdp, GridCellRole::SolidWall), 4);
    assert_ne!(tool.state_revision(), rev0);

    tool.brush_mode = GridBrushMode::Goal;
    tool.apply_brush_stroke(RasterBrushStroke {
        cells: vec![(2, 0)],
        secondary: false,
        via_keyboard: true,
        focus_cell: (2, 0),
    });
    assert_eq!(tool.mdp.goal, GridState { x: 2, y: 0 });

    tool.brush_mode = GridBrushMode::Wall;
    tool.apply_brush_stroke(RasterBrushStroke {
        cells: vec![(0, 0), (2, 0)],
        secondary: false,
        via_keyboard: false,
        focus_cell: (2, 0),
    });
    assert_ne!(tool.mdp.cell_role(tool.mdp.start), GridCellRole::SolidWall);
    assert_ne!(tool.mdp.cell_role(tool.mdp.goal), GridCellRole::SolidWall);

    assert_greedy_bellman_reroute(&mut tool);

    let saved = tool.save_state().expect("GridWorld save_state");
    assert!(saved.contains("rle_mask"));
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
    assert_eq!(count_role(&restored.mdp, GridCellRole::SolidWall), 4);
    assert_eq!(restored.mdp.goal, GridState { x: 2, y: 0 });
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

#[test]
fn test_grid_world_step_limit_timeout_resets_episode() {
    let mut tool = GridWorldTool::default();
    tool.mdp
        .set_cell_role(GridState { x: 1, y: 0 }, GridCellRole::SolidWall);
    tool.mdp
        .set_cell_role(GridState { x: 0, y: 1 }, GridCellRole::SolidWall);
    let limit = (tool.workspace.max_episode_steps.max(4)) as u32;
    for step_idx in 0..limit {
        tool.step_agent();
        if step_idx + 1 < limit {
            assert_eq!(tool.episodes, 0);
            assert_eq!(tool.steps, step_idx + 1);
        }
    }
    assert_eq!(tool.current_state, tool.mdp.start);
    assert_eq!(tool.steps, 0);
    assert_eq!(tool.episodes, 1);
}
