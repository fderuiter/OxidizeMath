#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{BrushMode, BrushStroke};
use crate::url_state::MAX_DECODED_URL_STATE_BYTES;
use math_explorer::ai::reinforcement_learning::{GridCellRole, GridState};

#[test]
fn test_q_table_inspector_initial_state() {
    let tool = QTableInspectorTool::default();
    assert_eq!(tool.episodes_trained, 0);
    let q = tool.get_q_value_for_view(&GridState { x: 0, y: 0 });
    assert_eq!(q, 0.0);
}

#[test]
fn test_q_table_inspector_sync_training_and_sonification() {
    let mut tool = QTableInspectorTool::default();
    tool.train(5);
    assert_eq!(tool.episodes_trained, 5);
    assert!(tool.latest_audio.is_audible());
}

#[test]
fn test_q_table_inspector_background_offload() {
    let mut tool = QTableInspectorTool::default();
    let _ = tool.worker.send_command(SimCommand::TrainEpisodes(10));
    let mut completed = false;
    for _ in 0..100 {
        std::thread::sleep(std::time::Duration::from_millis(15));
        if let Some(snap) = tool.worker.update() {
            if snap.custom_data.first().copied().unwrap_or(0.0) >= 10.0 {
                completed = true;
                break;
            }
        }
    }
    assert!(completed);
}

#[test]
fn test_q_table_inspector_sculpt_sync_and_rle_roundtrip() {
    let mut tool = QTableInspectorTool {
        view: HeatmapView::BellmanVStar,
        ..Default::default()
    };
    let before_v = tool.get_q_value_for_view(&GridState { x: 0, y: 0 });

    let stroke1 = BrushStroke {
        start: [1.5, 4.5],
        end: [1.5, 1.5],
        radius: 0.5,
        mode: BrushMode::Solid,
        drag_vector: [0.0, -3.0],
    };
    let stroke2 = BrushStroke {
        start: [3.5, 3.5],
        end: [3.5, 0.5],
        radius: 0.5,
        mode: BrushMode::Solid,
        drag_vector: [0.0, -3.0],
    };
    let _ = apply_rl_brush_stroke(&mut tool.mdp, &stroke1);
    let _ = apply_rl_brush_stroke(&mut tool.mdp, &stroke2);
    tool.mdp.wind_slip_prob = 0.25;
    tool.sync_sculpt_to_worker();
    assert_eq!(
        tool.mdp.cell_role(GridState { x: 1, y: 0 }),
        GridCellRole::SolidWall
    );
    let after_v = tool.get_q_value_for_view(&GridState { x: 0, y: 0 });
    assert_ne!(before_v, after_v);

    let saved = tool.save_state().expect("save QTableInspectorState");
    assert!(saved.len() < MAX_DECODED_URL_STATE_BYTES);

    let mut restored = QTableInspectorTool::default();
    restored
        .try_load_state(&saved)
        .expect("restore QTableInspectorState");
    assert_eq!(restored.mdp, tool.mdp);
    assert_eq!(restored.view, HeatmapView::BellmanVStar);
}

#[test]
fn test_q_table_inspector_policy_gradient_20x20_preserves_v_pi_across_sculpting() {
    let mut mdp = SculptedGridMdp::default_with_traps(20, 20, 0.95);
    mdp.wind_slip_prob = 0.1;
    let runner = SculptedQTableRunner::from_mdp(mdp.clone(), RlSolverMode::PolicyGradient);
    let worker = BackgroundWorker::new(runner);
    let latest_snap = worker
        .latest_snapshot()
        .and_then(|s| extract_snapshot_parts(s).1)
        .expect("seeded initial QTableSnapshot");
    let mut tool = QTableInspectorTool {
        mdp,
        solver_mode: RlSolverMode::PolicyGradient,
        maze_preset: RlMazePreset::BaselineTraps,
        brush_kit: new_rl_brush_kit(),
        latest_snap,
        episodes_trained: 0,
        view: HeatmapView::PolicyVPi,
        latest_audio: SonificationFrame::silent(),
        revision: 0,
        worker,
    };

    tool.train(30);
    assert_eq!(tool.episodes_trained, 30);
    assert!(
        tool.latest_snap
            .theta_logits
            .iter()
            .any(|row| row.iter().any(|&v| v.abs() > 1e-6)),
        "Expected trained PolicyGradient logits to be non-zero"
    );
    let pre_sculpt_logits = tool.latest_snap.theta_logits.clone();

    let stroke = BrushStroke {
        start: [10.5, 10.5],
        end: [10.5, 6.5],
        radius: 0.5,
        mode: BrushMode::Solid,
        drag_vector: [0.0, -4.0],
    };
    let _ = apply_rl_brush_stroke(&mut tool.mdp, &stroke);
    tool.sync_sculpt_to_worker();

    for y in 0..20 {
        for x in 0..20 {
            let idx = (y * 20 + x) as usize;
            let st = GridState { x, y };
            if tool.mdp.cell_role(st) == GridCellRole::SolidWall {
                assert_eq!(tool.latest_snap.theta_logits[idx], [0.0; 4]);
            } else {
                assert_eq!(
                    tool.latest_snap.theta_logits[idx], pre_sculpt_logits[idx],
                    "Sculpting a 20x20 grid must preserve learned PolicyGradient theta_logits at {st:?}"
                );
            }
        }
    }
    let near_goal = tool.get_q_value_for_view(&GridState { x: 18, y: 19 });
    let start_val = tool.get_q_value_for_view(&GridState { x: 0, y: 0 });
    assert!(near_goal.is_finite() && start_val.is_finite());
    assert!(
        (near_goal - start_val).abs() > 1e-4,
        "Expected non-degenerate V_pi values across 20x20 grid after sculpting, got start={start_val}, near_goal={near_goal}"
    );
}
