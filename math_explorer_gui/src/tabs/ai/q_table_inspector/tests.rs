#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{BrushMode, BrushStroke};
use crate::url_state::MAX_DECODED_URL_STATE_BYTES;
use math_explorer::ai::reinforcement_learning::GridCellRole;

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
        if let Some(snap) = tool.worker.update()
            && snap.custom_data.first().copied().unwrap_or(0.0) >= 10.0
        {
            completed = true;
            break;
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
