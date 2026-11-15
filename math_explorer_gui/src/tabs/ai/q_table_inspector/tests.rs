#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_q_table_inspector_initial_state() {
    let tool = QTableInspectorTool::default();
    assert_eq!(tool.episodes_trained, 0);
    let q = tool.get_q_value_for_view(&GridState { x: 0, y: 0 });
    assert_eq!(q, 0.0);
}

#[test]
fn test_q_table_inspector_sync_training() {
    let mut tool = QTableInspectorTool::default();
    tool.train(5);
    assert_eq!(tool.episodes_trained, 5);
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
