#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::SimulationFramework;
use crate::presets::{PresetDialogState, serialize_compact_preset_envelope};
use federated_registry::Severity;
use std::sync::mpsc::channel;

#[test]
fn test_serialized_graph_state_rejects_near_u32_max_and_stale_counter() {
    let valid_boundary = SerializedGraphState {
        node_positions: BTreeMap::new(),
        edges: Vec::new(),
        selected_node: None,
        next_node_id: snapshot::MAX_NEXT_NODE_ID,
        edge_latencies: Vec::new(),
        congestion: BraessCongestionConfig::default(),
    };
    assert!(valid_boundary.validate().is_ok());

    for bad_next in [
        snapshot::MAX_NEXT_NODE_ID + 1,
        (u32::MAX - 1) as usize,
        u32::MAX as usize,
    ] {
        let near_max = SerializedGraphState {
            node_positions: BTreeMap::new(),
            edges: Vec::new(),
            selected_node: None,
            next_node_id: bad_next,
            edge_latencies: Vec::new(),
            congestion: BraessCongestionConfig::default(),
        };
        assert!(
            near_max.validate().is_err(),
            "expected next_node_id {bad_next} to fail headroom check"
        );
    }

    let mut positions = BTreeMap::new();
    positions.insert(5, (10.0, 20.0));
    for stale_next in [0, 3, 5] {
        let stale = SerializedGraphState {
            node_positions: positions.clone(),
            edges: Vec::new(),
            selected_node: Some(5),
            next_node_id: stale_next,
            edge_latencies: Vec::new(),
            congestion: BraessCongestionConfig::default(),
        };
        assert!(
            stale.validate().is_err(),
            "expected next_node_id {stale_next} <= max node ID 5 to be rejected"
        );
    }
}

fn assert_graph_import_rejected(
    framework: &mut SimulationFramework,
    bad_cfg: &serde_json::Value,
    baseline: &str,
) {
    let raw_str = bad_cfg.to_string();
    assert!(
        framework
            .active_tool
            .as_mut()
            .unwrap()
            .try_load_state(&raw_str)
            .is_err()
    );
    assert_eq!(
        framework
            .active_tool
            .as_ref()
            .unwrap()
            .save_state()
            .as_deref(),
        Some(baseline)
    );

    let env = serialize_compact_preset_envelope("Graph Editor", "graph_theory", bad_cfg)
        .expect("envelope");
    let mut dialog = PresetDialogState::default();
    let (tx, rx) = channel();
    dialog.import_rx = Some(rx);
    tx.send(Ok(("bad_graph.json".to_string(), env.clone())))
        .unwrap();
    let loaded = dialog.poll_pending_import(|js| {
        framework
            .active_tool
            .as_mut()
            .unwrap()
            .import_preset_json(js)
    });
    assert!(!loaded);
    assert!(dialog.toast_error.is_some());
    let events = crate::telemetry::poll_app_telemetry();
    assert!(
        events
            .iter()
            .any(|e| e.source == "PresetSerialization" && e.severity == Severity::Warning)
    );

    assert!(
        framework
            .import_url_preset_json(Some("GraphEditorTool"), Some(&env))
            .is_err()
    );
    assert_eq!(
        framework
            .active_tool
            .as_ref()
            .unwrap()
            .save_state()
            .as_deref(),
        Some(baseline)
    );
}

#[test]
fn test_graph_editor_loader_and_url_preset_fail_fast_preserve_state() {
    let _ = crate::telemetry::poll_app_telemetry();
    let mut framework = SimulationFramework::new("graph_theory");
    assert!(framework.select_tool_by_name("Graph Editor"));
    {
        let tool = framework
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<GraphEditorTool>()
            .unwrap();
        tool.click_add_node(Pos2::new(120.0, 140.0));
    }
    let baseline = framework
        .active_tool
        .as_ref()
        .unwrap()
        .save_state()
        .unwrap();

    let overflow_cfg = serde_json::json!({
        "node_positions": {"0": [10.0, 20.0]},
        "edges": [],
        "selected_node": 0,
        "next_node_id": u32::MAX
    });
    let stale_cfg = serde_json::json!({
        "node_positions": {"4": [10.0, 20.0]},
        "edges": [],
        "selected_node": 4,
        "next_node_id": 4
    });

    assert_graph_import_rejected(&mut framework, &overflow_cfg, &baseline);
    assert_graph_import_rejected(&mut framework, &stale_cfg, &baseline);
}

#[test]
fn test_click_and_keyboard_node_insertion_saturate_cleanly_at_u32_max() {
    let mut tool = GraphEditorTool::default();
    let viewport = egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(800.0, 600.0));

    tool.next_node_id = (u32::MAX as usize) - 1;
    tool.click_add_node(Pos2::new(50.0, 60.0));
    assert_eq!(tool.node_positions.len(), 1);
    assert!(tool.node_positions.contains_key(&((u32::MAX as usize) - 1)));
    assert_eq!(tool.next_node_id, u32::MAX as usize);
    assert_eq!(tool.undo_stack.len(), 1);

    for capped_id in [u32::MAX as usize, usize::MAX] {
        tool.next_node_id = capped_id;
        crate::accessibility::clear_test_announcements();
        let undo_len_before = tool.undo_stack.len();

        tool.click_add_node(Pos2::new(200.0, 220.0));
        assert_eq!(tool.node_positions.len(), 1);
        assert_eq!(tool.graph.graph.node_count(), 1);
        assert_eq!(tool.undo_stack.len(), undo_len_before);

        tool.insert_node_from_keyboard(viewport);
        assert_eq!(tool.node_positions.len(), 1);
        assert_eq!(tool.graph.graph.node_count(), 1);
        assert_eq!(tool.undo_stack.len(), undo_len_before);

        let msgs = crate::accessibility::all_announcements();
        assert_eq!(
            msgs.iter()
                .filter(|(m, _)| m.contains("node identifier limit reached"))
                .count(),
            2
        );
    }
}
