#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::app::workspace_bus::{SharedWorkspaceBus, MAX_TOPOLOGY_NODES};

#[test]
fn test_graph_editor_normalizes_coordinates_and_publishes_revisions() {
    let mut tool = GraphEditorTool::default();
    let mut bus = SharedWorkspaceBus::default();

    // Single node maps cleanly to [0.0, 0.0] without NaN.
    tool.push_undo_snapshot();
    tool.node_positions.insert(0, Pos2::new(250.0, 180.0));
    tool.next_node_id = 1;
    tool.rebuild_graph();
    tool.publish_topology_to_bus(&mut bus);
    assert_eq!(bus.topology.revision(), 1);
    assert_eq!(bus.topology.positions(), &[[0.0, 0.0]]);

    // Adding a second node and edge updates normalized coordinates and increments revision.
    tool.push_undo_snapshot();
    tool.node_positions.insert(1, Pos2::new(450.0, 180.0));
    tool.edges.push((0, 1, 2.5));
    tool.next_node_id = 2;
    tool.rebuild_graph();
    tool.publish_topology_to_bus(&mut bus);
    assert_eq!(bus.topology.revision(), 2);
    assert_eq!(bus.topology.positions(), &[[-1.0, 0.0], [1.0, 0.0]]);
    assert_eq!(bus.topology.edges(), &[(0, 1, 2.5)]);

    // Calling publish again without local edits keeps revision unchanged.
    tool.publish_topology_to_bus(&mut bus);
    assert_eq!(bus.topology.revision(), 2);
}

#[test]
fn test_graph_editor_caps_published_nodes_at_256() {
    let mut tool = GraphEditorTool::default();
    tool.push_undo_snapshot();
    for i in 0..280 {
        tool.node_positions
            .insert(i, Pos2::new(i as f32 * 5.0, (i % 17) as f32 * 10.0));
    }
    tool.next_node_id = 280;
    let mut bus = SharedWorkspaceBus::default();
    tool.publish_topology_to_bus(&mut bus);
    assert_eq!(bus.topology.positions().len(), MAX_TOPOLOGY_NODES);
}
