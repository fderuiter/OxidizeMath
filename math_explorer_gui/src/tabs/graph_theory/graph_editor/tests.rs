#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;

#[test]
fn test_undo_redo_add_node() {
    let mut tool = GraphEditorTool::default();
    assert!(!tool.can_undo() && !tool.can_redo());

    tool.push_undo_snapshot();
    let pos = Pos2::new(10.0, 20.0);
    let node_idx = tool.graph.add_node(pos);
    tool.node_indices.insert(tool.next_node_id, node_idx);
    tool.node_positions.insert(tool.next_node_id, pos);
    tool.next_node_id += 1;

    assert_eq!(tool.node_positions.len(), 1);
    assert!(tool.can_undo() && !tool.can_redo());

    tool.undo();
    assert_eq!(tool.node_positions.len(), 0);
    assert!(!tool.can_undo() && tool.can_redo());

    tool.redo();
    assert_eq!(tool.node_positions.len(), 1);
    assert!(tool.can_undo() && !tool.can_redo());
}

#[test]
fn test_undo_redo_add_edge() {
    let mut tool = GraphEditorTool::default();
    tool.node_positions.insert(0, Pos2::new(0.0, 0.0));
    let idx0 = tool.graph.add_node(Pos2::new(0.0, 0.0));
    tool.node_indices.insert(0, idx0);

    tool.node_positions.insert(1, Pos2::new(50.0, 0.0));
    let idx1 = tool.graph.add_node(Pos2::new(50.0, 0.0));
    tool.node_indices.insert(1, idx1);
    tool.next_node_id = 2;

    tool.push_undo_snapshot();
    tool.edges.push((0, 1, 1.0));
    tool.rebuild_graph();
    assert_eq!(tool.edges.len(), 1);

    tool.undo();
    assert_eq!(tool.edges.len(), 0);
    assert_eq!(tool.node_positions.len(), 2);

    tool.redo();
    assert_eq!(tool.edges.len(), 1);
}

#[test]
fn test_clear_empty_graph_preserves_stacks() {
    let mut tool = GraphEditorTool::default();
    assert!(!tool.can_undo() && !tool.can_redo());
    if !tool.node_positions.is_empty() || !tool.edges.is_empty() {
        tool.push_undo_snapshot();
    }
    assert!(!tool.can_undo() && !tool.can_redo());
}

#[test]
fn test_new_edit_clears_redo_stack() {
    let mut tool = GraphEditorTool::default();
    tool.push_undo_snapshot();
    tool.node_positions.insert(0, Pos2::new(0.0, 0.0));
    tool.next_node_id = 1;

    tool.push_undo_snapshot();
    tool.node_positions.insert(1, Pos2::new(10.0, 10.0));
    tool.next_node_id = 2;

    tool.undo();
    assert!(tool.can_redo());

    tool.push_undo_snapshot();
    tool.node_positions.insert(2, Pos2::new(20.0, 20.0));
    tool.next_node_id = 3;

    assert!(!tool.can_redo());
}

#[test]
fn test_bounded_undo_stack() {
    let mut tool = GraphEditorTool::default();
    for i in 0..60 {
        tool.push_undo_snapshot();
        tool.node_positions.insert(i, Pos2::new(i as f32, i as f32));
    }
    assert_eq!(tool.undo_stack.len(), MAX_SNAPSHOTS);
}

#[test]
fn test_drag_node_saves_one_snapshot() {
    let mut tool = GraphEditorTool::default();
    let initial_pos = Pos2::new(10.0, 10.0);
    tool.node_positions.insert(0, initial_pos);

    tool.push_undo_snapshot();
    if let Some(pos) = tool.node_positions.get_mut(&0) { *pos += egui::vec2(5.0, 5.0); }
    if let Some(pos) = tool.node_positions.get_mut(&0) { *pos += egui::vec2(5.0, 5.0); }

    assert_eq!(tool.undo_stack.len(), 1);
    assert_eq!(tool.node_positions.get(&0), Some(&Pos2::new(20.0, 20.0)));

    tool.undo();
    assert_eq!(tool.node_positions.get(&0), Some(&initial_pos));
}
