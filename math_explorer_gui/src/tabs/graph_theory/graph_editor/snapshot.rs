#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GraphEditorTool;
use eframe::egui::Pos2;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub struct GraphSnapshot {
    pub node_positions: HashMap<usize, Pos2>,
    pub edges: Vec<(usize, usize, f64)>,
    pub selected_node: Option<usize>,
    pub next_node_id: usize,
}

pub const MAX_SNAPSHOTS: usize = 50;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SerializedGraphState {
    pub node_positions: HashMap<usize, (f32, f32)>,
    pub edges: Vec<(usize, usize, f64)>,
    pub selected_node: Option<usize>,
    pub next_node_id: usize,
}

impl GraphEditorTool {
    pub fn push_undo_snapshot(&mut self) {
        let snapshot = GraphSnapshot {
            node_positions: self.node_positions.clone(),
            edges: self.edges.clone(),
            selected_node: self.selected_node,
            next_node_id: self.next_node_id,
        };
        self.undo_stack.push(snapshot);
        if self.undo_stack.len() > MAX_SNAPSHOTS {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    pub fn undo(&mut self) {
        if let Some(previous_state) = self.undo_stack.pop() {
            let current_state = GraphSnapshot {
                node_positions: self.node_positions.clone(),
                edges: self.edges.clone(),
                selected_node: self.selected_node,
                next_node_id: self.next_node_id,
            };
            self.redo_stack.push(current_state);
            self.restore_snapshot(previous_state);
        }
    }

    pub fn redo(&mut self) {
        if let Some(next_state) = self.redo_stack.pop() {
            let current_state = GraphSnapshot {
                node_positions: self.node_positions.clone(),
                edges: self.edges.clone(),
                selected_node: self.selected_node,
                next_node_id: self.next_node_id,
            };
            self.undo_stack.push(current_state);
            self.restore_snapshot(next_state);
        }
    }

    pub(crate) fn restore_snapshot(&mut self, snapshot: GraphSnapshot) {
        self.node_positions = snapshot.node_positions;
        self.edges = snapshot.edges;
        self.selected_node = snapshot
            .selected_node
            .filter(|id| self.node_positions.contains_key(id));
        self.next_node_id = snapshot.next_node_id;
        self.rebuild_graph();
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Force-Directed Graph",
    "domain": "graph_theory",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "damping": 0.85,
    "repulsion": 50.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "graph_editor_starter",
        title: "Force-Directed Graph",
        description: "Spring-electrical equilibrium layout of nodes and edges with interactive adjacency manipulation.",
        category: "Transformers",
        icon: "🕸️",
        domain: "graph_theory",
        tool_name: Some("GraphEditorTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
