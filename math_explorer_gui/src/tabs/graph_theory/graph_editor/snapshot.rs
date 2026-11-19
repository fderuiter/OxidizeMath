#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GraphEditorTool;
use eframe::egui::Pos2;
use serde::ser::{Serialize, SerializeMap, Serializer};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct GraphSnapshot {
    pub node_positions: BTreeMap<usize, Pos2>,
    pub edges: Vec<(usize, usize, f64)>,
    pub selected_node: Option<usize>,
    pub next_node_id: usize,
}

pub const MAX_SNAPSHOTS: usize = 50;

pub(super) struct BorrowedNodePositions<'a>(pub(super) &'a BTreeMap<usize, Pos2>);

impl Serialize for BorrowedNodePositions<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (&id, pos) in self.0 {
            map.serialize_entry(&id, &(pos.x, pos.y))?;
        }
        map.end()
    }
}

#[derive(serde::Serialize)]
pub(super) struct SerializedGraphStateRef<'a> {
    pub(super) node_positions: BorrowedNodePositions<'a>,
    pub(super) edges: &'a [(usize, usize, f64)],
    pub(super) selected_node: Option<usize>,
    pub(super) next_node_id: usize,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SerializedGraphState {
    pub node_positions: std::collections::BTreeMap<usize, (f32, f32)>,
    pub edges: Vec<(usize, usize, f64)>,
    pub selected_node: Option<usize>,
    pub next_node_id: usize,
}

impl SerializedGraphState {
    pub fn validate(&self) -> Result<(), String> {
        if !self
            .node_positions
            .values()
            .all(|(x, y)| x.is_finite() && y.is_finite())
        {
            return Err("Node coordinates must be finite".to_string());
        }
        if !self.edges.iter().all(|&(u, v, w)| {
            self.node_positions.contains_key(&u)
                && self.node_positions.contains_key(&v)
                && w.is_finite()
        }) {
            return Err("Edges must reference existing nodes with finite weights".to_string());
        }
        if let Some(sel) = self.selected_node {
            if !self.node_positions.contains_key(&sel) {
                return Err(format!("selected_node {sel} does not exist"));
            }
        }
        Ok(())
    }
}

impl GraphEditorTool {
    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    pub(crate) fn serialize_borrowed_state(&self) -> Option<String> {
        serde_json::to_string(&SerializedGraphStateRef {
            node_positions: BorrowedNodePositions(&self.node_positions),
            edges: &self.edges,
            selected_node: self.selected_node,
            next_node_id: self.next_node_id,
        })
        .ok()
    }

    pub(crate) fn check_finite_graph(&self) -> bool {
        self.node_positions
            .values()
            .all(|p| p.x.is_finite() && p.y.is_finite())
            && self.edges.iter().all(|&(_, _, w)| w.is_finite())
    }

    pub fn push_undo_snapshot(&mut self) {
        self.bump_revision();
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
        self.bump_revision();
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
    "title": "Interactive Graph Layout",
    "domain": "graph_theory",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "node_positions": {
      "0": [300.0, 220.0],
      "1": [200.0, 140.0],
      "2": [400.0, 140.0],
      "3": [400.0, 300.0],
      "4": [200.0, 300.0]
    },
    "edges": [
      [0, 1, 1.0],
      [0, 2, 1.0],
      [0, 3, 1.0],
      [0, 4, 1.0],
      [1, 2, 1.0],
      [2, 3, 1.0],
      [3, 4, 1.0],
      [4, 1, 1.0]
    ],
    "selected_node": 0,
    "next_node_id": 5
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "GraphEditorTool",
        display_name: "Graph Editor",
        domain: "graph_theory",
        tags: &[],
        build: || Box::new(GraphEditorTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for GraphEditorTool {
    fn theory_description(&self) -> String {
        format!(
            "Combinatorial graph structure $G = (V, E)$ with $|V| = {}$ nodes and $|E| = {}$ edges under force-directed spring-electrical embedding $F = F_{{\\text{{attr}}}} + F_{{\\text{{rep}}}}.$",
            self.node_positions.len(),
            self.edges.len()
        )
    }
    fn phonetic_description(&self) -> String {
        "Combinatorial graph theory and force-directed network embedding".into()
    }
    fn theory_citation(&self) -> String {
        "Graph Drawing by Force-Directed Placement (Fruchterman & Reingold, 1991)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "graph_editor_starter",
        title: "Interactive Graph Layout",
        description: "Interactive graph editor with circular, grid, and Fruchterman-Reingold force-directed layouts.",
        category: "Graph Theory",
        icon: "🕸️",
        domain: "graph_theory",
        tool_name: Some("GraphEditorTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
