#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AffineLatency, BraessCongestionConfig, GraphEditorTool};
use eframe::egui::Pos2;
use serde::ser::{Serialize, SerializeMap, Serializer};
use std::collections::BTreeMap;

/// Undo/redo snapshot of node positions, edges, and affine congestion state.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphSnapshot {
    /// Node coordinates keyed by editor node ID.
    pub node_positions: BTreeMap<usize, Pos2>,
    /// Edge tuples `(u, v, weight)`.
    pub edges: Vec<(usize, usize, f64)>,
    /// Currently selected node ID.
    pub selected_node: Option<usize>,
    /// Next monotonic node ID counter.
    pub next_node_id: usize,
    /// Per-edge affine latency functions.
    pub edge_latencies: Vec<AffineLatency>,
    /// Congestion routing settings.
    pub congestion: BraessCongestionConfig,
}

/// Maximum undo history entries retained per `GraphEditorTool`.
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) edge_latencies: Option<&'a [AffineLatency]>,
    #[serde(skip_serializing_if = "BraessCongestionConfig::is_default")]
    pub(super) congestion: &'a BraessCongestionConfig,
}

/// Owned serializable snapshot of `GraphEditorTool` state for presets and URLs.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SerializedGraphState {
    /// Node coordinates keyed by ID.
    pub node_positions: BTreeMap<usize, (f32, f32)>,
    /// Edge tuples `(u, v, weight)`.
    pub edges: Vec<(usize, usize, f64)>,
    /// Optional selected node ID.
    pub selected_node: Option<usize>,
    /// Next monotonic node ID counter.
    pub next_node_id: usize,
    /// Optional per-edge affine congestion latency functions $l_e(f_e) = a_e f_e + b_e$.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edge_latencies: Vec<AffineLatency>,
    /// Optional Braess congestion routing settings.
    #[serde(default, skip_serializing_if = "BraessCongestionConfig::is_default")]
    pub congestion: BraessCongestionConfig,
}

impl SerializedGraphState {
    /// Validates finiteness, node index existence, and affine latency bounds.
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
        if !self.edge_latencies.is_empty() && self.edge_latencies.len() != self.edges.len() {
            return Err("edge_latencies length must match edges".to_string());
        }
        if !self
            .edge_latencies
            .iter()
            .all(|l| l.a.is_finite() && l.b.is_finite())
            || !self.congestion.total_demand.is_finite()
        {
            return Err("Congestion parameters must be finite".to_string());
        }
        Ok(())
    }
}

impl GraphEditorTool {
    pub(crate) fn compute_state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut h);
        self.node_positions.len().hash(&mut h);
        self.edges.len().hash(&mut h);
        self.selected_node.hash(&mut h);
        self.next_node_id.hash(&mut h);
        self.braess.config.enabled.hash(&mut h);
        self.braess.config.shortcut_enabled.hash(&mut h);
        self.braess.config.source_node.hash(&mut h);
        self.braess.config.sink_node.hash(&mut h);
        self.braess.config.total_demand.to_bits().hash(&mut h);
        self.braess.config.prediction.hash(&mut h);
        h.finish().max(1)
    }

    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    pub(crate) fn serialize_borrowed_state(&self) -> Option<String> {
        let include_lats = self.braess.has_custom_state(&self.edges)
            && self.braess.edge_latencies.len() == self.edges.len();
        serde_json::to_string(&SerializedGraphStateRef {
            node_positions: BorrowedNodePositions(&self.node_positions),
            edges: &self.edges,
            selected_node: self.selected_node,
            next_node_id: self.next_node_id,
            edge_latencies: include_lats.then_some(self.braess.edge_latencies.as_slice()),
            congestion: &self.braess.config,
        })
        .ok()
    }

    pub(crate) fn check_finite_graph(&self) -> bool {
        self.node_positions
            .values()
            .all(|p| p.x.is_finite() && p.y.is_finite())
            && self.edges.iter().all(|&(_, _, w)| w.is_finite())
            && self.braess.edge_latencies.iter().all(|l| l.is_valid())
            && self.braess.config.total_demand.is_finite()
    }

    pub(crate) fn matches_serialized_state(&self, loaded: &SerializedGraphState) -> bool {
        let lats_match = if loaded.edge_latencies.is_empty() {
            !self.braess.has_custom_state(&self.edges)
        } else {
            self.braess.edge_latencies == loaded.edge_latencies
        };
        lats_match
            && self.braess.config == loaded.congestion
            && self.next_node_id == loaded.next_node_id
            && self.selected_node == loaded.selected_node
            && self.edges == loaded.edges
            && self.node_positions.len() == loaded.node_positions.len()
            && self.node_positions.iter().all(|(&id, &pos)| {
                loaded
                    .node_positions
                    .get(&id)
                    .is_some_and(|&(x, y)| pos.x == x && pos.y == y)
            })
    }

    pub(crate) fn load_serialized_state(&mut self, state: &str) -> Result<(), String> {
        let loaded: SerializedGraphState =
            serde_json::from_str(state).map_err(|e| format!("Invalid graph state: {e}"))?;
        loaded.validate()?;
        if self.matches_serialized_state(&loaded) {
            return Ok(());
        }
        self.node_positions = loaded
            .node_positions
            .into_iter()
            .map(|(id, (x, y))| (id, Pos2::new(x, y)))
            .collect();
        self.edges = loaded.edges;
        self.selected_node = loaded.selected_node;
        self.next_node_id = loaded.next_node_id;
        self.braess.edge_latencies = loaded
            .edge_latencies
            .into_iter()
            .map(|l| AffineLatency::clamped(l.a, l.b))
            .collect();
        self.braess.sync_with_edges(&self.edges);
        self.braess.config = loaded.congestion;
        self.braess.config.total_demand = self.braess.config.total_demand.clamp(0.1, 10.0);
        self.dragged_node = None;
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.bump_revision();
        self.rebuild_graph();
        if self.braess.config.enabled {
            self.recompute_braess();
        }
        Ok(())
    }

    fn build_current_snapshot(&self) -> GraphSnapshot {
        GraphSnapshot {
            node_positions: self.node_positions.clone(),
            edges: self.edges.clone(),
            selected_node: self.selected_node,
            next_node_id: self.next_node_id,
            edge_latencies: self.braess.edge_latencies.clone(),
            congestion: self.braess.config.clone(),
        }
    }

    /// Saves the current graph and congestion state onto the bounded undo stack.
    pub fn push_undo_snapshot(&mut self) {
        self.bump_revision();
        let snapshot = self.build_current_snapshot();
        self.undo_stack.push(snapshot);
        if self.undo_stack.len() > MAX_SNAPSHOTS {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    /// Restores the most recent undo snapshot.
    pub fn undo(&mut self) {
        if let Some(previous_state) = self.undo_stack.pop() {
            let current_state = self.build_current_snapshot();
            self.redo_stack.push(current_state);
            self.restore_snapshot(previous_state);
        }
    }

    /// Reapplies the most recently undone snapshot.
    pub fn redo(&mut self) {
        if let Some(next_state) = self.redo_stack.pop() {
            let current_state = self.build_current_snapshot();
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
        self.braess.edge_latencies = snapshot.edge_latencies;
        self.braess.sync_with_edges(&self.edges);
        self.braess.config = snapshot.congestion;
        self.rebuild_graph();
        if self.braess.config.enabled {
            self.recompute_braess();
        }
    }
}

pub(super) const STARTER_PRESET_JSON: &str = r#"{
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
        tags: &["vertices", "edges", "adjacency", "planar", "network"],
        build: || Box::new(GraphEditorTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for GraphEditorTool {
    fn theory_description(&self) -> String {
        format!(
            "Combinatorial graph structure $G = (V, E)$ with $|V| = {}$ nodes and $|E| = {}$ edges under force-directed spring-electrical embedding and Wardrop congestion equilibrium $l_e(f_e) = a_e f_e + b_e$.",
            self.node_positions.len(),
            self.edges.len()
        )
    }
    fn phonetic_description(&self) -> String {
        "Combinatorial graph theory and Wardrop network congestion equilibrium".into()
    }
    fn theory_citation(&self) -> String {
        "Graph Drawing by Force-Directed Placement (Fruchterman & Reingold, 1991); Über ein Paradoxon aus der Verkehrsplanung (Braess, 1968)".into()
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
