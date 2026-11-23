//! Shared custom topology buffer, compact URL state, and cross-tool handoff across `GraphTheoryTab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::graph_editor::{AffineLatency, SerializedGraphState};
use eframe::egui::{self, Pos2};
use math_explorer::pure_math::graph_theory::{
    graph::Graph,
    interchange::{
        InterchangeGraph, InterchangeNode, MAX_INTERCHANGE_EDGES, MAX_INTERCHANGE_VERTICES,
        parse_dot, parse_edge_csv, parse_graphml,
    },
    layout::{circular_layout, grid_layout},
};
use std::collections::BTreeMap;

fn is_false(flag: &bool) -> bool {
    !*flag
}

/// Destination tool within `GraphTheoryTab` for one-click topology handoff.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TopologyHandoffTarget {
    /// Interactive `GraphEditorTool` (with Wardrop/Braess lab).
    #[default]
    GraphEditor,
    /// Stepped `AlgorithmVisualizerTool` (Dijkstra, BFS, DFS, Max-Flow, MST).
    AlgorithmVisualizer,
    /// Structural `NetworkMetricsTool` (centrality, degeneracy, vertex cover, treewidth).
    NetworkMetrics,
}

impl TopologyHandoffTarget {
    /// Canonical `InteractiveTool::name()` registered in `SimulationFramework`.
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::GraphEditor => "Graph Editor",
            Self::AlgorithmVisualizer => "Algorithm Visualizer",
            Self::NetworkMetrics => "Network Metrics",
        }
    }
}

/// Compact serializable custom topology embedded in URL/preset states within `URL_JSON_BOUNDS`.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactTopologyState {
    /// Finite 2D vertex coordinates keyed by vertex ID.
    pub nodes: BTreeMap<usize, (f32, f32)>,
    /// Weighted edge tuples `(source, target, weight)`.
    pub edges: Vec<(usize, usize, f64)>,
    /// Whether edges are directed (`true`) or undirected (`false`).
    #[serde(default, skip_serializing_if = "is_false")]
    pub directed: bool,
}

impl CompactTopologyState {
    /// Validates vertex/edge count caps, endpoint existence, and coordinate/weight finiteness.
    pub fn validate(&self) -> Result<(), String> {
        if self.nodes.is_empty() {
            return Err("Custom topology must contain at least 1 vertex".to_string());
        }
        if self.nodes.len() > MAX_INTERCHANGE_VERTICES {
            return Err(format!(
                "Vertex count {} exceeds maximum {}",
                self.nodes.len(),
                MAX_INTERCHANGE_VERTICES
            ));
        }
        if self.edges.len() > MAX_INTERCHANGE_EDGES {
            return Err(format!(
                "Edge count {} exceeds maximum {}",
                self.edges.len(),
                MAX_INTERCHANGE_EDGES
            ));
        }
        if !self
            .nodes
            .values()
            .all(|(x, y)| x.is_finite() && y.is_finite())
        {
            return Err("Topology coordinates must be finite".to_string());
        }
        if !self.edges.iter().all(|&(u, v, w)| {
            self.nodes.contains_key(&u) && self.nodes.contains_key(&v) && w.is_finite()
        }) {
            return Err("Edges must reference existing vertices with finite weights".to_string());
        }
        Ok(())
    }

    /// Quantizes coordinates to 0.1 UI points to minimize URL query payload bytes.
    #[must_use]
    pub fn quantized(&self) -> Self {
        let nodes = self
            .nodes
            .iter()
            .map(|(&id, &(x, y))| {
                let qx = (x * 10.0).round() / 10.0;
                let qy = (y * 10.0).round() / 10.0;
                (id, (qx, qy))
            })
            .collect();
        let edges = self
            .edges
            .iter()
            .map(|&(u, v, w)| (u, v, (w * 1e4).round() / 1e4))
            .collect();
        Self {
            nodes,
            edges,
            directed: self.directed,
        }
    }
}

/// Shared custom graph topology exchanged across `GraphEditorTool`, `AlgorithmVisualizerTool`, and `NetworkMetricsTool`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SharedGraphTopology {
    /// Vertex positions keyed by vertex ID.
    pub nodes: BTreeMap<usize, (f32, f32)>,
    /// Weighted edges `(u, v, weight)`.
    pub edges: Vec<(usize, usize, f64)>,
    /// Whether edges are directed.
    pub directed: bool,
    /// Optional source terminal `s`.
    pub source_node: Option<usize>,
    /// Optional sink terminal `t`.
    pub sink_node: Option<usize>,
    /// Optional affine congestion functions preserved when round-tripping with `GraphEditorTool`.
    pub edge_latencies: Vec<AffineLatency>,
    /// Monotonic revision counter bumped on each cross-tool publish.
    pub revision: u64,
}

impl SharedGraphTopology {
    /// Returns true when no vertices exist in the shared buffer.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Converts this topology into a validated [`InterchangeGraph`] for `.dot`, `.graphml`, or `.csv` export.
    pub fn to_interchange_graph(&self) -> Result<InterchangeGraph, String> {
        let mut g = InterchangeGraph::new(self.directed);
        for (&id, &(x, y)) in &self.nodes {
            g.insert_node(InterchangeNode {
                id,
                pos: Some((f64::from(x), f64::from(y))),
                label: Some(id.to_string()),
            })
            .map_err(|e| e.to_string())?;
        }
        for &(u, v, w) in &self.edges {
            g.push_edge(u, v, w).map_err(|e| e.to_string())?;
        }
        g.validate().map_err(|e| e.to_string())?;
        Ok(g)
    }

    /// Builds a [`SharedGraphTopology`] from a parsed [`InterchangeGraph`], computing layout positions if absent.
    pub fn from_interchange_graph(interchange: &InterchangeGraph) -> Result<Self, String> {
        interchange.validate().map_err(|e| e.to_string())?;
        if interchange.nodes.is_empty() {
            return Err("Imported topology contains 0 vertices".to_string());
        }
        let mut nodes = BTreeMap::new();
        let all_have_pos = interchange.nodes.values().all(|n| n.pos.is_some());
        if all_have_pos {
            for (&id, n) in &interchange.nodes {
                if let Some((x, y)) = n.pos {
                    nodes.insert(id, (x as f32, y as f32));
                }
            }
        } else {
            nodes = synthesize_missing_positions(interchange);
        }
        let edges: Vec<(usize, usize, f64)> = interchange
            .edges
            .iter()
            .map(|e| (e.source, e.target, e.weight))
            .collect();
        let source_node = nodes.keys().next().copied();
        let sink_node = nodes
            .keys()
            .next_back()
            .copied()
            .filter(|&t| Some(t) != source_node);
        Ok(Self {
            nodes,
            edges,
            directed: interchange.directed,
            source_node,
            sink_node,
            edge_latencies: Vec::new(),
            revision: 1,
        })
    }

    /// Converts this topology into a [`CompactTopologyState`] for URL/preset persistence.
    #[must_use]
    pub fn to_compact_state(&self) -> CompactTopologyState {
        CompactTopologyState {
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            directed: self.directed,
        }
        .quantized()
    }

    /// Builds a [`SharedGraphTopology`] from a validated [`CompactTopologyState`].
    pub fn from_compact_state(compact: &CompactTopologyState) -> Result<Self, String> {
        compact.validate()?;
        let source_node = compact.nodes.keys().next().copied();
        let sink_node = compact
            .nodes
            .keys()
            .next_back()
            .copied()
            .filter(|&t| Some(t) != source_node);
        Ok(Self {
            nodes: compact.nodes.clone(),
            edges: compact.edges.clone(),
            directed: compact.directed,
            source_node,
            sink_node,
            edge_latencies: Vec::new(),
            revision: 1,
        })
    }
}

fn synthesize_missing_positions(interchange: &InterchangeGraph) -> BTreeMap<usize, (f32, f32)> {
    let mut temp_graph: Graph<(), ()> = Graph::new();
    let mut idx_map = BTreeMap::new();
    for &id in interchange.nodes.keys() {
        idx_map.insert(id, temp_graph.add_node(()));
    }
    for e in &interchange.edges {
        if let (Some(&u), Some(&v)) = (idx_map.get(&e.source), idx_map.get(&e.target)) {
            temp_graph.add_edge(u, v, ());
        }
    }
    let bounds = (100.0, 100.0, 560.0, 440.0);
    let generated = if interchange.nodes.len() <= 12 {
        circular_layout(&temp_graph, bounds)
    } else {
        grid_layout(&temp_graph, bounds)
    };
    let mut out = BTreeMap::new();
    for (&id, node) in &interchange.nodes {
        if let Some((x, y)) = node.pos {
            out.insert(id, (x as f32, y as f32));
        } else if let Some(&n_idx) = idx_map.get(&id) {
            let pos = generated.get(&n_idx).copied().unwrap_or((300.0, 260.0));
            out.insert(id, pos);
        }
    }
    out
}

/// Parses `.dot`, `.graphml`, `.csv`, or `.json` file contents into a bounded [`SharedGraphTopology`].
pub fn parse_topology_document(
    filename_hint: &str,
    content: &str,
) -> Result<SharedGraphTopology, String> {
    let ext = std::path::Path::new(filename_hint)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Err("Imported file is empty".to_string());
    }
    match ext.as_str() {
        "dot" | "gv" => SharedGraphTopology::from_interchange_graph(
            &parse_dot(trimmed).map_err(|e| e.to_string())?,
        ),
        "graphml" | "xml" => SharedGraphTopology::from_interchange_graph(
            &parse_graphml(trimmed).map_err(|e| e.to_string())?,
        ),
        "csv" => SharedGraphTopology::from_interchange_graph(
            &parse_edge_csv(trimmed).map_err(|e| e.to_string())?,
        ),
        "json" => parse_json_topology(trimmed),
        _ => parse_autodetected_topology(trimmed),
    }
}

fn parse_autodetected_topology(trimmed: &str) -> Result<SharedGraphTopology, String> {
    if trimmed.starts_with('{') {
        parse_json_topology(trimmed)
    } else if trimmed.starts_with('<') {
        SharedGraphTopology::from_interchange_graph(
            &parse_graphml(trimmed).map_err(|e| e.to_string())?,
        )
    } else if trimmed.contains('{')
        && (trimmed.contains("->")
            || trimmed.contains("--")
            || trimmed.starts_with("graph")
            || trimmed.starts_with("digraph")
            || trimmed.starts_with("strict"))
    {
        SharedGraphTopology::from_interchange_graph(&parse_dot(trimmed).map_err(|e| e.to_string())?)
    } else {
        SharedGraphTopology::from_interchange_graph(
            &parse_edge_csv(trimmed).map_err(|e| e.to_string())?,
        )
    }
}

fn parse_json_topology(trimmed: &str) -> Result<SharedGraphTopology, String> {
    crate::framework::ToolStateBounds::DEFAULT.validate_json(trimmed)?;
    if let Ok(compact) = serde_json::from_str::<CompactTopologyState>(trimmed) {
        return SharedGraphTopology::from_compact_state(&compact);
    }
    if let Ok(editor_state) = serde_json::from_str::<SerializedGraphState>(trimmed) {
        return topology_from_editor_state(editor_state);
    }
    let val: serde_json::Value =
        serde_json::from_str(trimmed).map_err(|e| format!("Invalid JSON topology: {e}"))?;
    if let Some(cfg) = val.get("config") {
        let cfg_str = cfg.to_string();
        if let Ok(editor_state) = serde_json::from_str::<SerializedGraphState>(&cfg_str) {
            return topology_from_editor_state(editor_state);
        }
        if let Ok(compact) = serde_json::from_str::<CompactTopologyState>(&cfg_str) {
            return SharedGraphTopology::from_compact_state(&compact);
        }
    }
    Err("Unrecognized JSON graph topology schema".to_string())
}

fn topology_from_editor_state(state: SerializedGraphState) -> Result<SharedGraphTopology, String> {
    state.validate()?;
    if state.node_positions.is_empty() {
        return Err("Imported JSON graph contains 0 vertices".to_string());
    }
    if state.node_positions.len() > MAX_INTERCHANGE_VERTICES {
        return Err(format!(
            "Vertex count {} exceeds maximum {}",
            state.node_positions.len(),
            MAX_INTERCHANGE_VERTICES
        ));
    }
    if state.edges.len() > MAX_INTERCHANGE_EDGES {
        return Err(format!(
            "Edge count {} exceeds maximum {}",
            state.edges.len(),
            MAX_INTERCHANGE_EDGES
        ));
    }
    let directed = state.congestion.enabled;
    let source_node = state
        .node_positions
        .contains_key(&state.congestion.source_node)
        .then_some(state.congestion.source_node)
        .or_else(|| state.node_positions.keys().next().copied());
    let sink_node = state
        .node_positions
        .contains_key(&state.congestion.sink_node)
        .then_some(state.congestion.sink_node)
        .filter(|&t| Some(t) != source_node);
    Ok(SharedGraphTopology {
        nodes: state.node_positions,
        edges: state.edges,
        directed,
        source_node,
        sink_node,
        edge_latencies: state.edge_latencies,
        revision: 1,
    })
}

/// Converts a `BTreeMap<usize, Pos2>` into rounded `(f32, f32)` coordinates.
#[must_use]
pub fn btree_pos2_to_tuples(map: &BTreeMap<usize, Pos2>) -> BTreeMap<usize, (f32, f32)> {
    map.iter().map(|(&id, pos)| (id, (pos.x, pos.y))).collect()
}

#[derive(Clone, Default)]
pub(crate) struct PendingTopologyHandoff {
    pub(crate) target: TopologyHandoffTarget,
    pub(crate) topology: SharedGraphTopology,
}

fn handoff_key() -> egui::Id {
    egui::Id::new("graph_theory_pending_topology_handoff")
}

fn shared_buffer_key() -> egui::Id {
    egui::Id::new("graph_theory_shared_topology_buffer")
}

/// Publishes a cross-tool topology handoff request in `ctx` temporary data.
pub fn queue_topology_handoff(
    ctx: &egui::Context,
    target: TopologyHandoffTarget,
    mut topology: SharedGraphTopology,
) {
    let prev_rev = read_shared_topology_buffer(ctx).map_or(0, |t| t.revision);
    topology.revision = prev_rev.wrapping_add(1).max(1);
    ctx.data_mut(|d| {
        d.insert_temp(shared_buffer_key(), topology.clone());
        d.insert_temp(handoff_key(), PendingTopologyHandoff { target, topology });
    });
    ctx.request_repaint();
}

/// Reads the latest shared topology published in `ctx`, if non-empty.
#[must_use]
pub fn read_shared_topology_buffer(ctx: &egui::Context) -> Option<SharedGraphTopology> {
    ctx.data(|d| d.get_temp::<SharedGraphTopology>(shared_buffer_key()))
        .filter(|t| !t.is_empty())
}

/// Synchronizes `tab_buffer` into `ctx` before rendering `GraphTheoryTab`.
pub(crate) fn sync_shared_buffer_to_ctx(ctx: &egui::Context, tab_buffer: &SharedGraphTopology) {
    if tab_buffer.is_empty() {
        return;
    }
    let needs_write = ctx.data(|d| {
        d.get_temp::<SharedGraphTopology>(shared_buffer_key())
            .is_none_or(|cur| cur.revision < tab_buffer.revision)
    });
    if needs_write {
        ctx.data_mut(|d| d.insert_temp(shared_buffer_key(), tab_buffer.clone()));
    }
}

/// Consumes any queued [`PendingTopologyHandoff`] for `GraphTheoryTab` to switch tools immediately.
pub(crate) fn take_any_pending_handoff(ctx: &egui::Context) -> Option<PendingTopologyHandoff> {
    ctx.data_mut(|d| d.remove_temp::<PendingTopologyHandoff>(handoff_key()))
}

/// Consumes a queued [`PendingTopologyHandoff`] only if it targets `expected` (for split-pane mode).
pub(crate) fn take_pending_handoff_for(
    ctx: &egui::Context,
    expected: TopologyHandoffTarget,
) -> Option<SharedGraphTopology> {
    ctx.data_mut(|d| {
        let cur = d.get_temp::<PendingTopologyHandoff>(handoff_key())?;
        if cur.target == expected {
            d.remove_temp::<PendingTopologyHandoff>(handoff_key())
                .map(|p| p.topology)
        } else {
            None
        }
    })
}

// [cite:graph_parameters_rust]
