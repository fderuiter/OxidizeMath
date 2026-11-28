//! Data models (`TopologySource`, `SharedGraphTopology`, `SharedPointCloud3D`) for [`super::SharedSimulationBus`].

use serde::{Deserialize, Serialize};

/// Maximum node count published in [`SharedGraphTopology`].
#[cfg(any(
    feature = "pure_math",
    feature = "epidemiology",
    all(test, any(feature = "biology", feature = "physics"))
))]
pub const MAX_SHARED_GRAPH_NODES: usize = 256;

/// Maximum undirected/directed sculpted edge count published in [`SharedGraphTopology`].
#[cfg(any(
    feature = "pure_math",
    all(test, any(feature = "biology", feature = "physics"))
))]
pub const MAX_SHARED_GRAPH_EDGES: usize = 2_048;

/// Maximum neuron count ingested by `NeuralNetworkVizTool` from [`SharedGraphTopology`].
#[cfg(feature = "biology")]
pub const MAX_NEURAL_SHARED_NODES: usize = 32;

/// Maximum 3D chaotic landmark count published in [`SharedPointCloud3D`].
#[cfg(any(feature = "physics", all(test, feature = "pure_math")))]
pub const MAX_SHARED_LANDMARKS: usize = 128;

/// Per-tool topology ingestion mode persisted as a compact tag in tool JSON and URL state.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TopologySource {
    /// Subscribes to [`super::SharedSimulationBus`] revisions whenever workspace `Link Topology` is enabled.
    #[default]
    SharedBus,
    /// Opts this pane out of workspace topology sync and preserves its local preset topology.
    LocalPreset,
}

impl TopologySource {
    /// Predicate used by `#[serde(skip_serializing_if = "TopologySource::is_default")]`.
    #[must_use]
    pub fn is_default(&self) -> bool {
        matches!(self, Self::SharedBus)
    }

    /// Human-readable UI label for the topology source mode.
    #[cfg(any(
        feature = "applied",
        feature = "biology",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::SharedBus => "Shared Bus",
            Self::LocalPreset => "Local Preset",
        }
    }
}

/// Immutable copy-on-write Compressed Sparse Row (CSR) graph snapshot shared across panes.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedGraphTopology {
    /// Monotonically increasing bus revision assigned when published.
    pub revision: u64,
    /// Whether the authored graph had directed edges enabled.
    pub directed: bool,
    /// Sorted editor node IDs mapped to contiguous `0..n` CSR indices.
    pub node_ids: Vec<usize>,
    /// Raw 2D canvas positions `[x, y]` for each node in `0..n`.
    pub positions: Vec<[f32; 2]>,
    /// Viewport-normalized `[0.0, 1.0]^2` node coordinates for cross-tool 2D layout mapping.
    pub normalized_positions: Vec<[f32; 2]>,
    /// Authored CSR row offsets (`len == n + 1`; directed when `directed`, symmetric otherwise).
    pub row_ptr: Vec<usize>,
    /// Authored CSR column indices.
    pub col_idx: Vec<usize>,
    /// Authored CSR edge weights.
    pub weights: Vec<f64>,
    /// Symmetrized undirected CSR row offsets (`len == n + 1`).
    pub sym_row_ptr: Vec<usize>,
    /// Symmetrized undirected CSR column indices.
    pub sym_col_idx: Vec<usize>,
    /// Symmetrized undirected CSR edge weights.
    pub sym_weights: Vec<f64>,
    /// Canonical undirected edges `(u, v, weight)` with `u < v`.
    pub undirected_edges: Vec<(usize, usize, f64)>,
    /// Mean symmetric degree across active (`deg > 0`) nodes (`0.0` if edgeless).
    pub mean_active_degree: f64,
    /// Unclamped source node count before applying `MAX_SHARED_GRAPH_NODES`.
    pub original_node_count: usize,
    /// Unclamped source edge count before applying `MAX_SHARED_GRAPH_EDGES`.
    pub original_edge_count: usize,
    /// True if `original_node_count > MAX_SHARED_GRAPH_NODES`.
    pub clamped_nodes: bool,
    /// True if `original_edge_count > MAX_SHARED_GRAPH_EDGES`.
    pub clamped_edges: bool,
}

#[cfg(any(
    feature = "pure_math",
    all(test, any(feature = "biology", feature = "physics"))
))]
fn normalize_positions(positions: &[[f32; 2]]) -> Vec<[f32; 2]> {
    if positions.is_empty() {
        return Vec::new();
    }
    if positions.len() == 1 {
        return vec![[0.5, 0.5]];
    }
    let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
    for &[x, y] in positions {
        if x.is_finite() && y.is_finite() {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }
    if !min_x.is_finite() || !min_y.is_finite() {
        return vec![[0.5, 0.5]; positions.len()];
    }
    let (span_x, span_y) = (max_x - min_x, max_y - min_y);
    positions
        .iter()
        .map(|&[x, y]| {
            let nx = if span_x > 1e-4 && x.is_finite() {
                ((x - min_x) / span_x).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let ny = if span_y > 1e-4 && y.is_finite() {
                ((y - min_y) / span_y).clamp(0.0, 1.0)
            } else {
                0.5
            };
            [nx, ny]
        })
        .collect()
}

#[cfg(any(
    feature = "pure_math",
    all(test, any(feature = "biology", feature = "physics"))
))]
type AdjacencyPair = (Vec<Vec<(usize, f64)>>, Vec<Vec<(usize, f64)>>, bool);

#[cfg(any(
    feature = "pure_math",
    all(test, any(feature = "biology", feature = "physics"))
))]
fn build_bounded_adjacencies(
    node_ids: &[usize],
    directed: bool,
    raw_edges: &[(usize, usize, f64)],
) -> AdjacencyPair {
    let n = node_ids.len();
    let mut clamped_edges = raw_edges.len() > MAX_SHARED_GRAPH_EDGES;
    let mut authored_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut sym_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut accepted = 0usize;

    for &(u_id, v_id, w) in raw_edges {
        if accepted >= MAX_SHARED_GRAPH_EDGES {
            clamped_edges = true;
            break;
        }
        let weight = if w.is_finite() { w } else { 1.0 };
        let u_opt = node_ids.binary_search(&u_id).ok();
        let v_opt = node_ids.binary_search(&v_id).ok();
        if let (Some(u), Some(v)) = (u_opt, v_opt) {
            if u != v {
                upsert_edge(&mut authored_adj[u], v, weight);
                if !directed {
                    upsert_edge(&mut authored_adj[v], u, weight);
                }
                upsert_edge(&mut sym_adj[u], v, weight);
                upsert_edge(&mut sym_adj[v], u, weight);
                accepted += 1;
            }
        }
    }
    for row in &mut authored_adj {
        row.sort_by_key(|&(v, _)| v);
    }
    for row in &mut sym_adj {
        row.sort_by_key(|&(v, _)| v);
    }
    (authored_adj, sym_adj, clamped_edges)
}

#[cfg(any(
    feature = "pure_math",
    all(test, any(feature = "biology", feature = "physics"))
))]
fn pack_csr(n: usize, adj: &[Vec<(usize, f64)>]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let total: usize = adj.iter().map(Vec::len).sum();
    let mut col_idx = Vec::with_capacity(total);
    let mut weights = Vec::with_capacity(total);
    row_ptr.push(0);
    for row in adj.iter().take(n) {
        for &(v, w) in row {
            col_idx.push(v);
            weights.push(w);
        }
        row_ptr.push(col_idx.len());
    }
    (row_ptr, col_idx, weights)
}

#[cfg(any(
    feature = "pure_math",
    all(test, any(feature = "biology", feature = "physics"))
))]
fn upsert_edge(row: &mut Vec<(usize, f64)>, target: usize, weight: f64) {
    if let Some(existing) = row.iter_mut().find(|(v, _)| *v == target) {
        existing.1 = weight;
    } else {
        row.push((target, weight));
    }
}

#[cfg(any(
    feature = "pure_math",
    all(test, any(feature = "biology", feature = "physics"))
))]
fn summarize_symmetric_edges(sym_adj: &[Vec<(usize, f64)>]) -> (Vec<(usize, usize, f64)>, f64) {
    let mut undirected_edges = Vec::new();
    let mut active_nodes = 0usize;
    let mut deg_sum = 0usize;
    for (u, row) in sym_adj.iter().enumerate() {
        if !row.is_empty() {
            active_nodes += 1;
            deg_sum += row.len();
        }
        for &(v, w) in row {
            if u < v {
                undirected_edges.push((u, v, w));
            }
        }
    }
    let mean_deg = if active_nodes > 0 {
        (deg_sum as f64) / (active_nodes as f64)
    } else {
        0.0
    };
    (undirected_edges, mean_deg)
}

impl SharedGraphTopology {
    /// Builds an immutable [`SharedGraphTopology`] from raw `(node_id, [x, y])` and `(u_id, v_id, w)` slices.
    #[cfg(any(
        feature = "pure_math",
        all(test, any(feature = "biology", feature = "physics"))
    ))]
    #[must_use]
    pub fn from_raw(
        revision: u64,
        directed: bool,
        raw_nodes: &[(usize, [f32; 2])],
        raw_edges: &[(usize, usize, f64)],
    ) -> Self {
        let mut sorted_nodes: Vec<(usize, [f32; 2])> = raw_nodes.to_vec();
        sorted_nodes.sort_by_key(|&(id, _)| id);
        sorted_nodes.dedup_by_key(|(id, _)| *id);
        let original_node_count = sorted_nodes.len();
        let clamped_nodes = original_node_count > MAX_SHARED_GRAPH_NODES;
        sorted_nodes.truncate(MAX_SHARED_GRAPH_NODES);

        let n = sorted_nodes.len();
        let node_ids: Vec<usize> = sorted_nodes.iter().map(|&(id, _)| id).collect();
        let positions: Vec<[f32; 2]> = sorted_nodes.iter().map(|&(_, p)| p).collect();
        let normalized_positions = normalize_positions(&positions);
        let (authored_adj, sym_adj, clamped_edges) =
            build_bounded_adjacencies(&node_ids, directed, raw_edges);
        let (row_ptr, col_idx, weights) = pack_csr(n, &authored_adj);
        let (sym_row_ptr, sym_col_idx, sym_weights) = pack_csr(n, &sym_adj);
        let (undirected_edges, mean_active_degree) = summarize_symmetric_edges(&sym_adj);

        Self {
            revision,
            directed,
            node_ids,
            positions,
            normalized_positions,
            row_ptr,
            col_idx,
            weights,
            sym_row_ptr,
            sym_col_idx,
            sym_weights,
            undirected_edges,
            mean_active_degree,
            original_node_count,
            original_edge_count: raw_edges.len(),
            clamped_nodes,
            clamped_edges,
        }
    }

    /// Returns the number of nodes in the snapshot (`<= 256`).
    #[cfg(any(
        feature = "biology",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_ids.len()
    }

    /// Maps normalized node positions into `[min_x..=max_x] x [min_y..=max_y]`.
    #[cfg(any(
        feature = "biology",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    #[must_use]
    pub fn map_positions_to_bounds(
        &self,
        min_x: f32,
        min_y: f32,
        max_x: f32,
        max_y: f32,
    ) -> Vec<[f32; 2]> {
        let w = (max_x - min_x).max(1.0);
        let h = (max_y - min_y).max(1.0);
        self.normalized_positions
            .iter()
            .map(|&[nx, ny]| [min_x + nx * w, min_y + ny * h])
            .collect()
    }
}

/// Immutable 3D point-cloud frame published from chaotic attractors to topological data analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedPointCloud3D {
    /// Monotonically increasing bus revision assigned when published.
    pub revision: u64,
    /// 3D landmark coordinates `[x, y, z]` (capped at `128` landmarks).
    pub points: Vec<[f64; 3]>,
    /// Lorenz / flow parameter $\sigma$.
    pub sigma: f64,
    /// Lorenz / flow parameter $\rho$.
    pub rho: f64,
    /// Lorenz / flow parameter $\beta$.
    pub beta: f64,
    /// Integration step $\Delta t$.
    pub dt: f64,
}
