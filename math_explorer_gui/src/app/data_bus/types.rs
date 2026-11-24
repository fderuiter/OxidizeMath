//! Typed immutable signal payloads carried by [`super::WorkspaceDataBus`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[cfg(any(test, feature = "pure_math"))]
use eframe::egui;
use serde::{Deserialize, Serialize};
#[cfg(any(
    test,
    feature = "applied",
    feature = "biology",
    feature = "epidemiology",
    feature = "physics"
))]
use std::sync::Arc;

/// Maximum Hodgkin-Huxley neuron count simulated live from a shared workspace graph.
#[cfg(any(test, feature = "biology"))]
pub const MAX_HH_NEURON_BUDGET: usize = 32;

/// Maximum contact-network node count simulated live in `NetworkPropagationTool`.
#[cfg(any(test, feature = "epidemiology"))]
pub const MAX_EPIDEMIC_NODE_BUDGET: usize = 500;

/// Maximum spatial game node count simulated live in `ReplicatorDynamicsTool`.
#[cfg(any(test, feature = "applied"))]
pub const MAX_GAME_NODE_BUDGET: usize = 500;

/// Maximum graph spin count simulated live in `IsingModelTool`.
#[cfg(any(test, feature = "physics"))]
pub const MAX_SPIN_NODE_BUDGET: usize = 1_000;

/// Maximum node count accepted when validating `.mws` session bus snapshots.
pub const SESSION_MAX_BUS_NODES: usize = 2_000;

/// Maximum undirected edge count accepted when validating `.mws` session bus snapshots.
pub const SESSION_MAX_BUS_EDGES: usize = 10_000;

/// Maximum trajectory sample count accepted in `.mws` epidemic trace snapshots.
pub const SESSION_MAX_TRACE_POINTS: usize = 2_000;

/// Maximum node count accepted when validating `#state=` URL graph payloads.
pub const URL_MAX_BUS_NODES: usize = 500;

/// Maximum edge count accepted when validating `#state=` URL graph payloads.
pub const URL_MAX_BUS_EDGES: usize = 2_000;

/// Maximum trajectory sample count accepted in `#state=` URL epidemic trace payloads.
pub const URL_MAX_TRACE_POINTS: usize = 512;

/// Normalized 2D node record inside a [`SharedGraph`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedNode {
    /// Stable user-facing node identifier from `GraphEditorTool` or preset topology.
    pub id: usize,
    /// Normalized viewport coordinates `[x, y]` in `[0.0, 1.0]^2`.
    pub norm_pos: [f32; 2],
}

/// Immutable, monotonically revisioned signed-weight graph topology shared across panes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedGraph {
    /// Monotonically increasing topology revision (`>= 1`).
    pub revision: u64,
    /// Ordered node list (`0..nodes.len()`).
    pub nodes: Vec<SharedNode>,
    /// Compact undirected signed edges `(u_idx, v_idx, weight)` indexing into `nodes`.
    pub edges: Vec<(usize, usize, f64)>,
}

/// Immutable empirical compartment trace streamed from `NetworkPropagationTool` to `SirTool`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EpidemicTraceSignal {
    /// Monotonically increasing trace revision (`>= 1`).
    pub revision: u64,
    /// Network population size `N` used to normalize or rescale compartments.
    pub population: usize,
    /// Per-contact transmission probability `beta`.
    pub beta: f64,
    /// Per-step recovery probability `gamma`.
    pub gamma: f64,
    /// Analytical heterogeneous basic reproduction number $R_0^{\text{het}}$.
    pub heterogeneous_r0: f64,
    /// Empirical `[step, S(step)]` counts.
    pub susceptible: Vec<[f64; 2]>,
    /// Empirical `[step, I(step)]` counts.
    pub infected: Vec<[f64; 2]>,
    /// Empirical `[step, R(step)]` counts.
    pub recovered: Vec<[f64; 2]>,
}

impl SharedGraph {
    /// Validates structural bounds, coordinate finiteness, and endpoint indices.
    pub fn validate(&self, max_nodes: usize, max_edges: usize) -> Result<(), String> {
        if self.nodes.len() > max_nodes {
            return Err(format!(
                "SharedGraph node count ({}) exceeds limit ({max_nodes})",
                self.nodes.len()
            ));
        }
        if self.edges.len() > max_edges {
            return Err(format!(
                "SharedGraph edge count ({}) exceeds limit ({max_edges})",
                self.edges.len()
            ));
        }
        for node in &self.nodes {
            let [x, y] = node.norm_pos;
            if !x.is_finite()
                || !y.is_finite()
                || !(-0.01..=1.01).contains(&x)
                || !(-0.01..=1.01).contains(&y)
            {
                return Err("SharedGraph contains non-finite or out-of-range coordinates".into());
            }
        }
        let n = self.nodes.len();
        for &(u, v, w) in &self.edges {
            if u >= n || v >= n || u == v || !w.is_finite() {
                return Err(
                    "SharedGraph contains invalid edge endpoint or non-finite weight".into(),
                );
            }
        }
        Ok(())
    }

    /// Normalizes canvas positions and arbitrary node IDs into a compact [`SharedGraph`].
    #[cfg(any(test, feature = "pure_math"))]
    #[must_use]
    pub fn from_canvas_positions(
        revision: u64,
        sorted_nodes: &[(usize, egui::Pos2)],
        raw_edges: &[(usize, usize, f64)],
    ) -> Self {
        let rev = revision.max(1);
        if sorted_nodes.is_empty() {
            return Self {
                revision: rev,
                nodes: Vec::new(),
                edges: Vec::new(),
            };
        }

        let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
        for &(_, pos) in sorted_nodes {
            min_x = min_x.min(pos.x);
            max_x = max_x.max(pos.x);
            min_y = min_y.min(pos.y);
            max_y = max_y.max(pos.y);
        }
        let span_x = (max_x - min_x).max(1.0);
        let span_y = (max_y - min_y).max(1.0);
        let single = sorted_nodes.len() == 1;

        let mut nodes = Vec::with_capacity(sorted_nodes.len());
        let mut id_to_compact = std::collections::HashMap::with_capacity(sorted_nodes.len());
        for (compact_idx, &(id, pos)) in sorted_nodes.iter().enumerate() {
            id_to_compact.insert(id, compact_idx);
            let norm_pos = if single {
                [0.5, 0.5]
            } else {
                let nx = if max_x - min_x < 1e-4 {
                    0.5
                } else {
                    0.1 + 0.8 * ((pos.x - min_x) / span_x)
                };
                let ny = if max_y - min_y < 1e-4 {
                    0.5
                } else {
                    0.1 + 0.8 * ((pos.y - min_y) / span_y)
                };
                [nx.clamp(0.0, 1.0), ny.clamp(0.0, 1.0)]
            };
            nodes.push(SharedNode { id, norm_pos });
        }

        let edges = deduplicate_compact_edges(raw_edges, &id_to_compact);
        Self {
            revision: rev,
            nodes,
            edges,
        }
    }

    /// Clamps this graph to `max_nodes`, returning `(clamped_graph, Some(original_count))` if truncated.
    #[cfg(any(
        test,
        feature = "applied",
        feature = "biology",
        feature = "epidemiology",
        feature = "physics"
    ))]
    #[must_use]
    pub fn clamped_to_budget(self: &Arc<Self>, max_nodes: usize) -> (Arc<Self>, Option<usize>) {
        if self.nodes.len() <= max_nodes {
            return (Arc::clone(self), None);
        }
        let original = self.nodes.len();
        let nodes = self.nodes[..max_nodes].to_vec();
        let edges = self
            .edges
            .iter()
            .copied()
            .filter(|&(u, v, _)| u < max_nodes && v < max_nodes)
            .collect();
        (
            Arc::new(Self {
                revision: self.revision,
                nodes,
                edges,
            }),
            Some(original),
        )
    }

    /// Builds an undirected neighbor list from positive-weight (`w > 0.0`) active edges.
    #[cfg(any(test, feature = "epidemiology"))]
    #[must_use]
    pub fn positive_adjacency(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.nodes.len()];
        for &(u, v, w) in &self.edges {
            if w > 0.0 && u < adj.len() && v < adj.len() && u != v {
                if !adj[u].contains(&v) {
                    adj[u].push(v);
                }
                if !adj[v].contains(&u) {
                    adj[v].push(u);
                }
            }
        }
        adj
    }
}

/// Deduplicates undirected compact edges while preserving the latest finite weight.
#[cfg(any(test, feature = "pure_math"))]
fn deduplicate_compact_edges(
    raw_edges: &[(usize, usize, f64)],
    id_to_compact: &std::collections::HashMap<usize, usize>,
) -> Vec<(usize, usize, f64)> {
    let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(raw_edges.len());
    for &(u_id, v_id, w) in raw_edges {
        if !w.is_finite() {
            continue;
        }
        let (Some(&u), Some(&v)) = (id_to_compact.get(&u_id), id_to_compact.get(&v_id)) else {
            continue;
        };
        if u == v {
            continue;
        }
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        if let Some(slot) = edges.iter_mut().find(|(ea, eb, _)| *ea == a && *eb == b) {
            slot.2 = w;
        } else {
            edges.push((a, b, w));
        }
    }
    edges
}

/// Computes exact population degree mean $\langle k \rangle$ and variance $\operatorname{Var}(k)$
/// for an undirected adjacency list.
#[cfg(any(test, feature = "epidemiology"))]
#[must_use]
pub fn adjacency_degree_moments(adjacency: &[Vec<usize>]) -> (f64, f64) {
    if adjacency.is_empty() {
        return (0.0, 0.0);
    }
    let n = adjacency.len() as f64;
    let mut sum_k = 0.0_f64;
    let mut sum_k_sq = 0.0_f64;
    for neighbors in adjacency {
        let k = neighbors.len() as f64;
        sum_k += k;
        sum_k_sq += k * k;
    }
    let mean_k = sum_k / n;
    let var_k = (sum_k_sq / n - mean_k * mean_k).max(0.0);
    (mean_k, var_k)
}

impl EpidemicTraceSignal {
    /// Validates sample counts and numeric finiteness for session or URL deserialization.
    pub fn validate(&self, max_points: usize) -> Result<(), String> {
        if self.population == 0 || self.population > SESSION_MAX_BUS_NODES {
            return Err("EpidemicTraceSignal has invalid population bound".into());
        }
        if !self.beta.is_finite()
            || !self.gamma.is_finite()
            || !self.heterogeneous_r0.is_finite()
            || self.beta < 0.0
            || self.gamma < 0.0
        {
            return Err("EpidemicTraceSignal has non-finite or negative rate parameters".into());
        }
        for series in [&self.susceptible, &self.infected, &self.recovered] {
            if series.len() > max_points {
                return Err(format!(
                    "EpidemicTraceSignal series length ({}) exceeds limit ({max_points})",
                    series.len()
                ));
            }
            if series
                .iter()
                .any(|p| !p[0].is_finite() || !p[1].is_finite())
            {
                return Err("EpidemicTraceSignal contains non-finite sample point".into());
            }
        }
        Ok(())
    }
}
