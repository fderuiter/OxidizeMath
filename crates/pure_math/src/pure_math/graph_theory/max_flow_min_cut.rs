//! Preallocated zero-allocation Edmonds-Karp `s-t` maximum-flow and minimum-cut solver.

use super::error::GraphTheoryError;

const FLOW_EPS: f64 = 1e-12;

/// Summary of a single shortest augmenting path step discovered by Edmonds-Karp BFS.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AugmentingPathStep {
    /// Vertex sequence `[source, ..., sink]` along the augmenting path.
    pub nodes: Vec<usize>,
    /// Indices into the input edge slice traversed along the path.
    pub edge_indices: Vec<usize>,
    /// Bottleneck flow `> 0` pushed along this augmenting path.
    pub bottleneck: f64,
    /// Cumulative `s-t` flow after applying this augmentation.
    pub cumulative_flow: f64,
}

/// Zero-allocation scalar summary returned by [`solve_max_flow_min_cut_in_place`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MaxFlowSummary {
    /// Total maximum `s -> t` flow value.
    pub max_flow: f64,
    /// Total capacity of edges crossing the `s-t` minimum cut `(S, T)` (equals `max_flow`).
    pub min_cut_capacity: f64,
    /// Number of augmenting path iterations executed.
    pub augmenting_steps: usize,
}

/// Complete maximum-flow, minimum-cut partition, and augmenting-path trace for visualization.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MaxFlowMinCutResult {
    /// Maximum `s -> t` flow value.
    pub max_flow: f64,
    /// Capacity of the `s-t` minimum cut `(S, T)`.
    pub min_cut_capacity: f64,
    /// Signed net flow per input edge (`> 0` flows `u -> v`, `< 0` flows `v -> u` on undirected edges).
    pub edge_flows: Vec<f64>,
    /// Boolean mask of length `vertex_count`: `true` if vertex belongs to source side `S` of the min-cut.
    pub source_side_mask: Vec<bool>,
    /// Indices into the input edge slice that cross the minimum cut between `S` and `T = V \ S`.
    pub cut_edge_indices: Vec<usize>,
    /// Ordered Edmonds-Karp augmenting paths from initial zero flow to optimality.
    pub steps: Vec<AugmentingPathStep>,
}

/// Preallocated CSR residual-network workspace for zero-allocation Edmonds-Karp execution.
#[derive(Clone, Debug, Default)]
pub struct EdmondsKarpScratch {
    num_nodes: usize,
    num_edges: usize,
    directed: bool,
    edge_u: Vec<usize>,
    edge_v: Vec<usize>,
    edge_cap: Vec<f64>,
    head_offsets: Vec<usize>,
    csr_arcs: Vec<usize>,
    write_cursor: Vec<usize>,
    residual: Vec<f64>,
    bfs_queue: Vec<usize>,
    parent_arc: Vec<usize>,
}

impl EdmondsKarpScratch {
    /// Preallocates workspace buffers for graphs up to `max_nodes` vertices and `max_edges` edges.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_capacity(max_nodes: usize, max_edges: usize) -> Self {
        Self {
            num_nodes: 0,
            num_edges: 0,
            directed: true,
            edge_u: Vec::with_capacity(max_edges),
            edge_v: Vec::with_capacity(max_edges),
            edge_cap: Vec::with_capacity(max_edges),
            head_offsets: Vec::with_capacity(max_nodes + 1),
            csr_arcs: Vec::with_capacity(max_edges.saturating_mul(2)),
            write_cursor: Vec::with_capacity(max_nodes),
            residual: Vec::with_capacity(max_edges.saturating_mul(2)),
            bfs_queue: Vec::with_capacity(max_nodes),
            parent_arc: Vec::with_capacity(max_nodes),
        }
    }

    /// Validates and loads `edges` (`(u, v, capacity)`) into the CSR residual layout.
    ///
    /// Call this setup method once per topology; subsequent [`solve_max_flow_min_cut_in_place`] calls
    /// perform zero heap allocations.
    #[verified_engine::verified]
    pub fn prepare(
        &mut self,
        vertex_count: usize,
        edges: &[(usize, usize, f64)],
        directed: bool,
    ) -> Result<(), GraphTheoryError> {
        validate_flow_edges(vertex_count, edges)?;
        self.num_nodes = vertex_count;
        self.num_edges = edges.len();
        self.directed = directed;
        self.resize_topology_buffers(vertex_count, edges.len());
        self.populate_csr(edges);
        Ok(())
    }

    #[verified_engine::verified]
    fn resize_topology_buffers(&mut self, n: usize, m: usize) {
        self.edge_u.clear();
        self.edge_v.clear();
        self.edge_cap.clear();
        self.head_offsets.clear();
        self.head_offsets.resize(n + 1, 0);
        self.write_cursor.resize(n, 0);
        self.csr_arcs.resize(m * 2, 0);
        self.residual.resize(m * 2, 0.0);
        self.bfs_queue.resize(n, 0);
        self.parent_arc.resize(n, usize::MAX);
    }

    #[verified_engine::verified]
    fn populate_csr(&mut self, edges: &[(usize, usize, f64)]) {
        for &(u, v, cap) in edges {
            self.edge_u.push(u);
            self.edge_v.push(v);
            self.edge_cap.push(cap);
            if u != v {
                self.head_offsets[u + 1] += 1;
                self.head_offsets[v + 1] += 1;
            }
        }
        let mut idx = 0usize;
        while idx < self.num_nodes {
            self.head_offsets[idx + 1] += self.head_offsets[idx];
            self.write_cursor[idx] = self.head_offsets[idx];
            idx += 1;
        }
        let mut e = 0usize;
        while e < self.num_edges {
            let u = self.edge_u[e];
            let v = self.edge_v[e];
            if u != v {
                let pos_u = self.write_cursor[u];
                self.csr_arcs[pos_u] = 2 * e;
                self.write_cursor[u] = pos_u + 1;
                let pos_v = self.write_cursor[v];
                self.csr_arcs[pos_v] = 2 * e + 1;
                self.write_cursor[v] = pos_v + 1;
            }
            e += 1;
        }
    }

    #[verified_engine::verified]
    fn reset_residuals(&mut self) {
        let mut e = 0usize;
        while e < self.num_edges {
            let cap = if self.edge_u[e] == self.edge_v[e] {
                0.0
            } else {
                self.edge_cap[e]
            };
            self.residual[2 * e] = cap;
            self.residual[2 * e + 1] = if self.directed { 0.0 } else { cap };
            e += 1;
        }
    }

    #[verified_engine::verified]
    fn arc_endpoints(&self, arc: usize) -> (usize, usize) {
        let edge_idx = arc / 2;
        if arc.is_multiple_of(2) {
            (self.edge_u[edge_idx], self.edge_v[edge_idx])
        } else {
            (self.edge_v[edge_idx], self.edge_u[edge_idx])
        }
    }

    #[verified_engine::verified]
    fn bfs_shortest_augmenting(&mut self, source: usize, sink: usize) -> bool {
        self.parent_arc.fill(usize::MAX);
        self.parent_arc[source] = usize::MAX - 1;
        self.bfs_queue[0] = source;
        let mut head = 0usize;
        let mut tail = 1usize;
        while head < tail {
            let u = self.bfs_queue[head];
            head += 1;
            if u == sink {
                return true;
            }
            let mut cur = self.head_offsets[u];
            let end = self.head_offsets[u + 1];
            while cur < end {
                let arc = self.csr_arcs[cur];
                let (_, v) = self.arc_endpoints(arc);
                if self.parent_arc[v] == usize::MAX && self.residual[arc] > FLOW_EPS {
                    self.parent_arc[v] = arc;
                    self.bfs_queue[tail] = v;
                    tail += 1;
                }
                cur += 1;
            }
        }
        false
    }

    #[verified_engine::verified]
    fn augment_along_parent(&mut self, source: usize, sink: usize) -> f64 {
        let mut bottleneck = f64::INFINITY;
        let mut cur = sink;
        let mut guard = 0usize;
        while cur != source && guard < self.num_nodes {
            let arc = self.parent_arc[cur];
            bottleneck = bottleneck.min(self.residual[arc]);
            let (u, _) = self.arc_endpoints(arc);
            cur = u;
            guard += 1;
        }
        cur = sink;
        guard = 0;
        while cur != source && guard < self.num_nodes {
            let arc = self.parent_arc[cur];
            self.residual[arc] -= bottleneck;
            self.residual[arc ^ 1] += bottleneck;
            let (u, _) = self.arc_endpoints(arc);
            cur = u;
            guard += 1;
        }
        bottleneck
    }

    #[verified_engine::verified]
    fn mark_reachable_cut(&mut self, source: usize, out_source_side: &mut [bool]) {
        out_source_side.fill(false);
        out_source_side[source] = true;
        self.bfs_queue[0] = source;
        let mut head = 0usize;
        let mut tail = 1usize;
        while head < tail {
            let u = self.bfs_queue[head];
            head += 1;
            let mut cur = self.head_offsets[u];
            let end = self.head_offsets[u + 1];
            while cur < end {
                let arc = self.csr_arcs[cur];
                let (_, v) = self.arc_endpoints(arc);
                if !out_source_side[v] && self.residual[arc] > FLOW_EPS {
                    out_source_side[v] = true;
                    self.bfs_queue[tail] = v;
                    tail += 1;
                }
                cur += 1;
            }
        }
    }
}

#[verified_engine::verified]
fn validate_flow_edges(
    vertex_count: usize,
    edges: &[(usize, usize, f64)],
) -> Result<(), GraphTheoryError> {
    let mut i = 0usize;
    while i < edges.len() {
        let (u, v, cap) = edges[i];
        if u >= vertex_count {
            return Err(GraphTheoryError::InvalidTerminal {
                node: u,
                vertex_count,
            });
        }
        if v >= vertex_count {
            return Err(GraphTheoryError::InvalidTerminal {
                node: v,
                vertex_count,
            });
        }
        if !cap.is_finite() || cap < 0.0 {
            return Err(GraphTheoryError::InvalidCapacity {
                edge_index: i,
                value: cap,
            });
        }
        i += 1;
    }
    Ok(())
}

#[verified_engine::verified]
fn validate_in_place_buffers(
    scratch: &EdmondsKarpScratch,
    source: usize,
    sink: usize,
    out_edge_flows: &[f64],
    out_source_side: &[bool],
) -> Result<(), GraphTheoryError> {
    if source >= scratch.num_nodes {
        return Err(GraphTheoryError::InvalidTerminal {
            node: source,
            vertex_count: scratch.num_nodes,
        });
    }
    if sink >= scratch.num_nodes {
        return Err(GraphTheoryError::InvalidTerminal {
            node: sink,
            vertex_count: scratch.num_nodes,
        });
    }
    if source == sink {
        return Err(GraphTheoryError::SourceEqualsSink { node: source });
    }
    if out_edge_flows.len() != scratch.num_edges {
        return Err(GraphTheoryError::ScratchBufferMismatch {
            expected: scratch.num_edges,
            actual: out_edge_flows.len(),
        });
    }
    if out_source_side.len() != scratch.num_nodes {
        return Err(GraphTheoryError::ScratchBufferMismatch {
            expected: scratch.num_nodes,
            actual: out_source_side.len(),
        });
    }
    Ok(())
}

/// Solves `s-t` max-flow and min-cut in-place using preallocated `scratch` with zero heap allocations.
///
/// # Examples
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::graph_theory::max_flow_min_cut::{
///     EdmondsKarpScratch, solve_max_flow_min_cut_in_place,
/// };
///
/// let edges = [(0, 1, 3.0), (0, 2, 2.0), (1, 2, 1.0), (1, 3, 2.0), (2, 3, 4.0)];
/// let mut scratch = EdmondsKarpScratch::with_capacity(4, 5);
/// scratch.prepare(4, &edges, true)?;
/// let mut flows = [0.0; 5];
/// let mut cut_side = [false; 4];
/// let summary = solve_max_flow_min_cut_in_place(&mut scratch, 0, 3, &mut flows, &mut cut_side)?;
/// assert!((summary.max_flow - 5.0).abs() < 1e-9);
/// assert!((summary.max_flow - summary.min_cut_capacity).abs() < 1e-9);
/// # Ok::<(), math_explorer::error::Error>(())
/// ```
#[verified_engine::verified]
pub fn solve_max_flow_min_cut_in_place(
    scratch: &mut EdmondsKarpScratch,
    source: usize,
    sink: usize,
    out_edge_flows: &mut [f64],
    out_source_side: &mut [bool],
) -> Result<MaxFlowSummary, GraphTheoryError> {
    validate_in_place_buffers(scratch, source, sink, out_edge_flows, out_source_side)?;
    scratch.reset_residuals();
    let max_augmentations = scratch.num_nodes.saturating_mul(scratch.num_edges).max(1) + 1;
    let mut total_flow = 0.0;
    let mut steps = 0usize;
    while steps < max_augmentations && scratch.bfs_shortest_augmenting(source, sink) {
        total_flow += scratch.augment_along_parent(source, sink);
        steps += 1;
    }
    scratch.mark_reachable_cut(source, out_source_side);
    let min_cut_capacity = extract_flows_and_cut_capacity(scratch, out_source_side, out_edge_flows);
    Ok(MaxFlowSummary {
        max_flow: total_flow,
        min_cut_capacity,
        augmenting_steps: steps,
    })
}

#[verified_engine::verified]
fn extract_flows_and_cut_capacity(
    scratch: &EdmondsKarpScratch,
    source_side: &[bool],
    out_edge_flows: &mut [f64],
) -> f64 {
    let mut cut_cap = 0.0;
    let mut e = 0usize;
    while e < scratch.num_edges {
        let u = scratch.edge_u[e];
        let v = scratch.edge_v[e];
        let cap = scratch.edge_cap[e];
        out_edge_flows[e] = if u == v {
            0.0
        } else {
            cap - scratch.residual[2 * e]
        };
        let crosses = if scratch.directed {
            source_side[u] && !source_side[v]
        } else {
            source_side[u] != source_side[v]
        };
        if crosses {
            cut_cap += cap;
        }
        e += 1;
    }
    cut_cap
}

/// Computes `s-t` maximum flow, minimum-cut partition `(S, T)`, and augmenting-path steps.
///
/// # Examples
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::graph_theory::max_flow_min_cut::max_flow_min_cut;
///
/// let edges = [(0, 1, 10.0), (0, 2, 5.0), (1, 2, 15.0), (1, 3, 5.0), (2, 3, 10.0)];
/// let result = max_flow_min_cut(4, &edges, 0, 3, true)?;
/// assert!((result.max_flow - 15.0).abs() < 1e-9);
/// assert!((result.min_cut_capacity - 15.0).abs() < 1e-9);
/// # Ok::<(), math_explorer::error::Error>(())
/// ```
#[verified_engine::verified]
pub fn max_flow_min_cut(
    vertex_count: usize,
    edges: &[(usize, usize, f64)],
    source: usize,
    sink: usize,
    directed: bool,
) -> Result<MaxFlowMinCutResult, GraphTheoryError> {
    let mut scratch = EdmondsKarpScratch::with_capacity(vertex_count, edges.len());
    scratch.prepare(vertex_count, edges, directed)?;
    let mut edge_flows = vec![0.0; edges.len()];
    let mut source_side_mask = vec![false; vertex_count];
    validate_in_place_buffers(&scratch, source, sink, &edge_flows, &source_side_mask)?;
    scratch.reset_residuals();

    let max_augmentations = scratch.num_nodes.saturating_mul(scratch.num_edges).max(1) + 1;
    let mut total_flow = 0.0;
    let mut steps = Vec::new();
    let mut iter = 0usize;
    while iter < max_augmentations && scratch.bfs_shortest_augmenting(source, sink) {
        let (nodes, edge_indices) = reconstruct_augmenting_path(&scratch, source, sink);
        let bottleneck = scratch.augment_along_parent(source, sink);
        total_flow += bottleneck;
        steps.push(AugmentingPathStep {
            nodes,
            edge_indices,
            bottleneck,
            cumulative_flow: total_flow,
        });
        iter += 1;
    }
    scratch.mark_reachable_cut(source, &mut source_side_mask);
    let min_cut_capacity =
        extract_flows_and_cut_capacity(&scratch, &source_side_mask, &mut edge_flows);
    let cut_edge_indices = collect_cut_edges(&scratch, &source_side_mask);
    Ok(MaxFlowMinCutResult {
        max_flow: total_flow,
        min_cut_capacity,
        edge_flows,
        source_side_mask,
        cut_edge_indices,
        steps,
    })
}

#[path = "max_flow_trace.rs"]
mod max_flow_trace;
use max_flow_trace::{collect_cut_edges, reconstruct_augmenting_path};

// [cite:graph_parameters_rust]
