//! Zero-allocation Frank-Wolfe Wardrop congestion routing solver on directed affine-latency graphs.
//!
//! Models flow-dependent affine edge latencies $l_e(f_e) = a_e f_e + b_e$ ($a_e, b_e \ge 0$)
//! and computes Wardrop Nash equilibrium flows by minimizing Beckmann's convex potential
//! $\Phi(f) = \sum_e (\frac{1}{2} a_e f_e^2 + b_e f_e)$ via Frank-Wolfe over preallocated buffers.

use thiserror::Error;

const INITIAL_NODE_CAP: usize = 64;
const INITIAL_EDGE_CAP: usize = 128;
const MAX_FW_ITERS: usize = 200;

/// Non-negative affine congestion latency function $l_e(f_e) = a_e f_e + b_e$.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AffineLatency {
    /// Congestion slope $a_e \ge 0$.
    pub a: f64,
    /// Free-flow base latency $b_e \ge 0$.
    pub b: f64,
}

impl AffineLatency {
    /// Clamps `a` and `b` to finite non-negative numbers.
    #[must_use]
    #[verified_engine::verified]
    pub fn clamped(a: f64, b: f64) -> Self {
        let out = Self {
            a: if a.is_finite() { a.max(0.0) } else { 0.0 },
            b: if b.is_finite() { b.max(0.0) } else { 0.0 },
        };
        debug_assert!(out.a >= 0.0 && out.b >= 0.0);
        out
    }

    /// Evaluates $l_e(f_e) = a_e f_e + b_e$ for non-negative flow.
    #[must_use]
    #[verified_engine::verified]
    pub fn eval(self, flow: f64) -> f64 {
        let latency = self.a.max(0.0) * flow.max(0.0) + self.b.max(0.0);
        debug_assert!(!latency.is_nan());
        latency
    }

    /// Checks finiteness and non-negativity of both affine coefficients.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_valid(self) -> bool {
        self.a.is_finite() && self.b.is_finite() && self.a >= 0.0 && self.b >= 0.0
    }
}

/// Failure modes when solving for Wardrop equilibrium on a directed network.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum WardropError {
    /// Source or sink node does not exist in the graph or output slices are undersized.
    #[error("source or sink node does not exist in the graph")]
    MissingTerminal,
    /// No directed path connects `s` to `t`.
    #[error("no directed path connects source to sink")]
    Unreachable,
}

/// Summary metrics from a converged Frank-Wolfe Wardrop equilibrium run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WardropSummary {
    /// Average commuter travel latency across active routes.
    pub commuter_latency: f64,
    /// Minimized Beckmann convex potential $\Phi(f)$.
    pub beckmann_potential: f64,
    /// Terminal Frank-Wolfe duality gap.
    pub duality_gap: f64,
    /// Iterations executed.
    pub iterations: usize,
}

/// Preallocated scratch workspace for zero-allocation Frank-Wolfe routing.
#[derive(Clone, Debug)]
pub struct FrankWolfeScratch {
    dense_nodes: Vec<usize>,
    edge_u: Vec<usize>,
    edge_v: Vec<usize>,
    edge_active: Vec<bool>,
    clamped_a: Vec<f64>,
    clamped_b: Vec<f64>,
    dist: Vec<f64>,
    visited: Vec<bool>,
    prev_edge: Vec<Option<usize>>,
    aux_flow: Vec<f64>,
    /// Reusable counterfactual edge-flow buffer sized to the prepared topology.
    pub baseline_flows: Vec<f64>,
    /// Reusable counterfactual edge-latency buffer sized to the prepared topology.
    pub baseline_latencies: Vec<f64>,
    /// Reusable demand-sweep edge-flow buffer sized to the prepared topology.
    pub sweep_flows: Vec<f64>,
    /// Reusable demand-sweep edge-latency buffer sized to the prepared topology.
    pub sweep_latencies: Vec<f64>,
}

impl Default for FrankWolfeScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl FrankWolfeScratch {
    /// Allocates reusable scratch vectors with default node and edge capacities.
    #[must_use]
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self {
            dense_nodes: Vec::with_capacity(INITIAL_NODE_CAP),
            edge_u: Vec::with_capacity(INITIAL_EDGE_CAP),
            edge_v: Vec::with_capacity(INITIAL_EDGE_CAP),
            edge_active: Vec::with_capacity(INITIAL_EDGE_CAP),
            clamped_a: Vec::with_capacity(INITIAL_EDGE_CAP),
            clamped_b: Vec::with_capacity(INITIAL_EDGE_CAP),
            dist: Vec::with_capacity(INITIAL_NODE_CAP),
            visited: Vec::with_capacity(INITIAL_NODE_CAP),
            prev_edge: Vec::with_capacity(INITIAL_NODE_CAP),
            aux_flow: Vec::with_capacity(INITIAL_EDGE_CAP),
            baseline_flows: Vec::with_capacity(INITIAL_EDGE_CAP),
            baseline_latencies: Vec::with_capacity(INITIAL_EDGE_CAP),
            sweep_flows: Vec::with_capacity(INITIAL_EDGE_CAP),
            sweep_latencies: Vec::with_capacity(INITIAL_EDGE_CAP),
        }
    }

    /// Indexes `node_ids` and directed `edges` (`(u, v, default_b)`) with clamped `latencies`.
    #[verified_engine::verified]
    pub fn prepare_topology<I: IntoIterator<Item = usize>>(
        &mut self,
        node_ids: I,
        edges: &[(usize, usize, f64)],
        latencies: &[AffineLatency],
    ) {
        self.dense_nodes.clear();
        self.dense_nodes.extend(node_ids);
        self.dense_nodes.sort_unstable();
        self.dense_nodes.dedup();
        let n = self.dense_nodes.len();
        let m = edges.len();

        self.dist.resize(n, f64::INFINITY);
        self.visited.resize(n, false);
        self.prev_edge.resize(n, None);
        self.aux_flow.resize(m, 0.0);
        self.baseline_flows.resize(m, 0.0);
        self.baseline_latencies.resize(m, 0.0);
        self.sweep_flows.resize(m, 0.0);
        self.sweep_latencies.resize(m, 0.0);

        self.edge_u.clear();
        self.edge_v.clear();
        self.edge_active.clear();
        self.clamped_a.clear();
        self.clamped_b.clear();

        for (i, &(u, v, w)) in edges.iter().enumerate() {
            let lat = latencies
                .get(i)
                .copied()
                .unwrap_or_else(|| AffineLatency::clamped(0.0, w));
            let c = AffineLatency::clamped(lat.a, lat.b);
            let u_d = self.dense_nodes.binary_search(&u).ok();
            let v_d = self.dense_nodes.binary_search(&v).ok();
            self.edge_u.push(u_d.unwrap_or(0));
            self.edge_v.push(v_d.unwrap_or(0));
            self.edge_active.push(u_d.is_some() && v_d.is_some());
            self.clamped_a.push(c.a);
            self.clamped_b.push(c.b);
        }
        debug_assert_eq!(self.edge_u.len(), m);
    }

    #[verified_engine::verified]
    fn shortest_path_in_place(
        &mut self,
        s_idx: usize,
        t_idx: usize,
        flows: &[f64],
        skip_edge: Option<usize>,
    ) -> bool {
        let n = self.dense_nodes.len();
        self.dist.fill(f64::INFINITY);
        self.visited.fill(false);
        self.prev_edge.fill(None);
        self.dist[s_idx] = 0.0;

        for _ in 0..n {
            let mut best_u = None;
            let mut best_d = f64::INFINITY;
            for u in 0..n {
                if !self.visited[u] && self.dist[u] < best_d {
                    best_d = self.dist[u];
                    best_u = Some(u);
                }
            }
            let Some(u) = best_u else {
                break;
            };
            if u == t_idx {
                break;
            }
            self.visited[u] = true;

            for (e, flow) in flows.iter().copied().enumerate().take(self.edge_u.len()) {
                if !self.edge_active[e] || skip_edge == Some(e) || self.edge_u[e] != u {
                    continue;
                }
                let v = self.edge_v[e];
                let w = self.clamped_a[e] * flow + self.clamped_b[e];
                let cand = best_d + w;
                if cand < self.dist[v] {
                    self.dist[v] = cand;
                    self.prev_edge[v] = Some(e);
                }
            }
        }
        self.dist[t_idx].is_finite()
    }

    #[verified_engine::verified]
    fn route_demand_along_tree(&mut self, s_idx: usize, t_idx: usize, demand: f64) -> bool {
        self.aux_flow.fill(0.0);
        if s_idx == t_idx {
            return true;
        }
        let n = self.dense_nodes.len();
        let mut cur = t_idx;
        for _ in 0..n {
            if cur == s_idx {
                return true;
            }
            let Some(e) = self.prev_edge[cur] else {
                return false;
            };
            self.aux_flow[e] += demand;
            cur = self.edge_u[e];
        }
        cur == s_idx
    }
}

/// Solves Wardrop equilibrium in place using Frank-Wolfe with zero heap allocations.
///
/// Minimizes Beckmann's strictly convex congestion potential over directed $s$-$t$ path flows
/// $$\min_{f \in \mathcal{F}(d)} \Phi(f) = \sum_{e \in E} \int_0^{f_e} (a_e u + b_e)\,du = \sum_{e \in E} \Bigl(\tfrac{1}{2} a_e f_e^2 + b_e f_e\Bigr)$$
/// via Frank-Wolfe conditional-gradient steps with exact quadratic line search per Frank & Wolfe
/// (1956), *Naval Research Logistics Quarterly* 3(1-2), 95-110 (`DOI: 10.1002/nav.3800030109`)
/// and Braess, Nagurney & Wakolbinger (2005), *Transportation Science* 39(4), 446-450
/// (`DOI: 10.1287/trsc.1050.0127`).
///
/// # Stability
/// Clamps affine slopes and offsets to $a_e, b_e \ge 0$, ensuring convex quadratic subproblems,
/// bounded step sizes $\gamma \in [0, 1]$, and monotone decrease of $\Phi(f)$ until duality gap
/// $\le 10^{-10}$ or `200` iterations.
///
/// # Domain of Validity
/// Requires `scratch` prepared via [`FrankWolfeScratch::prepare_topology`], `out_flows` and
/// `out_latencies` slices of length at least `|E|`, and non-negative finite demand $d \ge 0$ on a
/// graph containing a directed $s \to t$ path after excluding `skip_edge`.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::graph_theory::wardrop::{
///     AffineLatency, FrankWolfeScratch, solve_wardrop_in_place,
/// };
///
/// let edges = [(0, 1, 0.0), (1, 3, 1.0), (0, 2, 1.0), (2, 3, 0.0), (1, 2, 0.0)];
/// let latencies = [
///     AffineLatency::clamped(1.0, 0.0),
///     AffineLatency::clamped(0.0, 1.0),
///     AffineLatency::clamped(0.0, 1.0),
///     AffineLatency::clamped(1.0, 0.0),
///     AffineLatency::clamped(0.0, 0.0),
/// ];
/// let mut scratch = FrankWolfeScratch::new();
/// scratch.prepare_topology(0..4, &edges, &latencies);
/// let mut flows = [0.0; 5];
/// let mut lats = [0.0; 5];
/// let closed = solve_wardrop_in_place(&mut scratch, 0, 3, 1.0, Some(4), &mut flows, &mut lats)
///     .unwrap();
/// let opened = solve_wardrop_in_place(&mut scratch, 0, 3, 1.0, None, &mut flows, &mut lats)
///     .unwrap();
/// assert!((closed.commuter_latency - 1.50).abs() < 1e-6);
/// assert!((opened.commuter_latency - 2.00).abs() < 1e-6);
/// ```
#[verified_engine::verified]
pub fn solve_wardrop_in_place(
    scratch: &mut FrankWolfeScratch,
    source_id: usize,
    sink_id: usize,
    demand: f64,
    skip_edge: Option<usize>,
    out_flows: &mut [f64],
    out_latencies: &mut [f64],
) -> Result<WardropSummary, WardropError> {
    let m = scratch.edge_u.len();
    if out_flows.len() < m || out_latencies.len() < m {
        return Err(WardropError::MissingTerminal);
    }
    let Ok(s_idx) = scratch.dense_nodes.binary_search(&source_id) else {
        return Err(WardropError::MissingTerminal);
    };
    let Ok(t_idx) = scratch.dense_nodes.binary_search(&sink_id) else {
        return Err(WardropError::MissingTerminal);
    };
    let d_pos = if demand.is_finite() {
        demand.max(0.0)
    } else {
        0.0
    };
    out_flows[..m].fill(0.0);

    if !scratch.shortest_path_in_place(s_idx, t_idx, &out_flows[..m], skip_edge)
        || !scratch.route_demand_along_tree(s_idx, t_idx, d_pos)
    {
        return Err(WardropError::Unreachable);
    }
    out_flows[..m].copy_from_slice(&scratch.aux_flow[..m]);

    let mut gap = 0.0;
    let mut iters = 0;
    for iter in 0..MAX_FW_ITERS {
        iters = iter + 1;
        if !scratch.shortest_path_in_place(s_idx, t_idx, &out_flows[..m], skip_edge)
            || !scratch.route_demand_along_tree(s_idx, t_idx, d_pos)
        {
            return Err(WardropError::Unreachable);
        }
        gap = compute_fw_gap(scratch, &out_flows[..m], skip_edge);
        if gap <= 1e-10 {
            break;
        }
        let gamma = exact_line_search_step(scratch, &out_flows[..m], gap, skip_edge);
        for (f, &y) in out_flows[..m].iter_mut().zip(scratch.aux_flow.iter()) {
            *f = (*f + gamma * (y - *f)).max(0.0);
        }
    }

    Ok(summarize_equilibrium(
        scratch,
        d_pos,
        gap,
        iters,
        skip_edge,
        &out_flows[..m],
        &mut out_latencies[..m],
    ))
}

#[verified_engine::verified]
fn compute_fw_gap(scratch: &FrankWolfeScratch, flows: &[f64], skip_edge: Option<usize>) -> f64 {
    let mut gap = 0.0;
    for (e, flow) in flows.iter().copied().enumerate() {
        if !scratch.edge_active[e] || skip_edge == Some(e) {
            continue;
        }
        let lat = scratch.clamped_a[e] * flow + scratch.clamped_b[e];
        gap += (flow - scratch.aux_flow[e]) * lat;
    }
    gap.max(0.0)
}

#[verified_engine::verified]
fn exact_line_search_step(
    scratch: &FrankWolfeScratch,
    flows: &[f64],
    gap: f64,
    skip_edge: Option<usize>,
) -> f64 {
    let mut curvature = 0.0;
    for (e, flow) in flows.iter().copied().enumerate() {
        if !scratch.edge_active[e] || skip_edge == Some(e) {
            continue;
        }
        let diff = scratch.aux_flow[e] - flow;
        curvature += scratch.clamped_a[e] * diff * diff;
    }
    if curvature > 1e-15 {
        (gap / curvature).clamp(0.0, 1.0)
    } else if gap > 0.0 {
        1.0
    } else {
        0.0
    }
}

#[verified_engine::verified]
fn summarize_equilibrium(
    scratch: &FrankWolfeScratch,
    demand: f64,
    gap: f64,
    iterations: usize,
    skip_edge: Option<usize>,
    flows: &[f64],
    out_latencies: &mut [f64],
) -> WardropSummary {
    let mut total_cost = 0.0;
    let mut potential = 0.0;
    for e in 0..flows.len() {
        if !scratch.edge_active[e] || skip_edge == Some(e) {
            out_latencies[e] = 0.0;
            continue;
        }
        let f = flows[e];
        let a = scratch.clamped_a[e];
        let b = scratch.clamped_b[e];
        let lat = a * f + b;
        out_latencies[e] = lat;
        total_cost += f * lat;
        potential += 0.5 * a * f * f + b * f;
    }
    let commuter_latency = if demand > 1e-12 {
        total_cost / demand
    } else {
        scratch.dist.first().copied().unwrap_or(0.0)
    };
    debug_assert!(commuter_latency.is_finite());
    WardropSummary {
        commuter_latency,
        beckmann_potential: potential,
        duality_gap: gap,
        iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    #[verified_engine::verified]
    fn test_wardrop_frank_wolfe_braess_paradox_and_zero_alloc() {
        let edges = [
            (0, 1, 0.0),
            (1, 3, 1.0),
            (0, 2, 1.0),
            (2, 3, 0.0),
            (1, 2, 0.0),
        ];
        let latencies = [
            AffineLatency::clamped(1.0, 0.0),
            AffineLatency::clamped(0.0, 1.0),
            AffineLatency::clamped(0.0, 1.0),
            AffineLatency::clamped(1.0, 0.0),
            AffineLatency::clamped(0.0, 0.0),
        ];
        let mut scratch = FrankWolfeScratch::new();
        scratch.prepare_topology([3, 1, 0, 2], &edges, &latencies);
        let mut flows = [0.0; 5];
        let mut lats = [0.0; 5];

        let closed =
            solve_wardrop_in_place(&mut scratch, 0, 3, 1.0, Some(4), &mut flows, &mut lats)
                .expect("closed network solvable");
        assert!((closed.commuter_latency - 1.50).abs() < 1e-6);
        assert!((closed.beckmann_potential - 1.25).abs() < 1e-6);

        let (opened_res, _) = VerifiedEngine::run_verified(|| {
            solve_wardrop_in_place(&mut scratch, 0, 3, 1.0, None, &mut flows, &mut lats)
        });
        let opened = opened_res.expect("opened network solvable");
        assert!((opened.commuter_latency - 2.00).abs() < 1e-6);
        assert!((opened.beckmann_potential - 1.00).abs() < 1e-6);
    }
}

// [cite:dijkstra]
