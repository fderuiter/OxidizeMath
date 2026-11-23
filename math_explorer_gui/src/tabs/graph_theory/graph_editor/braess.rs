//! Braess's Paradox Congestion Routing Lab and Zero-Allocation Frank-Wolfe Solver.
//!
//! Models flow-dependent affine edge latencies $l_e(f_e) = a_e f_e + b_e$ ($a_e, b_e \ge 0$)
//! and computes Wardrop Nash equilibrium flows by minimizing Beckmann's convex potential
//! $\Phi(f) = \sum_e (\frac{1}{2} a_e f_e^2 + b_e f_e)$ via Frank-Wolfe over preallocated buffers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::CalloutManager;
use eframe::egui::Pos2;
use egui_plot::PlotPoint;
use std::collections::BTreeMap;

const INITIAL_NODE_CAP: usize = 64;
const INITIAL_EDGE_CAP: usize = 128;
const MAX_FW_ITERS: usize = 200;
const DEMAND_CURVE_SAMPLES: usize = 21;

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
    pub fn clamped(a: f64, b: f64) -> Self {
        Self {
            a: if a.is_finite() { a.max(0.0) } else { 0.0 },
            b: if b.is_finite() { b.max(0.0) } else { 0.0 },
        }
    }

    /// Evaluates $l_e(f_e) = a_e f_e + b_e$ for non-negative flow.
    pub fn eval(self, flow: f64) -> f64 {
        self.a.max(0.0) * flow.max(0.0) + self.b.max(0.0)
    }

    /// Checks finiteness and non-negativity of both affine coefficients.
    pub fn is_valid(self) -> bool {
        self.a.is_finite() && self.b.is_finite() && self.a >= 0.0 && self.b >= 0.0
    }
}

/// User hypothesis choice before toggling the zero-cost `u -> v` shortcut.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ShortcutPrediction {
    /// User predicts adding a zero-cost road speeds up commuters.
    SpeedsUpCommuters,
    /// User predicts commuter travel time is unchanged.
    NoChange,
    /// User predicts selfish routing triggers a slower Nash equilibrium trap.
    SlowsAllCommutersNashTrap,
}

/// Serializable congestion routing configuration for `GraphEditorTool`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BraessCongestionConfig {
    /// Whether congestion routing and Wardrop equilibrium overlays are active.
    pub enabled: bool,
    /// Origin node ID `s`.
    pub source_node: usize,
    /// Destination node ID `t`.
    pub sink_node: usize,
    /// Total commuter demand `D` routed from `s` to `t`.
    pub total_demand: f64,
    /// Whether the `u -> v` (`1 -> 2`) zero-cost shortcut is enabled.
    pub shortcut_enabled: bool,
    /// Optional predict-before-toggle hypothesis selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prediction: Option<ShortcutPrediction>,
}

impl Default for BraessCongestionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            source_node: 0,
            sink_node: 3,
            total_demand: 1.0,
            shortcut_enabled: false,
            prediction: None,
        }
    }
}

impl BraessCongestionConfig {
    /// Returns true if congestion configuration matches its inactive default.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// Failure modes when solving for Wardrop equilibrium.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WardropError {
    /// Source or sink node does not exist in the graph.
    MissingTerminal,
    /// No directed path connects `s` to `t`.
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
    baseline_flows: Vec<f64>,
    baseline_latencies: Vec<f64>,
    sweep_flows: Vec<f64>,
    sweep_latencies: Vec<f64>,
}

impl FrankWolfeScratch {
    fn new() -> Self {
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

    fn prepare_topology(
        &mut self,
        node_positions: &BTreeMap<usize, Pos2>,
        edges: &[(usize, usize, f64)],
        latencies: &[AffineLatency],
    ) {
        self.dense_nodes.clear();
        self.dense_nodes.extend(node_positions.keys().copied());
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
    }

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

            for (e, &flow_e) in flows.iter().enumerate().take(self.edge_u.len()) {
                if !self.edge_active[e] || skip_edge == Some(e) || self.edge_u[e] != u {
                    continue;
                }
                let v = self.edge_v[e];
                let w = self.clamped_a[e] * flow_e + self.clamped_b[e];
                let cand = best_d + w;
                if cand < self.dist[v] {
                    self.dist[v] = cand;
                    self.prev_edge[v] = Some(e);
                }
            }
        }
        self.dist[t_idx].is_finite()
    }

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
pub fn solve_wardrop_in_place(
    scratch: &mut FrankWolfeScratch,
    source_id: usize,
    sink_id: usize,
    demand: f64,
    skip_edge: Option<usize>,
    out_flows: &mut [f64],
    out_latencies: &mut [f64],
) -> Result<WardropSummary, WardropError> {
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
    out_flows.fill(0.0);

    if !scratch.shortest_path_in_place(s_idx, t_idx, out_flows, skip_edge)
        || !scratch.route_demand_along_tree(s_idx, t_idx, d_pos)
    {
        return Err(WardropError::Unreachable);
    }
    out_flows.copy_from_slice(&scratch.aux_flow);

    let mut gap = 0.0;
    let mut iters = 0;
    for iter in 0..MAX_FW_ITERS {
        iters = iter + 1;
        if !scratch.shortest_path_in_place(s_idx, t_idx, out_flows, skip_edge)
            || !scratch.route_demand_along_tree(s_idx, t_idx, d_pos)
        {
            return Err(WardropError::Unreachable);
        }
        gap = compute_fw_gap(scratch, out_flows, skip_edge);
        if gap <= 1e-10 {
            break;
        }
        let gamma = exact_line_search_step(scratch, out_flows, gap, skip_edge);
        for (f, &y) in out_flows.iter_mut().zip(scratch.aux_flow.iter()) {
            *f = (*f + gamma * (y - *f)).max(0.0);
        }
    }

    Ok(summarize_equilibrium(
        scratch,
        d_pos,
        gap,
        iters,
        skip_edge,
        out_flows,
        out_latencies,
    ))
}

fn compute_fw_gap(scratch: &FrankWolfeScratch, flows: &[f64], skip_edge: Option<usize>) -> f64 {
    let mut gap = 0.0;
    for (e, &flow_e) in flows.iter().enumerate() {
        if !scratch.edge_active[e] || skip_edge == Some(e) {
            continue;
        }
        let lat = scratch.clamped_a[e] * flow_e + scratch.clamped_b[e];
        gap += (flow_e - scratch.aux_flow[e]) * lat;
    }
    gap.max(0.0)
}

fn exact_line_search_step(
    scratch: &FrankWolfeScratch,
    flows: &[f64],
    gap: f64,
    skip_edge: Option<usize>,
) -> f64 {
    let mut curvature = 0.0;
    for (e, &flow_e) in flows.iter().enumerate() {
        if !scratch.edge_active[e] || skip_edge == Some(e) {
            continue;
        }
        let diff = scratch.aux_flow[e] - flow_e;
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
    WardropSummary {
        commuter_latency,
        beckmann_potential: potential,
        duality_gap: gap,
        iterations,
    }
}

/// Interactive Braess's paradox congestion lab hosted inside [`super::GraphEditorTool`].
pub struct BraessCongestionLab {
    /// Active congestion parameters (`s`, `t`, demand `D`, shortcut toggle, hypothesis).
    pub config: BraessCongestionConfig,
    /// Per-edge affine latency coefficients aligned with `GraphEditorTool::edges`.
    pub edge_latencies: Vec<AffineLatency>,
    /// Default slope `a` used when adding new edges in the graph editor.
    pub new_edge_a: f64,
    /// Default base cost `b` used when adding new edges in the graph editor.
    pub new_edge_b: f64,
    /// Callout overlay manager marking the `1.50 -> 2.00` Nash trap.
    pub callout_manager: CalloutManager,
    /// Preallocated solver scratch workspace.
    pub scratch: FrankWolfeScratch,
    /// Equilibrium edge flows $f_e^*$.
    pub equilibrium_flows: Vec<f64>,
    /// Equilibrium edge latencies $l_e(f_e^*)$.
    pub equilibrium_latencies: Vec<f64>,
    /// Latest Wardrop solve result.
    pub status: Result<WardropSummary, WardropError>,
    /// Counterfactual latency without the `u -> v` (`1 -> 2`) shortcut when present.
    pub without_shortcut_latency: Option<f64>,
    active_curve: Vec<PlotPoint>,
    no_shortcut_curve: Vec<PlotPoint>,
}

impl Default for BraessCongestionLab {
    fn default() -> Self {
        Self {
            config: BraessCongestionConfig::default(),
            edge_latencies: Vec::new(),
            new_edge_a: 0.0,
            new_edge_b: 1.0,
            callout_manager: CalloutManager::new(),
            scratch: FrankWolfeScratch::new(),
            equilibrium_flows: Vec::with_capacity(INITIAL_EDGE_CAP),
            equilibrium_latencies: Vec::with_capacity(INITIAL_EDGE_CAP),
            status: Err(WardropError::MissingTerminal),
            without_shortcut_latency: None,
            active_curve: Vec::with_capacity(DEMAND_CURVE_SAMPLES),
            no_shortcut_curve: Vec::with_capacity(DEMAND_CURVE_SAMPLES),
        }
    }
}

#[path = "braess_ui.rs"]
mod braess_ui;
pub use braess_ui::BraessUiAction;

#[cfg(test)]
#[path = "braess_tests.rs"]
mod tests;

// [cite:dijkstra]
