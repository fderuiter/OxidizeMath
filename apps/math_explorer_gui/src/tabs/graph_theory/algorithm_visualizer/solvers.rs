//! Predecessor-tree, Edmonds-Karp max-flow/min-cut, and Kruskal MST solvers for `AlgorithmVisualizerTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{Algorithm, AlgorithmVisualizerTool};
use math_explorer::pure_math::graph_theory::{
    dijkstra::dijkstra,
    max_flow_min_cut::{max_flow_min_cut, AugmentingPathStep},
    minimum_spanning_tree::{minimum_spanning_tree, MstResult},
};
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::{HashMap, HashSet, VecDeque};

/// Cached visualization data for stepped Edmonds-Karp max-flow and `s-t` min-cut.
#[derive(Clone, Debug, Default)]
pub struct FlowVisualizationData {
    /// Total maximum `s -> t` flow value.
    pub max_flow: f64,
    /// Total capacity crossing the `s-t` minimum cut.
    pub min_cut_capacity: f64,
    /// Signed final flow per edge index.
    pub edge_flows: Vec<f64>,
    /// Original vertex IDs belonging to the source side `S` of the minimum cut.
    pub source_side_nodes: HashSet<usize>,
    /// Edge indices crossing between `S` and `T = V \ S`.
    pub cut_edge_indices: HashSet<usize>,
    /// Ordered augmenting-path steps (with node IDs mapped to tool node IDs).
    pub steps: Vec<AugmentingPathStep>,
    /// Cumulative flow per edge after each augmenting step `k` (`steps.len()` vectors).
    pub step_edge_flows: Vec<Vec<f64>>,
}

impl AlgorithmVisualizerTool {
    pub(super) fn run_dijkstra_solver(&mut self, start_id: usize, start_idx: NodeIndex) {
        if self.directed {
            self.run_directed_dijkstra(start_id);
            return;
        }
        let result = dijkstra(&self.graph.graph, start_idx);
        let idx_to_id: HashMap<NodeIndex, usize> = self
            .node_indices
            .iter()
            .map(|(&id, &ni)| (ni, id))
            .collect();
        let mut dist_vec = Vec::new();
        for (&id, &idx) in &self.node_indices {
            if let Some(&dist) = result.distances.get(&idx) {
                self.distances.insert(id, dist);
                dist_vec.push((id, dist));
            }
            if let Some(&pred_idx) = result.predecessors.get(&idx) {
                if let Some(&pred_id) = idx_to_id.get(&pred_idx) {
                    self.predecessors.insert(id, pred_id);
                }
            }
        }
        dist_vec.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        self.visit_order = dist_vec.into_iter().map(|(id, _)| id).collect();
        self.populate_tree_edge_steps();
    }

    fn run_directed_dijkstra(&mut self, start_id: usize) {
        let mut dig: DiGraph<(), f64> = DiGraph::new();
        let mut id_to_di = HashMap::new();
        let mut di_to_id = HashMap::new();
        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();
        for id in sorted_ids {
            let ni = dig.add_node(());
            id_to_di.insert(id, ni);
            di_to_id.insert(ni, id);
        }
        for &(u, v, w) in &self.edges {
            if let (Some(&ui), Some(&vi)) = (id_to_di.get(&u), id_to_di.get(&v)) {
                dig.add_edge(ui, vi, w.max(0.0));
            }
        }
        let Some(&s_di) = id_to_di.get(&start_id) else {
            return;
        };
        let result = dijkstra(&dig, s_di);
        let mut dist_vec = Vec::new();
        for (&id, &ni) in &id_to_di {
            if let Some(&dist) = result.distances.get(&ni) {
                self.distances.insert(id, dist);
                dist_vec.push((id, dist));
            }
            if let Some(&pred_ni) = result.predecessors.get(&ni) {
                if let Some(&pred_id) = di_to_id.get(&pred_ni) {
                    self.predecessors.insert(id, pred_id);
                }
            }
        }
        dist_vec.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        self.visit_order = dist_vec.into_iter().map(|(id, _)| id).collect();
        self.populate_tree_edge_steps();
    }

    pub(super) fn run_bfs_solver(&mut self, start_id: usize) {
        let adj = self.build_sorted_adjacency();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(start_id);
        queue.push_back(start_id);
        while let Some(u) = queue.pop_front() {
            self.visit_order.push(u);
            if let Some(nbrs) = adj.get(&u) {
                for &(v, _) in nbrs {
                    if visited.insert(v) {
                        self.predecessors.insert(v, u);
                        queue.push_back(v);
                    }
                }
            }
        }
        self.populate_tree_edge_steps();
    }

    pub(super) fn run_dfs_solver(&mut self, start_id: usize) {
        let adj = self.build_sorted_adjacency();
        let mut visited = HashSet::new();
        let mut stack = vec![(start_id, None::<usize>)];
        while let Some((u, parent)) = stack.pop() {
            if !visited.insert(u) {
                continue;
            }
            if let Some(p) = parent {
                self.predecessors.insert(u, p);
            }
            self.visit_order.push(u);
            if let Some(nbrs) = adj.get(&u) {
                for &(v, _) in nbrs.iter().rev() {
                    if !visited.contains(&v) {
                        stack.push((v, Some(u)));
                    }
                }
            }
        }
        self.populate_tree_edge_steps();
    }

    fn build_sorted_adjacency(&self) -> HashMap<usize, Vec<(usize, usize)>> {
        let mut adj: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (idx, &(u, v, _)) in self.edges.iter().enumerate() {
            if u == v {
                continue;
            }
            adj.entry(u).or_default().push((v, idx));
            if !self.directed {
                adj.entry(v).or_default().push((u, idx));
            }
        }
        for list in adj.values_mut() {
            list.sort_by_key(|&(nbr, idx)| (nbr, idx));
        }
        adj
    }

    pub(super) fn populate_tree_edge_steps(&mut self) {
        self.tree_edge_steps.clear();
        for (order_pos, &child) in self.visit_order.iter().enumerate() {
            let Some(&parent) = self.predecessors.get(&child) else {
                continue;
            };
            if let Some(edge_idx) = self.find_best_parent_edge(parent, child) {
                self.tree_edge_steps.insert(edge_idx, order_pos + 1);
            }
        }
    }

    fn find_best_parent_edge(&self, parent: usize, child: usize) -> Option<usize> {
        let expected_delta = match (self.distances.get(&parent), self.distances.get(&child)) {
            (Some(&dp), Some(&dc)) if self.selected_algorithm == Algorithm::Dijkstra => {
                Some(dc - dp)
            }
            _ => None,
        };
        let mut best: Option<(usize, f64)> = None;
        for (idx, &(u, v, w)) in self.edges.iter().enumerate() {
            let matches = if self.directed {
                u == parent && v == child
            } else {
                (u == parent && v == child) || (u == child && v == parent)
            };
            if !matches {
                continue;
            }
            let score = expected_delta.map_or(w, |delta| (w - delta).abs());
            if best.is_none_or(|(_, b_score)| score < b_score) {
                best = Some((idx, score));
            }
        }
        best.map(|(idx, _)| idx)
    }

    pub(super) fn run_max_flow_solver(&mut self, source_id: usize) {
        let Some(sink_id) = self.effective_sink_node(Some(source_id)) else {
            return;
        };
        let (sorted_ids, id_to_dense) = self.dense_node_mapping();
        let (Some(&s_dense), Some(&t_dense)) =
            (id_to_dense.get(&source_id), id_to_dense.get(&sink_id))
        else {
            return;
        };
        let dense_edges: Vec<(usize, usize, f64)> = self
            .edges
            .iter()
            .filter_map(|&(u, v, w)| {
                let &du = id_to_dense.get(&u)?;
                let &dv = id_to_dense.get(&v)?;
                Some((du, dv, w.max(0.0)))
            })
            .collect();
        if dense_edges.len() != self.edges.len() {
            return;
        }
        let Ok(res) = max_flow_min_cut(
            sorted_ids.len(),
            &dense_edges,
            s_dense,
            t_dense,
            self.directed,
        ) else {
            return;
        };
        self.store_flow_visualization(&sorted_ids, res);
    }

    fn store_flow_visualization(
        &mut self,
        sorted_ids: &[usize],
        res: math_explorer::pure_math::graph_theory::max_flow_min_cut::MaxFlowMinCutResult,
    ) {
        let mut source_side_nodes = HashSet::new();
        for (dense_idx, &in_s) in res.source_side_mask.iter().enumerate() {
            if in_s {
                source_side_nodes.insert(sorted_ids[dense_idx]);
            }
        }
        let mut running_flows = vec![0.0; self.edges.len()];
        let mut step_edge_flows = Vec::with_capacity(res.steps.len());
        let mut mapped_steps = Vec::with_capacity(res.steps.len());
        for step in res.steps {
            let mapped_nodes: Vec<usize> = step.nodes.iter().map(|&d| sorted_ids[d]).collect();
            for (hop, &edge_idx) in step.edge_indices.iter().enumerate() {
                if edge_idx < self.edges.len() && hop + 1 < mapped_nodes.len() {
                    let from = mapped_nodes[hop];
                    let sign = if self.edges[edge_idx].0 == from {
                        1.0
                    } else {
                        -1.0
                    };
                    running_flows[edge_idx] += sign * step.bottleneck;
                }
            }
            step_edge_flows.push(running_flows.clone());
            mapped_steps.push(AugmentingPathStep {
                nodes: mapped_nodes,
                edge_indices: step.edge_indices,
                bottleneck: step.bottleneck,
                cumulative_flow: step.cumulative_flow,
            });
        }
        self.flow_data = Some(FlowVisualizationData {
            max_flow: res.max_flow,
            min_cut_capacity: res.min_cut_capacity,
            edge_flows: res.edge_flows,
            source_side_nodes,
            cut_edge_indices: res.cut_edge_indices.into_iter().collect(),
            steps: mapped_steps,
            step_edge_flows,
        });
    }

    pub(super) fn run_mst_solver(&mut self) {
        let (sorted_ids, id_to_dense) = self.dense_node_mapping();
        if sorted_ids.is_empty() {
            return;
        }
        let dense_edges: Vec<(usize, usize, f64)> = self
            .edges
            .iter()
            .filter_map(|&(u, v, w)| {
                let &du = id_to_dense.get(&u)?;
                let &dv = id_to_dense.get(&v)?;
                Some((du, dv, w))
            })
            .collect();
        if dense_edges.len() != self.edges.len() {
            return;
        }
        if let Ok(mut mst) = minimum_spanning_tree(sorted_ids.len(), &dense_edges) {
            for step in &mut mst.steps {
                step.u = sorted_ids[step.u];
                step.v = sorted_ids[step.v];
            }
            self.mst_data = Some(mst);
        }
    }

    fn dense_node_mapping(&self) -> (Vec<usize>, HashMap<usize, usize>) {
        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();
        let id_to_dense: HashMap<usize, usize> = sorted_ids
            .iter()
            .enumerate()
            .map(|(dense, &id)| (id, dense))
            .collect();
        (sorted_ids, id_to_dense)
    }
}

/// Re-export `MstResult` for parent module field typing.
pub(super) type StoredMstData = MstResult;

// [cite:graph_parameters_rust]
