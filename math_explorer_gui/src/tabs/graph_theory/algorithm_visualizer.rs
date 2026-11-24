#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::shared_topology::CompactTopologyState;
use crate::app::TopologySource;
use crate::framework::InteractiveTool;
use eframe::egui;
use eframe::egui::Pos2;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::graph_theory::graph::Graph;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

#[path = "algorithm_visualizer/canvas.rs"]
mod canvas;
#[path = "algorithm_visualizer/controls.rs"]
mod controls;
#[path = "algorithm_visualizer/io_export.rs"]
mod io_export;
#[path = "algorithm_visualizer/solvers.rs"]
mod solvers;

pub use io_export::visualizer_state_json_from_shared;
pub(crate) use solvers::FlowVisualizationData;

/// Stepped graph traversal, shortest-path, network-flow, and spanning-tree modes.
#[derive(Debug, PartialEq, Eq, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum Algorithm {
    /// Priority-queue single-source shortest-path relaxation (`DijkstraResult::predecessors`).
    Dijkstra,
    /// Level-order Breadth-First Search discovery tree.
    Bfs,
    /// Stack-based Depth-First Search discovery tree.
    Dfs,
    /// Stepped Edmonds-Karp `s-t` maximum flow and terminal minimum-cut partition `(S, T)`.
    MaxFlowMinCut,
    /// Stepped Kruskal union-find minimum spanning tree/forest.
    MinimumSpanningTree,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AlgorithmVisualizerState {
    pub(crate) selected_algorithm: Algorithm,
    pub(crate) start_node: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) sink_node: Option<usize>,
    pub(crate) playback_speed: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) custom_topology: Option<CompactTopologyState>,
    #[serde(default, skip_serializing_if = "TopologySource::is_default")]
    pub(crate) topology_source: TopologySource,
}

/// Interactive stepped visualizer for shortest-path predecessor trees, `s-t` min-cuts, and MSTs.
pub struct AlgorithmVisualizerTool {
    graph: Graph<Pos2, f64>,
    node_indices: HashMap<usize, NodeIndex>,
    node_positions: HashMap<usize, Pos2>,
    edges: Vec<(usize, usize, f64)>,
    directed: bool,
    is_custom_topology: bool,

    selected_algorithm: Algorithm,
    pub topology_source: TopologySource,
    synced_bus_revision: Option<u64>,
    start_node: Option<usize>,
    local_start_node: Option<usize>,
    sink_node: Option<usize>,
    focused_node: usize,

    animation_step: usize,
    visit_order: Vec<usize>,
    distances: HashMap<usize, f64>,
    predecessors: HashMap<usize, usize>,
    tree_edge_steps: HashMap<usize, usize>,
    flow_data: Option<FlowVisualizationData>,
    mst_data: Option<solvers::StoredMstData>,

    is_playing: bool,
    playback_speed: f32,
    step_timer: f32,
    revision: u64,
    file_io: super::io_helpers::GraphFileIoState,
}

impl Default for AlgorithmVisualizerTool {
    fn default() -> Self {
        let mut tool = Self {
            graph: Graph::new(),
            node_indices: HashMap::new(),
            node_positions: HashMap::new(),
            edges: Vec::new(),
            directed: false,
            is_custom_topology: false,
            selected_algorithm: Algorithm::Dijkstra,
            topology_source: TopologySource::SharedBus,
            synced_bus_revision: None,
            start_node: None,
            local_start_node: Some(0),
            sink_node: None,
            focused_node: 0,
            animation_step: 0,
            visit_order: Vec::new(),
            distances: HashMap::new(),
            predecessors: HashMap::new(),
            tree_edge_steps: HashMap::new(),
            flow_data: None,
            mst_data: None,
            is_playing: false,
            playback_speed: 2.0,
            step_timer: 0.0,
            revision: 1,
            file_io: super::io_helpers::GraphFileIoState::default(),
        };
        tool.build_sample_graph();
        tool
    }
}

impl AlgorithmVisualizerTool {
    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    pub(crate) fn effective_sink_node(&self, source: Option<usize>) -> Option<usize> {
        if let Some(t) = self.sink_node {
            if self.node_positions.contains_key(&t) && Some(t) != source {
                return Some(t);
            }
        }
        self.node_positions
            .keys()
            .copied()
            .filter(|&id| Some(id) != source)
            .max()
    }

    /// Total discrete stepper frames available for the active algorithm.
    #[must_use]
    pub fn total_steps(&self) -> usize {
        match self.selected_algorithm {
            Algorithm::Dijkstra | Algorithm::Bfs | Algorithm::Dfs => self.visit_order.len(),
            Algorithm::MaxFlowMinCut => self.flow_data.as_ref().map_or(0, |f| f.steps.len() + 1),
            Algorithm::MinimumSpanningTree => self.mst_data.as_ref().map_or(0, |m| m.steps.len()),
        }
    }

    /// Switches the active algorithm mode and recomputes the stepper trace.
    #[cfg(test)]
    pub fn set_algorithm(&mut self, algorithm: Algorithm) {
        self.selected_algorithm = algorithm;
        self.bump_revision();
        self.run_algorithm();
    }

    /// Sets the current animation step (`0..=total_steps()`).
    #[cfg(test)]
    pub fn set_animation_step(&mut self, step: usize) {
        self.animation_step = step.min(self.total_steps());
    }

    /// Returns true if `edge_idx` belongs to the reconstructed predecessor/spanning tree.
    #[cfg(test)]
    #[must_use]
    pub fn is_tree_edge(&self, edge_idx: usize) -> bool {
        match self.selected_algorithm {
            Algorithm::Dijkstra | Algorithm::Bfs | Algorithm::Dfs => {
                self.tree_edge_steps.contains_key(&edge_idx)
            }
            Algorithm::MinimumSpanningTree => self
                .mst_data
                .as_ref()
                .is_some_and(|m| m.tree_edge_indices.contains(&edge_idx)),
            Algorithm::MaxFlowMinCut => self
                .flow_data
                .as_ref()
                .is_some_and(|f| f.cut_edge_indices.contains(&edge_idx)),
        }
    }

    /// Returns true if `edge_idx` is highlighted as a settled tree edge at the current `animation_step`.
    #[must_use]
    pub fn is_edge_highlighted_at_step(&self, edge_idx: usize) -> bool {
        match self.selected_algorithm {
            Algorithm::Dijkstra | Algorithm::Bfs | Algorithm::Dfs => self
                .tree_edge_steps
                .get(&edge_idx)
                .is_some_and(|&step| step <= self.animation_step),
            Algorithm::MinimumSpanningTree => self.mst_data.as_ref().is_some_and(|m| {
                let shown = self.animation_step.min(m.steps.len());
                m.steps[..shown]
                    .iter()
                    .any(|s| s.edge_index == edge_idx && s.accepted)
            }),
            Algorithm::MaxFlowMinCut => self.flow_data.as_ref().is_some_and(|f| {
                self.animation_step > f.steps.len() && f.cut_edge_indices.contains(&edge_idx)
            }),
        }
    }

    /// Counts how many predecessor/MST tree edges are highlighted at `animation_step`.
    #[cfg(test)]
    #[must_use]
    pub fn highlighted_tree_edge_count(&self) -> usize {
        (0..self.edges.len())
            .filter(|&i| self.is_tree_edge(i) && self.is_edge_highlighted_at_step(i))
            .count()
    }

    /// Counts how many non-tree edges are falsely highlighted at `animation_step` (always `0`).
    #[cfg(test)]
    #[must_use]
    pub fn highlighted_non_tree_edge_count(&self) -> usize {
        (0..self.edges.len())
            .filter(|&i| !self.is_tree_edge(i) && self.is_edge_highlighted_at_step(i))
            .count()
    }

    /// Returns computed max-flow visualization data when in `MaxFlowMinCut` mode.
    #[cfg(test)]
    #[must_use]
    pub fn flow_visualization(&self) -> Option<&FlowVisualizationData> {
        self.flow_data.as_ref()
    }

    fn build_sample_graph(&mut self) {
        self.graph = Graph::new();
        self.node_indices.clear();
        self.node_positions.clear();
        self.edges.clear();
        self.directed = false;
        self.is_custom_topology = false;
        let rows = 4;
        let cols = 4;
        let spacing = 80.0;
        let offset = Pos2::new(100.0, 100.0);
        for r in 0..rows {
            for c in 0..cols {
                let id = flatten_2d_index(c, r, cols);
                let pos = offset + egui::vec2(c as f32 * spacing, r as f32 * spacing);
                let idx = self.graph.add_node(pos);
                self.node_indices.insert(id, idx);
                self.node_positions.insert(id, pos);
            }
        }
        for r in 0..rows {
            for c in 0..cols {
                let u = flatten_2d_index(c, r, cols);
                if c < cols - 1 {
                    let v = u + 1;
                    let w = 1.0 + ((u + v) % 5) as f64;
                    self.edges.push((u, v, w));
                    self.graph
                        .add_edge(self.node_indices[&u], self.node_indices[&v], w);
                }
                if r < rows - 1 {
                    let v = u + cols;
                    let w = 1.0 + ((u * v) % 5) as f64;
                    self.edges.push((u, v, w));
                    self.graph
                        .add_edge(self.node_indices[&u], self.node_indices[&v], w);
                }
            }
        }
        if self.synced_bus_revision.is_none() {
            self.start_node = self.local_start_node;
        }
        if self
            .start_node
            .is_none_or(|id| !self.node_indices.contains_key(&id))
        {
            self.start_node = Some(0);
            self.local_start_node = Some(0);
            self.focused_node = 0;
        } else if let Some(id) = self.start_node {
            self.focused_node = id;
        }
        self.sink_node = Some(15);
        if self.sink_node == self.start_node {
            self.sink_node = self.effective_sink_node(self.start_node);
        }
        self.run_algorithm();
    }

    fn run_algorithm(&mut self) {
        self.visit_order.clear();
        self.distances.clear();
        self.predecessors.clear();
        self.tree_edge_steps.clear();
        self.flow_data = None;
        self.mst_data = None;
        self.animation_step = 0;
        self.is_playing = false;
        self.step_timer = 0.0;

        if self.selected_algorithm == Algorithm::MinimumSpanningTree {
            self.run_mst_solver();
            return;
        }
        let Some(start_id) = self.start_node else {
            return;
        };
        let Some(&start_idx) = self.node_indices.get(&start_id) else {
            return;
        };
        match self.selected_algorithm {
            Algorithm::Dijkstra => self.run_dijkstra_solver(start_id, start_idx),
            Algorithm::Bfs => self.run_bfs_solver(start_id),
            Algorithm::Dfs => self.run_dfs_solver(start_id),
            Algorithm::MaxFlowMinCut => self.run_max_flow_solver(start_id),
            Algorithm::MinimumSpanningTree => {}
        }
    }

    fn update_animation(&mut self, ctx: &egui::Context) {
        let total = self.total_steps();
        if self.is_playing && total > 0 {
            let dt = ctx.input(|i| i.stable_dt);
            self.step_timer += dt;
            let step_interval = 1.0 / self.playback_speed;
            while self.step_timer >= step_interval && self.is_playing {
                self.step_timer -= step_interval;
                if self.animation_step < total {
                    self.animation_step += 1;
                }
                if self.animation_step >= total {
                    self.animation_step = total;
                    self.is_playing = false;
                    self.step_timer = 0.0;
                    break;
                }
            }
            if self.is_playing {
                ctx.request_repaint();
            }
        }
    }
}

impl InteractiveTool for AlgorithmVisualizerTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Algorithm Visualizer"
    }
    fn domain(&self) -> &'static str {
        "graph_theory"
    }
    fn observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("node_count", self.node_positions.len() as f64),
            ("edge_count", self.edges.len() as f64),
            ("visited_count", self.visit_order.len() as f64),
            (
                "synced_bus_revision",
                self.synced_bus_revision.unwrap_or(0) as f64,
            ),
        ]
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::plot()
    }
    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.build_visualizer_plot_primitives()
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_visualizer_datasets()
    }
    fn step_simulation(&mut self) {
        let total = self.total_steps();
        if total > 0 {
            self.animation_step = (self.animation_step + 1) % (total + 1);
        }
    }
    fn state_revision(&self) -> u64 {
        self.revision.max(1)
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn save_state(&self) -> Option<String> {
        let on_bus = self.synced_bus_revision.is_some();
        let include_sink = !on_bus
            && (self.selected_algorithm == Algorithm::MaxFlowMinCut || self.is_custom_topology);
        let custom_topology = (!on_bus && self.is_custom_topology)
            .then(|| self.export_shared_topology().to_compact_state());
        serde_json::to_string(&AlgorithmVisualizerState {
            selected_algorithm: self.selected_algorithm,
            start_node: if on_bus {
                self.local_start_node
            } else {
                self.start_node
            },
            sink_node: include_sink
                .then(|| self.effective_sink_node(self.start_node))
                .flatten(),
            playback_speed: self.playback_speed,
            custom_topology,
            topology_source: self.topology_source,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let loaded: AlgorithmVisualizerState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid AlgorithmVisualizer state: {e}"))?;
        if !loaded.playback_speed.is_finite() || !(0.5..=5.0).contains(&loaded.playback_speed) {
            return Err(format!(
                "playback_speed out of [0.5, 5.0]: {}",
                loaded.playback_speed
            ));
        }
        if let Some(compact) = &loaded.custom_topology {
            compact.validate()?;
            self.synced_bus_revision = None;
            self.load_compact_topology(compact);
        } else if self.is_custom_topology || self.synced_bus_revision.take().is_some() {
            self.build_sample_graph();
        }
        if let Some(id) = loaded.start_node {
            if !self.node_indices.contains_key(&id) {
                return Err(format!("Unknown start_node id: {id}"));
            }
        }
        if let Some(sink) = loaded.sink_node {
            if !self.node_indices.contains_key(&sink) || Some(sink) == loaded.start_node {
                return Err(format!("Invalid sink_node id: {sink}"));
            }
        }
        self.selected_algorithm = loaded.selected_algorithm;
        self.topology_source = loaded.topology_source;
        self.start_node = loaded.start_node;
        self.local_start_node = loaded.start_node;
        if let Some(id) = loaded.start_node {
            self.focused_node = id;
        }
        self.sink_node = loaded
            .sink_node
            .or_else(|| self.effective_sink_node(self.start_node));
        self.playback_speed = loaded.playback_speed;
        self.bump_revision();
        self.run_algorithm();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.sync_shared_topology(&ctx);
        self.update_animation(&ctx);
        self.render_controls(ui);
        self.render_canvas(ui);
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "AlgorithmVisualizerTool",
        display_name: "Algorithm Visualizer",
        domain: "graph_theory",
        tags: &["dijkstra", "bfs", "dfs", "max-flow", "min-cut", "mst", "kruskal", "algorithm"],
        build: || Box::new(AlgorithmVisualizerTool::default()),
    }
}

#[cfg(test)]
#[path = "algorithm_visualizer_tests.rs"]
mod algorithm_visualizer_tests;
