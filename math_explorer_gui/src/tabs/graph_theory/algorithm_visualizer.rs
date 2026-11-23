use crate::app::{SharedGraphTopology, SharedSimulationBus, TopologySource};
use crate::framework::InteractiveTool;
use eframe::egui;
use eframe::egui::Pos2;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::graph_theory::dijkstra::dijkstra;
use math_explorer::pure_math::graph_theory::graph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::visit::{Bfs, Dfs};
use std::collections::HashMap;

#[derive(PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum Algorithm {
    Dijkstra,
    Bfs,
    Dfs,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AlgorithmVisualizerState {
    selected_algorithm: Algorithm,
    start_node: Option<usize>,
    playback_speed: f32,
    #[serde(default, skip_serializing_if = "TopologySource::is_default")]
    topology_source: TopologySource,
}

pub struct AlgorithmVisualizerTool {
    graph: Graph<Pos2, f64>,
    node_indices: HashMap<usize, NodeIndex>,
    node_positions: HashMap<usize, Pos2>,
    edges: Vec<(usize, usize, f64)>,

    selected_algorithm: Algorithm,
    pub topology_source: TopologySource,
    synced_bus_revision: Option<u64>,
    start_node: Option<usize>,
    local_start_node: Option<usize>,
    focused_node: usize,

    // Animation state
    animation_step: usize,
    visit_order: Vec<usize>, // List of node IDs in the order they are visited
    distances: HashMap<usize, f64>, // For Dijkstra
    is_playing: bool,
    playback_speed: f32,
    step_timer: f32,
}

impl Default for AlgorithmVisualizerTool {
    fn default() -> Self {
        let mut tool = Self {
            graph: Graph::new(),
            node_indices: HashMap::new(),
            node_positions: HashMap::new(),
            edges: Vec::new(),
            selected_algorithm: Algorithm::Dijkstra,
            topology_source: TopologySource::SharedBus,
            synced_bus_revision: None,
            start_node: None,
            local_start_node: Some(0),
            focused_node: 0,
            animation_step: 0,
            visit_order: Vec::new(),
            distances: HashMap::new(),
            is_playing: false,
            playback_speed: 2.0,
            step_timer: 0.0,
        };
        tool.build_sample_graph();
        tool
    }
}

impl AlgorithmVisualizerTool {
    fn build_sample_graph(&mut self) {
        self.graph = Graph::new();
        self.node_indices.clear();
        self.node_positions.clear();
        self.edges.clear();
        // Create a 4x4 grid graph for visualization
        let rows = 4;
        let cols = 4;
        let spacing = 80.0;
        let offset = Pos2::new(100.0, 100.0);

        // We store relative positions from (0,0) and offset them during rendering
        for r in 0..rows {
            for c in 0..cols {
                let id = flatten_2d_index(c, r, cols);
                let pos = offset + eframe::egui::vec2(c as f32 * spacing, r as f32 * spacing);
                let idx = self.graph.add_node(pos);
                self.node_indices.insert(id, idx);
                self.node_positions.insert(id, pos);
            }
        }

        let mut add_edge = |u: usize, v: usize, w: f64| {
            self.edges.push((u, v, w));
            if let (Some(&u_idx), Some(&v_idx)) =
                (self.node_indices.get(&u), self.node_indices.get(&v))
            {
                self.graph.add_edge(u_idx, v_idx, w);
            }
        };

        // Add grid edges with arbitrary weights
        for r in 0..rows {
            for c in 0..cols {
                let u = flatten_2d_index(c, r, cols);
                if c < cols - 1 {
                    let v = u + 1;
                    add_edge(u, v, 1.0 + ((u + v) % 5) as f64);
                }
                if r < rows - 1 {
                    let v = u + cols;
                    add_edge(u, v, 1.0 + ((u * v) % 5) as f64);
                }
            }
        }

        // Restore local start node when reverting from SharedSimulationBus
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
        self.run_algorithm();
    }

    /// Synchronizes this tool's graph with `SharedSimulationBus` or restores the 4x4 sample grid.
    pub fn sync_shared_topology(&mut self, ctx: &egui::Context) {
        match SharedSimulationBus::active_graph_for(ctx, self.topology_source) {
            Some(snap) => {
                if self.synced_bus_revision != Some(snap.revision) {
                    self.apply_shared_graph(&snap);
                    self.synced_bus_revision = Some(snap.revision);
                }
            }
            None => {
                if self.synced_bus_revision.take().is_some() {
                    self.build_sample_graph();
                }
            }
        }
    }

    fn apply_shared_graph(&mut self, snap: &SharedGraphTopology) {
        if snap.node_count() == 0 {
            return;
        }
        self.graph = Graph::new();
        self.node_indices.clear();
        self.node_positions.clear();
        self.edges.clear();

        let mapped = snap.map_positions_to_bounds(70.0, 70.0, 380.0, 340.0);
        for (idx_in_snap, &orig_id) in snap.node_ids.iter().enumerate() {
            let [x, y] = mapped[idx_in_snap];
            let pos = Pos2::new(x, y);
            let g_idx = self.graph.add_node(pos);
            self.node_indices.insert(orig_id, g_idx);
            self.node_positions.insert(orig_id, pos);
        }
        for &(u, v, w) in &snap.undirected_edges {
            let u_id = snap.node_ids[u];
            let v_id = snap.node_ids[v];
            if let (Some(&u_idx), Some(&v_idx)) =
                (self.node_indices.get(&u_id), self.node_indices.get(&v_id))
            {
                let safe_w = w.abs().max(0.01);
                self.edges.push((u_id, v_id, safe_w));
                self.graph.add_edge(u_idx, v_idx, safe_w);
            }
        }
        if self
            .start_node
            .is_none_or(|id| !self.node_indices.contains_key(&id))
        {
            let first_id = snap.node_ids[0];
            self.start_node = Some(first_id);
            self.focused_node = first_id;
        }
        self.run_algorithm();
    }

    #[cfg(test)]
    pub fn synced_bus_revision(&self) -> Option<u64> {
        self.synced_bus_revision
    }
    #[cfg(test)]
    pub fn active_node_count(&self) -> usize {
        self.node_positions.len()
    }

    fn run_algorithm(&mut self) {
        self.visit_order.clear();
        self.distances.clear();
        self.animation_step = 0;
        self.is_playing = false;
        self.step_timer = 0.0;

        let start_id = match self.start_node {
            Some(id) => id,
            None => return,
        };

        let start_idx = match self.node_indices.get(&start_id) {
            Some(&idx) => idx,
            None => return,
        };

        match self.selected_algorithm {
            Algorithm::Dijkstra => {
                // Dijkstra's algorithm from math_explorer
                let result = dijkstra(&self.graph.graph, start_idx);

                // Map node indices back to our IDs and store distances
                let mut dist_vec = Vec::new();
                for (&id, &idx) in &self.node_indices {
                    if let Some(&dist) = result.distances.get(&idx) {
                        self.distances.insert(id, dist);
                        dist_vec.push((id, dist));
                    }
                }

                // Sort by distance to simulate the visit order
                dist_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                self.visit_order = dist_vec.into_iter().map(|(id, _)| id).collect();
            }
            Algorithm::Bfs => {
                let mut bfs = Bfs::new(&self.graph.graph, start_idx);
                while let Some(nx) = bfs.next(&self.graph.graph) {
                    if let Some((&id, _)) = self.node_indices.iter().find(|(_, &idx)| idx == nx) {
                        self.visit_order.push(id);
                    }
                }
            }
            Algorithm::Dfs => {
                let mut dfs = Dfs::new(&self.graph.graph, start_idx);
                while let Some(nx) = dfs.next(&self.graph.graph) {
                    if let Some((&id, _)) = self.node_indices.iter().find(|(_, &idx)| idx == nx) {
                        self.visit_order.push(id);
                    }
                }
            }
        }
    }
    fn update_animation(&mut self, ctx: &egui::Context) {
        if self.is_playing && !self.visit_order.is_empty() {
            let dt = ctx.input(|i| i.stable_dt);
            self.step_timer += dt;
            let step_interval = 1.0 / self.playback_speed;
            while self.step_timer >= step_interval && self.is_playing {
                self.step_timer -= step_interval;
                if self.animation_step < self.visit_order.len() {
                    self.animation_step += 1;
                }
                if self.animation_step >= self.visit_order.len() {
                    self.animation_step = self.visit_order.len();
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

#[path = "algorithm_visualizer/canvas.rs"]
mod canvas;

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
        crate::framework::ExportCapabilities::none()
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&AlgorithmVisualizerState {
            selected_algorithm: self.selected_algorithm,
            start_node: if self.synced_bus_revision.is_some() {
                self.local_start_node
            } else {
                self.start_node
            },
            playback_speed: self.playback_speed,
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
        if self.synced_bus_revision.take().is_some() {
            self.build_sample_graph();
        }
        if let Some(id) = loaded.start_node {
            if !self.node_indices.contains_key(&id) {
                return Err(format!("Unknown start_node id: {id}"));
            }
        }
        self.selected_algorithm = loaded.selected_algorithm;
        self.topology_source = loaded.topology_source;
        self.start_node = loaded.start_node;
        self.local_start_node = loaded.start_node;
        if let Some(id) = loaded.start_node {
            self.focused_node = id;
        }
        self.playback_speed = loaded.playback_speed;
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
        tags: &["dijkstra", "bfs", "dfs", "algorithm"],
        build: || Box::new(AlgorithmVisualizerTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for AlgorithmVisualizerTool {
    fn theory_description(&self) -> String {
        let algo_str = match self.selected_algorithm {
            Algorithm::Dijkstra => "Dijkstra shortest path algorithm via priority relaxation $d(v) = \\min(d(v), d(u) + w(u, v))$",
            Algorithm::Bfs => "Breadth-First Search level-order queue traversal computing unweighted shortest path distances",
            Algorithm::Dfs => "Depth-First Search recursive stack traversal exploring graph connectivity and cycles",
        };
        format!(
            "Graph traversal and pathfinding analysis executing {} across $|V| = {}$ nodes and $|E| = {}$ edges.",
            algo_str, self.node_positions.len(), self.edges.len()
        )
    }
    fn phonetic_description(&self) -> String {
        "Graph search and shortest path algorithm theory".into()
    }
    fn theory_citation(&self) -> String {
        "A Note on Two Problems in Connexion with Graphs (Dijkstra, 1959); Introduction to Algorithms (Cormen et al., 2022)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Dijkstra".into(),
            "Greedy shortest path in non-negatively weighted graphs".into(),
        );
        map.insert(
            "BFS".into(),
            "Queue-based breadth traversal finding minimum hop counts".into(),
        );
        map.insert(
            "DFS".into(),
            "Stack-based depth exploration identifying connected components".into(),
        );
        map
    }
}

#[cfg(test)]
#[path = "algorithm_visualizer_tests.rs"]
mod algorithm_visualizer_tests;

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Dijkstra Shortest Path",
    "domain": "graph_theory",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "selected_algorithm": "Dijkstra",
    "start_node": 5,
    "playback_speed": 3.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "algorithm_visualizer_starter",
        title: "Dijkstra Shortest Path",
        description: "Priority-queue shortest path relaxation across weighted undirected graph topologies.",
        category: "Graph Algorithms",
        icon: "🛤️",
        domain: "graph_theory",
        tool_name: Some("AlgorithmVisualizerTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
