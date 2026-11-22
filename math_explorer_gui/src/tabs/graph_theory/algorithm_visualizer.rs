#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "algorithm_visualizer/bus_sync.rs"]
mod bus_sync;

use crate::accessibility::AccessibleHoverText;
use crate::app::data_bus::ui::render_follow_workspace_toggle;
use crate::app::data_bus::{SharedGraph, SESSION_MAX_BUS_EDGES, SESSION_MAX_BUS_NODES};
use crate::framework::InteractiveTool;
use eframe::egui;
use eframe::egui::Pos2;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::graph_theory::dijkstra::dijkstra;
use math_explorer::pure_math::graph_theory::graph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::visit::{Bfs, Dfs};
use std::collections::HashMap;
use std::sync::Arc;

fn default_true() -> bool {
    true
}

const fn is_true(value: &bool) -> bool {
    *value
}

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
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    follow_workspace_graph: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frozen_graph: Option<SharedGraph>,
}

pub struct AlgorithmVisualizerTool {
    graph: Graph<Pos2, f64>,
    node_indices: HashMap<usize, NodeIndex>,
    node_positions: HashMap<usize, Pos2>,
    edges: Vec<(usize, usize, f64)>,

    selected_algorithm: Algorithm,
    start_node: Option<usize>,
    focused_node: usize,

    animation_step: usize,
    visit_order: Vec<usize>,
    distances: HashMap<usize, f64>,
    is_playing: bool,
    playback_speed: f32,
    step_timer: f32,
    pub(crate) follow_workspace_graph: bool,
    pub(crate) applied_bus_revision: u64,
    active_shared_graph: Option<Arc<SharedGraph>>,
}

impl Default for AlgorithmVisualizerTool {
    fn default() -> Self {
        let mut tool = Self {
            graph: Graph::new(),
            node_indices: HashMap::new(),
            node_positions: HashMap::new(),
            edges: Vec::new(),
            selected_algorithm: Algorithm::Dijkstra,
            start_node: None,
            focused_node: 0,
            animation_step: 0,
            visit_order: Vec::new(),
            distances: HashMap::new(),
            is_playing: false,
            playback_speed: 2.0,
            step_timer: 0.0,
            follow_workspace_graph: true,
            applied_bus_revision: 0,
            active_shared_graph: None,
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

        let rows = 4;
        let cols = 4;
        let spacing = 80.0;
        let offset = Pos2::new(100.0, 100.0);

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

        self.start_node = Some(0);
        self.run_algorithm();
    }

    pub(super) fn run_algorithm(&mut self) {
        self.visit_order.clear();
        self.distances.clear();
        self.animation_step = 0;
        self.is_playing = false;
        self.step_timer = 0.0;

        let Some(start_id) = self.start_node else {
            return;
        };
        let Some(&start_idx) = self.node_indices.get(&start_id) else {
            return;
        };

        match self.selected_algorithm {
            Algorithm::Dijkstra => {
                let result = dijkstra(&self.graph.graph, start_idx);
                let mut dist_vec = Vec::new();
                for (&id, &idx) in &self.node_indices {
                    if let Some(&dist) = result.distances.get(&idx) {
                        self.distances.insert(id, dist);
                        dist_vec.push((id, dist));
                    }
                }
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

    fn render_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id =
            crate::framework::scoped_panel_id(parent_ui, "algorithm_visualizer_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Algorithm Visualizer");
            ui.separator();

            let rev_opt = (self.applied_bus_revision > 0).then_some(self.applied_bus_revision);
            if render_follow_workspace_toggle(
                ui,
                &mut self.follow_workspace_graph,
                "Algorithm Visualizer",
                rev_opt,
            ) && self.follow_workspace_graph
            {
                self.applied_bus_revision = 0;
                self.sync_from_bus(ui.ctx());
            }
            ui.separator();

            ui.label("Select Algorithm:");
            let mut changed = false;
            changed |= ui
                .radio_value(&mut self.selected_algorithm, Algorithm::Dijkstra, "Dijkstra")
                .clicked();
            changed |= ui
                .radio_value(&mut self.selected_algorithm, Algorithm::Bfs, "BFS")
                .clicked();
            changed |= ui
                .radio_value(&mut self.selected_algorithm, Algorithm::Dfs, "DFS")
                .clicked();
            if changed {
                self.run_algorithm();
            }
            ui.separator();
            self.render_start_node_selector(ui);
            ui.separator();
            self.render_playback_controls(ui);
            ui.separator();
            ui.label("Click a node or use Arrow keys + Enter on the canvas to set the start node.");
        });
    }

    fn render_playback_buttons(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .button("↻ Reset")
                .accessible_hover_text("Restart the algorithm visualization")
                .clicked()
            {
                self.animation_step = 0;
                self.is_playing = false;
                self.step_timer = 0.0_f32;
            }
            let (lbl, hov) = if self.is_playing {
                ("⏸ Pause", "Pause the algorithm visualization")
            } else {
                ("▶ Play", "Start or resume continuous algorithm visualization")
            };
            if ui.button(lbl).accessible_hover_text(hov).clicked() {
                if self.is_playing {
                    self.is_playing = false;
                } else {
                    if self.animation_step >= self.visit_order.len() {
                        self.animation_step = 0;
                    }
                    self.is_playing = true;
                }
                self.step_timer = 0.0_f32;
            }
            let can_step = self.animation_step < self.visit_order.len();
            if ui
                .add_enabled(can_step, egui::Button::new("▶ Step"))
                .accessible_hover_text("Advance visualization by one step")
                .clicked()
            {
                self.animation_step += 1;
                self.is_playing = false;
                self.step_timer = 0.0_f32;
            }
            if ui
                .add_enabled(can_step, egui::Button::new("⏹ Finish"))
                .accessible_hover_text("Skip to the end of the visualization")
                .clicked()
            {
                self.animation_step = self.visit_order.len();
                self.is_playing = false;
                self.step_timer = 0.0_f32;
            }
        });
    }

    fn render_playback_controls(&mut self, ui: &mut egui::Ui) {
        if self.visit_order.is_empty() {
            return;
        }
        ui.label(format!(
            "Animation Step: {} / {}",
            self.animation_step,
            self.visit_order.len()
        ));
        self.render_playback_buttons(ui);

        let slider =
            egui::Slider::new(&mut self.animation_step, 0..=self.visit_order.len()).text("Step");
        if ui.add(slider).changed() {
            self.step_timer = 0.0_f32;
        }
        ui.add(
            egui::Slider::new(&mut self.playback_speed, 0.5_f32..=5.0_f32)
                .text("Speed")
                .suffix("x"),
        );
    }

    fn render_start_node_selector(&mut self, ui: &mut egui::Ui) {
        let mut node_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        node_ids.sort_unstable();
        let selected_text = self
            .start_node
            .map_or_else(|| "None".to_string(), |id| format!("Node {id}"));
        ui.horizontal(|ui| {
            ui.label("Start Node:");
            egui::ComboBox::from_id_salt("algorithm_visualizer_start_node")
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    for id in node_ids {
                        let is_selected = self.start_node == Some(id);
                        if ui.selectable_label(is_selected, format!("Node {id}")).clicked() {
                            self.start_node = Some(id);
                            self.focused_node = id;
                            self.run_algorithm();
                            crate::accessibility::announce_live_status(
                                ui.ctx(),
                                format!("Start node set to Node {id}"),
                            );
                        }
                    }
                });
        });
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

    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::none()
    }

    fn save_state(&self) -> Option<String> {
        let frozen_graph = (!self.follow_workspace_graph)
            .then(|| self.active_shared_graph.as_deref().cloned())
            .flatten();
        serde_json::to_string(&AlgorithmVisualizerState {
            selected_algorithm: self.selected_algorithm,
            start_node: self.start_node,
            playback_speed: self.playback_speed,
            follow_workspace_graph: self.follow_workspace_graph,
            frozen_graph,
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
        if let Some(ref g) = loaded.frozen_graph {
            g.validate(SESSION_MAX_BUS_NODES, SESSION_MAX_BUS_EDGES)?;
        }
        if let Some(g) = loaded.frozen_graph {
            self.apply_shared_graph(Arc::new(g));
        }
        if let Some(id) = loaded.start_node {
            if !loaded.follow_workspace_graph && !self.node_indices.contains_key(&id) {
                return Err(format!("Unknown start_node id: {id}"));
            }
            if id >= SESSION_MAX_BUS_NODES {
                return Err(format!("start_node id out of bounds: {id}"));
            }
        }
        self.selected_algorithm = loaded.selected_algorithm;
        self.follow_workspace_graph = loaded.follow_workspace_graph;
        self.start_node = loaded.start_node;
        if let Some(id) = loaded.start_node {
            self.focused_node = id;
        }
        self.playback_speed = loaded.playback_speed;
        if self.follow_workspace_graph {
            self.applied_bus_revision = 0;
        }
        self.run_algorithm();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.sync_from_bus(&ctx);
        self.update_animation(&ctx);
        self.render_controls(ui);
        self.sync_from_bus(&ctx);
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
            "Graph traversal and pathfinding analysis executing {algo_str} across $|V| = {}$ nodes and $|E| = {}$ edges.",
            self.node_positions.len(),
            self.edges.len()
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
