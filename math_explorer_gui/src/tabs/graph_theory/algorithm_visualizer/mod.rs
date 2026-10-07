#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[cfg(test)]
mod tests;

use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use eframe::egui::Pos2;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::graph_theory::dijkstra::dijkstra;
use math_explorer::pure_math::graph_theory::graph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::visit::{Bfs, Dfs};
use std::collections::HashMap;

#[derive(PartialEq)]
enum Algorithm {
    Dijkstra,
    Bfs,
    Dfs,
}

pub struct AlgorithmVisualizerTool {
    graph: Graph<Pos2, f64>,
    node_indices: HashMap<usize, NodeIndex>,
    node_positions: HashMap<usize, Pos2>,
    edges: Vec<(usize, usize, f64)>,

    selected_algorithm: Algorithm,
    start_node: Option<usize>,

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
            start_node: None,
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

        // Set default start node
        self.start_node = Some(0);
        self.run_algorithm();
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
}

impl InteractiveTool for AlgorithmVisualizerTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Algorithm Visualizer"
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
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

        egui::SidePanel::left("algorithm_visualizer_controls").show(ctx, |ui| {
            ui.heading("Algorithm Visualizer");
            ui.separator();

            ui.label("Select Algorithm:");
            let mut changed = false;
            if ui
                .radio_value(
                    &mut self.selected_algorithm,
                    Algorithm::Dijkstra,
                    "Dijkstra",
                )
                .clicked()
            {
                changed = true;
            }
            if ui
                .radio_value(&mut self.selected_algorithm, Algorithm::Bfs, "BFS")
                .clicked()
            {
                changed = true;
            }
            if ui
                .radio_value(&mut self.selected_algorithm, Algorithm::Dfs, "DFS")
                .clicked()
            {
                changed = true;
            }

            if changed {
                self.run_algorithm();
            }

            ui.separator();

            if !self.visit_order.is_empty() {
                ui.label(format!(
                    "Animation Step: {} / {}",
                    self.animation_step,
                    self.visit_order.len()
                ));

                ui.horizontal(|ui| {
                    if ui
                        .button("↻ Reset")
                        .accessible_hover_text("Restart the algorithm visualization")
                        .clicked()
                    {
                        self.animation_step = 0;
                        self.is_playing = false;
                        self.step_timer = 0.0;
                    }

                    let play_pause_label = if self.is_playing { "⏸ Pause" } else { "▶ Play" };
                    let play_pause_hover = if self.is_playing {
                        "Pause the algorithm visualization"
                    } else {
                        "Start or resume continuous algorithm visualization"
                    };
                    if ui
                        .button(play_pause_label)
                        .accessible_hover_text(play_pause_hover)
                        .clicked()
                    {
                        if self.is_playing {
                            self.is_playing = false;
                            self.step_timer = 0.0;
                        } else {
                            if self.animation_step >= self.visit_order.len() {
                                self.animation_step = 0;
                            }
                            self.is_playing = true;
                            self.step_timer = 0.0;
                        }
                    }

                    let can_step = self.animation_step < self.visit_order.len();
                    if ui
                        .add_enabled(can_step, eframe::egui::Button::new("▶ Step"))
                        .accessible_hover_text("Advance visualization by one step")
                        .clicked()
                    {
                        self.animation_step += 1;
                        self.is_playing = false;
                        self.step_timer = 0.0;
                    }
                    if ui
                        .add_enabled(can_step, eframe::egui::Button::new("⏹ Finish"))
                        .accessible_hover_text("Skip to the end of the visualization")
                        .clicked()
                    {
                        self.animation_step = self.visit_order.len();
                        self.is_playing = false;
                        self.step_timer = 0.0;
                    }
                });

                let slider =
                    egui::Slider::new(&mut self.animation_step, 0..=self.visit_order.len())
                        .text("Step");
                if ui.add(slider).changed() {
                    self.step_timer = 0.0;
                }

                ui.add(
                    egui::Slider::new(&mut self.playback_speed, 0.5..=5.0)
                        .text("Speed")
                        .suffix("x"),
                );
            }

            ui.separator();
            ui.label("Click a node to set as start node.");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::click());
            let _ = response
                .clone()
                .accessible_hover_text("Graph Theory Algorithm Visualization");
            let pointer_pos = response.interact_pointer_pos();
            let node_radius = 15.0;

            let to_screen = eframe::egui::emath::RectTransform::from_to(
                eframe::egui::Rect::from_min_size(Pos2::ZERO, response.rect.size()),
                response.rect,
            );

            let mut clicked_node = None;
            if response.clicked() {
                if let Some(pos) = pointer_pos {
                    for (&id, &node_pos) in &self.node_positions {
                        let screen_pos = to_screen.transform_pos(node_pos);
                        if (pos - screen_pos).length() <= node_radius {
                            clicked_node = Some(id);
                            break;
                        }
                    }
                }
            }

            if let Some(id) = clicked_node {
                self.start_node = Some(id);
                self.run_algorithm();
            }

            // Draw edges
            for &(u, v, weight) in &self.edges {
                if let (Some(&pos_u), Some(&pos_v)) =
                    (self.node_positions.get(&u), self.node_positions.get(&v))
                {
                    let screen_pos_u = to_screen.transform_pos(pos_u);
                    let screen_pos_v = to_screen.transform_pos(pos_v);

                    // Check if both nodes have been visited
                    let u_visited = self
                        .visit_order
                        .iter()
                        .take(self.animation_step)
                        .any(|&id| id == u);
                    let v_visited = self
                        .visit_order
                        .iter()
                        .take(self.animation_step)
                        .any(|&id| id == v);

                    let color = if u_visited && v_visited {
                        egui::Color32::from_rgb(100, 200, 100) // Visited edge
                    } else {
                        egui::Color32::GRAY
                    };

                    painter.line_segment([screen_pos_u, screen_pos_v], (2.0, color));

                    // Draw weight
                    let mid_point = screen_pos_u + (screen_pos_v - screen_pos_u) * 0.5;
                    painter.text(
                        mid_point,
                        egui::Align2::CENTER_CENTER,
                        format!("{:.1}", weight),
                        egui::FontId::proportional(12.0),
                        egui::Color32::WHITE,
                    );
                }
            }

            // Draw nodes
            for (&id, &pos) in &self.node_positions {
                let screen_pos = to_screen.transform_pos(pos);
                let is_start = self.start_node == Some(id);

                // Determine visitation status based on animation step
                let visit_index = self.visit_order.iter().position(|&vid| vid == id);
                let is_visited = visit_index.is_some_and(|idx| idx < self.animation_step);
                let is_current = visit_index.is_some_and(|idx| {
                    idx == self.animation_step.saturating_sub(1) && self.animation_step > 0
                });

                let fill_color = if is_start {
                    egui::Color32::YELLOW
                } else if is_current {
                    egui::Color32::RED
                } else if is_visited {
                    egui::Color32::GREEN
                } else {
                    egui::Color32::LIGHT_BLUE
                };

                painter.circle(
                    screen_pos,
                    node_radius,
                    fill_color,
                    (1.0, egui::Color32::WHITE),
                );

                let label = if self.selected_algorithm == Algorithm::Dijkstra && is_visited {
                    if let Some(dist) = self.distances.get(&id) {
                        format!("{:.1}", dist)
                    } else {
                        id.to_string()
                    }
                } else {
                    id.to_string()
                };

                painter.text(
                    screen_pos,
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::proportional(12.0),
                    egui::Color32::BLACK,
                );
            }
        });
    }
}

// [cite:graph_parameters_rust]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "AlgorithmVisualizerTool",
        domain: "graph_theory",
        tags: &[],
        build: || Box::new(AlgorithmVisualizerTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for AlgorithmVisualizerTool {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}
