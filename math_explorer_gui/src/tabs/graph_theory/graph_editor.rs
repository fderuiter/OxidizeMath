use crate::accessibility::AccessibleHoverText;
use crate::framework::{InteractionContext, InteractiveTool};
use eframe::egui;
use eframe::egui::Pos2;
use math_explorer::pure_math::graph_theory::{
    graph::Graph,
    layout::{circular_layout, force_directed_layout, grid_layout},
};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

#[derive(PartialEq, Default)]
enum EditorMode {
    #[default]
    AddNode,
    AddEdge,
    Remove,
}

// We use egui's Id for dragging and dropping nodes
#[derive(Default)]
pub struct GraphEditorTool {
    graph: Graph<Pos2, f64>,
    node_indices: HashMap<usize, NodeIndex>,
    node_positions: HashMap<usize, Pos2>,
    selected_node: Option<usize>,
    next_node_id: usize,
    edges: Vec<(usize, usize, f64)>, // We store edge representations here since graph wrapper hides index access nicely
    dragged_node: Option<usize>,
    mode: EditorMode,
    hovered_node: Option<usize>,
    hovered_edge: Option<usize>,
}

impl InteractiveTool for GraphEditorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Graph Editor"
    }

    fn show_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Graph Editor");
        ui.separator();

        ui.horizontal(|ui| {
            ui.radio_value(&mut self.mode, EditorMode::AddNode, "Add Node/Move");
            ui.radio_value(&mut self.mode, EditorMode::AddEdge, "Add Edge");
            ui.radio_value(&mut self.mode, EditorMode::Remove, "Remove");
        });

        ui.separator();

        match self.mode {
            EditorMode::AddNode => {
                ui.label("Click empty space to add a node.");
                ui.label("Drag nodes to move them.");
            }
            EditorMode::AddEdge => {
                ui.label("Click a node to select it.");
                ui.label("Click another node to add an edge.");
            }
            EditorMode::Remove => {
                ui.label("Click a node or an edge to remove it.");
            }
        }

        ui.separator();

        if ui
            .button("🔄 Clear Graph")
            .accessible_hover_text("Remove all nodes and edges and start with a fresh graph")
            .clicked()
        {
            self.graph = Graph::new();
            self.node_indices.clear();
            self.node_positions.clear();
            self.edges.clear();
            self.selected_node = None;
            self.next_node_id = 0;
        }

        ui.separator();
        self.show_layout_toolbar(ui);

        ui.separator();
        ui.label(format!("Nodes: {}", self.node_positions.len()));
        ui.label(format!("Edges: {}", self.edges.len()));
    }

    fn on_hover(&mut self, ctx: &InteractionContext) {
        let node_radius = 15.0;
        self.hovered_node = None;
        self.hovered_edge = None;

        if let Some(pos) = ctx.pointer_pos {
            for (&id, &node_pos) in &self.node_positions {
                if (pos - node_pos).length() <= node_radius {
                    self.hovered_node = Some(id);
                    break;
                }
            }

            // If not hovering over a node, check edges
            if self.hovered_node.is_none() {
                for (i, &(u, v, _weight)) in self.edges.iter().enumerate() {
                    if let (Some(&pos_u), Some(&pos_v)) =
                        (self.node_positions.get(&u), self.node_positions.get(&v))
                    {
                        // Distance from point to line segment
                        let dir = pos_v - pos_u;
                        let len_sq = dir.length_sq();
                        if len_sq > 0.0 {
                            let t = ((pos - pos_u).dot(dir) / len_sq).clamp(0.0, 1.0);
                            let proj = pos_u + dir * t;
                            if (pos - proj).length() <= 5.0 {
                                // Edge detection radius
                                self.hovered_edge = Some(i);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    fn on_drag(&mut self, ctx: &InteractionContext) {
        if self.mode == EditorMode::AddNode {
            if ctx.response.drag_started() {
                if let Some(id) = self.hovered_node {
                    self.dragged_node = Some(id);
                }
            }

            if ctx.response.dragged() {
                if let Some(dragged_id) = self.dragged_node {
                    if let Some(node_pos) = self.node_positions.get_mut(&dragged_id) {
                        *node_pos += ctx.delta;
                    }
                }
            }

            if ctx.response.drag_stopped() {
                self.dragged_node = None;
            }
        } else {
            self.dragged_node = None; // Reset dragging if mode changed
        }
    }

    fn on_click(&mut self, ctx: &InteractionContext) {
        if ctx.is_clicked && self.dragged_node.is_none() {
            if let Some(pos) = ctx.pointer_pos {
                match self.mode {
                    EditorMode::AddNode => {
                        if self.hovered_node.is_none() {
                            // Add Node
                            let node_index = self.graph.add_node(pos);
                            self.node_indices.insert(self.next_node_id, node_index);
                            self.node_positions.insert(self.next_node_id, pos);
                            self.next_node_id += 1;
                        }
                    }
                    EditorMode::AddEdge => {
                        if let Some(clicked_id) = self.hovered_node {
                            if let Some(selected_id) = self.selected_node {
                                if selected_id != clicked_id {
                                    // Check if edge already exists
                                    let edge_exists = self.edges.iter().any(|&(u, v, _)| {
                                        (u == selected_id && v == clicked_id)
                                            || (v == selected_id && u == clicked_id)
                                    });

                                    if !edge_exists {
                                        let weight = 1.0;
                                        self.edges.push((selected_id, clicked_id, weight));
                                        if let (Some(&u_idx), Some(&v_idx)) = (
                                            self.node_indices.get(&selected_id),
                                            self.node_indices.get(&clicked_id),
                                        ) {
                                            self.graph.add_edge(u_idx, v_idx, weight);
                                        }
                                    }
                                    self.selected_node = None;
                                } else {
                                    // Deselect
                                    self.selected_node = None;
                                }
                            } else {
                                // Select
                                self.selected_node = Some(clicked_id);
                            }
                        } else {
                            self.selected_node = None; // Clicked on empty space
                        }
                    }
                    EditorMode::Remove => {
                        if let Some(clicked_id) = self.hovered_node {
                            // Remove Node
                            self.node_positions.remove(&clicked_id);
                            self.node_indices.remove(&clicked_id);

                            // Remove edges associated with this node visually
                            self.edges
                                .retain(|&(u, v, _)| u != clicked_id && v != clicked_id);

                            // To remove from petgraph cleanly while keeping indices intact,
                            // we'd ideally rebuild the petgraph from scratch visually, since petgraph removes shift indices.
                            self.rebuild_graph();
                        } else if let Some(edge_idx) = self.hovered_edge {
                            // Remove Edge
                            self.edges.remove(edge_idx);
                            self.rebuild_graph();
                        }
                    }
                }
            }
        }
    }

    fn draw(&mut self, _ui: &mut egui::Ui, _response: &egui::Response, painter: &egui::Painter) {
        let node_radius = 15.0;

        // Draw edges
        for (i, &(u, v, weight)) in self.edges.iter().enumerate() {
            if let (Some(&pos_u), Some(&pos_v)) =
                (self.node_positions.get(&u), self.node_positions.get(&v))
            {
                let color = if self.mode == EditorMode::Remove && self.hovered_edge == Some(i) {
                    egui::Color32::RED
                } else {
                    egui::Color32::GRAY
                };

                painter.line_segment([pos_u, pos_v], (2.0, color));

                // Draw weight
                let mid_point = pos_u + (pos_v - pos_u) * 0.5;
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
            let is_selected = self.selected_node == Some(id);
            let is_hovered = self.hovered_node == Some(id);

            let fill_color = if self.mode == EditorMode::Remove && is_hovered {
                egui::Color32::from_rgb(255, 100, 100)
            } else if is_selected {
                egui::Color32::GREEN
            } else {
                egui::Color32::LIGHT_BLUE
            };

            painter.circle(pos, node_radius, fill_color, (1.0, egui::Color32::WHITE));

            painter.text(
                pos,
                egui::Align2::CENTER_CENTER,
                id.to_string(),
                egui::FontId::proportional(14.0),
                egui::Color32::BLACK,
            );
        }
    }
}

impl GraphEditorTool {
    fn show_layout_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.label("Auto Layout:");
        ui.horizontal(|ui| {
            let bounds = (100.0, 100.0, 700.0, 500.0);

            if ui
                .button("⭕ Circular")
                .accessible_hover_text("Arrange nodes in a circle")
                .clicked()
            {
                let layout = circular_layout(&self.graph, bounds);
                self.apply_layout(&layout);
            }

            if ui
                .button("▦ Grid")
                .accessible_hover_text("Arrange nodes in a grid layout")
                .clicked()
            {
                let layout = grid_layout(&self.graph, bounds);
                self.apply_layout(&layout);
            }

            if ui
                .button("🧲 Force-Directed")
                .accessible_hover_text("Arrange nodes using force-directed physics")
                .clicked()
            {
                let layout = force_directed_layout(&self.graph, bounds, 100);
                self.apply_layout(&layout);
            }
        });
    }

    fn apply_layout(&mut self, layout: &HashMap<NodeIndex, (f32, f32)>) {
        for (&id, &node_idx) in &self.node_indices {
            if let Some(&(x, y)) = layout.get(&node_idx) {
                self.node_positions.insert(id, Pos2::new(x, y));
            }
        }
        self.rebuild_graph();
    }

    fn rebuild_graph(&mut self) {
        self.graph = Graph::new();
        self.node_indices.clear();

        // Re-add nodes
        for (&id, &pos) in &self.node_positions {
            let idx = self.graph.add_node(pos);
            self.node_indices.insert(id, idx);
        }

        // Re-add edges
        for &(u, v, weight) in &self.edges {
            if let (Some(&u_idx), Some(&v_idx)) =
                (self.node_indices.get(&u), self.node_indices.get(&v))
            {
                self.graph.add_edge(u_idx, v_idx, weight);
            }
        }
    }
}

// [cite:graph_parameters_rust]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "GraphEditorTool",
        domain: "graph_theory",
        tags: &[],
        build: || Box::new(GraphEditorTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for GraphEditorTool {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}
