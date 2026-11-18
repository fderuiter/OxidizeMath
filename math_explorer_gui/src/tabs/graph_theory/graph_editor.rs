#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::framework::{InteractionContext, InteractiveTool};
use eframe::egui;
use eframe::egui::Pos2;
use egui_plot::commands::{CommandRegistryData, CommandTrigger};
use math_explorer::pure_math::graph_theory::{
    graph::Graph,
    layout::{circular_layout, force_directed_layout, grid_layout},
};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

mod keyboard;
mod snapshot;
#[allow(unused_imports)]
pub use keyboard::{TRANSLATE_STEP_NORMAL, TRANSLATE_STEP_SHIFT};
#[allow(unused_imports)]
pub use snapshot::{GraphSnapshot, SerializedGraphState, MAX_SNAPSHOTS};

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
    undo_stack: Vec<GraphSnapshot>,
    redo_stack: Vec<GraphSnapshot>,
}

impl InteractiveTool for GraphEditorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Graph Editor"
    }

    fn domain(&self) -> &'static str {
        "graph_theory"
    }

    fn save_state(&self) -> Option<String> {
        let node_pos = self
            .node_positions
            .iter()
            .map(|(&id, &p)| (id, (p.x, p.y)))
            .collect();
        serde_json::to_string(&SerializedGraphState {
            node_positions: node_pos,
            edges: self.edges.clone(),
            selected_node: self.selected_node,
            next_node_id: self.next_node_id,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(loaded) = serde_json::from_str::<SerializedGraphState>(state) {
            self.node_positions = loaded
                .node_positions
                .into_iter()
                .map(|(id, (x, y))| (id, Pos2::new(x, y)))
                .collect();
            self.edges = loaded.edges;
            self.selected_node = loaded.selected_node;
            self.next_node_id = loaded.next_node_id;
            self.undo_stack.clear();
            self.redo_stack.clear();
            self.rebuild_graph();
        }
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
        self.show_undo_redo_toolbar(ui);

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
        self.show_clear_button(ui);

        ui.separator();
        self.show_layout_toolbar(ui);

        ui.separator();
        ui.label(format!("Nodes: {}", self.node_positions.len()));
        ui.label(format!("Edges: {}", self.edges.len()));
    }

    fn on_keyboard(&mut self, ctx: &InteractionContext) {
        self.handle_keyboard(ctx);
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
                    self.push_undo_snapshot();
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
                            self.push_undo_snapshot();
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
                                        self.push_undo_snapshot();
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
                            self.push_undo_snapshot();
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
                            self.push_undo_snapshot();
                            self.edges.remove(edge_idx);
                            self.rebuild_graph();
                        }
                    }
                }
            }
        }
    }

    fn draw(&mut self, ui: &mut egui::Ui, response: &egui::Response, painter: &egui::Painter) {
        self.configure_canvas_focus(ui, response);
        let node_radius = 15.0;
        let text_color = ui.visuals().text_color();
        let window_stroke = ui.visuals().window_stroke();

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
                    text_color,
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

            painter.circle(pos, node_radius, fill_color, window_stroke);

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
    pub fn can_undo(&self) -> bool { !self.undo_stack.is_empty() }
    pub fn can_redo(&self) -> bool { !self.redo_stack.is_empty() }

    fn show_undo_redo_toolbar(&mut self, ui: &mut egui::Ui) {
        self.register_keyboard_commands(ui.ctx());
        let m = if cfg!(target_os = "macos") { egui::Modifiers::MAC_CMD } else { egui::Modifiers::CTRL };
        CommandRegistryData::register(ui.ctx(), "Undo", "Undo the last graph modification", CommandTrigger::Shortcut(m, egui::Key::Z), true, "Graph Editor");
        CommandRegistryData::register(ui.ctx(), "Redo", "Redo the last undone graph modification", CommandTrigger::Shortcut(m | egui::Modifiers::SHIFT, egui::Key::Z), true, "Graph Editor");
        if !cfg!(target_os = "macos") {
            CommandRegistryData::register(ui.ctx(), "Redo", "Redo the last undone graph modification", CommandTrigger::Shortcut(m, egui::Key::Y), true, "Graph Editor");
        }
        ui.horizontal(|ui| {
            if ui
                .add_enabled(self.can_undo(), egui::Button::new("↩ Undo"))
                .accessible_hover_text("Undo the last graph modification (Ctrl+Z / Cmd+Z)")
                .clicked()
            {
                self.undo();
            }

            if ui
                .add_enabled(self.can_redo(), egui::Button::new("↪ Redo"))
                .accessible_hover_text("Redo the last undone graph modification (Ctrl+Y / Cmd+Shift+Z)")
                .clicked()
            {
                self.redo();
            }
        });

        ui.input_mut(|i| {
            let ctrl_or_cmd = i.modifiers.command || i.modifiers.ctrl || i.modifiers.mac_cmd;
            if ctrl_or_cmd && i.key_pressed(egui::Key::Z) {
                if i.modifiers.shift {
                    if self.can_redo() {
                        self.redo();
                    }
                } else if self.can_undo() {
                    self.undo();
                }
            } else if ctrl_or_cmd && i.key_pressed(egui::Key::Y) && self.can_redo() {
                self.redo();
            }
        });
    }

    fn show_clear_button(&mut self, ui: &mut egui::Ui) {
        if ui
            .button("🔄 Clear Graph")
            .accessible_hover_text("Remove all nodes and edges and start with a fresh graph")
            .clicked()
            && (!self.node_positions.is_empty() || !self.edges.is_empty())
        {
            self.push_undo_snapshot();
            self.graph = Graph::new();
            self.node_indices.clear();
            self.node_positions.clear();
            self.edges.clear();
            self.selected_node = None;
            self.next_node_id = 0;
        }
    }

    fn show_layout_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.label("Auto Layout:");
        ui.horizontal(|ui| {
            let bounds = (100.0, 100.0, 700.0, 500.0);

            if ui
                .button("⭕ Circular")
                .accessible_hover_text("Arrange nodes in a circle")
                .clicked()
                && !self.node_positions.is_empty()
            {
                self.push_undo_snapshot();
                let layout = circular_layout(&self.graph, bounds);
                self.apply_layout(&layout);
            }

            if ui
                .button("▦ Grid")
                .accessible_hover_text("Arrange nodes in a grid layout")
                .clicked()
                && !self.node_positions.is_empty()
            {
                self.push_undo_snapshot();
                let layout = grid_layout(&self.graph, bounds);
                self.apply_layout(&layout);
            }

            if ui
                .button("🧲 Force-Directed")
                .accessible_hover_text("Arrange nodes using force-directed physics")
                .clicked()
                && !self.node_positions.is_empty()
            {
                self.push_undo_snapshot();
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
        for (&id, &pos) in &self.node_positions {
            let idx = self.graph.add_node(pos);
            self.node_indices.insert(id, idx);
        }
        for &(u, v, weight) in &self.edges {
            if let (Some(&u_idx), Some(&v_idx)) = (self.node_indices.get(&u), self.node_indices.get(&v)) {
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
    fn theory_description(&self) -> String {
        format!(
            "Combinatorial graph structure $G = (V, E)$ with $|V| = {}$ nodes and $|E| = {}$ edges under force-directed spring-electrical embedding $F = F_{{\\text{{attr}}}} + F_{{\\text{{rep}}}}.$",
            self.node_positions.len(), self.edges.len()
        )
    }
    fn phonetic_description(&self) -> String { "Combinatorial graph theory and force-directed network embedding".into() }
    fn theory_citation(&self) -> String { "Graph Drawing by Force-Directed Placement (Fruchterman & Reingold, 1991)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

#[cfg(test)]
mod tests;

