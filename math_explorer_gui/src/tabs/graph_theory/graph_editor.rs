#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::framework::{InteractionContext, InteractiveTool};
use eframe::egui;
use eframe::egui::Pos2;
use math_explorer::pure_math::graph_theory::{
    graph::Graph,
    layout::{circular_layout, force_directed_layout, grid_layout},
};
use petgraph::graph::NodeIndex;
use std::collections::{BTreeMap, HashMap};

pub mod braess;
mod braess_host;
mod keyboard;
mod snapshot;
pub use braess::{AffineLatency, BraessCongestionConfig, BraessCongestionLab};
#[allow(unused_imports)]
pub use keyboard::{TRANSLATE_STEP_NORMAL, TRANSLATE_STEP_SHIFT};
#[allow(unused_imports)]
pub use snapshot::{GraphSnapshot, SerializedGraphState, MAX_SNAPSHOTS};

#[derive(PartialEq, Default)]
pub(crate) enum EditorMode {
    #[default]
    AddNode,
    AddEdge,
    Remove,
}

/// Interactive graph editor with layouts and Braess congestion routing laboratory.
#[derive(Default)]
pub struct GraphEditorTool {
    graph: Graph<Pos2, f64>,
    node_indices: HashMap<usize, NodeIndex>,
    node_positions: BTreeMap<usize, Pos2>,
    selected_node: Option<usize>,
    next_node_id: usize,
    pub(crate) edges: Vec<(usize, usize, f64)>,
    dragged_node: Option<usize>,
    mode: EditorMode,
    hovered_node: Option<usize>,
    hovered_edge: Option<usize>,
    undo_stack: Vec<GraphSnapshot>,
    redo_stack: Vec<GraphSnapshot>,
    canvas_id: Option<egui::Id>,
    pub(crate) revision: u64,
    /// Hosted Braess's paradox congestion routing laboratory.
    pub braess: BraessCongestionLab,
}

impl InteractiveTool for GraphEditorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Graph Editor"
    }
    fn domain(&self) -> &'static str {
        "graph_theory"
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        if self.braess.config.enabled {
            crate::framework::ExportCapabilities::plot()
        } else {
            crate::framework::ExportCapabilities::none()
        }
    }

    fn state_revision(&self) -> u64 {
        self.compute_state_revision()
    }

    fn has_finite_parameters(&self) -> bool {
        self.check_finite_graph()
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_borrowed_state()
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.canvas_id
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.load_serialized_state(state)
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
        self.show_mode_instructions(ui);
        ui.separator();
        self.show_clear_button(ui);
        ui.separator();
        self.show_layout_toolbar(ui);
        ui.separator();
        ui.label(format!("Nodes: {}", self.node_positions.len()));
        ui.label(format!("Edges: {}", self.edges.len()));
        ui.separator();

        let action = self.braess.show_sidebar(ui, &mut self.edges);
        self.apply_braess_action(action);
    }

    fn on_keyboard(&mut self, ctx: &InteractionContext) {
        self.handle_keyboard(ctx);
    }

    fn on_hover(&mut self, ctx: &InteractionContext) {
        self.update_hover_targets(ctx.pointer_pos);
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
                        self.bump_revision();
                    }
                }
            }
            if ctx.response.drag_stopped() {
                self.dragged_node = None;
            }
        } else {
            self.dragged_node = None;
        }
    }

    fn on_click(&mut self, ctx: &InteractionContext) {
        if !ctx.is_clicked || self.dragged_node.is_some() {
            return;
        }
        let Some(pos) = ctx.pointer_pos else {
            return;
        };
        match self.mode {
            EditorMode::AddNode => self.click_add_node(pos),
            EditorMode::AddEdge => self.click_add_edge(),
            EditorMode::Remove => self.click_remove(),
        }
    }

    fn draw(&mut self, ui: &mut egui::Ui, response: &egui::Response, painter: &egui::Painter) {
        self.configure_canvas_focus(ui, response);
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let text_color = ui.visuals().text_color();
        let window_stroke = ui.visuals().window_stroke();

        self.draw_edges(painter, &palette, text_color);
        self.draw_nodes(painter, &palette, window_stroke);
        if self.braess.config.enabled {
            self.braess
                .draw_canvas_banner(painter, response.rect, &palette);
        }
    }
}

impl GraphEditorTool {
    /// Returns true if an undo snapshot is available.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }
    /// Returns true if a redo snapshot is available.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    fn show_mode_instructions(&mut self, ui: &mut egui::Ui) {
        match self.mode {
            EditorMode::AddNode => {
                ui.label("Click empty space to add a node.");
                ui.label("Drag nodes to move them.");
            }
            EditorMode::AddEdge => {
                ui.label("Click a node to select it, then another to add an edge.");
                ui.horizontal(|ui| {
                    ui.label("New edge:");
                    ui.add(
                        egui::DragValue::new(&mut self.braess.new_edge_a)
                            .speed(0.1)
                            .range(0.0..=20.0)
                            .prefix("a="),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.braess.new_edge_b)
                            .speed(0.1)
                            .range(0.0..=20.0)
                            .prefix("b="),
                    );
                });
            }
            EditorMode::Remove => {
                ui.label("Click a node or an edge to remove it.");
            }
        }
    }

    fn update_hover_targets(&mut self, pointer_pos: Option<Pos2>) {
        self.hovered_node = None;
        self.hovered_edge = None;
        let Some(pos) = pointer_pos else {
            return;
        };
        for (&id, &node_pos) in &self.node_positions {
            if (pos - node_pos).length() <= 15.0 {
                self.hovered_node = Some(id);
                return;
            }
        }
        for (i, &(u, v, _)) in self.edges.iter().enumerate() {
            if let (Some(&pos_u), Some(&pos_v)) =
                (self.node_positions.get(&u), self.node_positions.get(&v))
            {
                let dir = pos_v - pos_u;
                let len_sq = dir.length_sq();
                if len_sq > 0.0 {
                    let t = ((pos - pos_u).dot(dir) / len_sq).clamp(0.0, 1.0);
                    if (pos - (pos_u + dir * t)).length() <= 5.0 {
                        self.hovered_edge = Some(i);
                        break;
                    }
                }
            }
        }
    }

    fn click_add_node(&mut self, pos: Pos2) {
        if self.hovered_node.is_none() {
            self.push_undo_snapshot();
            let node_index = self.graph.add_node(pos);
            self.node_indices.insert(self.next_node_id, node_index);
            self.node_positions.insert(self.next_node_id, pos);
            self.next_node_id += 1;
        }
    }

    fn draw_edges(
        &self,
        painter: &egui::Painter,
        palette: &crate::app::SemanticPalette,
        text_color: egui::Color32,
    ) {
        for (i, &(u, v, weight)) in self.edges.iter().enumerate() {
            let (Some(&pos_u), Some(&pos_v)) =
                (self.node_positions.get(&u), self.node_positions.get(&v))
            else {
                continue;
            };
            if braess_host::should_draw_congestion_edge(
                self.braess.config.enabled,
                &self.mode,
                self.hovered_edge,
                i,
            ) {
                self.braess
                    .draw_congestion_edge(painter, palette, text_color, i, pos_u, pos_v);
                continue;
            }
            let color = if self.mode == EditorMode::Remove && self.hovered_edge == Some(i) {
                palette.error
            } else {
                palette.graph_edge
            };
            painter.line_segment([pos_u, pos_v], (2.0, color));
            let mid_point = pos_u + (pos_v - pos_u) * 0.5;
            painter.text(
                mid_point,
                egui::Align2::CENTER_CENTER,
                format!("{weight:.1}"),
                egui::FontId::proportional(12.0),
                text_color,
            );
        }
    }

    fn draw_nodes(
        &self,
        painter: &egui::Painter,
        palette: &crate::app::SemanticPalette,
        window_stroke: egui::Stroke,
    ) {
        for (&id, &pos) in &self.node_positions {
            let is_selected = self.selected_node == Some(id);
            let is_hovered = self.hovered_node == Some(id);
            let fill_color = if self.mode == EditorMode::Remove && is_hovered {
                palette.error
            } else if is_selected {
                palette.success
            } else {
                palette.series_color(0)
            };
            painter.circle(pos, 15.0, fill_color, window_stroke);
            let label = if self.braess.config.enabled && id == self.braess.config.source_node {
                format!("{id}:s")
            } else if self.braess.config.enabled && id == self.braess.config.sink_node {
                format!("{id}:t")
            } else {
                id.to_string()
            };
            painter.text(
                pos,
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(13.0),
                crate::app::contrasting_text_color(fill_color),
            );
        }
    }

    fn show_undo_redo_toolbar(&mut self, ui: &mut egui::Ui) {
        let m = self.register_keyboard_commands(ui.ctx());
        let bridge = crate::app::session::shortcuts::read_workspace_history_bridge(ui.ctx());
        let can_undo = if bridge.active {
            bridge.can_undo
        } else {
            self.can_undo()
        };
        let can_redo = if bridge.active {
            bridge.can_redo
        } else {
            self.can_redo()
        };
        ui.horizontal(|ui| {
            if ui
                .add_enabled(can_undo, egui::Button::new("↩ Undo"))
                .accessible_hover_text("Undo the last graph modification (Ctrl+Z / Cmd+Z)")
                .clicked()
            {
                if bridge.active {
                    crate::app::session::shortcuts::request_workspace_undo(ui.ctx());
                    crate::accessibility::announce_status("Undid graph modification");
                } else {
                    self.undo();
                }
            }
            if ui
                .add_enabled(can_redo, egui::Button::new("↪ Redo"))
                .accessible_hover_text(
                    "Redo the last undone graph modification (Ctrl+Y / Cmd+Shift+Z)",
                )
                .clicked()
            {
                if bridge.active {
                    crate::app::session::shortcuts::request_workspace_redo(ui.ctx());
                    crate::accessibility::announce_status("Redid graph modification");
                } else {
                    self.redo();
                }
            }
        });
        if !bridge.active {
            self.consume_toolbar_undo_redo(ui, m);
        }
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
            self.braess.edge_latencies.clear();
            self.selected_node = None;
            self.next_node_id = 0;
            if self.braess.config.enabled {
                self.recompute_braess();
            }
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

    pub(crate) fn rebuild_graph(&mut self) {
        self.graph = Graph::new();
        self.node_indices.clear();
        for (&id, &pos) in &self.node_positions {
            let idx = self.graph.add_node(pos);
            self.node_indices.insert(id, idx);
        }
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

#[cfg(test)]
mod tests;
