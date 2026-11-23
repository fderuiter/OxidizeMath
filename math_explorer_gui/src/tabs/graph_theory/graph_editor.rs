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
mod brushes;
mod bus_sync;
mod canvas;
mod keyboard;
mod snapshot;

pub use braess::{AffineLatency, BraessCongestionConfig, BraessCongestionLab};
#[allow(unused_imports)]
pub use keyboard::{TRANSLATE_STEP_NORMAL, TRANSLATE_STEP_SHIFT};
#[allow(unused_imports)]
pub use snapshot::{GraphSnapshot, SerializedGraphState, MAX_SNAPSHOTS};

#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub(crate) enum EditorMode {
    #[default]
    AddNode,
    AddEdge,
    ExcitatoryBrush,
    InhibitoryBrush,
    HubStamp,
    SeverStroke,
    Remove,
}

/// Interactive graph editor with tactile signed brushes and Braess congestion routing laboratory.
pub struct GraphEditorTool {
    graph: Graph<Pos2, f64>,
    node_indices: HashMap<usize, NodeIndex>,
    node_positions: BTreeMap<usize, Pos2>,
    selected_node: Option<usize>,
    next_node_id: usize,
    pub(crate) edges: Vec<(usize, usize, f64)>,
    dragged_node: Option<usize>,
    mode: EditorMode,
    brush_weight: f64,
    hub_spokes: usize,
    hub_inhibitory: bool,
    brush_trail: Vec<Pos2>,
    last_brushed_node: Option<usize>,
    brush_stroke_undo_pushed: bool,
    hovered_node: Option<usize>,
    hovered_edge: Option<usize>,
    undo_stack: Vec<GraphSnapshot>,
    redo_stack: Vec<GraphSnapshot>,
    canvas_id: Option<egui::Id>,
    pub(crate) revision: u64,
    pub(crate) last_published_revision: u64,
    /// Hosted Braess's paradox congestion routing laboratory.
    pub braess: BraessCongestionLab,
}

impl Default for GraphEditorTool {
    fn default() -> Self {
        Self {
            graph: Graph::new(),
            node_indices: HashMap::new(),
            node_positions: BTreeMap::new(),
            selected_node: None,
            next_node_id: 0,
            edges: Vec::new(),
            dragged_node: None,
            mode: EditorMode::AddNode,
            brush_weight: 1.0,
            hub_spokes: 5,
            hub_inhibitory: false,
            brush_trail: Vec::new(),
            last_brushed_node: None,
            brush_stroke_undo_pushed: false,
            hovered_node: None,
            hovered_edge: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            canvas_id: None,
            revision: 0,
            last_published_revision: 0,
            braess: BraessCongestionLab::default(),
        }
    }
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
        self.check_finite_graph() && self.brush_weight.is_finite()
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
        self.sync_workspace_bus(ui.ctx());
        ui.heading("Graph Editor");
        ui.separator();

        self.render_mode_and_brush_controls(ui);

        ui.separator();
        self.show_undo_redo_toolbar(ui);
        ui.separator();
        self.render_mode_instructions(ui);
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
        self.sync_workspace_bus(ui.ctx());
    }

    fn on_keyboard(&mut self, ctx: &InteractionContext) {
        self.handle_keyboard(ctx);
    }

    fn on_hover(&mut self, ctx: &InteractionContext) {
        self.hovered_node = None;
        self.hovered_edge = None;
        if let Some(pos) = ctx.pointer_pos {
            self.hovered_node = self.node_at_pos(pos, 15.0);
            if self.hovered_node.is_none() {
                self.hovered_edge = self.edge_at_pos(pos, 5.0);
            }
        }
    }

    fn on_drag(&mut self, ctx: &InteractionContext) {
        match self.mode {
            EditorMode::AddNode => {
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
            }
            EditorMode::ExcitatoryBrush | EditorMode::InhibitoryBrush | EditorMode::SeverStroke => {
                self.dragged_node = None;
                self.handle_brush_or_sever_drag(ctx);
            }
            _ => {
                self.dragged_node = None;
            }
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
            EditorMode::AddNode => {
                if self.hovered_node.is_none() {
                    self.push_undo_snapshot();
                    let node_index = self.graph.add_node(pos);
                    self.node_indices.insert(self.next_node_id, node_index);
                    self.node_positions.insert(self.next_node_id, pos);
                    self.next_node_id += 1;
                    self.bump_revision();
                }
            }
            EditorMode::AddEdge if self.braess.config.enabled => {
                self.click_add_edge();
            }
            EditorMode::AddEdge | EditorMode::ExcitatoryBrush | EditorMode::InhibitoryBrush => {
                self.handle_edge_click_selection();
            }
            EditorMode::HubStamp => {
                self.stamp_hub_at(pos);
            }
            EditorMode::Remove | EditorMode::SeverStroke => {
                self.click_remove();
            }
        }
    }

    fn draw(&mut self, ui: &mut egui::Ui, response: &egui::Response, painter: &egui::Painter) {
        self.draw_graph_canvas(ui, response, painter);
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

    fn handle_edge_click_selection(&mut self) {
        let Some(clicked_id) = self.hovered_node else {
            self.selected_node = None;
            return;
        };
        let Some(selected_id) = self.selected_node else {
            self.selected_node = Some(clicked_id);
            return;
        };
        if selected_id != clicked_id {
            let weight = self.active_signed_weight();
            let exists_same = self.edges.iter().any(|&(u, v, w)| {
                ((u == selected_id && v == clicked_id) || (v == selected_id && u == clicked_id))
                    && (w - weight).abs() < 1e-9
            });
            if !exists_same {
                self.push_undo_snapshot();
                let _ = self.upsert_signed_edge(selected_id, clicked_id, weight);
            }
        }
        self.selected_node = None;
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
            self.bump_revision();
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
        self.bump_revision();
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
mod brush_tests;
#[cfg(test)]
mod tests;
