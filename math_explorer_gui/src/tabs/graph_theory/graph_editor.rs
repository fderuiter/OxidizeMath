#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{InteractionContext, InteractiveTool};
use eframe::egui;
use eframe::egui::Pos2;
use math_explorer::pure_math::graph_theory::graph::Graph;
use petgraph::graph::NodeIndex;
use std::collections::{BTreeMap, HashMap};

mod bus_sync;
mod interaction;
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

#[derive(Default)]
pub struct GraphEditorTool {
    graph: Graph<Pos2, f64>,
    node_indices: HashMap<usize, NodeIndex>,
    node_positions: BTreeMap<usize, Pos2>,
    selected_node: Option<usize>,
    next_node_id: usize,
    edges: Vec<(usize, usize, f64)>,
    dragged_node: Option<usize>,
    mode: EditorMode,
    hovered_node: Option<usize>,
    hovered_edge: Option<usize>,
    undo_stack: Vec<GraphSnapshot>,
    redo_stack: Vec<GraphSnapshot>,
    canvas_id: Option<egui::Id>,
    pub(crate) revision: u64,
    last_published_local_rev: Option<u64>,
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
        crate::framework::ExportCapabilities::none()
    }

    fn sync_workspace_bus(&mut self, bus: &mut crate::app::workspace_bus::SharedWorkspaceBus) {
        self.publish_topology_to_bus(bus);
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut h);
        self.node_positions.len().hash(&mut h);
        self.edges.len().hash(&mut h);
        self.selected_node.hash(&mut h);
        self.next_node_id.hash(&mut h);
        h.finish().max(1)
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
        if crate::app::workspace_bus::is_current_pane_linked() {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            ui.label(
                egui::RichText::new("🔗 Publishing topology to Workspace Bus")
                    .small()
                    .color(palette.success),
            );
        }
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
        self.handle_hover(ctx);
    }

    fn on_drag(&mut self, ctx: &InteractionContext) {
        self.handle_drag(ctx);
    }

    fn on_click(&mut self, ctx: &InteractionContext) {
        self.handle_click(ctx);
    }

    fn draw(&mut self, ui: &mut egui::Ui, response: &egui::Response, painter: &egui::Painter) {
        self.configure_canvas_focus(ui, response);
        self.sync_active_pane_bus();
        self.paint_graph_elements(ui, painter);
    }
}

impl GraphEditorTool {
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    fn paint_graph_elements(&self, ui: &egui::Ui, painter: &egui::Painter) {
        let node_radius = 15.0;
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let text_color = ui.visuals().text_color();
        let window_stroke = ui.visuals().window_stroke();

        for (i, &(u, v, weight)) in self.edges.iter().enumerate() {
            if let (Some(&pos_u), Some(&pos_v)) =
                (self.node_positions.get(&u), self.node_positions.get(&v))
            {
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
                    format!("{:.1}", weight),
                    egui::FontId::proportional(12.0),
                    text_color,
                );
            }
        }

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
            painter.circle(pos, node_radius, fill_color, window_stroke);
            painter.text(
                pos,
                egui::Align2::CENTER_CENTER,
                id.to_string(),
                egui::FontId::proportional(14.0),
                crate::app::contrasting_text_color(fill_color),
            );
        }
    }
}

// [cite:graph_parameters_rust]

#[cfg(test)]
mod bus_tests;
#[cfg(test)]
mod tests;
