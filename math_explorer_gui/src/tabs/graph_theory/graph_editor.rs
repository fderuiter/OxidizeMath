#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{InteractionContext, InteractiveTool};
use eframe::egui;
use eframe::egui::Pos2;
use math_explorer::pure_math::graph_theory::graph::Graph;
use petgraph::graph::NodeIndex;
use std::collections::{BTreeMap, HashMap};

pub mod braess;
mod braess_host;
mod bus_sync;
mod interaction;
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

/// Interactive graph editor with layouts, workspace bus coupling, and Braess congestion laboratory.
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
    last_published_local_rev: Option<u64>,
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

    fn sync_workspace_bus(&mut self, bus: &mut crate::app::workspace_bus::SharedWorkspaceBus) {
        self.publish_topology_to_bus(bus);
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
}

// [cite:graph_parameters_rust]

#[cfg(test)]
mod bus_tests;
#[cfg(test)]
mod tests;
