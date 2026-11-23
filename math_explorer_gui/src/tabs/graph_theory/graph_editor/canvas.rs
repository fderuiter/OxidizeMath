//! WCAG 2.2 AA canvas and sidebar rendering for [`super::GraphEditorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{EditorMode, GraphEditorTool};
use crate::framework::bounded_drag_value;
use eframe::egui::{self, Pos2};

impl GraphEditorTool {
    pub(super) fn render_mode_and_brush_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Interaction & Tactile Brushes:");
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.mode, EditorMode::AddNode, "Add/Move");
            ui.radio_value(&mut self.mode, EditorMode::AddEdge, "Add Edge");
            ui.radio_value(&mut self.mode, EditorMode::ExcitatoryBrush, "+w Brush");
            ui.radio_value(&mut self.mode, EditorMode::InhibitoryBrush, "-w Brush");
            ui.radio_value(&mut self.mode, EditorMode::HubStamp, "Hub Stamp");
            ui.radio_value(&mut self.mode, EditorMode::SeverStroke, "Sever Stroke");
            ui.radio_value(&mut self.mode, EditorMode::Remove, "Remove");
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Weight |w|:");
            let _ = ui.add(bounded_drag_value(&mut self.brush_weight, 0.1..=10.0, 0.1));
            if self.mode == EditorMode::HubStamp {
                ui.separator();
                ui.add(egui::Slider::new(&mut self.hub_spokes, 3..=8).text("Spokes"));
                ui.checkbox(&mut self.hub_inhibitory, "Inhibitory (-w)");
            }
        });
    }

    pub(super) fn render_mode_instructions(&mut self, ui: &mut egui::Ui) {
        match self.mode {
            EditorMode::AddNode => {
                ui.label("Click empty space to add a node. Drag nodes to move them.");
            }
            EditorMode::AddEdge => {
                if self.braess.config.enabled {
                    ui.label("Click a node to select it, then another to add a directed edge.");
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
                } else {
                    ui.label("Click two nodes to connect them with weight +|w|.");
                }
            }
            EditorMode::ExcitatoryBrush => {
                ui.label("Drag across nodes or edges to paint excitatory (+w) couplings.");
            }
            EditorMode::InhibitoryBrush => {
                ui.label("Drag across nodes or edges to paint inhibitory (-w) couplings.");
            }
            EditorMode::HubStamp => {
                ui.label("Click canvas to stamp a central hub linked to radial spokes.");
            }
            EditorMode::SeverStroke => {
                ui.label("Drag a cut stroke across edges to sever intersecting connections.");
            }
            EditorMode::Remove => {
                ui.label("Click a node or an edge to remove it.");
            }
        }
    }

    fn draw_graph_edges(
        &self,
        painter: &egui::Painter,
        palette: &crate::app::SemanticPalette,
        text_color: egui::Color32,
    ) {
        for (i, &(u, v, weight)) in self.edges.iter().enumerate() {
            if let (Some(&pos_u), Some(&pos_v)) =
                (self.node_positions.get(&u), self.node_positions.get(&v))
            {
                if super::braess_host::should_draw_congestion_edge(
                    self.braess.config.enabled,
                    &self.mode,
                    self.hovered_edge,
                    i,
                ) {
                    self.braess
                        .draw_congestion_edge(painter, palette, text_color, i, pos_u, pos_v);
                    continue;
                }
                let is_cut_hover =
                    matches!(self.mode, EditorMode::Remove | EditorMode::SeverStroke)
                        && self.hovered_edge == Some(i);
                let color = if is_cut_hover || weight < 0.0 {
                    palette.error
                } else {
                    palette.graph_edge
                };
                let stroke_width = (2.0_f32
                    + (weight.abs() as f32 - 1.0_f32).clamp(0.0_f32, 2.0_f32) * 0.4_f32)
                    .max(2.0_f32);
                painter.line_segment([pos_u, pos_v], (stroke_width, color));

                let mid_point = pos_u + (pos_v - pos_u) * 0.5_f32;
                painter.text(
                    mid_point,
                    egui::Align2::CENTER_CENTER,
                    format!("{weight:.1}"),
                    egui::FontId::proportional(12.0_f32),
                    text_color,
                );
            }
        }
    }

    fn draw_graph_nodes(
        &self,
        painter: &egui::Painter,
        palette: &crate::app::SemanticPalette,
        window_stroke: egui::Stroke,
    ) {
        let node_radius = 15.0_f32;
        for (&id, &pos) in &self.node_positions {
            let is_selected = self.selected_node == Some(id) || self.last_brushed_node == Some(id);
            let is_hovered = self.hovered_node == Some(id);

            let fill_color = if self.mode == EditorMode::Remove && is_hovered {
                palette.error
            } else if is_selected {
                palette.success
            } else {
                palette.series_color(0)
            };

            painter.circle(pos, node_radius, fill_color, window_stroke);
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
                egui::FontId::proportional(14.0_f32),
                crate::app::contrasting_text_color(fill_color),
            );
        }
    }

    pub(super) fn draw_graph_canvas(
        &mut self,
        ui: &mut egui::Ui,
        response: &egui::Response,
        painter: &egui::Painter,
    ) {
        self.sync_workspace_bus(ui.ctx());
        self.configure_canvas_focus(ui, response);
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let text_color = ui.visuals().text_color();
        let window_stroke = ui.visuals().window_stroke();

        self.draw_graph_edges(painter, &palette, text_color);

        if self.brush_trail.len() >= 2 {
            let trail_color = match self.mode {
                EditorMode::InhibitoryBrush | EditorMode::SeverStroke => palette.error,
                _ => palette.success,
            };
            painter.add(egui::Shape::line(
                self.brush_trail.clone(),
                egui::Stroke::new(2.5_f32, trail_color),
            ));
        }

        self.draw_graph_nodes(painter, &palette, window_stroke);
        if self.braess.config.enabled {
            self.braess
                .draw_canvas_banner(painter, response.rect, &palette);
        }
    }

    #[must_use]
    pub(super) fn node_at_pos(&self, pos: Pos2, radius: f32) -> Option<usize> {
        self.node_positions
            .iter()
            .find_map(|(&id, &node_pos)| ((pos - node_pos).length() <= radius).then_some(id))
    }

    #[must_use]
    pub(super) fn edge_at_pos(&self, pos: Pos2, radius: f32) -> Option<usize> {
        for (i, &(u, v, _)) in self.edges.iter().enumerate() {
            if let (Some(&pos_u), Some(&pos_v)) =
                (self.node_positions.get(&u), self.node_positions.get(&v))
            {
                let dir = pos_v - pos_u;
                let len_sq = dir.length_sq();
                if len_sq > 0.0 {
                    let t = ((pos - pos_u).dot(dir) / len_sq).clamp(0.0, 1.0);
                    let proj = pos_u + dir * t;
                    if (pos - proj).length() <= radius {
                        return Some(i);
                    }
                }
            }
        }
        None
    }
}
