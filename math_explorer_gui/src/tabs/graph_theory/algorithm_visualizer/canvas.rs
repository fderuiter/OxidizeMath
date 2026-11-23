#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{Algorithm, AlgorithmVisualizerTool};
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility, AccessibleHoverText,
};
use eframe::egui::{self, Pos2, Vec2};
use std::collections::HashMap;

fn next_node_in_direction(
    positions: &HashMap<usize, Pos2>,
    current: usize,
    dir: Vec2,
) -> Option<usize> {
    let &origin = positions.get(&current)?;
    positions
        .iter()
        .filter(|(&id, _)| id != current)
        .filter_map(|(&id, &pos)| {
            let delta = pos - origin;
            let forward = delta.dot(dir);
            if forward <= 1.0 {
                return None;
            }
            let lateral = (delta - dir * forward).length();
            if lateral > forward * 1.1 {
                return None;
            }
            let score = forward + lateral * 2.5;
            Some((id, score))
        })
        .min_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        })
        .map(|(id, _)| id)
}

impl AlgorithmVisualizerTool {
    pub(super) fn render_canvas(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::click());
            let _ = response
                .clone()
                .accessible_hover_text("Graph Theory Algorithm Visualization");
            let node_radius = 15.0;
            let to_screen = eframe::egui::emath::RectTransform::from_to(
                eframe::egui::Rect::from_min_size(Pos2::ZERO, response.rect.size()),
                response.rect,
            );

            self.handle_pointer_selection(ui.ctx(), &response, &to_screen, node_radius);
            if response.has_focus() {
                lock_canvas_arrow_keys(ui, &response);
                self.handle_keyboard_selection(ui);
                paint_canvas_focus_ring(ui, &response);
            }

            let summary = format!(
                "Focused Node {}, Start Node {}",
                self.focused_node,
                self.start_node
                    .map_or_else(|| "None".to_string(), |id| id.to_string())
            );
            register_canvas_accessibility(
                ui,
                &response,
                egui::accesskit::Role::GraphicsDocument,
                "Graph Theory Algorithm Visualization",
                Some(&summary),
            );

            let visuals = ui.visuals().clone();
            self.draw_edges(&painter, &to_screen, &visuals);
            self.draw_nodes(
                &painter,
                &to_screen,
                node_radius,
                &visuals,
                response.has_focus(),
            );
        });
    }

    fn handle_pointer_selection(
        &mut self,
        ctx: &egui::Context,
        response: &egui::Response,
        to_screen: &eframe::egui::emath::RectTransform,
        node_radius: f32,
    ) {
        if !response.clicked() {
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        let clicked_node = self.node_positions.iter().find_map(|(&id, &node_pos)| {
            let screen_pos = to_screen.transform_pos(node_pos);
            ((pos - screen_pos).length() <= node_radius).then_some(id)
        });
        if let Some(id) = clicked_node {
            self.focused_node = id;
            self.start_node = Some(id);
            self.run_algorithm();
            response.request_focus();
            announce_live_status(ctx, format!("Start node set to Node {id}"));
        }
    }

    fn handle_keyboard_selection(&mut self, ui: &egui::Ui) {
        let ctx = ui.ctx();
        let arrow_dir = ctx.input_mut(|i| {
            if i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp) {
                Some(Vec2::new(0.0, -1.0))
            } else if i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown) {
                Some(Vec2::new(0.0, 1.0))
            } else if i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft) {
                Some(Vec2::new(-1.0, 0.0))
            } else if i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight) {
                Some(Vec2::new(1.0, 0.0))
            } else {
                None
            }
        });
        if let Some(dir) = arrow_dir {
            ui.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
            if let Some(next_id) =
                next_node_in_direction(&self.node_positions, self.focused_node, dir)
            {
                self.focused_node = next_id;
            }
            announce_live_status(ctx, format!("Focused Node {}", self.focused_node));
        }

        let activate = ctx.input_mut(|i| {
            i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)
                || i.consume_key(egui::Modifiers::NONE, egui::Key::Space)
        });
        if activate {
            self.start_node = Some(self.focused_node);
            self.run_algorithm();
            announce_live_status(ctx, format!("Start node set to Node {}", self.focused_node));
        }
    }

    fn draw_edges(
        &self,
        painter: &egui::Painter,
        to_screen: &eframe::egui::emath::RectTransform,
        visuals: &egui::Visuals,
    ) {
        let palette = crate::app::SemanticPalette::from_visuals(visuals);
        for &(u, v, weight) in &self.edges {
            if let (Some(&pos_u), Some(&pos_v)) =
                (self.node_positions.get(&u), self.node_positions.get(&v))
            {
                let screen_pos_u = to_screen.transform_pos(pos_u);
                let screen_pos_v = to_screen.transform_pos(pos_v);
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
                    palette.success
                } else {
                    palette.graph_edge
                };
                painter.line_segment([screen_pos_u, screen_pos_v], (2.0, color));

                let mid_point = screen_pos_u + (screen_pos_v - screen_pos_u) * 0.5;
                painter.text(
                    mid_point,
                    egui::Align2::CENTER_CENTER,
                    format!("{:.1}", weight),
                    egui::FontId::proportional(12.0),
                    visuals.text_color(),
                );
            }
        }
    }

    fn draw_nodes(
        &self,
        painter: &egui::Painter,
        to_screen: &eframe::egui::emath::RectTransform,
        node_radius: f32,
        visuals: &egui::Visuals,
        canvas_focused: bool,
    ) {
        let palette = crate::app::SemanticPalette::from_visuals(visuals);
        for (&id, &pos) in &self.node_positions {
            let screen_pos = to_screen.transform_pos(pos);
            let is_start = self.start_node == Some(id);
            let visit_index = self.visit_order.iter().position(|&vid| vid == id);
            let is_visited = visit_index.is_some_and(|idx| idx < self.animation_step);
            let is_current = visit_index.is_some_and(|idx| {
                idx == self.animation_step.saturating_sub(1) && self.animation_step > 0
            });

            let fill_color = if is_start {
                palette.accent
            } else if is_current {
                palette.error
            } else if is_visited {
                palette.success
            } else {
                palette.series_color(0)
            };

            painter.circle(
                screen_pos,
                node_radius,
                fill_color,
                (1.0, palette.node_stroke),
            );
            if canvas_focused && self.focused_node == id {
                painter.circle_stroke(screen_pos, node_radius + 4.0_f32, (2.5_f32, palette.accent));
            }
            let label = if self.selected_algorithm == Algorithm::Dijkstra && is_visited {
                self.distances
                    .get(&id)
                    .map_or_else(|| id.to_string(), |dist| format!("{:.1}", dist))
            } else {
                id.to_string()
            };
            painter.text(
                screen_pos,
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(12.0),
                crate::app::contrasting_text_color(fill_color),
            );
        }
    }
}
