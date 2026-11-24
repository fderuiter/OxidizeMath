#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{Algorithm, AlgorithmVisualizerTool};
use crate::accessibility::{
    AccessibleHoverText, announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility,
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
        .filter(|&(&id, _)| id != current)
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
                "Focused Node {}, Start Node {}, Step {}/{}",
                self.focused_node,
                self.start_node
                    .map_or_else(|| "None".to_string(), |id| id.to_string()),
                self.animation_step,
                self.total_steps()
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
            self.draw_mode_banner(&painter, response.rect, &visuals);
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
            let shift = ctx.input(|i| i.modifiers.shift);
            if shift
                && self.selected_algorithm == Algorithm::MaxFlowMinCut
                && Some(id) != self.start_node
            {
                self.sink_node = Some(id);
                self.bump_revision();
                self.run_algorithm();
                response.request_focus();
                announce_live_status(ctx, format!("Sink node set to Node {id}"));
            } else {
                self.start_node = Some(id);
                if self.synced_bus_revision.is_none() {
                    self.local_start_node = Some(id);
                }
                if self.sink_node == Some(id) {
                    self.sink_node = self.effective_sink_node(Some(id));
                }
                self.bump_revision();
                self.run_algorithm();
                response.request_focus();
                announce_live_status(ctx, format!("Start node set to Node {id}"));
            }
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
            if self.synced_bus_revision.is_none() {
                self.local_start_node = Some(self.focused_node);
            }
            if self.sink_node == Some(self.focused_node) {
                self.sink_node = self.effective_sink_node(self.start_node);
            }
            self.bump_revision();
            self.run_algorithm();
            announce_live_status(ctx, format!("Start node set to Node {}", self.focused_node));
        }
    }

    pub(super) fn classify_edge_visual(
        &self,
        edge_idx: usize,
        palette: &crate::app::SemanticPalette,
    ) -> (egui::Color32, f32, String) {
        let (_, _, weight) = self.edges[edge_idx];
        match self.selected_algorithm {
            Algorithm::Dijkstra | Algorithm::Bfs | Algorithm::Dfs => {
                if self.is_edge_highlighted_at_step(edge_idx) {
                    (palette.success, 3.2, format!("{weight:.1}"))
                } else {
                    (palette.graph_edge, 2.0, format!("{weight:.1}"))
                }
            }
            Algorithm::MaxFlowMinCut => self.classify_flow_edge_visual(edge_idx, weight, palette),
            Algorithm::MinimumSpanningTree => {
                self.classify_mst_edge_visual(edge_idx, weight, palette)
            }
        }
    }

    fn classify_flow_edge_visual(
        &self,
        edge_idx: usize,
        cap: f64,
        palette: &crate::app::SemanticPalette,
    ) -> (egui::Color32, f32, String) {
        let Some(flow_data) = &self.flow_data else {
            return (palette.graph_edge, 2.0, format!("{cap:.1}"));
        };
        if self.animation_step == 0 {
            return (palette.graph_edge, 2.0, format!("0.0/{cap:.1}"));
        }
        let cut_step = flow_data.steps.len() + 1;
        if self.animation_step >= cut_step {
            let f = flow_data
                .edge_flows
                .get(edge_idx)
                .copied()
                .unwrap_or(0.0)
                .abs();
            if flow_data.cut_edge_indices.contains(&edge_idx) {
                return (palette.error, 4.0, format!("CUT {f:.1}/{cap:.1}"));
            }
            let col = if f > 1e-6 {
                palette.success
            } else {
                palette.graph_edge
            };
            return (
                col,
                if f > 1e-6 { 3.0 } else { 1.8 },
                format!("{f:.1}/{cap:.1}"),
            );
        }
        let step_idx = self.animation_step - 1;
        let f = flow_data
            .step_edge_flows
            .get(step_idx)
            .and_then(|v| v.get(edge_idx))
            .copied()
            .unwrap_or(0.0)
            .abs();
        let on_active_aug = flow_data.steps[step_idx].edge_indices.contains(&edge_idx);
        let col = if on_active_aug {
            palette.accent
        } else if f > 1e-6 {
            palette.success
        } else {
            palette.graph_edge
        };
        let width = if on_active_aug {
            3.8
        } else if f > 1e-6 {
            2.8
        } else {
            1.8
        };
        (col, width, format!("{f:.1}/{cap:.1}"))
    }

    fn classify_mst_edge_visual(
        &self,
        edge_idx: usize,
        weight: f64,
        palette: &crate::app::SemanticPalette,
    ) -> (egui::Color32, f32, String) {
        let Some(mst) = &self.mst_data else {
            return (palette.graph_edge, 2.0, format!("{weight:.1}"));
        };
        let shown = self.animation_step.min(mst.steps.len());
        if shown > 0 && mst.steps[shown - 1].edge_index == edge_idx {
            let col = if mst.steps[shown - 1].accepted {
                palette.accent
            } else {
                palette.error
            };
            return (col, 3.8, format!("{weight:.1}"));
        }
        let accepted = mst.steps[..shown]
            .iter()
            .any(|s| s.edge_index == edge_idx && s.accepted);
        if accepted {
            (palette.success, 3.2, format!("{weight:.1}"))
        } else {
            (palette.graph_edge, 1.8, format!("{weight:.1}"))
        }
    }

    fn draw_edges(
        &self,
        painter: &egui::Painter,
        to_screen: &eframe::egui::emath::RectTransform,
        visuals: &egui::Visuals,
    ) {
        let palette = crate::app::SemanticPalette::from_visuals(visuals);
        for (idx, &(u, v, _)) in self.edges.iter().enumerate() {
            let (Some(&pos_u), Some(&pos_v)) =
                (self.node_positions.get(&u), self.node_positions.get(&v))
            else {
                continue;
            };
            let screen_u = to_screen.transform_pos(pos_u);
            let screen_v = to_screen.transform_pos(pos_v);
            let (color, width, label) = self.classify_edge_visual(idx, &palette);
            painter.line_segment([screen_u, screen_v], (width, color));
            if self.directed {
                draw_canvas_arrowhead(painter, screen_u, screen_v, width, color);
            }
            let mid_point = screen_u + (screen_v - screen_u) * 0.5;
            painter.text(
                mid_point + egui::vec2(0.0, -8.0),
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(11.5),
                visuals.text_color(),
            );
        }
    }

    pub(super) fn classify_node_visual(
        &self,
        id: usize,
        palette: &crate::app::SemanticPalette,
    ) -> (egui::Color32, String) {
        if self.selected_algorithm == Algorithm::MaxFlowMinCut {
            return self.classify_flow_node_visual(id, palette);
        }
        if self.selected_algorithm == Algorithm::MinimumSpanningTree {
            return self.classify_mst_node_visual(id, palette);
        }
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
        let label = if self.selected_algorithm == Algorithm::Dijkstra && is_visited {
            self.distances
                .get(&id)
                .map_or_else(|| id.to_string(), |dist| format!("{dist:.1}"))
        } else {
            id.to_string()
        };
        (fill_color, label)
    }

    fn classify_flow_node_visual(
        &self,
        id: usize,
        palette: &crate::app::SemanticPalette,
    ) -> (egui::Color32, String) {
        let sink = self.effective_sink_node(self.start_node);
        let lbl = if self.start_node == Some(id) {
            format!("{id}:s")
        } else if sink == Some(id) {
            format!("{id}:t")
        } else {
            id.to_string()
        };
        if let Some(flow) = &self.flow_data
            && self.animation_step > flow.steps.len()
            && self.animation_step > 0
        {
            let col = if flow.source_side_nodes.contains(&id) {
                palette.accent
            } else {
                palette.series_color(2)
            };
            return (col, lbl);
        }
        let col = if self.start_node == Some(id) {
            palette.accent
        } else if sink == Some(id) {
            palette.warning
        } else {
            palette.series_color(0)
        };
        (col, lbl)
    }

    fn classify_mst_node_visual(
        &self,
        id: usize,
        palette: &crate::app::SemanticPalette,
    ) -> (egui::Color32, String) {
        let covered = self.mst_data.as_ref().is_some_and(|mst| {
            let shown = self.animation_step.min(mst.steps.len());
            mst.steps[..shown]
                .iter()
                .any(|s| s.accepted && (s.u == id || s.v == id))
        });
        let col = if covered {
            palette.success
        } else {
            palette.series_color(0)
        };
        (col, id.to_string())
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
            let (fill_color, label) = self.classify_node_visual(id, &palette);
            painter.circle(
                screen_pos,
                node_radius,
                fill_color,
                (1.0, palette.node_stroke),
            );
            if canvas_focused && self.focused_node == id {
                painter.circle_stroke(screen_pos, node_radius + 4.0_f32, (2.5_f32, palette.accent));
            }
            painter.text(
                screen_pos,
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(11.5),
                crate::app::contrasting_text_color(fill_color),
            );
        }
    }

    fn draw_mode_banner(&self, painter: &egui::Painter, rect: egui::Rect, visuals: &egui::Visuals) {
        let palette = crate::app::SemanticPalette::from_visuals(visuals);
        let text = match self.selected_algorithm {
            Algorithm::MaxFlowMinCut => self.flow_data.as_ref().map(|f| {
                format!(
                    "Edmonds-Karp Max-Flow = {:.2} | s-t Min-Cut Capacity = {:.2} ({} cut edges)",
                    f.max_flow,
                    f.min_cut_capacity,
                    f.cut_edge_indices.len()
                )
            }),
            Algorithm::MinimumSpanningTree => self.mst_data.as_ref().map(|m| {
                format!(
                    "Kruskal MST Total Weight = {:.2} ({} tree edges, {} component(s))",
                    m.total_weight,
                    m.tree_edge_indices.len(),
                    m.component_count
                )
            }),
            _ => None,
        };
        if let Some(msg) = text {
            painter.text(
                rect.min + egui::vec2(14.0, 18.0),
                egui::Align2::LEFT_CENTER,
                msg,
                egui::FontId::proportional(13.0),
                palette.accent,
            );
        }
    }
}

fn draw_canvas_arrowhead(
    painter: &egui::Painter,
    from: Pos2,
    to: Pos2,
    width: f32,
    color: egui::Color32,
) {
    let dir = to - from;
    let len = dir.length();
    if len <= 34.0 {
        return;
    }
    let unit = dir / len;
    let tip = to - unit * 16.0;
    let perp = egui::vec2(-unit.y, unit.x);
    painter.line_segment([tip, tip - unit * 8.0 + perp * 4.5], (width, color));
    painter.line_segment([tip, tip - unit * 8.0 - perp * 4.5], (width, color));
}

// [cite:graph_parameters_rust]
