#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::analytics::compute_network_analytics;
use super::{ContainmentMode, NetworkPropagationTool};
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::SemanticPalette;
use eframe::egui::{self, Pos2, Vec2};
use math_explorer::epidemiology::networks::NodeState;

const NODE_RADIUS: f32 = 5.5;
const NODE_HIT_RADIUS: f32 = 9.5;
const EDGE_HIT_RADIUS: f32 = 7.5;

fn point_to_segment_distance(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_sq();
    if len_sq <= 1e-6 {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

fn step_spatial_focus(points: &[[f32; 2]], current: usize, dir: Vec2) -> usize {
    if points.is_empty() {
        return 0;
    }
    let cur = current.min(points.len() - 1);
    let origin = points[cur];
    let best = points
        .iter()
        .enumerate()
        .filter(|&(idx, _)| idx != cur)
        .filter_map(|(idx, pos)| {
            let dx = pos[0] - origin[0];
            let dy = pos[1] - origin[1];
            let forward = dx * dir.x + dy * dir.y;
            if forward <= 0.5 {
                return None;
            }
            let lat_x = dx - dir.x * forward;
            let lat_y = dy - dir.y * forward;
            let lateral = (lat_x * lat_x + lat_y * lat_y).sqrt();
            (lateral <= forward * 1.35).then_some((idx, forward + lateral * 2.0))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)))
        .map(|(idx, _)| idx);

    best.unwrap_or_else(|| {
        if dir.x + dir.y >= 0.0 {
            (cur + 1) % points.len()
        } else {
            (cur + points.len() - 1) % points.len()
        }
    })
}

fn node_state_label(state: NodeState) -> &'static str {
    match state {
        NodeState::Susceptible => "Susceptible",
        NodeState::Infected => "Infected",
        NodeState::Recovered => "Recovered",
        NodeState::Vaccinated => "Vaccinated",
    }
}

impl NetworkPropagationTool {
    pub(super) fn canvas_scale_and_center(rect: egui::Rect) -> (Pos2, f32) {
        let min_dim = rect.width().min(rect.height()).max(60.0);
        let scale = (min_dim / 430.0).clamp(0.35, 1.0);
        (rect.center(), scale)
    }

    pub(super) fn node_screen_pos(&self, node: usize, center: Pos2, scale: f32) -> Pos2 {
        let p = self.model.positions()[node];
        center + egui::vec2(p[0] * scale, p[1] * scale)
    }

    pub(super) fn draw_graph_column(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Network Visualization");
        ui.small(
            egui::RichText::new(
                "Click nodes/edges or focus canvas (Arrows navigate, M switches mode, Space/Enter toggles).",
            )
            .color(palette.muted_text),
        );

        let (response, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::click());
        let label = "Network Propagation Visualization";
        let _ = response.clone().accessible_hover_text(label);
        let (center, scale) = Self::canvas_scale_and_center(response.rect);

        self.handle_canvas_pointer(ui.ctx(), &response, center, scale);
        let has_focus = response.has_focus();
        self.canvas_id = has_focus.then_some(response.id);
        if has_focus {
            lock_canvas_arrow_keys(ui, &response);
            self.handle_canvas_keyboard(ui);
        }

        let summary = self.build_accessibility_summary();
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            label,
            Some(&summary),
        );

        self.paint_edges(&painter, center, scale, &palette, has_focus);
        self.paint_nodes(&painter, center, scale, &palette, has_focus);
        paint_canvas_focus_ring(ui, &response);
    }

    fn build_accessibility_summary(&self) -> String {
        let analytics = compute_network_analytics(&self.model);
        let focus_desc = match self.interaction_mode {
            ContainmentMode::VaccinateNode if self.focused_node < self.model.num_nodes() => {
                format!(
                    "Node {} ({})",
                    self.focused_node,
                    node_state_label(self.model.states()[self.focused_node])
                )
            }
            ContainmentMode::QuarantineEdge => {
                self.model.edges().get(self.focused_edge).map_or_else(
                    || "No edges".to_string(),
                    |&[u, v]| format!("Edge {} ({u}-{v})", self.focused_edge),
                )
            }
            ContainmentMode::VaccinateNode => "None".to_string(),
        };
        format!(
            "{} network nodes, R0 {:.2}, budget {}/{}, focus {}",
            self.model.num_nodes(),
            analytics.effective_r0,
            self.used_budget(),
            self.budget,
            focus_desc
        )
    }

    fn hit_test_node(&self, pointer: Pos2, center: Pos2, scale: f32) -> Option<usize> {
        (0..self.model.num_nodes())
            .filter_map(|i| {
                let d = (pointer - self.node_screen_pos(i, center, scale)).length();
                (d <= NODE_HIT_RADIUS).then_some((i, d))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    fn hit_test_edge(&self, pointer: Pos2, center: Pos2, scale: f32) -> Option<usize> {
        self.model
            .edges()
            .iter()
            .enumerate()
            .filter_map(|(idx, &[u, v])| {
                let p1 = self.node_screen_pos(u, center, scale);
                let p2 = self.node_screen_pos(v, center, scale);
                let d = point_to_segment_distance(pointer, p1, p2);
                (d <= EDGE_HIT_RADIUS).then_some((idx, d))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(idx, _)| idx)
    }

    fn handle_canvas_pointer(
        &mut self,
        ctx: &egui::Context,
        response: &egui::Response,
        center: Pos2,
        scale: f32,
    ) {
        if !response.clicked() {
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        response.request_focus();

        let node_hit = self.hit_test_node(pos, center, scale);
        let edge_hit = self.hit_test_edge(pos, center, scale);
        match (self.interaction_mode, node_hit, edge_hit) {
            (ContainmentMode::QuarantineEdge, _, Some(edge_idx))
            | (ContainmentMode::VaccinateNode, None, Some(edge_idx)) => {
                self.interaction_mode = ContainmentMode::QuarantineEdge;
                self.focused_edge = edge_idx;
                self.announce_edge_toggle(ctx, edge_idx);
            }
            (_, Some(node_idx), _) => {
                self.interaction_mode = ContainmentMode::VaccinateNode;
                self.focused_node = node_idx;
                self.announce_node_toggle(ctx, node_idx);
            }
            _ => {}
        }
    }

    fn handle_canvas_keyboard(&mut self, ui: &egui::Ui) {
        let ctx = ui.ctx();
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::M)) {
            self.interaction_mode = match self.interaction_mode {
                ContainmentMode::VaccinateNode if !self.model.edges().is_empty() => {
                    ContainmentMode::QuarantineEdge
                }
                _ => ContainmentMode::VaccinateNode,
            };
            let mode_name = match self.interaction_mode {
                ContainmentMode::VaccinateNode => "Vaccinate Node mode",
                ContainmentMode::QuarantineEdge => "Quarantine Edge mode",
            };
            announce_live_status(ctx, format!("Switched to {mode_name}"));
        }

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
            self.move_keyboard_focus(ctx, dir);
        }

        let activate = ctx.input_mut(|i| {
            i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)
                || i.consume_key(egui::Modifiers::NONE, egui::Key::Space)
        });
        if activate {
            match self.interaction_mode {
                ContainmentMode::VaccinateNode => self.announce_node_toggle(ctx, self.focused_node),
                ContainmentMode::QuarantineEdge => {
                    self.announce_edge_toggle(ctx, self.focused_edge);
                }
            }
        }
    }

    fn move_keyboard_focus(&mut self, ctx: &egui::Context, dir: Vec2) {
        match self.interaction_mode {
            ContainmentMode::VaccinateNode => {
                self.focused_node =
                    step_spatial_focus(self.model.positions(), self.focused_node, dir);
                if self.focused_node < self.model.num_nodes() {
                    let st = node_state_label(self.model.states()[self.focused_node]);
                    let deg = self.model.active_degree(self.focused_node);
                    announce_live_status(
                        ctx,
                        format!(
                            "Focused Node {} ({st}, active degree {deg})",
                            self.focused_node
                        ),
                    );
                }
            }
            ContainmentMode::QuarantineEdge => {
                let midpoints: Vec<[f32; 2]> = self
                    .model
                    .edges()
                    .iter()
                    .map(|&[u, v]| {
                        let pu = self.model.positions()[u];
                        let pv = self.model.positions()[v];
                        [0.5 * (pu[0] + pv[0]), 0.5 * (pu[1] + pv[1])]
                    })
                    .collect();
                if let Some(&[u, v]) =
                    self.model
                        .edges()
                        .get(step_spatial_focus(&midpoints, self.focused_edge, dir))
                {
                    self.focused_edge = step_spatial_focus(&midpoints, self.focused_edge, dir);
                    let status = if self.model.is_edge_quarantined(u, v) {
                        "quarantined"
                    } else {
                        "active"
                    };
                    announce_live_status(
                        ctx,
                        format!("Focused Edge {} ({u}-{v}, {status})", self.focused_edge),
                    );
                }
            }
        }
    }

    fn announce_node_toggle(&mut self, ctx: &egui::Context, node: usize) {
        match self.toggle_node_vaccination(node) {
            Ok(true) => announce_live_status(
                ctx,
                format!(
                    "Vaccinated Node {node} ({} budget remaining)",
                    self.remaining_budget()
                ),
            ),
            Ok(false) => announce_live_status(
                ctx,
                format!(
                    "Removed vaccination from Node {node} ({} budget remaining)",
                    self.remaining_budget()
                ),
            ),
            Err(err) => announce_live_status(ctx, err),
        }
    }

    fn announce_edge_toggle(&mut self, ctx: &egui::Context, edge_idx: usize) {
        let endpoints = self.model.edges().get(edge_idx).copied();
        match self.toggle_edge_quarantine(edge_idx) {
            Ok(true) => {
                let [u, v] = endpoints.unwrap_or([0, 0]);
                announce_live_status(
                    ctx,
                    format!(
                        "Quarantined Edge {edge_idx} ({u}-{v}, {} budget remaining)",
                        self.remaining_budget()
                    ),
                );
            }
            Ok(false) => {
                let [u, v] = endpoints.unwrap_or([0, 0]);
                announce_live_status(
                    ctx,
                    format!(
                        "Restored Edge {edge_idx} ({u}-{v}, {} budget remaining)",
                        self.remaining_budget()
                    ),
                );
            }
            Err(err) => announce_live_status(ctx, err),
        }
    }

    fn paint_edges(
        &self,
        painter: &egui::Painter,
        center: Pos2,
        scale: f32,
        palette: &SemanticPalette,
        canvas_focused: bool,
    ) {
        for (idx, &[u, v]) in self.model.edges().iter().enumerate() {
            let p1 = self.node_screen_pos(u, center, scale);
            let p2 = self.node_screen_pos(v, center, scale);
            let quarantined = self.model.is_edge_quarantined(u, v);
            let is_focused_edge = canvas_focused
                && self.interaction_mode == ContainmentMode::QuarantineEdge
                && self.focused_edge == idx;

            if is_focused_edge {
                painter.line_segment([p1, p2], (4.0, palette.accent));
            }
            if quarantined {
                painter.line_segment([p1, p2], (1.8, palette.warning));
                let mid = p1 + (p2 - p1) * 0.5;
                let d = 3.5;
                painter.line_segment(
                    [mid + egui::vec2(-d, -d), mid + egui::vec2(d, d)],
                    (2.0, palette.warning),
                );
                painter.line_segment(
                    [mid + egui::vec2(-d, d), mid + egui::vec2(d, -d)],
                    (2.0, palette.warning),
                );
            } else {
                painter.line_segment([p1, p2], (1.2, palette.graph_edge));
            }
        }
    }

    fn paint_nodes(
        &self,
        painter: &egui::Painter,
        center: Pos2,
        scale: f32,
        palette: &SemanticPalette,
        canvas_focused: bool,
    ) {
        for i in 0..self.model.num_nodes() {
            let p = self.node_screen_pos(i, center, scale);
            let state = self.model.states()[i];
            let (fill, radius) = match state {
                NodeState::Susceptible => (palette.axis_z, NODE_RADIUS),
                NodeState::Infected => (palette.axis_x, NODE_RADIUS + 1.2),
                NodeState::Recovered => (palette.axis_y, NODE_RADIUS),
                NodeState::Vaccinated => (palette.accent, NODE_RADIUS + 1.0),
            };
            painter.circle_filled(p, radius, fill);
            painter.circle_stroke(p, radius, (1.2, palette.node_stroke));
            if state == NodeState::Vaccinated {
                painter.circle_stroke(p, radius + 2.5, (1.6, palette.info));
            }
            if canvas_focused
                && self.interaction_mode == ContainmentMode::VaccinateNode
                && self.focused_node == i
            {
                painter.circle_stroke(p, radius + 4.5, (2.5, palette.accent));
            }
        }
    }
}
