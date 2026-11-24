#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::braess::{AffineLatency, BraessCongestionLab, BraessUiAction, WardropError};
use super::{EditorMode, GraphEditorTool};
use eframe::egui::{self, Pos2};

impl GraphEditorTool {
    /// Recomputes Wardrop congestion routing on the active graph.
    pub fn recompute_braess(&mut self) {
        self.braess.recompute(&self.node_positions, &self.edges);
    }

    /// Populates the 4-node Braess diamond network (`s=0, u=1, v=2, t=3`) with optional shortcut.
    pub fn load_braess_preset(&mut self, shortcut_enabled: bool) {
        self.push_undo_snapshot();
        self.node_positions.clear();
        self.node_positions.insert(0, Pos2::new(140.0, 260.0));
        self.node_positions.insert(1, Pos2::new(360.0, 120.0));
        self.node_positions.insert(2, Pos2::new(360.0, 400.0));
        self.node_positions.insert(3, Pos2::new(580.0, 260.0));
        self.selected_node = None;
        self.next_node_id = 4;

        self.edges = vec![(0, 1, 0.0), (1, 3, 1.0), (0, 2, 1.0), (2, 3, 0.0)];
        self.braess.edge_latencies = vec![
            AffineLatency { a: 1.0, b: 0.0 },
            AffineLatency { a: 0.0, b: 1.0 },
            AffineLatency { a: 0.0, b: 1.0 },
            AffineLatency { a: 1.0, b: 0.0 },
        ];
        if shortcut_enabled {
            self.edges.push((1, 2, 0.0));
            self.braess
                .edge_latencies
                .push(AffineLatency { a: 0.0, b: 0.0 });
        }
        self.braess.config.enabled = true;
        self.braess.config.source_node = 0;
        self.braess.config.sink_node = 3;
        self.braess.config.total_demand = 1.0;
        self.braess.config.shortcut_enabled = shortcut_enabled;
        self.rebuild_graph();
        self.recompute_braess();
        self.bump_revision();
    }

    /// Enables or removes the zero-cost `u -> v` (`1 -> 2`) shortcut edge.
    pub fn set_braess_shortcut(&mut self, enabled: bool) {
        self.push_undo_snapshot();
        self.braess.sync_with_edges(&self.edges);
        if enabled {
            let exists = self.edges.iter().any(|&(u, v, _)| u == 1 && v == 2);
            if !exists
                && self.node_positions.contains_key(&1)
                && self.node_positions.contains_key(&2)
            {
                self.edges.push((1, 2, 0.0));
                self.braess
                    .edge_latencies
                    .push(AffineLatency { a: 0.0, b: 0.0 });
            }
        } else {
            let mut i = 0;
            while i < self.edges.len() {
                if self.edges[i].0 == 1 && self.edges[i].1 == 2 {
                    self.edges.remove(i);
                    self.braess.edge_latencies.remove(i);
                } else {
                    i += 1;
                }
            }
        }
        self.braess.config.shortcut_enabled = enabled;
        self.rebuild_graph();
        self.recompute_braess();
        self.bump_revision();
    }

    pub(crate) fn insert_new_edge(&mut self, u: usize, v: usize) {
        self.braess.sync_with_edges(&self.edges);
        let lat = AffineLatency::clamped(self.braess.new_edge_a, self.braess.new_edge_b);
        let weight = lat.b;
        self.edges.push((u, v, weight));
        self.braess.edge_latencies.push(lat);
        if let (Some(&u_idx), Some(&v_idx)) = (self.node_indices.get(&u), self.node_indices.get(&v))
        {
            self.graph.add_edge(u_idx, v_idx, weight);
        } else {
            self.rebuild_graph();
        }
        if self.braess.config.enabled {
            self.recompute_braess();
        }
        self.bump_revision();
    }

    pub(crate) fn apply_braess_action(&mut self, action: BraessUiAction) {
        match action {
            BraessUiAction::None => {}
            BraessUiAction::LoadPreset => self.load_braess_preset(false),
            BraessUiAction::ToggleShortcut(on) => self.set_braess_shortcut(on),
            BraessUiAction::Recompute => {
                self.bump_revision();
                self.rebuild_graph();
                self.recompute_braess();
            }
        }
    }

    pub(crate) fn click_add_edge(&mut self) {
        let Some(clicked_id) = self.hovered_node else {
            self.selected_node = None;
            return;
        };
        let Some(selected_id) = self.selected_node else {
            self.selected_node = Some(clicked_id);
            return;
        };
        if selected_id != clicked_id {
            let edge_exists = self.edges.iter().any(|&(u, v, _)| {
                if self.braess.config.enabled {
                    u == selected_id && v == clicked_id
                } else {
                    (u == selected_id && v == clicked_id) || (v == selected_id && u == clicked_id)
                }
            });
            if !edge_exists {
                self.push_undo_snapshot();
                self.insert_new_edge(selected_id, clicked_id);
            }
        }
        self.selected_node = None;
    }

    pub(crate) fn click_remove(&mut self) {
        self.braess.sync_with_edges(&self.edges);
        if let Some(clicked_id) = self.hovered_node {
            self.push_undo_snapshot();
            self.node_positions.remove(&clicked_id);
            self.node_indices.remove(&clicked_id);
            let mut i = 0;
            while i < self.edges.len() {
                if self.edges[i].0 == clicked_id || self.edges[i].1 == clicked_id {
                    self.edges.remove(i);
                    self.braess.edge_latencies.remove(i);
                } else {
                    i += 1;
                }
            }
            self.rebuild_graph();
            if self.braess.config.enabled {
                self.recompute_braess();
            }
            self.bump_revision();
        } else if let Some(edge_idx) = self.hovered_edge {
            self.push_undo_snapshot();
            self.edges.remove(edge_idx);
            if edge_idx < self.braess.edge_latencies.len() {
                self.braess.edge_latencies.remove(edge_idx);
            }
            self.rebuild_graph();
            if self.braess.config.enabled {
                self.recompute_braess();
            }
            self.bump_revision();
        }
    }
}

impl BraessCongestionLab {
    /// Draws flow-weighted directed edges, arrowheads, and latency badges on the canvas.
    pub fn draw_congestion_edge(
        &self,
        painter: &egui::Painter,
        palette: &crate::app::SemanticPalette,
        text_color: egui::Color32,
        idx: usize,
        pos_u: Pos2,
        pos_v: Pos2,
    ) {
        let f = self.equilibrium_flows.get(idx).copied().unwrap_or(0.0);
        let lat_fn = self
            .edge_latencies
            .get(idx)
            .copied()
            .unwrap_or(AffineLatency { a: 0.0, b: 1.0 });
        let lat_val = self
            .equilibrium_latencies
            .get(idx)
            .copied()
            .unwrap_or_else(|| lat_fn.eval(f));
        let ratio = (f / self.config.total_demand.max(0.1)).clamp(0.0, 1.5);
        let width = 2.0 + 3.5 * ratio as f32;
        let color = if f > 1e-3 {
            palette.series_color(3)
        } else {
            palette.graph_edge
        };
        painter.line_segment([pos_u, pos_v], (width, color));
        draw_arrowhead(painter, pos_u, pos_v, width, color);

        let mid = pos_u + (pos_v - pos_u) * 0.5;
        let label = format!("l={lat_val:.2} (f={f:.2})");
        painter.text(
            mid + egui::vec2(0.0, -10.0),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(11.5),
            text_color,
        );
    }

    /// Draws a canvas banner summarizing Wardrop latency or unreachable status.
    pub fn draw_canvas_banner(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        palette: &crate::app::SemanticPalette,
    ) {
        let (msg, col) = match self.status {
            Ok(s) => {
                if let Some(base) = self
                    .without_shortcut_latency
                    .filter(|&b| s.commuter_latency > b + 1e-6)
                {
                    (
                        format!(
                            "Braess Nash Trap Active: Commuter Latency {:.2} -> {:.2} (u->v shortcut slows all drivers!)",
                            base, s.commuter_latency
                        ),
                        palette.error,
                    )
                } else {
                    (
                        format!(
                            "Wardrop Equilibrium Latency: {:.2} (Demand D = {:.2})",
                            s.commuter_latency, self.config.total_demand
                        ),
                        palette.success,
                    )
                }
            }
            Err(WardropError::Unreachable) => (
                format!(
                    "Unreachable s->t Pair ({} -> {}): No directed path available",
                    self.config.source_node, self.config.sink_node
                ),
                palette.error,
            ),
            Err(WardropError::MissingTerminal) => (
                "Select existing Source s and Sink t node IDs".to_string(),
                palette.warning,
            ),
        };
        painter.text(
            rect.min + egui::vec2(14.0, 18.0),
            egui::Align2::LEFT_CENTER,
            msg,
            egui::FontId::proportional(13.5),
            col,
        );
    }
}

pub(crate) fn should_draw_congestion_edge(
    enabled: bool,
    mode: &EditorMode,
    hovered_edge: Option<usize>,
    idx: usize,
) -> bool {
    enabled
        && !(matches!(mode, EditorMode::Remove | EditorMode::SeverStroke)
            && hovered_edge == Some(idx))
}

fn draw_arrowhead(
    painter: &egui::Painter,
    pos_u: Pos2,
    pos_v: Pos2,
    width: f32,
    color: egui::Color32,
) {
    let dir = pos_v - pos_u;
    let len = dir.length();
    if len <= 34.0 {
        return;
    }
    let unit = dir / len;
    let tip = pos_v - unit * 16.0;
    let perp = egui::vec2(-unit.y, unit.x);
    let left = tip - unit * 9.0 + perp * 5.0;
    let right = tip - unit * 9.0 - perp * 5.0;
    painter.line_segment([tip, left], (width, color));
    painter.line_segment([tip, right], (width, color));
}

// [cite:dijkstra]
