//! Tactile signed-edge sculpting brushes, hub stamping, and stroke severing for [`super::GraphEditorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{EditorMode, GraphEditorTool};
use crate::framework::InteractionContext;
use eframe::egui::{self, Pos2};
use std::f32::consts::TAU;

fn cross_2d(a: egui::Vec2, b: egui::Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

fn point_segment_distance(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_sq();
    if len_sq <= 1e-6 {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

/// Returns `true` when segment `[a, b]` intersects or passes within `tol` pixels of `[c, d]`.
#[must_use]
pub(super) fn segments_intersect_or_near(a: Pos2, b: Pos2, c: Pos2, d: Pos2, tol: f32) -> bool {
    let r = b - a;
    let s = d - c;
    let denom = cross_2d(r, s);
    let qp = c - a;

    if denom.abs() > 1e-6 {
        let t = cross_2d(qp, s) / denom;
        let u = cross_2d(qp, r) / denom;
        if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
            return true;
        }
    }

    point_segment_distance(a, c, d) <= tol
        || point_segment_distance(b, c, d) <= tol
        || point_segment_distance(c, a, b) <= tol
        || point_segment_distance(d, a, b) <= tol
}

impl GraphEditorTool {
    #[cfg(test)]
    pub(crate) fn with_brush_config(
        mode: EditorMode,
        hub_spokes: usize,
        brush_weight: f64,
    ) -> Self {
        Self {
            mode,
            hub_spokes,
            brush_weight,
            ..Default::default()
        }
    }

    /// Returns the signed weight corresponding to the active brush or stamp mode.
    #[must_use]
    pub fn active_signed_weight(&self) -> f64 {
        let mag = if self.brush_weight.is_finite() && self.brush_weight > 0.0 {
            self.brush_weight
        } else {
            1.0
        };
        match self.mode {
            EditorMode::InhibitoryBrush => -mag,
            EditorMode::HubStamp if self.hub_inhibitory => -mag,
            _ => mag,
        }
    }

    /// Inserts or updates a signed undirected edge `(u, v, signed_weight)` without self-loops.
    pub fn upsert_signed_edge(&mut self, u: usize, v: usize, signed_weight: f64) -> bool {
        if u == v
            || !signed_weight.is_finite()
            || !self.node_positions.contains_key(&u)
            || !self.node_positions.contains_key(&v)
        {
            return false;
        }
        if let Some(slot) = self
            .edges
            .iter_mut()
            .find(|(eu, ev, _)| (*eu == u && *ev == v) || (*eu == v && *ev == u))
        {
            if (slot.2 - signed_weight).abs() < 1e-9 {
                return false;
            }
            slot.2 = signed_weight;
            self.rebuild_graph();
            self.braess.sync_with_edges(&self.edges);
            if self.braess.config.enabled {
                self.recompute_braess();
            }
            self.bump_revision();
            return true;
        }

        self.edges.push((u, v, signed_weight));
        if let (Some(&u_idx), Some(&v_idx)) = (self.node_indices.get(&u), self.node_indices.get(&v))
        {
            self.graph.add_edge(u_idx, v_idx, signed_weight);
        }
        self.braess.sync_with_edges(&self.edges);
        if self.braess.config.enabled {
            self.recompute_braess();
        }
        self.bump_revision();
        true
    }

    /// Stamps a central hub at `center` with `spoke_count` radial spokes and links nearby nodes.
    pub fn stamp_hub_at(&mut self, center: Pos2) {
        self.push_undo_snapshot();
        let weight = self.active_signed_weight();
        let spokes = self.hub_spokes.clamp(3, 8);
        let radius = 70.0_f32;

        let hub_id = self.next_node_id;
        let hub_idx = self.graph.add_node(center);
        self.node_indices.insert(hub_id, hub_idx);
        self.node_positions.insert(hub_id, center);
        self.next_node_id += 1;

        let existing_nearby: Vec<usize> = self
            .node_positions
            .iter()
            .filter_map(|(&id, &p)| (id != hub_id && (p - center).length() <= 115.0).then_some(id))
            .collect();

        for i in 0..spokes {
            let angle = TAU * (i as f32) / (spokes as f32);
            let spoke_pos = Pos2::new(
                center.x + radius * angle.cos(),
                center.y + radius * angle.sin(),
            );
            let spoke_id = self.next_node_id;
            let spoke_idx = self.graph.add_node(spoke_pos);
            self.node_indices.insert(spoke_id, spoke_idx);
            self.node_positions.insert(spoke_id, spoke_pos);
            self.next_node_id += 1;
            let _ = self.upsert_signed_edge(hub_id, spoke_id, weight);
        }

        for nearby_id in existing_nearby {
            let _ = self.upsert_signed_edge(hub_id, nearby_id, weight);
        }
        self.selected_node = Some(hub_id);
        self.bump_revision();
    }

    /// Severs all undirected edges intersected by stroke segment `[from, to]`.
    pub fn sever_edges_along_segment(&mut self, from: Pos2, to: Pos2, record_undo: bool) -> usize {
        let positions = &self.node_positions;
        let mut to_keep = Vec::with_capacity(self.edges.len());
        let mut severed = 0usize;

        for &(u, v, w) in &self.edges {
            let cut = match (positions.get(&u), positions.get(&v)) {
                (Some(&pu), Some(&pv)) => segments_intersect_or_near(from, to, pu, pv, 6.0),
                _ => false,
            };
            if cut {
                severed += 1;
            } else {
                to_keep.push((u, v, w));
            }
        }

        if severed > 0 {
            if record_undo {
                self.push_undo_snapshot();
            }
            self.edges = to_keep;
            self.rebuild_graph();
            self.braess.sync_with_edges(&self.edges);
            if self.braess.config.enabled {
                self.recompute_braess();
            }
            self.bump_revision();
        }
        severed
    }

    pub(super) fn handle_brush_or_sever_drag(&mut self, ctx: &InteractionContext) {
        let Some(pos) = ctx.pointer_pos else {
            return;
        };
        if ctx.response.drag_started() {
            self.brush_trail.clear();
            self.brush_trail.push(pos);
            self.brush_stroke_undo_pushed = false;
            self.last_brushed_node = self.node_at_pos(pos, 20.0);
        }

        if ctx.response.dragged() {
            let prev = self.brush_trail.last().copied().unwrap_or(pos);
            if (pos - prev).length_sq() > 1.0 {
                self.brush_trail.push(pos);
            }
            match self.mode {
                EditorMode::ExcitatoryBrush | EditorMode::InhibitoryBrush => {
                    self.step_weight_brush_stroke(pos);
                }
                EditorMode::SeverStroke => {
                    let record_undo = !self.brush_stroke_undo_pushed;
                    if self.sever_edges_along_segment(prev, pos, record_undo) > 0 {
                        self.brush_stroke_undo_pushed = true;
                    }
                }
                _ => {}
            }
        }

        if ctx.response.drag_stopped() {
            self.brush_trail.clear();
            self.last_brushed_node = None;
            self.brush_stroke_undo_pushed = false;
        }
    }

    fn step_weight_brush_stroke(&mut self, pos: Pos2) {
        let weight = self.active_signed_weight();
        if let Some(hit_id) = self.node_at_pos(pos, 20.0) {
            if let Some(prev_id) = self.last_brushed_node {
                if prev_id != hit_id {
                    if !self.brush_stroke_undo_pushed {
                        self.push_undo_snapshot();
                        self.brush_stroke_undo_pushed = true;
                    }
                    let _ = self.upsert_signed_edge(prev_id, hit_id, weight);
                }
            }
            self.last_brushed_node = Some(hit_id);
            return;
        }

        if let Some(edge_idx) = self.edge_at_pos(pos, 7.0) {
            let (u, v, old_w) = self.edges[edge_idx];
            if (old_w - weight).abs() > 1e-9 {
                if !self.brush_stroke_undo_pushed {
                    self.push_undo_snapshot();
                    self.brush_stroke_undo_pushed = true;
                }
                let _ = self.upsert_signed_edge(u, v, weight);
            }
        }
    }
}
