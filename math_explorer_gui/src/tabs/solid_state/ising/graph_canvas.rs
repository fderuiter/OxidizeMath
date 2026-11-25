//! Interactive canvas, frustrated-bond highlighting, and transport controls for CSR [`IsingModelTool`] graph mode.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::preset::IsingInteractionMode;
use super::{ising_spin_color, IsingModelTool, DEFAULT_ISING_SIZE};
use crate::accessibility::{
    announce_live_status, draw_legend_swatch, paint_canvas_focus_ring,
    register_canvas_accessibility, AccessibleHoverText, AccessibleTheoryHover,
};
use crate::app::data_bus::ui::{render_follow_workspace_toggle, render_node_budget_banner};
use crate::app::data_bus::MAX_SPIN_NODE_BUDGET;
use crate::app::{
    contrasting_text_color, render_topology_source_bar, SemanticPalette, SharedSimulationBus,
};
use crate::framework::{cell_screen_rect, PlaybackAction, PlaybackTransport, RasterBrushCanvas};
use eframe::egui::{self, Color32, Pos2, Rect, Stroke};
use math_commons::math_kernel::types::flatten_2d_index;

type MappedSpinGraph<'a> = (Vec<[f32; 2]>, &'a [(usize, usize, f64)]);

impl IsingModelTool {
    pub(super) fn draw_topology_source_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let mut changed = false;
        ui.add_enabled_ui(self.active_challenge.is_none(), |ui| {
            changed = render_topology_source_bar(
                ui,
                "ising_model",
                &mut self.topology_source,
                self.synced_bus_revision,
                None,
            );
            let rev_opt = (self.graph_mode.applied_bus_revision > 0)
                .then_some(self.graph_mode.applied_bus_revision);
            if render_follow_workspace_toggle(
                ui,
                &mut self.graph_mode.follow_workspace_graph,
                "Ising Model",
                rev_opt,
            ) && self.graph_mode.follow_workspace_graph
            {
                self.graph_mode.applied_bus_revision = 0;
                if !SharedSimulationBus::is_link_topology_enabled(ctx) {
                    self.sync_graph_from_bus(ctx);
                }
            }
        });
        if changed {
            self.sync_shared_topology(ctx);
            if !SharedSimulationBus::is_link_topology_enabled(ctx) {
                self.sync_graph_from_bus(ctx);
            }
        }
        render_node_budget_banner(
            ui,
            self.graph_mode.clamped_source_nodes,
            MAX_SPIN_NODE_BUDGET,
            "graph spins",
        );
        if let Some(ref snap) = self.shared_topology {
            ui.small(format!(
                "Shared CSR Graph: |V| = {}, |E| = {}",
                snap.node_count(),
                snap.undirected_edges.len()
            ));
        } else if let Some(ref g) = self.graph_mode.active_graph {
            ui.small(format!(
                "Workspace Spin Graph: |V| = {}, |E| = {}",
                g.nodes.len(),
                g.edges.len()
            ));
        }
    }

    pub(super) fn draw_graph_transport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if let Some(action) = PlaybackTransport::in_thread(
            "ising_graph_transport",
            "Ising CSR Graph",
            self.graph_running,
        )
        .show(ui)
        {
            match action {
                PlaybackAction::Play => self.graph_running = true,
                PlaybackAction::Pause => self.graph_running = false,
                PlaybackAction::Step => {
                    self.graph_running = false;
                    let trials = self.active_graph_node_count().unwrap_or(16).max(16);
                    self.step_graph_trials(trials);
                }
                PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
                PlaybackAction::Reset => {
                    let _ = self.reset_graph_spins();
                }
            }
            ctx.request_repaint();
        }
    }

    pub(super) fn draw_graph_spin_canvas(&mut self, ui: &mut egui::Ui, side: f32) {
        let (response, painter) = ui.allocate_painter(egui::vec2(side, side), egui::Sense::click());
        let n = self.active_graph_node_count().unwrap_or(0);
        let label = "CSR Graph Ising Spin & Fermi Strategy Visualization";
        let _ = response.clone().accessible_hover_text(label);
        let summary = format!(
            "{n} CSR nodes, {} frustrated edges, M = {:.3}, E = {:.3}, C = {:.1}%; Legend: +1 Up/C bright amber circle, -1 Down/D deep cobalt circle",
            self.frustrated_edge_count(),
            self.magnetization,
            self.energy,
            self.cooperator_fraction() * 100.0
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            label,
            Some(&summary),
        );

        let palette = SemanticPalette::from_visuals(ui.visuals());
        painter.rect_filled(response.rect, 4.0, ui.visuals().extreme_bg_color);
        painter.rect_stroke(
            response.rect,
            4.0,
            egui::Stroke::new(1.25_f32, palette.graph_edge),
            egui::StrokeKind::Inside,
        );
        if let Some(idx) = self.paint_graph_spin_elements(&response, &painter, side, &palette) {
            if let Some(ref mut sys) = self.graph_system {
                sys.flip_spin(idx);
            }
            self.refresh_graph_observables();
            announce_live_status(ui.ctx(), format!("Flipped spin on CSR node {idx}"));
        }
        paint_canvas_focus_ring(ui, &response);
        ui.horizontal_wrapped(|ui| {
            crate::accessibility::draw_legend_swatch(
                ui,
                ising_spin_color(1, &palette),
                Some("↑"),
                "+1 Up / C (warm amber)",
            );
            crate::accessibility::draw_legend_swatch(
                ui,
                ising_spin_color(-1, &palette),
                Some("↓"),
                "-1 Down / D (deep cobalt)",
            );
            ui.small("Click a node to flip its spin");
        });
    }

    fn mapped_graph_positions_and_edges(
        &self,
        rect: egui::Rect,
        pad: f32,
    ) -> Option<MappedSpinGraph<'_>> {
        if let Some(ref snap) = self.shared_topology {
            let mapped = snap.map_positions_to_bounds(
                rect.min.x + pad,
                rect.min.y + pad,
                rect.max.x - pad,
                rect.max.y - pad,
            );
            return Some((mapped, &snap.undirected_edges));
        }
        let graph = self.graph_mode.active_graph.as_ref()?;
        let inner = rect.shrink(pad);
        let mapped = graph
            .nodes
            .iter()
            .map(|nd| {
                [
                    inner.left() + nd.norm_pos[0].clamp(0.0, 1.0) * inner.width().max(1.0),
                    inner.top() + nd.norm_pos[1].clamp(0.0, 1.0) * inner.height().max(1.0),
                ]
            })
            .collect();
        Some((mapped, &graph.edges))
    }

    fn paint_graph_spin_elements(
        &self,
        response: &egui::Response,
        painter: &egui::Painter,
        side: f32,
        palette: &SemanticPalette,
    ) -> Option<usize> {
        let sys_spins = self.graph_system.as_ref()?.spins();
        let pad = 22.0_f32.min(side * 0.15);
        let (mapped, edges) = self.mapped_graph_positions_and_edges(response.rect, pad)?;
        let show_frustration = self.mode == IsingInteractionMode::Ferromagnetic;
        for &(u, v, w) in edges {
            if u < mapped.len() && v < mapped.len() && u != v {
                let pu = Pos2::new(mapped[u][0], mapped[u][1]);
                let pv = Pos2::new(mapped[v][0], mapped[v][1]);
                let su = f64::from(sys_spins.get(u).copied().unwrap_or(1));
                let sv = f64::from(sys_spins.get(v).copied().unwrap_or(1));
                let frustrated = show_frustration && (self.j_coupling * w * su * sv < -1e-12);
                let stroke = if frustrated {
                    (2.6, palette.warning)
                } else {
                    (1.6, palette.graph_edge)
                };
                painter.line_segment([pu, pv], stroke);
            }
        }
        let node_r = if mapped.len() > 64 { 6.0 } else { 11.0 };
        let click_pos = response
            .clicked()
            .then(|| response.interact_pointer_pos())
            .flatten();
        let mut clicked_idx = None;
        for (i, &[x, y]) in mapped.iter().enumerate() {
            let center = Pos2::new(x, y);
            if click_pos.is_some_and(|cp| (cp - center).length() <= node_r + 4.0) {
                clicked_idx = Some(i);
            }
            let spin = sys_spins.get(i).copied().unwrap_or(1);
            let fill = ising_spin_color(spin, palette);
            painter.circle(center, node_r, fill, (1.0, palette.node_stroke));
            if node_r >= 10.0 {
                painter.text(
                    center,
                    egui::Align2::CENTER_CENTER,
                    if spin > 0 { "+" } else { "-" },
                    egui::FontId::proportional(10.0),
                    contrasting_text_color(fill),
                );
            }
        }
        clicked_idx
    }

    pub(super) fn draw_lattice_canvas(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        side: f32,
    ) {
        let Some(texture) = self.texture.clone() else {
            return;
        };
        let pinned_up = self.pin_cells.iter().filter(|&&c| c == 1).count();
        let pinned_down = self.pin_cells.iter().filter(|&&c| c == 2).count();
        let pinned_count = pinned_up + pinned_down;
        let alt_label = format!(
            "2D Ising Model spin lattice visualization (T = {:.3}, M = {:.3}, E = {:.3}, C = {:.1}%; +1 warm amber, -1 deep cobalt, pinned +1 gold '+' cross, pinned -1 teal 'x' cross)",
            self.temperature,
            self.magnetization,
            self.energy,
            self.cooperator_fraction() * 100.0
        );
        let summary = format!(
            "M = {:.3}, E = {:.1}, C = {:.1}%, {pinned_count} pinned sites ({pinned_up} +1, {pinned_down} -1); Legend: +1 Up warm amber, -1 Down deep cobalt, Pin +1 gold '+', Pin -1 teal 'x'",
            self.magnetization,
            self.energy,
            self.cooperator_fraction() * 100.0
        );
        let pin_snapshot = self.pin_cells.clone();
        let (response, stroke) = RasterBrushCanvas::new(
            "ising_lattice_canvas",
            DEFAULT_ISING_SIZE,
            DEFAULT_ISING_SIZE,
            &alt_label,
        )
        .role(egui::accesskit::Role::GraphicsDocument)
        .value_summary(&summary)
        .max_canvas_size(egui::vec2(side, side))
        .show_custom(ui, &mut self.brush_state, |painter, rect, palette| {
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            painter.image(texture.id(), rect, uv, Color32::WHITE);
            paint_pinned_site_markers(painter, rect, &pin_snapshot, DEFAULT_ISING_SIZE, palette);
        });
        let response = response.accessible_theory_hover(self);
        self.last_canvas_rect = Some(response.rect);
        self.sync_brush_fields_from_palette();
        if let Some(stroke) = stroke {
            self.apply_brush_stroke(stroke, Some(ctx));
        }

        let palette = SemanticPalette::from_visuals(ui.visuals());
        draw_ising_spin_legend(ui, &palette);
    }
}

/// Paints non-color vector markers (`+` orthogonal cross for pinned `+1`, `×` diagonal cross for
/// pinned `-1`) over pinned lattice cells at `>= 4.5:1` contrast against the underlying spin fill.
pub(super) fn paint_pinned_site_markers(
    painter: &egui::Painter,
    canvas_rect: Rect,
    pin_cells: &[u8],
    grid_size: usize,
    palette: &SemanticPalette,
) {
    if pin_cells.len() != grid_size * grid_size {
        return;
    }
    for y in 0..grid_size {
        for x in 0..grid_size {
            let pin = pin_cells[flatten_2d_index(x, y, grid_size)];
            if pin == 0 {
                continue;
            }
            let cell_r = cell_screen_rect(canvas_rect, x, y, grid_size, grid_size, false);
            let spin_fill = ising_spin_color(if pin == 1 { 2 } else { -2 }, palette);
            let fg = contrasting_text_color(spin_fill);
            let arm = (cell_r.width().min(cell_r.height()) * 0.42).max(2.0_f32);
            let c = cell_r.center();
            let stroke = Stroke::new(1.4_f32, fg);
            if pin == 1 {
                painter.line_segment(
                    [Pos2::new(c.x - arm, c.y), Pos2::new(c.x + arm, c.y)],
                    stroke,
                );
                painter.line_segment(
                    [Pos2::new(c.x, c.y - arm), Pos2::new(c.x, c.y + arm)],
                    stroke,
                );
            } else {
                let d = arm * 0.85_f32;
                painter.line_segment(
                    [Pos2::new(c.x - d, c.y - d), Pos2::new(c.x + d, c.y + d)],
                    stroke,
                );
                painter.line_segment(
                    [Pos2::new(c.x - d, c.y + d), Pos2::new(c.x + d, c.y - d)],
                    stroke,
                );
            }
        }
    }
}

/// Renders explicit color-polarity + shape swatch labels for the four discrete Ising spin states.
pub(super) fn draw_ising_spin_legend(ui: &mut egui::Ui, palette: &SemanticPalette) {
    ui.horizontal_wrapped(|ui| {
        draw_legend_swatch(
            ui,
            ising_spin_color(1, palette),
            None,
            "+1 Up / C (warm amber)",
        );
        draw_legend_swatch(
            ui,
            ising_spin_color(-1, palette),
            None,
            "-1 Down / D (deep cobalt)",
        );
        draw_legend_swatch(
            ui,
            ising_spin_color(2, palette),
            Some("+"),
            "Pin +1 (+ cross, bright gold)",
        );
        draw_legend_swatch(
            ui,
            ising_spin_color(-2, palette),
            Some("×"),
            "Pin -1 (× cross, deep teal)",
        );
    });
}
