//! Heatmap painter and `SpatialBrushKit::sample_rect` sculpting canvas for [`super::QTableInspectorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{HeatmapView, QTableInspectorTool, q_value_cell_color};
use crate::accessibility::{AccessibleHoverText, register_canvas_accessibility};
use crate::app::{SemanticPalette, contrasting_text_color};
use crate::async_sim::ui::render_empty_plot_guidance;
use crate::framework::{BrushMode, InteractionContext, InteractiveTool};
use crate::tabs::ai::rl_sculpt::{paint_policy_arrow, role_base_fill, role_glyph};
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::ai::reinforcement_learning::{
    GridCellRole, GridState, MarkovDecisionProcess, Move,
};

impl QTableInspectorTool {
    pub(super) fn draw_view_and_legend(&mut self, ui: &mut egui::Ui, legend_size: f32) {
        ui.separator();
        ui.heading("Heatmap View");
        for view in [
            HeatmapView::MaxQ,
            HeatmapView::BellmanVStar,
            HeatmapView::PolicyVPi,
            HeatmapView::BellmanResidual,
            HeatmapView::Up,
            HeatmapView::Down,
            HeatmapView::Left,
            HeatmapView::Right,
        ] {
            if ui.radio_value(&mut self.view, view, view.label()).changed() {
                self.revision = self.revision.wrapping_add(1);
            }
        }

        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.separator();
        ui.label("Legend:");
        for (label, color) in [
            ("Goal ★:", palette.success),
            ("Trap ✕:", palette.error),
            ("Wall ■:", palette.graph_edge),
        ] {
            ui.horizontal(|ui| {
                ui.label(label);
                let (resp, painter) =
                    ui.allocate_painter(egui::vec2(legend_size, legend_size), egui::Sense::hover());
                painter.rect_filled(resp.rect, 0.0, color);
            });
        }
    }

    fn compute_q_bounds(&self) -> (f64, f64) {
        let mut min_q = f64::INFINITY;
        let mut max_q = f64::NEG_INFINITY;
        for x in 0..self.mdp.width {
            for y in 0..self.mdp.height {
                let st = GridState { x, y };
                if self.mdp.is_terminal(&st) || self.mdp.cell_role(st) == GridCellRole::SolidWall {
                    continue;
                }
                let q = self.get_q_value_for_view(&st);
                if q.is_finite() {
                    min_q = min_q.min(q);
                    max_q = max_q.max(q);
                }
            }
        }
        if !min_q.is_finite() || !max_q.is_finite() {
            return (0.0, 1.0);
        }
        if (max_q - min_q).abs() < 1e-6 {
            max_q = min_q + 1.0;
        }
        (min_q, max_q)
    }

    fn action_overlay_for_cell(&self, state: GridState) -> Option<Move> {
        if self.mdp.is_terminal(&state) || self.mdp.cell_role(state) == GridCellRole::SolidWall {
            return None;
        }
        match self.view {
            HeatmapView::BellmanVStar => self.latest_snap.bellman.optimal_action_at(state),
            HeatmapView::MaxQ | HeatmapView::PolicyVPi | HeatmapView::BellmanResidual => {
                if self.episodes_trained == 0 {
                    return None;
                }
                let idx =
                    flatten_2d_index(state.x as usize, state.y as usize, self.mdp.width as usize);
                self.latest_snap.greedy_actions.get(idx).copied().flatten()
            }
            HeatmapView::Up | HeatmapView::Down | HeatmapView::Left | HeatmapView::Right => None,
        }
    }

    fn paint_heatmap_cell(
        &self,
        painter: &egui::Painter,
        cell_rect: egui::Rect,
        state: GridState,
        bounds: (f64, f64),
        palette: &SemanticPalette,
    ) {
        let role = self.mdp.cell_role(state);
        let q_value = self.get_q_value_for_view(&state);
        let is_special = matches!(
            role,
            GridCellRole::Goal | GridCellRole::Trap | GridCellRole::SolidWall
        );
        let fill_color = if is_special {
            role_base_fill(role, palette)
        } else {
            let norm = ((q_value - bounds.0) / (bounds.1 - bounds.0)).clamp(0.0, 1.0);
            q_value_cell_color(norm, palette)
        };

        painter.rect_filled(cell_rect, 0.0, fill_color);
        painter.rect_stroke(
            cell_rect,
            0.0,
            egui::Stroke::new(1.0_f32, palette.graph_edge),
            egui::StrokeKind::Middle,
        );
        self.paint_heatmap_cell_annotations(
            painter,
            cell_rect,
            state,
            role,
            is_special,
            q_value,
            contrasting_text_color(fill_color),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn paint_heatmap_cell_annotations(
        &self,
        painter: &egui::Painter,
        cell_rect: egui::Rect,
        state: GridState,
        role: GridCellRole,
        is_special: bool,
        q_value: f64,
        text_color: egui::Color32,
    ) {
        if let Some(glyph) = role_glyph(role) {
            let pos = if is_special {
                cell_rect.center()
            } else {
                cell_rect.left_top() + egui::vec2(9.0, 9.0)
            };
            painter.text(
                pos,
                egui::Align2::CENTER_CENTER,
                glyph,
                egui::FontId::proportional(13.0),
                text_color,
            );
        }

        if !is_special {
            let act = self.action_overlay_for_cell(state);
            if let Some(move_dir) = act {
                paint_policy_arrow(
                    painter,
                    cell_rect.center() - egui::vec2(0.0, cell_rect.height() * 0.20),
                    move_dir,
                    cell_rect.width() * 0.18,
                    egui::Stroke::new(2.0_f32, text_color),
                );
            }
            let text_pos = if act.is_some() {
                cell_rect.center() + egui::vec2(0.0, cell_rect.height() * 0.22)
            } else {
                cell_rect.center()
            };
            painter.text(
                text_pos,
                egui::Align2::CENTER_CENTER,
                format!("{q_value:.2}"),
                egui::FontId::proportional(13.0),
                text_color,
            );
        }
    }

    fn paint_heatmap_grid(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        cell_size: f32,
        palette: &SemanticPalette,
    ) {
        let bounds = self.compute_q_bounds();
        for x in 0..self.mdp.width {
            for y in 0..self.mdp.height {
                let top_left = rect.min + egui::vec2(x as f32 * cell_size, y as f32 * cell_size);
                let cell_rect =
                    egui::Rect::from_min_size(top_left, egui::vec2(cell_size, cell_size));
                self.paint_heatmap_cell(painter, cell_rect, GridState { x, y }, bounds, palette);
            }
        }
    }

    pub(super) fn draw_heatmap(&mut self, parent_ui: &mut egui::Ui, cell_size: f32) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading(format!("Q-Value & Bellman Heatmap ({})", self.view.label()));
            if self.episodes_trained == 0 && !matches!(self.view, HeatmapView::BellmanVStar) {
                render_empty_plot_guidance(
                    ui,
                    "Untrained baseline Q-table (0.00) — click '▶ Train 100', sculpt walls/wind, or switch to 'Bellman Optimal V*(s)'.",
                );
            }

            let grid_size = egui::vec2(
                self.mdp.width as f32 * cell_size,
                self.mdp.height as f32 * cell_size,
            );
            let (response, painter) = ui.allocate_painter(grid_size, egui::Sense::click_and_drag());
            let _ = response
                .clone()
                .accessible_hover_text("Q-Table Heatmap Visualization");
            let summary = format!(
                "Episodes trained: {}, View: {}, Solver: {}, Wind: {:.2}",
                self.episodes_trained,
                self.view.label(),
                self.solver_mode.label(),
                self.mdp.wind_slip_prob
            );
            register_canvas_accessibility(
                ui,
                &response,
                egui::accesskit::Role::GraphicsDocument,
                "Q-Table Heatmap Visualization",
                Some(&summary),
            );

            let palette = SemanticPalette::from_visuals(ui.visuals());
            self.paint_heatmap_grid(&painter, response.rect, cell_size, &palette);

            let domain_max = [f64::from(self.mdp.width), f64::from(self.mdp.height)];
            let strokes = self.brush_kit.sample_rect(
                ui,
                &response,
                [0.0, 0.0],
                domain_max,
                Some(BrushMode::Erase),
            );
            for stroke in strokes {
                let brush_ctx = InteractionContext::from_brush_stroke(&response, stroke);
                self.on_brush(&brush_ctx);
            }
        });
    }
}
