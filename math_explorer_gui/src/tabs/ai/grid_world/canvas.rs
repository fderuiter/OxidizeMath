//! Central canvas painter and `SpatialBrushKit::sample_rect` integration for [`super::GridWorldTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GridWorldTool;
use crate::accessibility::{register_canvas_accessibility, AccessibleHoverText};
use crate::app::{contrasting_text_color, SemanticPalette};
use crate::framework::{BrushMode, InputMode, InteractionContext, InteractiveTool};
use crate::tabs::ai::q_table_inspector::q_value_cell_color;
use crate::tabs::ai::rl_sculpt::{
    mean_bellman_residual, paint_policy_arrow, role_base_fill, role_glyph, RlOverlayMode,
    RlSolverMode,
};
use eframe::egui;
use math_explorer::ai::reinforcement_learning::{GridCellRole, GridState, MarkovDecisionProcess};

impl GridWorldTool {
    fn overlay_scalar(&self, state: GridState) -> Option<f64> {
        if self.mdp.is_terminal(&state) || self.mdp.cell_role(state) == GridCellRole::SolidWall {
            return None;
        }
        match self.overlay_mode {
            RlOverlayMode::EmpiricalPolicy => None,
            RlOverlayMode::BellmanVStar => Some(self.bellman.v_star_at(state)),
            RlOverlayMode::PolicyValueVPi => Some(self.bellman.v_pi_at(state)),
            RlOverlayMode::BellmanResidual => {
                let empirical = match self.solver_mode {
                    RlSolverMode::TabularQ => self.workspace.max_q_at(state),
                    RlSolverMode::PolicyGradient => self.workspace.state_value_at(state),
                };
                Some((empirical - self.bellman.v_star_at(state)).abs())
            }
        }
    }

    fn overlay_bounds(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for y in 0..self.mdp.height {
            for x in 0..self.mdp.width {
                if let Some(v) = self.overlay_scalar(GridState { x, y }) {
                    if v.is_finite() {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
            }
        }
        if !lo.is_finite() || !hi.is_finite() {
            return (0.0, 1.0);
        }
        if (hi - lo).abs() < 1e-6 {
            hi = lo + 1.0;
        }
        (lo, hi)
    }

    fn cell_fill(
        &self,
        state: GridState,
        bounds: (f64, f64),
        palette: &SemanticPalette,
    ) -> egui::Color32 {
        let role = self.mdp.cell_role(state);
        if matches!(
            role,
            GridCellRole::Goal | GridCellRole::Trap | GridCellRole::SolidWall
        ) {
            return role_base_fill(role, palette);
        }
        if let Some(val) = self.overlay_scalar(state) {
            let norm = ((val - bounds.0) / (bounds.1 - bounds.0)).clamp(0.0, 1.0);
            return q_value_cell_color(norm, palette);
        }
        if state == self.current_state {
            palette.accent
        } else if state == self.mdp.start {
            palette.info
        } else {
            role_base_fill(role, palette)
        }
    }

    fn paint_single_cell(
        &self,
        painter: &egui::Painter,
        cell_rect: egui::Rect,
        state: GridState,
        bounds: (f64, f64),
        palette: &SemanticPalette,
    ) {
        let fill = self.cell_fill(state, bounds, palette);
        painter.rect_filled(cell_rect, 0.0, fill);
        painter.rect_stroke(
            cell_rect,
            0.0,
            egui::Stroke::new(1.0_f32, palette.graph_edge),
            egui::StrokeKind::Middle,
        );

        let fg = contrasting_text_color(fill);
        let role = self.mdp.cell_role(state);
        if let Some(glyph) = role_glyph(role) {
            let pos = if matches!(
                role,
                GridCellRole::Goal | GridCellRole::Trap | GridCellRole::SolidWall
            ) {
                cell_rect.center()
            } else {
                cell_rect.left_top() + egui::vec2(8.0, 9.0)
            };
            painter.text(
                pos,
                egui::Align2::CENTER_CENTER,
                glyph,
                egui::FontId::proportional(13.0),
                fg,
            );
        }

        if let Some(action) = self.greedy_action_for_display(state) {
            let arrow_center = if self.overlay_scalar(state).is_some() {
                cell_rect.center() - egui::vec2(0.0, cell_rect.height() * 0.16)
            } else {
                cell_rect.center()
            };
            paint_policy_arrow(
                painter,
                arrow_center,
                action,
                cell_rect.width() * 0.22,
                egui::Stroke::new(2.0_f32, fg),
            );
        }

        if let Some(val) = self.overlay_scalar(state) {
            painter.text(
                cell_rect.center() + egui::vec2(0.0, cell_rect.height() * 0.24),
                egui::Align2::CENTER_CENTER,
                format!("{val:.2}"),
                egui::FontId::proportional(11.0),
                fg,
            );
        }

        if state == self.current_state {
            let ring_stroke = egui::Stroke::new(2.5_f32, fg);
            painter.circle_stroke(cell_rect.center(), cell_rect.width() * 0.34, ring_stroke);
        }
    }

    fn draw_telemetry_header(&self, ui: &mut egui::Ui) {
        ui.heading(format!(
            "Grid World Navigation ({} — {})",
            self.solver_mode.label(),
            self.overlay_mode.label()
        ));
        ui.separator();
        let residual =
            mean_bellman_residual(&self.mdp, &self.workspace, self.solver_mode, &self.bellman);
        ui.horizontal_wrapped(|ui| {
            ui.label(format!("Episode: {}", self.episodes));
            ui.label(format!("Steps: {}", self.steps));
            ui.label(format!("Reward: {:.2}", self.total_reward));
            ui.label(format!("|TD δ|: {:.3}", self.last_td_error.abs()));
            ui.label(format!("Mean |V - V*|: {residual:.3}"));
            if self.solver_mode == RlSolverMode::PolicyGradient {
                ui.label(format!("‖∇θ‖: {:.3}", self.last_pg_grad_norm));
            }
        });
    }

    fn compute_grid_layout(&self, ui: &egui::Ui, input_mode: InputMode) -> (f32, egui::Vec2) {
        let base_cell = if input_mode == InputMode::Touch {
            48.0_f32
        } else {
            44.0_f32
        };
        let max_dim = self.mdp.width.max(self.mdp.height).max(1) as f32;
        let fit_cell =
            ((ui.available_width().min(ui.available_height()) - 16.0) / max_dim).clamp(24.0, 64.0);
        let cell_size = base_cell.max(fit_cell).min(64.0);
        let grid_size = egui::vec2(
            self.mdp.width as f32 * cell_size,
            self.mdp.height as f32 * cell_size,
        );
        (cell_size, grid_size)
    }

    pub(super) fn draw_central_canvas(&mut self, parent_ui: &mut egui::Ui, input_mode: InputMode) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.draw_telemetry_header(ui);
            let (cell_size, grid_size) = self.compute_grid_layout(ui, input_mode);
            let (response, painter) = ui.allocate_painter(grid_size, egui::Sense::click_and_drag());
            let _ = response
                .clone()
                .accessible_hover_text("Grid World Visualization");
            let summary = format!(
                "Agent at ({}, {}), Episode {}, Reward {:.2}, Overlay {}, Wind {:.2}",
                self.current_state.x,
                self.current_state.y,
                self.episodes,
                self.total_reward,
                self.overlay_mode.label(),
                self.mdp.wind_slip_prob
            );
            register_canvas_accessibility(
                ui,
                &response,
                egui::accesskit::Role::GraphicsDocument,
                "Grid World Visualization",
                Some(&summary),
            );

            let palette = SemanticPalette::from_visuals(ui.visuals());
            let bounds = self.overlay_bounds();
            for x in 0..self.mdp.width {
                for y in 0..self.mdp.height {
                    let top_left =
                        response.rect.min + egui::vec2(x as f32 * cell_size, y as f32 * cell_size);
                    let cell_rect =
                        egui::Rect::from_min_size(top_left, egui::vec2(cell_size, cell_size));
                    self.paint_single_cell(
                        &painter,
                        cell_rect,
                        GridState { x, y },
                        bounds,
                        &palette,
                    );
                }
            }

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
