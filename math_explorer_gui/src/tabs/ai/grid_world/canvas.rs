//! Central canvas painter combining Bellman/policy/wind overlays with [`RasterBrushCanvas`]
//! tactile and keyboard sculpting for [`super::GridWorldTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GridWorldTool;
use crate::accessibility::{
    draw_legend_swatch, paint_two_tone_circle, paint_two_tone_rect_frame, AccessibleTheoryHover,
};
use crate::app::{contrasting_text_color, SemanticPalette};
use crate::framework::{
    cell_screen_rect, BrushMode, InputMode, RasterBrushCanvas, RasterBrushStroke,
};
use crate::tabs::ai::q_table_inspector::q_value_cell_color;
pub use crate::tabs::ai::rl_sculpt::START_BADGE_GLYPH;
use crate::tabs::ai::rl_sculpt::{
    brush_mode_to_cell_role, cell_role_or_start_glyph, mean_bellman_residual, paint_policy_arrow,
    role_base_fill, role_glyph, RlOverlayMode, RlSolverMode,
};
use eframe::egui;
use math_explorer::ai::reinforcement_learning::{GridCellRole, GridState, MarkovDecisionProcess};

fn paint_start_marker(painter: &egui::Painter, cell_rect: egui::Rect, palette: &SemanticPalette) {
    paint_two_tone_rect_frame(
        painter,
        cell_rect.shrink(2.0_f32),
        2.0_f32,
        palette.info,
        1.6_f32,
    );
    let badge_size = (cell_rect.width().min(cell_rect.height()) * 0.34).clamp(11.0_f32, 16.0_f32);
    let badge_rect = egui::Rect::from_min_size(
        egui::pos2(
            cell_rect.right() - badge_size - 2.0_f32,
            cell_rect.top() + 2.0_f32,
        ),
        egui::vec2(badge_size, badge_size),
    );
    let badge_fg = contrasting_text_color(palette.info);
    painter.rect_filled(badge_rect, 2.0_f32, palette.info);
    painter.rect_stroke(
        badge_rect,
        2.0_f32,
        egui::Stroke::new(1.0_f32, badge_fg),
        egui::StrokeKind::Inside,
    );
    painter.text(
        badge_rect.center(),
        egui::Align2::CENTER_CENTER,
        cell_role_or_start_glyph(GridCellRole::Free, true).unwrap_or(START_BADGE_GLYPH),
        egui::FontId::proportional(10.0),
        badge_fg,
    );
}

fn draw_grid_world_legend(ui: &mut egui::Ui, palette: &SemanticPalette) {
    ui.horizontal_wrapped(|ui| {
        draw_legend_swatch(ui, palette.info, Some(START_BADGE_GLYPH), "Start [S]");
        draw_legend_swatch(ui, palette.success, Some("★"), "Goal [★]");
        draw_legend_swatch(ui, palette.error, Some("✕"), "Trap [✕]");
        draw_legend_swatch(ui, palette.graph_edge, Some("■"), "Wall [■]");
        draw_legend_swatch(ui, palette.surface_fill, Some("↑"), "Wind [↑↓←→]");
        draw_legend_swatch(ui, palette.accent, Some("◎"), "Agent [◎]");
    });
}

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
        paint_cell_role_glyph(painter, cell_rect, self.mdp.cell_role(state), fg);

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

        if state == self.mdp.start {
            paint_start_marker(painter, cell_rect, palette);
        }

        if state == self.current_state {
            paint_two_tone_circle(
                painter,
                cell_rect.center(),
                cell_rect.width() * 0.34,
                palette.accent,
                2.2_f32,
            );
        }
    }

    fn build_grid_summary(&self) -> String {
        let (mut traps, mut walls, mut winds) = (0_usize, 0_usize, 0_usize);
        for y in 0..self.mdp.height {
            for x in 0..self.mdp.width {
                match self.mdp.cell_role(GridState { x, y }) {
                    GridCellRole::Trap => traps += 1,
                    GridCellRole::SolidWall => walls += 1,
                    GridCellRole::WindUp
                    | GridCellRole::WindDown
                    | GridCellRole::WindLeft
                    | GridCellRole::WindRight => winds += 1,
                    GridCellRole::Free | GridCellRole::Goal => {}
                }
            }
        }
        format!(
            "Agent at ({}, {}) [◎], Start ({}, {}) [S], Goal ({}, {}) [★], Traps {traps} [✕], Walls {walls} [■], Wind cells {winds} [↑↓←→] (slip {:.2}), Episode {}, Reward {:.2}, Overlay {}; Color/shape key: Start [S] framed badge, Goal [★], Trap [✕], Wall [■], Wind [↑↓←→], Agent [◎] two-tone ring",
            self.current_state.x,
            self.current_state.y,
            self.mdp.start.x,
            self.mdp.start.y,
            self.mdp.goal.x,
            self.mdp.goal.y,
            self.mdp.wind_slip_prob,
            self.episodes,
            self.total_reward,
            self.overlay_mode.label()
        )
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

    fn compute_grid_layout(&self, ui: &egui::Ui, input_mode: InputMode) -> egui::Vec2 {
        let base_cell = if input_mode == InputMode::Touch {
            48.0_f32
        } else {
            44.0_f32
        };
        let max_dim = self.mdp.width.max(self.mdp.height).max(1) as f32;
        let fit_cell = ((ui.available_width().min(ui.available_height() - 28.0) - 16.0) / max_dim)
            .clamp(24.0, 64.0);
        let cell_size = base_cell.max(fit_cell).min(64.0);
        egui::vec2(
            self.mdp.width as f32 * cell_size,
            self.mdp.height as f32 * cell_size,
        )
    }

    fn apply_canvas_stroke(&mut self, stroke: RasterBrushStroke) {
        if !stroke.secondary
            && matches!(
                self.brush_kit.mode,
                BrushMode::WindUp
                    | BrushMode::WindDown
                    | BrushMode::WindLeft
                    | BrushMode::WindRight
            )
        {
            let role = brush_mode_to_cell_role(self.brush_kit.mode);
            for &(cx, cy) in &stroke.cells {
                let st = GridState {
                    x: cx as i32,
                    y: cy as i32,
                };
                if self.mdp.in_bounds(st) && st != self.mdp.goal {
                    self.mdp.set_cell_role(st, role);
                }
            }
            self.bellman_dirty = true;
            self.ensure_bellman_up_to_date();
            self.revision = self.revision.wrapping_add(1);
            return;
        }
        self.apply_brush_stroke(stroke);
    }

    pub(super) fn draw_central_canvas(&mut self, parent_ui: &mut egui::Ui, input_mode: InputMode) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.draw_telemetry_header(ui);
            let grid_size = self.compute_grid_layout(ui, input_mode);
            let gw = self.mdp.width.max(1) as usize;
            let gh = self.mdp.height.max(1) as usize;
            let summary = self.build_grid_summary();
            let bounds = self.overlay_bounds();
            let mut brush_state = std::mem::take(&mut self.brush_state);
            let (response, stroke) =
                RasterBrushCanvas::new("grid_world_canvas", gw, gh, "Grid World Visualization")
                    .role(egui::accesskit::Role::GraphicsDocument)
                    .value_summary(&summary)
                    .max_canvas_size(grid_size)
                    .show_custom(ui, &mut brush_state, |painter, rect, palette| {
                        for x in 0..gw {
                            for y in 0..gh {
                                let cell_rect = cell_screen_rect(rect, x, y, gw, gh, false);
                                let st = GridState {
                                    x: x as i32,
                                    y: y as i32,
                                };
                                self.paint_single_cell(painter, cell_rect, st, bounds, palette);
                            }
                        }
                    });
            self.brush_state = brush_state;
            let _ = response.accessible_theory_hover(self);
            let legend_palette = SemanticPalette::from_visuals(ui.visuals());
            draw_grid_world_legend(ui, &legend_palette);
            if let Some(stroke) = stroke {
                self.apply_canvas_stroke(stroke);
            }
        });
    }
}

fn paint_cell_role_glyph(
    painter: &egui::Painter,
    cell_rect: egui::Rect,
    role: GridCellRole,
    fg: egui::Color32,
) {
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
}
