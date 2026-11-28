#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    BifurcationDiagram, BifurcationMapMode, CUSTOM_ESCAPE_GUARD, MAX_SWEEP_PREVIEW_POINTS,
};
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility, AccessibleTheoryHover,
};
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{HLine, Line, LineStyle, Plot, Points, VLine};

impl BifurcationDiagram {
    pub(super) fn draw_coupled_plots(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let avail = ui.available_size();
        if avail.x < 20.0 || avail.y < 20.0 {
            return;
        }
        let gap = 8.0_f32;
        let left_w = ((avail.x - gap) * 0.58).max(40.0);
        let right_w = (avail.x - left_w - gap).max(40.0);
        let total_h = avail.y.max(60.0);
        let link_id = ui.id().with("bifurcation_r_axis_link");

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.set_width(left_w);
                let bif_h = ((total_h - gap) * 0.64).max(30.0);
                let lyap_h = (total_h - bif_h - gap).max(24.0);
                self.draw_bifurcation_plot(ui, &palette, left_w, bif_h, link_id);
                ui.add_space(gap);
                self.draw_lyapunov_plot(ui, &palette, left_w, lyap_h, link_id);
            });
            ui.add_space(gap);
            ui.vertical(|ui| {
                ui.set_width(right_w);
                let cob_h = ((total_h - gap) * 0.56).max(30.0);
                let orb_h = (total_h - cob_h - gap).max(24.0);
                self.draw_cobweb_plot(ui, &palette, right_w, cob_h);
                ui.add_space(gap);
                self.draw_orbit_plot(ui, &palette, right_w, orb_h);
            });
        });
    }

    fn apply_r_plot_interaction(&mut self, plot_ui: &egui_plot::PlotUi<'_>) -> bool {
        let mut r_changed = false;
        let mut x0_changed = false;
        let mut announce = false;
        let resp = plot_ui.response();

        let pointer_released = plot_ui
            .ctx()
            .input(|i| i.pointer.button_released(egui::PointerButton::Primary));

        if resp.has_focus() {
            self.focused_canvas = Some(resp.id);
            let (k_r, k_x0, k_ann) =
                self.consume_focused_r_keys(plot_ui.ctx(), resp, pointer_released);
            r_changed |= k_r;
            x0_changed |= k_x0;
            announce |= k_ann;
        }

        if !announce && resp.clicked() && pointer_released {
            resp.request_focus();
            if let Some(coord) = plot_ui.pointer_coordinate() {
                if coord.x.is_finite() {
                    let target_r = coord.x.clamp(self.r_min, self.r_max);
                    let tol = (self.r_max - self.r_min).abs() * 0.015;
                    if self.r_pinned && (self.selected_r - target_r).abs() <= tol {
                        self.r_pinned = false;
                    } else {
                        self.selected_r = target_r;
                        self.r_pinned = true;
                    }
                    self.r_from_hover = false;
                    r_changed = true;
                    announce = true;
                }
            }
        } else if !announce && resp.hovered() && !resp.dragged() && !self.r_pinned {
            if let Some(coord) = plot_ui.pointer_coordinate() {
                if coord.x.is_finite() {
                    let next_r = coord.x.clamp(self.r_min, self.r_max);
                    self.r_from_hover = true;
                    if (next_r - self.selected_r).abs() > 1e-9 {
                        self.selected_r = next_r;
                        r_changed = true;
                    }
                }
            }
        }

        if r_changed {
            self.probe_r = self.selected_r;
        }
        if x0_changed {
            self.recompute();
        } else if r_changed {
            self.recompute_coupled_views();
        }
        announce
    }

    fn consume_focused_r_keys(
        &mut self,
        ctx: &egui::Context,
        resp: &egui::Response,
        pointer_released: bool,
    ) -> (bool, bool, bool) {
        let (left, right, up, down, enter_key) = ctx.input_mut(|i| {
            (
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft),
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight),
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
                i.consume_key(egui::Modifiers::NONE, egui::Key::Enter),
            )
        });
        let any_ptr_released = ctx.input(|i| i.pointer.any_released());
        let enter = enter_key || (resp.clicked() && !pointer_released && !any_ptr_released);
        if left || right || up || down {
            ctx.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
        }
        let step_r = ((self.r_max - self.r_min) * 0.01).clamp(1e-4, 0.05);
        let (x_lo, x_hi) = match self.map_mode {
            BifurcationMapMode::Logistic => (0.001, 0.999),
            BifurcationMapMode::Custom => (-10.0, 10.0),
        };
        let mut r_changed = false;
        let mut x0_changed = false;
        if left {
            self.selected_r = (self.selected_r - step_r).clamp(self.r_min, self.r_max);
            r_changed = true;
        }
        if right {
            self.selected_r = (self.selected_r + step_r).clamp(self.r_min, self.r_max);
            r_changed = true;
        }
        if up {
            self.x0 = (self.x0 + 0.01).clamp(x_lo, x_hi);
            x0_changed = true;
        }
        if down {
            self.x0 = (self.x0 - 0.01).clamp(x_lo, x_hi);
            x0_changed = true;
        }
        if enter {
            self.r_pinned = !self.r_pinned;
        }
        if r_changed || enter {
            self.r_from_hover = false;
        }
        (r_changed, x0_changed, r_changed || x0_changed || enter)
    }

    fn sync_sweep_preview_points(&mut self, ctx: &egui::Context) -> bool {
        let previewing = self.completed_generation != self.request_generation
            || crate::reflective_ui::slider::has_active_sweeps(ctx);
        if !previewing || self.points.len() <= MAX_SWEEP_PREVIEW_POINTS {
            return false;
        }
        let stride = self.points.len().div_ceil(MAX_SWEEP_PREVIEW_POINTS);
        self.sweep_preview_points.clear();
        for pt in self.points.iter().step_by(stride) {
            self.sweep_preview_points.push(*pt);
        }
        true
    }

    fn draw_bifurcation_plot(
        &mut self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        width: f32,
        height: f32,
        link_id: egui::Id,
    ) {
        let use_preview = self.sync_sweep_preview_points(ui.ctx());
        let mut should_announce = false;
        let response = Plot::new(ui.id().with("bifurcation_plot"))
            .width(width)
            .height(height)
            .link_axis(link_id, [true, false])
            .x_axis_label("Growth Rate (r)")
            .y_axis_label("Attractor (x)")
            .show(ui, |plot_ui| {
                should_announce = self.apply_r_plot_interaction(plot_ui);
                let attractor_pts = if use_preview {
                    &self.sweep_preview_points[..]
                } else {
                    &self.points[..]
                };
                plot_ui.points(
                    Points::new("Attractor", attractor_pts)
                        .radius(1.0_f32)
                        .color(palette.series_color(0)),
                );
                let style = if self.r_pinned {
                    LineStyle::Solid
                } else {
                    LineStyle::dashed_dense()
                };
                plot_ui.vline(
                    VLine::new("Selected r", self.selected_r)
                        .color(palette.accent)
                        .width(if self.r_pinned { 2.0_f32 } else { 1.4_f32 })
                        .style(style),
                );
                if self.orbit_step_count > 0
                    && self.orbit_x.is_finite()
                    && self.orbit_x.abs() <= CUSTOM_ESCAPE_GUARD
                {
                    plot_ui.points(
                        Points::new("Live Orbit x_n", vec![[self.selected_r, self.orbit_x]])
                            .radius(4.5_f32)
                            .color(palette.warning),
                    );
                }
            })
            .response;

        self.finish_bifurcation_canvas(ui, response, should_announce);
    }

    fn finish_bifurcation_canvas(
        &self,
        ui: &mut egui::Ui,
        response: egui::Response,
        should_announce: bool,
    ) {
        lock_canvas_arrow_keys(ui, &response);
        paint_canvas_focus_ring(ui, &response);
        let status = format!(
            "r = {:.4} ({}), x0 = {:.3}, Lyapunov = {:.3}",
            self.selected_r,
            if self.r_pinned { "pinned" } else { "unpinned" },
            self.x0,
            self.selected_lyapunov
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "Bifurcation Diagram Canvas",
            Some(&status),
        );
        if should_announce {
            announce_live_status(ui.ctx(), format!("Bifurcation crosshair: {status}"));
        }
        response.accessible_theory_hover(self);
    }

    fn draw_lyapunov_plot(
        &mut self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        width: f32,
        height: f32,
        link_id: egui::Id,
    ) {
        let weak_color = ui.visuals().weak_text_color();
        let response = Plot::new(ui.id().with("bifurcation_lyapunov_plot"))
            .width(width)
            .height(height)
            .link_axis(link_id, [true, false])
            .x_axis_label("Growth Rate (r)")
            .y_axis_label("Lyapunov (λ)")
            .show(ui, |plot_ui| {
                let _ = self.apply_r_plot_interaction(plot_ui);
                plot_ui.hline(
                    HLine::new("Chaos Threshold (λ = 0)", 0.0)
                        .color(weak_color)
                        .style(LineStyle::dashed_loose()),
                );
                plot_ui.line(
                    Line::new("Lyapunov Exponent", &self.lyapunov_points[..])
                        .color(palette.series_color(1))
                        .width(1.5_f32),
                );
                plot_ui.vline(
                    VLine::new("Selected r", self.selected_r)
                        .color(palette.accent)
                        .width(if self.r_pinned { 2.0_f32 } else { 1.4_f32 }),
                );
                plot_ui.points(
                    Points::new(
                        "Selected λ",
                        vec![[self.selected_r, self.selected_lyapunov]],
                    )
                    .radius(4.0_f32)
                    .color(palette.accent),
                );
            })
            .response;
        lock_canvas_arrow_keys(ui, &response);
        paint_canvas_focus_ring(ui, &response);
    }

    fn draw_cobweb_plot(
        &mut self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        width: f32,
        height: f32,
    ) {
        let weak_color = ui.visuals().weak_text_color();
        let (x_lo, x_hi) = match self.map_mode {
            BifurcationMapMode::Logistic => (0.001, 0.999),
            BifurcationMapMode::Custom => (-10.0, 10.0),
        };
        let _ = Plot::new(ui.id().with("bifurcation_cobweb_plot"))
            .width(width)
            .height(height)
            .data_aspect(1.0)
            .x_axis_label("x_n")
            .y_axis_label("x_{n+1} = f(x_n, r)")
            .show(ui, |plot_ui| {
                if plot_ui.response().clicked() {
                    if let Some(coord) = plot_ui.pointer_coordinate() {
                        if coord.x.is_finite() {
                            self.x0 = coord.x.clamp(x_lo, x_hi);
                            self.recompute();
                        }
                    }
                }
                plot_ui.line(
                    Line::new("y = x", &self.cobweb_diagonal[..])
                        .color(weak_color)
                        .style(LineStyle::dashed_loose()),
                );
                plot_ui.line(
                    Line::new("y = f(x, r)", &self.cobweb_curve[..])
                        .color(palette.series_color(0))
                        .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new("Transient Staircase", &self.cobweb_transient[..])
                        .color(palette.series_color(2).gamma_multiply(0.45_f32))
                        .width(1.0_f32),
                );
                plot_ui.line(
                    Line::new("Cobweb Cycle", &self.cobweb_cycle[..])
                        .color(palette.accent)
                        .width(2.2_f32),
                );
            });
    }

    fn draw_orbit_plot(
        &self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        width: f32,
        height: f32,
    ) {
        let _ = Plot::new(ui.id().with("bifurcation_orbit_plot"))
            .width(width)
            .height(height)
            .x_axis_label("Iteration (n)")
            .y_axis_label("State (x_n)")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Orbit x_n", &self.orbit_points[..])
                        .color(palette.series_color(2))
                        .width(1.5_f32),
                );
                plot_ui.points(
                    Points::new("Orbit Samples", &self.orbit_points[..])
                        .radius(2.2_f32)
                        .color(palette.series_color(2)),
                );
                if self.orbit_step_count > 0
                    && self.orbit_x.is_finite()
                    && self.orbit_x.abs() <= CUSTOM_ESCAPE_GUARD
                {
                    plot_ui.points(
                        Points::new(
                            "Active Step",
                            vec![[self.orbit_step_count as f64, self.orbit_x]],
                        )
                        .radius(4.5_f32)
                        .color(palette.accent),
                    );
                }
            });
    }
}
