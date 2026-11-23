#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Accessible pointer click-to-seed, keyboard `Arrow` + `Enter` cursor seeding,
//! and `PhasePortrait` `egui_plot` drawing for [`OdeSolverTool`].

use super::PlotViewMode;
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility,
};
use crate::framework::recovery::FiniteStateGuard;
use crate::tabs::analysis::ode::OdeSolverTool;
use eframe::egui;
use egui_plot::{Arrows, Line, LineStyle, Plot, PlotPoint, PlotPoints, PlotUi, Points, VLine};
use std::sync::Arc;

impl OdeSolverTool {
    /// Seeds initial conditions from canvas coordinates, pushing the prior run to `snapshot_manager`.
    pub fn seed_from_canvas_point(&mut self, ctx: Option<&egui::Context>, x: f64, y: f64) {
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        self.pending_sweep_run = None;
        let prev_summary = self.last_param_summary.clone();
        let prev_series = self.build_current_trace_series();
        self.snapshot_manager
            .push_history_run(prev_summary, prev_series);

        let msg = if self.is_2d_phase_portrait() {
            self.ic_y0 = x.clamp(-1e3, 1e3);
            self.ic_v0 = y.clamp(-1e3, 1e3);
            self.phase_cursor = Some([self.ic_y0, self.ic_v0]);
            format!(
                "Seeded phase trajectory at y(0) = {:.3}, v(0) = {:.3}",
                self.ic_y0, self.ic_v0
            )
        } else {
            self.ic_y0 = y.clamp(-1e3, 1e3);
            self.phase_cursor = Some([x.clamp(0.0, self.total_time.max(0.0)), self.ic_y0]);
            format!("Seeded trajectory at y(0) = {:.3}", self.ic_y0)
        };
        self.recalculate();
        if let Some(egui_ctx) = ctx {
            announce_live_status(egui_ctx, msg);
        }
    }

    fn consume_canvas_keyboard_cursor(&mut self, ui: &mut egui::Ui) -> Option<bool> {
        let (xb, yb) = self.phase_window_bounds();
        let default_cur = if self.is_2d_phase_portrait() {
            [self.ic_y0.clamp(xb.0, xb.1), self.ic_v0.clamp(yb.0, yb.1)]
        } else {
            [self.t_slice.clamp(xb.0, xb.1), self.ic_y0.clamp(yb.0, yb.1)]
        };
        let mut cur = self.phase_cursor.unwrap_or(default_cur);
        let dx = ((xb.1 - xb.0) * 0.04).max(1e-3);
        let dy = ((yb.1 - yb.0) * 0.04).max(1e-3);
        let mut moved = false;
        let mut seed_now = false;

        ui.input_mut(|i| {
            for (key, idx, delta) in [
                (egui::Key::ArrowLeft, 0usize, -dx),
                (egui::Key::ArrowRight, 0usize, dx),
                (egui::Key::ArrowDown, 1usize, -dy),
                (egui::Key::ArrowUp, 1usize, dy),
            ] {
                if i.consume_key(egui::Modifiers::NONE, key) {
                    cur[idx] = if idx == 0 {
                        (cur[0] + delta).clamp(xb.0, xb.1)
                    } else {
                        (cur[1] + delta).clamp(yb.0, yb.1)
                    };
                    moved = true;
                }
            }
            if i.consume_key(egui::Modifiers::NONE, egui::Key::Enter) {
                seed_now = true;
            }
        });

        if moved || self.phase_cursor.is_none() {
            self.phase_cursor = Some(cur);
        }
        (moved || seed_now).then_some(seed_now)
    }

    /// Consumes `Arrow` + `Enter` keys for seed-cursor navigation when `ode_plot` holds focus,
    /// running before `egui_plot::Plot::show` so arrow presses move the seed cursor cleanly.
    pub(crate) fn handle_pre_plot_keyboard_seeding(
        &mut self,
        ui: &mut egui::Ui,
        changed: &mut bool,
    ) {
        let is_plot_focused = self
            .focused_canvas
            .is_some_and(|id| ui.memory(|m| m.has_focus(id)));
        if !is_plot_focused {
            return;
        }
        let ctx = ui.ctx().clone();
        if let Some(seed_now) = self.consume_canvas_keyboard_cursor(ui) {
            if let Some([kx, ky]) = self.phase_cursor {
                if seed_now {
                    self.seed_from_canvas_point(Some(&ctx), kx, ky);
                    self.sync_series_cache();
                    *changed = true;
                } else {
                    let spec = self.build_custom_ode_spec();
                    let [fx, fy] = self.evaluate_field_vector(kx, ky, spec.as_ref());
                    announce_live_status(
                        &ctx,
                        format!(
                            "Phase cursor ({kx:.2}, {ky:.2}) — vector ({fx:.3}, {fy:.3}); press Enter to seed"
                        ),
                    );
                }
            }
        }
    }

    pub(crate) fn attach_ode_plot_accessibility_and_seeding(
        &mut self,
        ui: &mut egui::Ui,
        raw_resp: egui::Response,
        hover_coord: Option<[f64; 2]>,
        changed: &mut bool,
    ) {
        let focus_resp = raw_resp.interact(egui::Sense::click());
        let mode_label = match self.view_mode {
            PlotViewMode::TimeSeries => "ODE Solution Trajectory Canvas",
            PlotViewMode::PhasePortrait if self.has_second_state() => {
                "ODE 2D Phase Portrait Canvas"
            }
            PlotViewMode::PhasePortrait => "ODE 1D Slope Field Canvas",
        };
        let a11y_val = format!(
            "y(0) = {:.2}, v(0) = {:.2}, t_slice = {:.2}",
            self.ic_y0, self.ic_v0, self.t_slice
        );
        register_canvas_accessibility(
            ui,
            &focus_resp,
            egui::accesskit::Role::Canvas,
            mode_label,
            Some(&a11y_val),
        );
        lock_canvas_arrow_keys(ui, &focus_resp);
        paint_canvas_focus_ring(ui, &focus_resp);
        let has_focus = focus_resp.has_focus() || ui.memory(|m| m.has_focus(focus_resp.id));
        self.focused_canvas = has_focus.then_some(focus_resp.id);

        if focus_resp.clicked() {
            focus_resp.request_focus();
            self.focused_canvas = Some(focus_resp.id);
            if let Some([hx, hy]) = hover_coord {
                let ctx = ui.ctx().clone();
                self.seed_from_canvas_point(Some(&ctx), hx, hy);
                *changed = true;
            }
        }
    }

    pub(crate) fn render_phase_portrait_canvas(
        &mut self,
        ui: &mut egui::Ui,
        palette: &crate::app::SemanticPalette,
        changed: &mut bool,
    ) {
        self.handle_pre_plot_keyboard_seeding(ui, changed);
        let is_2d = self.has_second_state();
        let (origins, tips) = self.quiver_vectors();
        let (x_label, y_label, aspect) = if is_2d {
            ("y(t)", self.second_state_label(), 1.25_f32)
        } else {
            ("t \\text{ (Time)}", "y(t)", 2.0_f32)
        };
        let weak = ui.visuals().weak_text_color();
        let quiver_color = egui::Color32::from_rgba_unmultiplied(weak.r(), weak.g(), weak.b(), 190);
        let mut hover_coord = None;
        let plot_resp = Plot::new("ode_plot")
            .view_aspect(aspect)
            .x_axis_label(x_label)
            .y_axis_label(y_label)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                self.draw_phase_plot_contents(plot_ui, palette, is_2d, quiver_color, origins, tips);
                hover_coord = plot_ui.pointer_coordinate().map(|pt| [pt.x, pt.y]);
            });

        self.attach_ode_plot_accessibility_and_seeding(
            ui,
            plot_resp.response,
            hover_coord,
            changed,
        );
    }

    fn draw_phase_plot_contents(
        &mut self,
        plot_ui: &mut PlotUi<'_>,
        palette: &crate::app::SemanticPalette,
        is_2d: bool,
        quiver_color: egui::Color32,
        origins: Arc<Vec<PlotPoint>>,
        tips: Arc<Vec<PlotPoint>>,
    ) {
        let quiver_name = if is_2d { "Vector Field" } else { "Slope Field" };
        plot_ui.arrows(
            Arrows::new(
                quiver_name,
                PlotPoints::from_shared(origins),
                PlotPoints::from_shared(tips),
            )
            .color(quiver_color),
        );
        if !is_2d {
            plot_ui.vline(
                VLine::new("t_slice", self.t_slice)
                    .color(palette.info)
                    .style(LineStyle::Dashed { length: 4.0 }),
            );
        }
        if is_2d {
            self.draw_2d_phase_trajectories(plot_ui, palette);
        } else {
            self.draw_1d_slope_trajectories(plot_ui, palette);
        }
        if let Some(kb) = self.phase_cursor {
            plot_ui.points(
                Points::new("Seed Cursor", vec![kb])
                    .radius(4.5_f32)
                    .color(palette.accent),
            );
        }
        if self.show_annotations && !FiniteStateGuard::diverged(self) {
            crate::tabs::analysis::ode::annotations::render_ode_callouts(plot_ui, self);
        }
        self.callout_manager.render_on_plot(plot_ui);
    }

    fn draw_2d_phase_trajectories(
        &self,
        plot_ui: &mut PlotUi<'_>,
        palette: &crate::app::SemanticPalette,
    ) {
        if !self.comparison_mode {
            plot_ui.line(
                Line::new(
                    "Phase Orbit (y, v)",
                    PlotPoints::from_shared(Arc::clone(&self.display_phase)),
                )
                .color(palette.series_color(0))
                .width(2.0_f32),
            );
            if self.ic_y0.is_finite() && self.ic_v0.is_finite() {
                plot_ui.points(
                    Points::new("Initial State", vec![[self.ic_y0, self.ic_v0]])
                        .radius(5.0_f32)
                        .color(palette.series_color(0)),
                );
            }
        } else {
            for (idx, (name, pts)) in self.projected_comparison_orbits().into_iter().enumerate() {
                plot_ui.line(
                    Line::new(name, PlotPoints::from_shared(pts))
                        .color(palette.series_color(idx))
                        .width(1.8_f32),
                );
            }
        }
        for (idx, (label, pts, is_history)) in
            self.projected_snapshot_orbits().into_iter().enumerate()
        {
            let base = palette.series_color(idx + 2);
            let (col, dash, width) = if is_history {
                (
                    egui::Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 160),
                    5.0_f32,
                    1.2_f32,
                )
            } else {
                (base, 6.0_f32, 1.5_f32)
            };
            plot_ui.line(
                Line::new(label, PlotPoints::from_shared(pts))
                    .color(col)
                    .style(LineStyle::Dashed { length: dash })
                    .width(width),
            );
        }
    }

    fn draw_1d_slope_trajectories(
        &self,
        plot_ui: &mut PlotUi<'_>,
        palette: &crate::app::SemanticPalette,
    ) {
        if !self.comparison_mode {
            plot_ui.line(
                Line::new("y(t)", PlotPoints::from_shared(Arc::clone(&self.display_y)))
                    .color(palette.series_color(0))
                    .width(2.0_f32),
            );
        } else {
            self.comparison_manager.render_on_plot(plot_ui);
        }
        self.snapshot_manager.render_on_plot(plot_ui);
    }
}
