#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Synchronized 3D twin-trajectory curves and scrubbable `log10(||delta x(t)||)` subplot.

use super::twin_state::{AttractorSystemKind, MIN_LOG_SEPARATION};
use super::AttractorPlotter;
use crate::accessibility::{announce_status, AccessibleTheoryHover};
use crate::app::SemanticPalette;
use crate::async_sim::ui::render_empty_plot_guidance;
use crate::framework::Plot3DViewport;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints, Points, VLine};

impl AttractorPlotter {
    pub(super) fn draw_twin_central_views(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        self.draw_central_fault_banner(ui, ctx);

        if self.history.is_empty() && !self.diverged() {
            render_empty_plot_guidance(
                ui,
                "Initial twin trajectories ready — click ▶ Play or ⏭ Step, then scrub or press Comma to rewind across chaotic divergence.",
            );
        }

        let avail = ui.available_size_before_wrap();
        if self.shadow_enabled {
            let top_height = (avail.y * 0.64).clamp(180.0, (avail.y - 125.0).max(180.0));
            ui.allocate_ui_with_layout(
                egui::vec2(avail.x, top_height),
                egui::Layout::top_down(egui::Align::Min),
                |top_ui| {
                    self.draw_3d_twin_viewport(top_ui);
                },
            );
            ui.add_space(4.0);
            let bottom_height = ui.available_height().max(110.0);
            self.draw_divergence_subplot(ui, ctx, bottom_height);
        } else {
            self.draw_3d_twin_viewport(ui);
        }
    }

    fn collect_projected_twin_markers(&self) -> (Option<[f64; 2]>, Option<[f64; 2]>) {
        let active_frame = self.timeline.current_copied();
        let primary_marker = active_frame
            .map(|f| self.project(f.primary_vec3()))
            .or_else(|| {
                self.is_finite_state()
                    .then(|| self.project(self.system.state.vec))
            });
        let shadow_marker = self
            .shadow_enabled
            .then(|| active_frame.map(|f| self.project(f.shadow_vec3())))
            .flatten();
        (primary_marker, shadow_marker)
    }

    fn draw_3d_twin_viewport(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let primary_color = palette.series_color(0);
        let shadow_color = palette.series_color(1);
        let primary_pts: Vec<[f64; 2]> = self.history.iter().map(|p| self.project(*p)).collect();
        let shadow_pts: Vec<[f64; 2]> = if self.shadow_enabled {
            self.shadow_history
                .iter()
                .map(|p| self.project(*p))
                .collect()
        } else {
            Vec::new()
        };
        let (primary_marker, shadow_marker) = self.collect_projected_twin_markers();

        let half_extent = self.system_kind.viewport_half_extent();
        let mut camera = self.camera;
        let response = Plot3DViewport::new("attractor_plot", half_extent)
            .show(ui, &mut camera, |plot_ui| {
                plot_ui.line(
                    Line::new("Trajectory", PlotPoints::new(primary_pts))
                        .color(primary_color)
                        .width(1.8_f32),
                );
                if self.shadow_enabled && !shadow_pts.is_empty() {
                    plot_ui.line(
                        Line::new("Shadow Trajectory", PlotPoints::new(shadow_pts))
                            .color(shadow_color)
                            .width(1.6_f32),
                    );
                }
                if let Some(pt) = shadow_marker {
                    plot_ui.points(
                        Points::new("Shadow State", vec![pt])
                            .radius(5.0_f32)
                            .color(shadow_color),
                    );
                }
                if let Some(pt) = primary_marker {
                    plot_ui.points(
                        Points::new("Primary State", vec![pt])
                            .radius(4.5_f32)
                            .color(primary_color),
                    );
                }
                if self.system_kind == AttractorSystemKind::Lorenz {
                    self.draw_fixed_point_callouts(plot_ui);
                } else {
                    self.clear_callout_drag(plot_ui);
                }
            })
            .response;
        self.camera = camera;
        response.accessible_theory_hover(self);
    }

    fn render_divergence_plot_series(
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &SemanticPalette,
        log_points: Vec<[f64; 2]>,
        slope_points: Vec<[f64; 2]>,
        current_time: f64,
        cursor_dot: Option<[f64; 2]>,
    ) -> Option<f64> {
        if !slope_points.is_empty() {
            plot_ui.line(
                Line::new("Wolf λ₁·t / ln(10)", PlotPoints::new(slope_points))
                    .color(palette.series_color(2))
                    .style(LineStyle::dashed_dense())
                    .width(1.5_f32),
            );
        }
        plot_ui.line(
            Line::new("log₁₀(||δx(t)||)", PlotPoints::new(log_points))
                .color(palette.series_color(1))
                .width(2.0_f32),
        );
        plot_ui.vline(
            VLine::new("Timeline Cursor", current_time)
                .color(palette.accent)
                .width(1.8_f32),
        );
        if let Some(dot) = cursor_dot {
            plot_ui.points(
                Points::new("Active Frame", vec![dot])
                    .radius(4.5_f32)
                    .color(palette.accent),
            );
        }
        let resp = plot_ui.response();
        (resp.clicked() || resp.dragged())
            .then(|| plot_ui.pointer_coordinate().map(|c| c.x))
            .flatten()
    }

    fn draw_divergence_subplot(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, height: f32) {
        self.ensure_lyapunov_cache();
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let log_points = self.build_log10_divergence_series();
        let slope_points = self.build_wolf_reference_series();
        let active_frame = self.timeline.current_copied();
        let current_time = active_frame.map_or(0.0, |f| f.time);
        let cursor_dot = active_frame.map(|f| [f.time, f.log10_separation]);
        let can_scrub = self.timeline.len() > 1;

        let mut scrubbed_time = None;
        Plot::new("attractor_divergence_plot")
            .height(height)
            .x_axis_label("Time t")
            .y_axis_label("log₁₀(||δx(t)||)")
            .legend(Legend::default())
            .allow_2d_analysis(false)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .show(ui, |plot_ui| {
                let maybe_t = Self::render_divergence_plot_series(
                    plot_ui,
                    &palette,
                    log_points,
                    slope_points,
                    current_time,
                    cursor_dot,
                );
                if can_scrub {
                    scrubbed_time = maybe_t;
                }
            });

        if let Some(target_t) = scrubbed_time {
            if let Some(idx) = self.nearest_frame_for_time(target_t) {
                if idx != self.timeline.cursor() {
                    self.seek_timeline_frame(idx, ctx);
                    announce_status(&format!(
                        "Scrubbed Attractor Plotter divergence cursor to frame {idx}"
                    ));
                }
            }
        }
    }

    pub(super) fn build_log10_divergence_series(&self) -> Vec<[f64; 2]> {
        let mut pts: Vec<[f64; 2]> = self
            .timeline
            .iter_all()
            .filter(|f| f.time.is_finite() && f.log10_separation.is_finite())
            .map(|f| [f.time, f.log10_separation])
            .collect();
        if pts.is_empty() {
            let y0 = self.epsilon.abs().max(MIN_LOG_SEPARATION).log10();
            pts.push([0.0, y0]);
        }
        pts
    }

    pub(super) fn build_wolf_reference_series(&self) -> Vec<[f64; 2]> {
        let Some(first) = self.timeline.get(0) else {
            return Vec::new();
        };
        let Some(last) = self.timeline.get(self.timeline.len().saturating_sub(1)) else {
            return Vec::new();
        };
        if last.time <= first.time || !self.metrics_cache.wolf_lambda1.is_finite() {
            return Vec::new();
        }
        let y0 = first.log10_separation;
        let slope = self.metrics_cache.wolf_lambda1 / std::f64::consts::LN_10;
        let dt_span = last.time - first.time;
        let y_end = (y0 + slope * dt_span).clamp(-16.0, 2.5);
        let t_cap = if slope.abs() > 1e-12 {
            first.time + ((y_end - y0) / slope).clamp(0.0, dt_span)
        } else {
            last.time
        };
        vec![[first.time, y0], [t_cap, y0 + slope * (t_cap - first.time)]]
    }

    fn nearest_frame_for_time(&self, target_t: f64) -> Option<usize> {
        if !target_t.is_finite() || self.timeline.is_empty() {
            return None;
        }
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        for (idx, frame) in self.timeline.iter_all().enumerate() {
            let dist = (frame.time - target_t).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        Some(best_idx)
    }
}
