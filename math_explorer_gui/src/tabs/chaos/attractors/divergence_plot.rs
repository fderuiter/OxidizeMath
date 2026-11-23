#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Synchronized 3D twin-trajectory curves and scrubbable `log10(||delta x(t)||)` subplot.

use super::twin_state::{AttractorSystemKind, MIN_LOG_SEPARATION};
use super::AttractorPlotter;
use crate::accessibility::{announce_status, AccessibleTheoryHover};
use crate::app::SemanticPalette;
use crate::async_sim::ui::render_empty_plot_guidance;
use crate::framework::Plot3DViewport;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, MarkerShape, Plot, PlotPoints, Points, VLine};
use nalgebra::Point3;

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

    fn draw_3d_twin_viewport(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let primary_color = palette.series_color(0);
        let shadow_color = palette.series_color(1);
        self.populate_twin_scene(primary_color, shadow_color);

        let half_extent = match self.mode {
            super::AttractorMode::Lorenz => self.system_kind.viewport_half_extent(),
            super::AttractorMode::Custom => 18.0,
        };
        let mut camera = self.camera;
        let response = Plot3DViewport::new("attractor_plot", half_extent)
            .show(ui, &mut camera, |plot_ui| {
                self.scene_builder.finish(plot_ui);
                if self.mode == super::AttractorMode::Lorenz
                    && self.system_kind == AttractorSystemKind::Lorenz
                {
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
            .allow_csv_import(false)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .allow_double_click_reset(false)
            .show_navigation_controls(false)
            .show_toolbar(false)
            .show_settings_button(false)
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

    /// Populates `self.scene_builder` with world `Point3<f64>` attractor trajectory segments.
    pub fn populate_scene(&mut self, series_color: egui::Color32) {
        let origin_offset = match self.mode {
            super::AttractorMode::Lorenz => {
                let c = self.system_kind.center_offset();
                Point3::new(c.x, c.y, c.z)
            }
            super::AttractorMode::Custom => Point3::origin(),
        };
        self.scene_builder.begin(self.camera);
        self.scene_builder.set_origin_offset(origin_offset);
        self.add_trajectory_curve("Trajectory", true, series_color, 1.8_f32);
    }

    pub(crate) fn populate_twin_scene(
        &mut self,
        primary_color: egui::Color32,
        shadow_color: egui::Color32,
    ) {
        self.populate_scene(primary_color);
        if self.shadow_enabled && !self.shadow_history.is_empty() {
            self.add_trajectory_curve("Shadow Trajectory", false, shadow_color, 1.6_f32);
        }
        self.add_active_twin_markers(primary_color, shadow_color);
    }

    fn add_trajectory_curve(
        &mut self,
        name: &str,
        primary: bool,
        color: egui::Color32,
        width: f32,
    ) {
        self.trajectory_scratch.clear();
        let iter = if primary {
            self.history.iter()
        } else {
            self.shadow_history.iter()
        };
        self.trajectory_scratch
            .extend(iter.map(|v| Point3::new(v.x, v.y, v.z)));
        if !self.trajectory_scratch.is_empty() {
            if self.trajectory_scratch.len() == 1 {
                self.scene_builder.add_points(
                    name,
                    &self.trajectory_scratch,
                    4.0_f32,
                    color,
                    MarkerShape::Circle,
                    true,
                );
            } else {
                self.scene_builder.add_polyline(
                    name,
                    &self.trajectory_scratch,
                    egui::Stroke::new(width, color),
                    LineStyle::Solid,
                    true,
                );
            }
        } else if primary && self.is_finite_state() {
            let v = self.system.state.vec;
            self.scene_builder.add_points(
                "Initial State",
                &[Point3::new(v.x, v.y, v.z)],
                4.0_f32,
                color,
                MarkerShape::Circle,
                true,
            );
        }
    }

    fn add_active_twin_markers(
        &mut self,
        primary_color: egui::Color32,
        shadow_color: egui::Color32,
    ) {
        let active_frame = self.timeline.current_copied();
        if self.shadow_enabled {
            if let Some(sv) = active_frame.map(|f| f.shadow_vec3()) {
                self.scene_builder.add_points(
                    "Shadow State",
                    &[Point3::new(sv.x, sv.y, sv.z)],
                    5.0_f32,
                    shadow_color,
                    MarkerShape::Circle,
                    false,
                );
            }
        }
        let primary_pt = active_frame
            .map(|f| f.primary_vec3())
            .or_else(|| self.is_finite_state().then_some(self.system.state.vec));
        if let Some(pv) = primary_pt {
            self.scene_builder.add_points(
                "Primary State",
                &[Point3::new(pv.x, pv.y, pv.z)],
                4.5_f32,
                primary_color,
                MarkerShape::Circle,
                false,
            );
        }
    }
}
