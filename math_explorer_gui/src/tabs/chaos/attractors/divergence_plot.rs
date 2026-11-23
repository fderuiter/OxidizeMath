//! Initial-state / twin perturbation controls and synchronized `log10(||Δx(t)||)` divergence plot.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::twin::{
    log10_separation, MAX_INITIAL_COORD, MAX_TWIN_EPSILON, MIN_INITIAL_COORD, MIN_SEPARATION_FLOOR,
    MIN_TWIN_EPSILON,
};
use super::{AttractorMode, AttractorPlotter};
use crate::app::SemanticPalette;
use crate::framework::bounded_drag_value;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, MarkerShape, Plot, PlotPoints, Points, VLine};

impl AttractorPlotter {
    pub(super) fn draw_initial_and_twin_controls(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
    ) {
        egui::CollapsingHeader::new("Initial State & Twin Perturbation")
            .default_open(true)
            .show(ui, |ui| {
                let mut x0 = self.twin_cfg.initial_state.x;
                let mut y0 = self.twin_cfg.initial_state.y;
                let mut z0 = self.twin_cfg.initial_state.z;
                let mut changed = false;

                ui.label("Initial Coordinates (x₀, y₀, z₀):");
                ui.horizontal(|ui| {
                    changed |= ui
                        .add(
                            bounded_drag_value(&mut x0, MIN_INITIAL_COORD..=MAX_INITIAL_COORD, 0.1)
                                .prefix("x₀: "),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            bounded_drag_value(&mut y0, MIN_INITIAL_COORD..=MAX_INITIAL_COORD, 0.1)
                                .prefix("y₀: "),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            bounded_drag_value(&mut z0, MIN_INITIAL_COORD..=MAX_INITIAL_COORD, 0.1)
                                .prefix("z₀: "),
                        )
                        .changed();
                });

                changed |= ui
                    .checkbox(&mut self.twin_cfg.enabled, "Perturbed Twin Trajectory")
                    .changed();

                changed |= ui
                    .add_enabled(
                        self.twin_cfg.enabled,
                        egui::Slider::new(
                            &mut self.twin_cfg.epsilon,
                            MIN_TWIN_EPSILON..=MAX_TWIN_EPSILON,
                        )
                        .logarithmic(true)
                        .text("Perturbation ε"),
                    )
                    .changed();

                if changed {
                    self.twin_cfg.initial_state.x = x0;
                    self.twin_cfg.initial_state.y = y0;
                    self.twin_cfg.initial_state.z = z0;
                    self.twin_cfg = self.twin_cfg.sanitized();
                    self.reset();
                    ctx.request_repaint();
                }
            });
    }

    pub(super) fn draw_divergence_panel(&mut self, parent_ui: &mut egui::Ui) {
        if !self.twin_cfg.enabled {
            return;
        }
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "twin_divergence_panel");
        egui::TopBottomPanel::bottom(panel_id)
            .resizable(true)
            .default_height(185.0)
            .height_range(120.0..=280.0)
            .show_inside(parent_ui, |ui| {
                let lyap = if self.mode == AttractorMode::Lorenz {
                    self.current_lyapunov_exponent()
                } else {
                    None
                };
                self.render_divergence_plot(ui, lyap);
            });
    }

    fn render_divergence_plot(&self, ui: &mut egui::Ui, lyapunov: Option<f64>) {
        let dt = if self.dt.is_finite() && self.dt > 0.0 {
            self.dt
        } else {
            0.01
        };
        let current_t = self.step_count as f64 * dt;
        let max_timeline_t = self
            .worker
            .timeline_metadata()
            .map_or(current_t, |m| m.max_step as f64 * dt)
            .max(current_t)
            .max(dt * 10.0);

        ui.horizontal(|ui| {
            ui.strong("Twin Separation: log₁₀ ||Δx(t)||");
            let current_sep = (self.twin_state.shadow_vec - self.system.state.vec).norm();
            if current_sep.is_finite() {
                ui.label(format!("| t = {current_t:.2} | ||Δx|| = {current_sep:.2e}"));
            }
        });

        let y0 = self.twin_cfg.epsilon.max(MIN_SEPARATION_FLOOR).log10();
        let (obs_min, obs_max) = observed_y_bounds(&self.twin_state.divergence_history, y0);
        let slope_segment = lyapunov.and_then(|lam| {
            build_lyapunov_slope_segment(y0, lam, max_timeline_t, obs_min, obs_max)
        });
        self.draw_divergence_egui_plot(
            ui,
            (current_t, max_timeline_t, y0),
            lyapunov.zip(slope_segment),
        );
    }

    fn draw_divergence_egui_plot(
        &self,
        ui: &mut egui::Ui,
        axes: (f64, f64, f64),
        slope: Option<(f64, Vec<[f64; 2]>)>,
    ) {
        let (current_t, max_timeline_t, y0) = axes;
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let curve_pts: PlotPoints<'_> = if self.twin_state.divergence_history.is_empty() {
            PlotPoints::from(vec![[0.0, y0]])
        } else {
            PlotPoints::from_iter(self.twin_state.divergence_history.iter().copied())
        };
        let current_log_d = log10_separation(self.system.state.vec, self.twin_state.shadow_vec);
        let plot_id = crate::framework::scoped_panel_id(ui, "twin_divergence_plot");
        Plot::new(plot_id)
            .legend(Legend::default())
            .x_axis_label("Time t")
            .y_axis_label("log₁₀ ||Δx(t)||")
            .include_x(0.0)
            .include_x(max_timeline_t)
            .include_y(y0 - 0.5)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("log₁₀ ||Δx(t)||", curve_pts)
                        .color(palette.series_color(1))
                        .width(1.8_f32),
                );
                if let Some((lam, seg)) = slope {
                    plot_ui.line(
                        Line::new(format!("Wolf λ₁ ≈ {lam:+.3} ref"), PlotPoints::from(seg))
                            .color(palette.series_color(2))
                            .style(LineStyle::Dashed { length: 6.0_f32 })
                            .width(1.5_f32),
                    );
                }
                plot_ui.vline(
                    VLine::new("Scrub Cursor", current_t)
                        .color(palette.accent)
                        .style(LineStyle::Dotted { spacing: 4.0_f32 })
                        .width(1.2_f32),
                );
                if current_log_d.is_finite() {
                    plot_ui.points(
                        Points::new("Current Step", vec![[current_t, current_log_d]])
                            .color(palette.series_color(1))
                            .radius(4.5_f32)
                            .shape(MarkerShape::Diamond),
                    );
                }
            });
    }
}

fn observed_y_bounds(
    history: &std::collections::VecDeque<[f64; 2]>,
    fallback_y: f64,
) -> (f64, f64) {
    let mut min_y = fallback_y;
    let mut max_y = fallback_y;
    for &[_, y] in history {
        if y.is_finite() {
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }
    (min_y, max_y)
}

fn build_lyapunov_slope_segment(
    y0: f64,
    lambda_1: f64,
    t_horizon: f64,
    obs_min: f64,
    obs_max: f64,
) -> Option<Vec<[f64; 2]>> {
    if !lambda_1.is_finite() || !y0.is_finite() || t_horizon <= 0.0 {
        return None;
    }
    let slope = lambda_1 / std::f64::consts::LN_10;
    let t_end = if slope > 1e-9 {
        let y_cap = obs_max.max(y0 + 1.0).max(1.5);
        ((y_cap - y0) / slope).clamp(0.0, t_horizon)
    } else if slope < -1e-9 {
        let y_floor = obs_min.min(y0 - 2.0).min(-15.0);
        ((y_floor - y0) / slope).clamp(0.0, t_horizon)
    } else {
        t_horizon
    };
    if t_end <= 0.0 {
        return None;
    }
    Some(vec![[0.0, y0], [t_end, y0 + slope * t_end]])
}
