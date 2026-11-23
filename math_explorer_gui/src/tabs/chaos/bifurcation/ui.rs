#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    build_cobweb_and_orbit, build_custom_cobweb_and_orbit, BifurcationDiagram, BifurcationMapMode,
    COBWEB_ORBIT_STEPS,
};
use crate::accessibility::{AccessibleHoverText, AccessibleTheoryHover};
use crate::framework::bounded_drag_value;
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use egui_plot::{HLine, Line, Plot, PlotPoints, Points, VLine};

impl BifurcationDiagram {
    fn draw_map_mode_selector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            if ui
                .radio_value(&mut self.map_mode, BifurcationMapMode::Logistic, "Logistic")
                .clicked()
            {
                self.r_min = self.r_min.clamp(0.0, 3.95);
                self.r_max = self.r_max.clamp(self.r_min + 0.05, 4.0);
                changed = true;
            }
            if ui
                .radio_value(
                    &mut self.map_mode,
                    BifurcationMapMode::Custom,
                    "Custom f(r, x)",
                )
                .clicked()
            {
                changed = true;
            }
        });
        if self.map_mode == BifurcationMapMode::Custom {
            ui.add_space(4.0);
            changed |= self.math_input.ui(ui);
            changed |= ui
                .add(bounded_drag_value(&mut self.x0, -10.0..=10.0, 0.02).prefix("Initial x₀: "))
                .changed();
        }
        changed
    }

    fn draw_lyapunov_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        let (sel_lo, sel_hi) = (self.r_min, self.r_max.max(self.r_min));
        self.selected_r = self.selected_r.clamp(sel_lo, sel_hi);
        ui.add(egui::Slider::new(&mut self.selected_r, sel_lo..=sel_hi).text("Selected r"));

        if self.map_mode == BifurcationMapMode::Logistic {
            ui.checkbox(&mut self.show_lyapunov, "Overlay Lyapunov λ(r)");
            ui.add(egui::Slider::new(&mut self.orbit_x0, 0.01..=0.99).text("Orbit Seed (x₀)"));
            let lambda = self.selected_lyapunov();
            let regime = if lambda > 0.0 {
                "Chaotic (λ > 0)"
            } else {
                "Stable / Periodic (λ ≤ 0)"
            };
            ui.label(format!(
                "λ({:.3}) = {:+.4} — {regime}",
                self.selected_r, lambda
            ));
        }
        if let Some((done, total)) = self.sweep_progress {
            ui.small(format!("Sweeping r: {done}/{total}"));
        }
        ui.separator();
        ui.small("Click bifurcation plot to inspect cobweb & x_n orbit.");
    }

    pub(super) fn draw_side_panel(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "bifurcation_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("1D Map Bifurcation");
            ui.separator();

            let mut sweep_changed = self.draw_map_mode_selector(ui);
            let (lo_bound, hi_bound): (f64, f64) = match self.map_mode {
                BifurcationMapMode::Logistic => (0.0, 4.0),
                BifurcationMapMode::Custom => (-6.0, 6.0),
            };
            let max_r_min = (hi_bound - 0.01).max(lo_bound);
            sweep_changed |= ui
                .add(
                    AnimatedSlider::new(&mut self.r_min, lo_bound..=max_r_min)
                        .text("Min r")
                        .id_salt("bifurcation_r_min"),
                )
                .changed();

            let min_bound = (self.r_min + 0.01).min(hi_bound);
            if self.r_max < min_bound {
                self.r_max = min_bound;
            }
            sweep_changed |= ui
                .add(
                    AnimatedSlider::new(&mut self.r_max, min_bound..=hi_bound)
                        .text("Max r")
                        .id_salt("bifurcation_r_max"),
                )
                .changed();
            sweep_changed |= ui
                .add(egui::Slider::new(&mut self.steps, 100..=2000).text("Resolution (steps)"))
                .changed();

            if ui
                .button("▶ Recompute")
                .accessible_hover_text(
                    "Regenerate the bifurcation diagram with the current parameters",
                )
                .clicked()
                || sweep_changed
            {
                self.sanitize_bounds();
                self.dirty = true;
            }
            self.draw_lyapunov_controls(ui);
            self.draw_audio_probe(ui);
        });
    }

    fn draw_audio_probe(&mut self, ui: &mut egui::Ui) {
        self.probe_r = self.probe_r.clamp(self.r_min, self.r_max);
        ui.separator();
        let _ = ui.add(
            AnimatedSlider::new(&mut self.probe_r, self.r_min..=self.r_max)
                .text("Sweep r (Audio)")
                .id_salt("bifurcation_probe_r"),
        );
        let frame = crate::audio::mappings::bifurcation_frame(self.probe_r);
        crate::audio::render_sonification_panel(ui, "bifurcation", &frame, false);
    }

    fn draw_inspector_plots(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        let bottom_h = ui.available_height().max(100.0);
        let (series, diag_bounds) = match self.map_mode {
            BifurcationMapMode::Logistic => (
                build_cobweb_and_orbit(self.selected_r, self.orbit_x0, COBWEB_ORBIT_STEPS),
                [0.0, 1.0],
            ),
            BifurcationMapMode::Custom => {
                let s = build_custom_cobweb_and_orbit(
                    self.selected_r,
                    self.x0,
                    COBWEB_ORBIT_STEPS,
                    &self.math_input,
                );
                let lo = s.parabola.first().map_or(-1.0, |p| p[0]);
                let hi = s.parabola.last().map_or(1.0, |p| p[0]);
                (s, [lo, hi])
            }
        };
        ui.columns(2, |cols| {
            Plot::new("logistic_cobweb_plot")
                .height(bottom_h)
                .x_axis_label("x_n")
                .y_axis_label("x_{n+1}")
                .show(&mut cols[0], |plot_ui| {
                    plot_ui.line(
                        Line::new("f_r(x)", PlotPoints::new(series.parabola))
                            .color(palette.series_color(0)),
                    );
                    plot_ui.line(
                        Line::new(
                            "y = x",
                            PlotPoints::new(vec![
                                [diag_bounds[0], diag_bounds[0]],
                                [diag_bounds[1], diag_bounds[1]],
                            ]),
                        )
                        .color(palette.series_color(2)),
                    );
                    plot_ui.line(
                        Line::new("Cobweb", PlotPoints::new(series.cobweb))
                            .color(palette.series_color(1)),
                    );
                });
            Plot::new("logistic_orbit_plot")
                .height(bottom_h)
                .x_axis_label("Iteration (n)")
                .y_axis_label("State (x_n)")
                .show(&mut cols[1], |plot_ui| {
                    plot_ui.line(
                        Line::new("Orbit x_n", PlotPoints::new(series.orbit))
                            .color(palette.series_color(3)),
                    );
                });
        });
    }

    pub(super) fn draw_central_panel(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let top_h = (ui.available_height().max(240.0) * 0.56).max(130.0);
            let mut clicked_r: Option<f64> = None;
            let is_logistic = self.map_mode == BifurcationMapMode::Logistic;

            let plot_resp = Plot::new("bifurcation_plot")
                .height(top_h)
                .x_axis_label("Control Parameter (r)")
                .y_axis_label("Population (x) / Lyapunov (λ)")
                .show(ui, |plot_ui| {
                    plot_ui.points(
                        Points::new("Attractor", &self.points[..])
                            .radius(1.0_f32)
                            .color(palette.series_color(0)),
                    );
                    if is_logistic && self.show_lyapunov && !self.lyapunov_points.is_empty() {
                        plot_ui.hline(
                            HLine::new("Stability Threshold (λ = 0)", 0.0)
                                .color(palette.series_color(2)),
                        );
                        plot_ui.line(
                            Line::new("Lyapunov λ(r)", &self.lyapunov_points[..])
                                .color(palette.series_color(1)),
                        );
                    }
                    plot_ui.vline(
                        VLine::new("Selected r", self.selected_r).color(palette.series_color(3)),
                    );
                    plot_ui.vline(VLine::new("Audio Probe r", self.probe_r).color(palette.warning));
                    if plot_ui.response().clicked() && !plot_ui.response().dragged() {
                        clicked_r = plot_ui.pointer_coordinate().map(|c| c.x);
                    }
                });
            plot_resp.response.accessible_theory_hover(self);
            if let Some(r_val) = clicked_r {
                self.select_r(r_val);
            }
            ui.separator();
            self.draw_inspector_plots(ui, &palette);
        });
    }
}
