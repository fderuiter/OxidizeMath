#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BifurcationDiagram, BifurcationMapMode, CUSTOM_ESCAPE_GUARD};
use crate::accessibility::{AccessibleHoverText, AccessibleTheoryHover};
use crate::framework::bounded_drag_value;
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use egui_plot::{HLine, Line, LineStyle, Plot, Points, VLine};

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
                self.probe_r = self.probe_r.clamp(self.r_min, self.r_max);
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
        ui.add_space(4.0);
        if self.map_mode == BifurcationMapMode::Custom {
            changed |= self.math_input.ui(ui);
        }
        if ui
            .add(bounded_drag_value(&mut self.x0, -10.0..=10.0, 0.02).prefix("Initial x₀: "))
            .changed()
        {
            self.reset_live_orbit();
            changed |= self.map_mode == BifurcationMapMode::Custom;
        }
        changed
    }

    pub(super) fn draw_controls(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.heading("1D Map Bifurcation");
        ui.separator();
        let _ = self.atlas.show_toggle_button(ui);

        let mut changed = self.draw_map_mode_selector(ui);
        let (lo_bound, hi_bound) = match self.map_mode {
            BifurcationMapMode::Logistic => (0.0, 4.0),
            BifurcationMapMode::Custom => (-6.0, 6.0),
        };
        changed |= ui
            .add(
                AnimatedSlider::new(&mut self.r_min, lo_bound..=hi_bound)
                    .text("Min r")
                    .id_salt("bifurcation_r_min"),
            )
            .changed();
        if self.r_max <= self.r_min {
            self.r_max = (self.r_min + 0.05).min(hi_bound);
        }
        changed |= ui
            .add(
                AnimatedSlider::new(&mut self.r_max, self.r_min..=hi_bound)
                    .text("Max r")
                    .id_salt("bifurcation_r_max"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.steps, 100..=2000).text("Resolution (steps)"))
            .changed();

        if ui
            .button("▶ Recompute")
            .accessible_hover_text("Regenerate the bifurcation diagram on the background worker")
            .clicked()
            || changed
        {
            self.probe_r = self.probe_r.clamp(self.r_min, self.r_max);
            self.reset_live_orbit();
            self.dispatch_sweep_request(&ctx);
        }

        self.draw_audio_and_orbit_probe(ui);

        ui.separator();
        ui.label("Click diagram to set probe r • Wheel to zoom.");
    }

    fn draw_audio_and_orbit_probe(&mut self, ui: &mut egui::Ui) {
        self.probe_r = self.probe_r.clamp(self.r_min, self.r_max);
        ui.separator();
        if ui
            .add(
                AnimatedSlider::new(&mut self.probe_r, self.r_min..=self.r_max)
                    .text("Sweep r (Probe & Audio)")
                    .id_salt("bifurcation_probe_r"),
            )
            .changed()
        {
            self.reset_live_orbit();
        }
        ui.horizontal(|ui| {
            if ui.button("⏭ Step Orbit").clicked() {
                <Self as crate::framework::InteractiveTool>::step_simulation(self);
            }
            if ui.button("↺ Reset Orbit").clicked() {
                self.reset_live_orbit();
            }
        });
        let lam_txt = self
            .probe_lyapunov()
            .map_or_else(|| "divergent".to_string(), |l| format!("{l:+.3}"));
        ui.small(format!(
            "Step {}: x_n = {:.4}  |  λ(r={:.3}) = {}",
            self.orbit_step_count, self.orbit_x, self.probe_r, lam_txt
        ));
        let frame = crate::audio::mappings::bifurcation_frame(self.probe_r);
        crate::audio::render_sonification_panel(ui, "bifurcation", &frame, false);
    }

    pub(super) fn draw_bifurcation_and_cobweb(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let mut clicked_r: Option<f64> = None;
        let top_h = (ui.available_height() * 0.58).max(180.0);

        let response = Plot::new("bifurcation_plot")
            .height(top_h)
            .x_axis_label("Control Parameter (r)")
            .y_axis_label("State (x) / Lyapunov (λ)")
            .show(ui, |plot_ui| {
                plot_ui.points(
                    Points::new("Attractor", &self.points[..])
                        .radius(1.0_f32)
                        .color(palette.series_color(0)),
                );
                if !self.lyapunov_points.is_empty() {
                    plot_ui.line(
                        Line::new("Lyapunov λ(r)", &self.lyapunov_points[..])
                            .color(palette.series_color(1))
                            .width(1.5_f32),
                    );
                }
                plot_ui.hline(
                    HLine::new("λ = 0", 0.0)
                        .color(palette.graph_edge)
                        .style(LineStyle::Dashed { length: 4.0 }),
                );
                plot_ui.vline(VLine::new("Probe r", self.probe_r).color(palette.warning));
                if self.orbit_x.is_finite() && self.orbit_x.abs() <= CUSTOM_ESCAPE_GUARD {
                    plot_ui.points(
                        Points::new("Live Orbit x_n", vec![[self.probe_r, self.orbit_x]])
                            .radius(4.5_f32)
                            .color(palette.accent),
                    );
                }
                if plot_ui.response().clicked()
                    && let Some(pt) = plot_ui.pointer_coordinate()
                {
                    clicked_r = Some(pt.x.clamp(self.r_min, self.r_max));
                }
            })
            .response;
        response.accessible_theory_hover(self);
        if let Some(nr) = clicked_r {
            self.probe_r = nr;
            self.reset_live_orbit();
        }

        ui.separator();
        self.draw_cobweb_plot(ui, palette);
    }

    fn draw_cobweb_plot(&self, ui: &mut egui::Ui, palette: crate::app::SemanticPalette) {
        let next_y = self.eval_active_map(self.probe_r, self.orbit_x);
        let (d_lo, d_hi) = match self.map_mode {
            BifurcationMapMode::Logistic => (0.0, 1.0),
            BifurcationMapMode::Custom => (-2.0, 2.0),
        };
        Plot::new("bifurcation_cobweb_plot")
            .x_axis_label("x_n")
            .y_axis_label("x_{n+1}")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Identity y = x", vec![[d_lo, d_lo], [d_hi, d_hi]])
                        .color(palette.graph_edge)
                        .style(LineStyle::Dashed { length: 4.0 }),
                );
                plot_ui.line(
                    Line::new("Return Map f(r, x)", &self.cobweb_curve[..])
                        .color(palette.series_color(0))
                        .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new("Cobweb Staircase", &self.cobweb_points[..])
                        .color(palette.warning)
                        .width(1.4_f32),
                );
                if self.orbit_x.is_finite() && next_y.is_finite() {
                    plot_ui.points(
                        Points::new("Current Iterate", vec![[self.orbit_x, next_y]])
                            .radius(5.0_f32)
                            .color(palette.accent),
                    );
                }
            });
    }
}
