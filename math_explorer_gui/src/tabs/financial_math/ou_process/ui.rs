//! Interactive UI controls, confidence-band plot, and snippet bar for [`super::OuProcessTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{OuProcessTool, MAX_OU_PATHS, MAX_OU_STEPS};
use crate::accessibility::AccessibleHoverText;
use crate::framework::bounded_drag_value;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, MarkerShape, Plot, Points};

impl OuProcessTool {
    fn show_sde_grid(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::Grid::new("ou_params_grid")
            .num_columns(6)
            .spacing([14.0, 6.0])
            .show(ui, |ui| {
                ui.label("Equilibrium (μ):");
                changed |= ui
                    .add(bounded_drag_value(&mut self.mu, -1000.0..=1000.0, 0.01))
                    .accessible_hover_text("Long-term mean-reversion target mu")
                    .changed();
                ui.label("Reversion (θ):");
                changed |= ui
                    .add(bounded_drag_value(&mut self.theta, 0.01..=50.0, 0.02))
                    .accessible_hover_text("Mean-reversion speed theta (> 0)")
                    .changed();
                ui.label("Volatility (σ):");
                changed |= ui
                    .add(bounded_drag_value(&mut self.sigma, 0.0..=50.0, 0.01))
                    .accessible_hover_text("Diffusion volatility sigma (>= 0)")
                    .changed();
                ui.end_row();
                ui.label("Initial State (X₀):");
                changed |= ui
                    .add(bounded_drag_value(&mut self.x0, -1000.0..=1000.0, 0.01))
                    .changed();
                ui.label("Solver Δt:");
                changed |= ui
                    .add(bounded_drag_value(&mut self.dt, 0.001..=2.0, 0.005))
                    .changed();
                ui.label("CSV Sample Δt:");
                changed |= ui
                    .add(bounded_drag_value(&mut self.sample_dt, 0.001..=100.0, 0.01))
                    .changed();
                ui.end_row();
            });
        changed
    }

    fn show_parameter_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = self.show_sde_grid(ui);
        ui.horizontal_wrapped(|ui| {
            ui.label("Steps:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.n_steps).range(10..=MAX_OU_STEPS))
                .changed();
            ui.label("MC Paths:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.n_paths).range(8..=MAX_OU_PATHS))
                .changed();
            ui.label("Target Threshold:");
            changed |= ui
                .add(bounded_drag_value(
                    &mut self.threshold,
                    -1000.0..=1000.0,
                    0.01,
                ))
                .changed();
            ui.label("RNG Seed:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.rng_seed).speed(1.0))
                .accessible_hover_text("Deterministic OxidizeRng seed for Euler-Maruyama fan")
                .changed();
            if ui
                .button("📂 Upload CSV & Calibrate")
                .accessible_hover_text(
                    "Upload UTF-8 or BOM CSV series to fit (mu, theta, sigma) via estimate_ou_params",
                )
                .clicked()
            {
                self.trigger_open_csv();
            }
            if ui
                .button("🔄 Recalibrate Observed")
                .accessible_hover_text("Re-run estimate_ou_params on loaded series with current CSV Sample dt")
                .clicked()
            {
                if let Err(err) = self.recalibrate_from_observed() {
                    self.io_state.status = Some((err, true));
                }
            }
        });
        changed
    }

    fn show_diagnostics_banner(
        &self,
        ui: &mut egui::Ui,
        palette: &crate::app::SemanticPalette,
    ) -> bool {
        if let Some(warn) = self.url_share_warning() {
            ui.colored_label(palette.warning, format!("⚠ {warn}"));
        }
        if let Some(ref err) = self.calc_error {
            ui.colored_label(palette.error, err);
            return false;
        }
        ui.horizontal_wrapped(|ui| {
            ui.strong(format!("Regime: {}", self.momentum_class.description()));
            ui.separator();
            ui.label(format!("Half-Life t₁/₂: {:.3}", self.half_life));
            ui.separator();
            ui.label(format!(
                "P(X_T > {:.2}): {:.1}%",
                self.threshold,
                self.exceedance_prob * 100.0
            ));
            ui.separator();
            ui.label(format!(
                "Observed CSV Points: {}",
                self.observed_series.len()
            ));
        });
        true
    }

    fn show_diagnostics_and_snippets(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if !self.show_diagnostics_banner(ui, &palette) {
            return;
        }
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("🦀 Copy Rust")
                .accessible_hover_text("Copy calibrated Rust EulerMaruyama snippet")
                .clicked()
            {
                let code = self.rust_snippet();
                self.copy_payload(ui.ctx(), "calibrated Rust OU snippet", code);
            }
            if ui
                .button("🐍 Copy Python")
                .accessible_hover_text("Copy calibrated Python NumPy SDE snippet")
                .clicked()
            {
                let code = self.python_snippet();
                self.copy_payload(ui.ctx(), "calibrated Python OU snippet", code);
            }
            if ui
                .button("∑ Copy LaTeX")
                .accessible_hover_text("Copy calibrated LaTeX SDE formula")
                .clicked()
            {
                let tex = self.latex_snippet();
                self.copy_payload(ui.ctx(), "calibrated LaTeX SDE", tex);
            }
            ui.separator();
            if ui
                .button("💾 Save CSV Bands")
                .accessible_hover_text("Download or save Euler-Maruyama percentile bands as CSV")
                .clicked()
            {
                self.trigger_save_csv();
            }
            if ui
                .button("💾 Save JSON")
                .accessible_hover_text("Download or save OU calibration & bands as JSON")
                .clicked()
            {
                self.trigger_save_json();
            }
        });
        if let Some((ref msg, is_err)) = self.io_state.status {
            let color = if is_err { palette.error } else { palette.info };
            ui.colored_label(color, msg);
        }
    }

    fn show_confidence_bands_plot(&self, ui: &mut egui::Ui) {
        if self.mean_points.is_empty() {
            return;
        }
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        Plot::new("ou_confidence_bands_plot")
            .view_aspect(2.15)
            .legend(Legend::default())
            .x_axis_label("Simulation Time (t)")
            .y_axis_label("Process State X(t)")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Equilibrium μ", &self.eq_line_points[..])
                        .color(palette.graph_edge)
                        .style(LineStyle::Dashed { length: 6.0 }),
                );
                plot_ui.line(
                    Line::new("95th Percentile Band", &self.p95_points[..])
                        .color(palette.series_color(4))
                        .style(LineStyle::Dotted { spacing: 5.0 }),
                );
                plot_ui.line(
                    Line::new("75th Percentile Band", &self.p75_points[..])
                        .color(palette.series_color(2)),
                );
                plot_ui.line(
                    Line::new("Monte Carlo Mean E[X_t]", &self.mean_points[..])
                        .color(palette.series_color(0))
                        .width(2.3_f32),
                );
                plot_ui.line(
                    Line::new("25th Percentile Band", &self.p25_points[..])
                        .color(palette.series_color(2)),
                );
                plot_ui.line(
                    Line::new("5th Percentile Band", &self.p05_points[..])
                        .color(palette.series_color(4))
                        .style(LineStyle::Dotted { spacing: 5.0 }),
                );
                if !self.sample_path_points.is_empty() {
                    plot_ui.line(
                        Line::new(
                            "Sample Euler-Maruyama Trajectory",
                            &self.sample_path_points[..],
                        )
                        .color(palette.series_color(1))
                        .width(1.4_f32),
                    );
                }
                if !self.observed_points.is_empty() {
                    plot_ui.points(
                        Points::new("Observed CSV Series", &self.observed_points[..])
                            .color(palette.series_color(3))
                            .radius(4.8_f32)
                            .shape(MarkerShape::Circle),
                    );
                }
            });
    }

    pub(super) fn render_workbench_ui(&mut self, ui: &mut egui::Ui) {
        self.poll_io(ui.ctx());
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Ornstein-Uhlenbeck CSV Calibration & Confidence Bands");
            ui.label(
                "Fit mean-reverting (μ, θ, σ) parameters from empirical CSV series, simulate zero-allocation Euler-Maruyama confidence fans, and copy calibrated snippets.",
            );
            ui.separator();

            if self.show_parameter_controls(ui) {
                self.bump_revision();
                self.recalculate();
            }
            ui.separator();
            self.show_diagnostics_and_snippets(ui);
            ui.separator();
            self.show_confidence_bands_plot(ui);
        });
    }
}

// [cite:clinical_trials_statistics]
