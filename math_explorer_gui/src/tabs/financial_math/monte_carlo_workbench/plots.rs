//! Log-log error-decay plots and `Percentile` vs `BCa` bootstrap histogram views.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MonteCarloWorkbenchTool, WorkbenchViewMode};
use crate::accessibility::{paint_canvas_focus_ring, register_canvas_accessibility};
use crate::app::SemanticPalette;
use eframe::egui;
use eframe::egui::accesskit::Role;
use egui_plot::{Bar, BarChart, Legend, Line, LineStyle, Plot};

impl MonteCarloWorkbenchTool {
    pub(super) fn render_central_views(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let avail_h = ui.available_height().max(260.0);

        match self.view_mode {
            WorkbenchViewMode::SplitOverview => {
                let card_h = (avail_h * 0.47).max(170.0);
                self.render_convergence_card(ui, &palette, card_h);
                ui.add_space(8.0);
                self.render_bootstrap_card(ui, &palette, card_h);
            }
            WorkbenchViewMode::ErrorDecayLogPlot => {
                self.render_convergence_card(ui, &palette, avail_h - 28.0);
            }
            WorkbenchViewMode::BootstrapHistogram => {
                self.render_bootstrap_card(ui, &palette, avail_h - 28.0);
            }
        }
    }

    fn render_convergence_card(
        &self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        height: f32,
    ) {
        ui.horizontal_wrapped(|ui| {
            ui.strong(format!(
                "Log-Log RMSE Error Decay — {} (Exact = {:.5})",
                self.benchmark.label(),
                self.exact_reference
            ));
            if let Some(sob) = self.est_sobol {
                ui.colored_label(
                    palette.success,
                    format!("Sobol SE: {:.2e}", sob.standard_error),
                );
            }
            if let Some(iid) = self.est_iid {
                ui.label(format!("IID SE: {:.2e}", iid.standard_error));
            }
        });

        let plot_resp = Plot::new("mc_workbench_error_decay_plot")
            .height(height.max(130.0))
            .x_axis_label("log10(Sample Budget N)")
            .y_axis_label("log10(Empirical RMSE vs Exact)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                self.draw_decay_series(plot_ui, palette);
            });

        let summary = format!(
            "Monte Carlo log10 RMSE decay up to N = {}: IID last {:.3}, Sobol RQMC last {:.3}",
            self.num_samples,
            self.decay_iid.last().map_or(0.0, |p| p.y),
            self.decay_sobol.last().map_or(0.0, |p| p.y),
        );
        register_canvas_accessibility(
            ui,
            &plot_resp.response,
            Role::GraphicsDocument,
            "Monte Carlo and Quasi-Monte Carlo Log-Log Error Decay Plot",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &plot_resp.response);
    }

    fn draw_decay_series<'a>(
        &'a self,
        plot_ui: &mut egui_plot::PlotUi<'a>,
        palette: &SemanticPalette,
    ) {
        let dashed = LineStyle::Dashed { length: 7.0 };
        plot_ui.line(
            Line::new("IID MC", &self.decay_iid[..])
                .color(palette.series_color(0))
                .width(2.0_f32),
        );
        plot_ui.line(
            Line::new("Antithetic", &self.decay_antithetic[..]).color(palette.series_color(1)),
        );
        plot_ui.line(
            Line::new("Control Variate", &self.decay_control[..]).color(palette.series_color(2)),
        );
        plot_ui.line(
            Line::new("Latin Hypercube", &self.decay_lhs[..]).color(palette.series_color(3)),
        );
        plot_ui.line(
            Line::new("Halton RQMC", &self.decay_halton[..]).color(palette.series_color(4)),
        );
        plot_ui.line(
            Line::new("Joe-Kuo Sobol RQMC", &self.decay_sobol[..])
                .color(palette.success)
                .width(2.4_f32),
        );
        plot_ui.line(
            Line::new("Ref Slope -1/2", &self.slope_half_ref[..])
                .color(palette.series_color(0))
                .style(dashed),
        );
        plot_ui.line(
            Line::new("Ref Slope -1", &self.slope_one_ref[..])
                .color(palette.success)
                .style(dashed),
        );
    }

    fn render_bootstrap_card(
        &self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        height: f32,
    ) {
        ui.horizontal_wrapped(|ui| {
            ui.strong(format!(
                "Nonparametric Bootstrap Replicate Histogram ({} — B = {})",
                self.bootstrap_statistic.label(),
                self.bootstrap_replicates
            ));
            if let Some(comp) = self.boot_comparison {
                ui.colored_label(
                    palette.warning,
                    format!(
                        "Percentile: [{:.4}, {:.4}]",
                        comp.percentile.ci_lower, comp.percentile.ci_upper
                    ),
                );
                ui.colored_label(
                    palette.success,
                    format!(
                        "BCa: [{:.4}, {:.4}] (z0={:+.3}, a={:+.4})",
                        comp.bca.ci_lower,
                        comp.bca.ci_upper,
                        comp.bca.bias_correction_z0,
                        comp.bca.acceleration
                    ),
                );
            }
        });

        let plot_resp = Plot::new("mc_workbench_bootstrap_hist_plot")
            .height(height.max(130.0))
            .x_axis_label("Resampled Statistic θ*")
            .y_axis_label("Empirical Density")
            .include_y(0.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                self.draw_bootstrap_series(plot_ui, palette);
            });

        let summary = if let Some(comp) = self.boot_comparison {
            format!(
                "Bootstrap {} histogram: point {:.4}, Percentile CI [{:.4}, {:.4}], BCa CI [{:.4}, {:.4}]",
                self.bootstrap_statistic.label(),
                comp.bca.point_estimate,
                comp.percentile.ci_lower,
                comp.percentile.ci_upper,
                comp.bca.ci_lower,
                comp.bca.ci_upper
            )
        } else {
            "Bootstrap replicate histogram".to_string()
        };
        register_canvas_accessibility(
            ui,
            &plot_resp.response,
            Role::GraphicsDocument,
            "Percentile versus BCa Bootstrap Replicate Histogram",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &plot_resp.response);
    }

    fn draw_bootstrap_series<'a>(
        &'a self,
        plot_ui: &mut egui_plot::PlotUi<'a>,
        palette: &SemanticPalette,
    ) {
        let bar_width = (0.88 * self.hist_bin_width).max(1e-6);
        let bars: Vec<Bar> = self
            .hist_bins
            .iter()
            .map(|&(cx, density)| Bar::new(cx, density).width(bar_width))
            .collect();
        plot_ui.bar_chart(
            BarChart::new("Bootstrap Replicates", bars).color(palette.series_color(2)),
        );

        if self.boot_comparison.is_none() {
            return;
        };
        let dashed = LineStyle::Dashed { length: 6.0 };

        plot_ui.line(
            Line::new("Percentile Lower", &self.line_pct_lo[..])
                .color(palette.warning)
                .style(dashed)
                .width(2.0_f32),
        );
        plot_ui.line(
            Line::new("Percentile Upper", &self.line_pct_hi[..])
                .color(palette.warning)
                .style(dashed)
                .width(2.0_f32),
        );
        plot_ui.line(
            Line::new("BCa Lower", &self.line_bca_lo[..])
                .color(palette.success)
                .width(2.4_f32),
        );
        plot_ui.line(
            Line::new("BCa Upper", &self.line_bca_hi[..])
                .color(palette.success)
                .width(2.4_f32),
        );
        plot_ui.line(
            Line::new("Point Estimate θ̂", &self.line_point_est[..])
                .color(palette.accent)
                .width(1.8_f32),
        );
    }
}

// [cite:clinical_trials_statistics]
