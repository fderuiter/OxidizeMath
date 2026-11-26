#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::WorkbenchView;
use super::PortfolioCopulaWorkbenchTool;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, MarkerShape, Plot, Points};

impl PortfolioCopulaWorkbenchTool {
    pub(super) fn show_main_workbench(&self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| match self.state.view {
            WorkbenchView::KellyFrontier => self.show_kelly_frontier_view(ui),
            WorkbenchView::CopulaTailCrash => self.show_copula_crash_view(ui),
            WorkbenchView::OuMeanReversion => self.show_ou_cone_view(ui),
            WorkbenchView::BlackScholesGreeks => self.show_bsm_greeks_view(ui),
        });
    }

    fn show_kelly_frontier_view(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Multi-Asset Covariance Kelly Frontier & VaR / CVaR");
        if let Some(ref err) = self.kelly_cache.error {
            ui.colored_label(palette.error, err);
            return;
        }
        Plot::new("pw_kelly_frontier_plot")
            .height(310.0)
            .legend(Legend::default())
            .x_axis_label("Allocation Scale Multiplier c (1.0 = Full Strategy)")
            .y_axis_label("Excess Logarithmic Growth g_excess(c · w)")
            .show(ui, |plot_ui| {
                self.draw_kelly_frontier_series(plot_ui, &palette)
            });
        ui.separator();
        self.show_kelly_and_var_tables(ui);
    }

    fn draw_kelly_frontier_series<'a>(
        &'a self,
        plot_ui: &mut egui_plot::PlotUi<'a>,
        palette: &crate::app::SemanticPalette,
    ) {
        plot_ui.line(
            Line::new("Break-Even (g = 0)", &self.kelly_cache.zero_line[..])
                .color(palette.graph_edge)
                .style(LineStyle::Dashed { length: 6.0 }),
        );
        plot_ui.line(
            Line::new(
                "Covariance Kelly (Σ⁻¹(μ - r1))",
                &self.kelly_cache.cov_frontier[..],
            )
            .color(palette.series_color(0))
            .width(2.2_f32),
        );
        plot_ui.line(
            Line::new(
                "Long-Only Kelly (Projected Gradient)",
                &self.kelly_cache.long_frontier[..],
            )
            .color(palette.series_color(2))
            .width(2.0_f32),
        );
        plot_ui.line(
            Line::new(
                "Independent Kelly (Ignores ρ)",
                &self.kelly_cache.indep_frontier[..],
            )
            .color(palette.series_color(1))
            .style(LineStyle::Dashed { length: 5.0 })
            .width(2.0_f32),
        );
        plot_ui.points(
            Points::new(
                "Full Covariance Kelly (c=1)",
                &self.kelly_cache.cov_marker[..],
            )
            .color(palette.series_color(0))
            .radius(6.0_f32)
            .shape(MarkerShape::Circle),
        );
        plot_ui.points(
            Points::new(
                "Independent Overbet (c=1)",
                &self.kelly_cache.indep_marker[..],
            )
            .color(palette.series_color(1))
            .radius(6.0_f32)
            .shape(MarkerShape::Diamond),
        );
    }

    fn show_kelly_and_var_tables(&self, ui: &mut egui::Ui) {
        ui.strong("Strategy Leverage, Log-Growth & Tail-Risk Comparison");
        egui::Grid::new("pw_kelly_summary_grid")
            .num_columns(7)
            .striped(true)
            .spacing([16.0, 5.0])
            .show(ui, |ui| {
                for hdr in [
                    "Strategy",
                    "Gross Lev",
                    "Excess Growth",
                    "95% VaR ($)",
                    "95% CVaR ($)",
                    "99% VaR ($)",
                    "99% CVaR ($)",
                ] {
                    ui.strong(hdr);
                }
                ui.end_row();
                for (name, sol_opt, tail_opt) in [
                    (
                        "Covariance Kelly",
                        self.kelly_cache.cov_kelly.as_ref(),
                        self.kelly_cache.cov_sim_tail,
                    ),
                    (
                        "Long-Only Kelly",
                        self.kelly_cache.long_only_capped.as_ref(),
                        self.kelly_cache.long_sim_tail,
                    ),
                    (
                        "Independent Kelly",
                        self.kelly_cache.indep_kelly.as_ref(),
                        self.kelly_cache.indep_sim_tail,
                    ),
                ] {
                    if let Some(sol) = sol_opt {
                        Self::show_kelly_strategy_row(ui, name, sol, tail_opt);
                    }
                }
                if let Some(emp) = self.kelly_cache.empirical_tail {
                    Self::show_tail_cells(
                        ui,
                        "Empirical Imported (1/N)",
                        "1.0000x",
                        "Historical",
                        Some(emp),
                    );
                }
            });
    }

    fn show_kelly_strategy_row(
        ui: &mut egui::Ui,
        name: &str,
        sol: &math_explorer::pure_math::statistics::kelly::PortfolioKellySolution,
        tail_opt: Option<math_explorer::pure_math::statistics::kelly::TailRiskMetrics>,
    ) {
        let lev = format!("{:.4}x", sol.metrics.gross_leverage);
        let growth = format!("{:+.2}%", sol.metrics.excess_log_growth * 100.0);
        Self::show_tail_cells(ui, name, &lev, &growth, tail_opt);
    }

    fn show_tail_cells(
        ui: &mut egui::Ui,
        name: &str,
        lev: &str,
        growth: &str,
        tail_opt: Option<math_explorer::pure_math::statistics::kelly::TailRiskMetrics>,
    ) {
        ui.label(name);
        ui.label(lev);
        ui.label(growth);
        if let Some(t) = tail_opt {
            ui.label(format!("${:.2}", t.var_95));
            ui.label(format!("${:.2}", t.cvar_95));
            ui.label(format!("${:.2}", t.var_99));
            ui.label(format!("${:.2}", t.cvar_99));
        } else {
            for _ in 0..4 {
                ui.label("—");
            }
        }
        ui.end_row();
    }

    fn show_copula_crash_view(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.heading(format!(
            "{}-Leg Gaussian Copula Joint Probability & Simultaneous Tail Crash",
            self.state.copula_legs
        ));
        if let Some(ref err) = self.copula_cache.error {
            ui.colored_label(palette.error, err);
            return;
        }
        Plot::new("pw_copula_crash_plot")
            .height(320.0)
            .legend(Legend::default())
            .x_axis_label("Pairwise Equicorrelation (ρ)")
            .y_axis_label("Joint Cumulative Probability C(u₁, ..., u_d; ρ)")
            .show(ui, |plot_ui| {
                self.draw_copula_crash_series(plot_ui, &palette)
            });
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.label(format!(
                "SGP Joint Prob: {:.4}% (Naive: {:.4}%)",
                self.copula_cache.active_sgp_prob * 100.0,
                self.copula_cache.naive_sgp_prob * 100.0
            ));
            ui.separator();
            ui.colored_label(
                palette.warning,
                format!(
                    "Simultaneous {:.1}% Tail Crash: {:.5} vs Naive {:.6} ({:.2}x amplification)",
                    self.state.crash_quantile * 100.0,
                    self.copula_cache.active_crash_prob,
                    self.copula_cache.naive_crash_prob,
                    self.copula_cache.crash_multiplier
                ),
            );
        });
    }

    fn draw_copula_crash_series<'a>(
        &'a self,
        plot_ui: &mut egui_plot::PlotUi<'a>,
        palette: &crate::app::SemanticPalette,
    ) {
        plot_ui.line(
            Line::new(
                "Copula Joint SGP Probability",
                &self.copula_cache.sgp_curve[..],
            )
            .color(palette.series_color(0))
            .width(2.2_f32),
        );
        plot_ui.line(
            Line::new(
                "Naive Independent SGP",
                &self.copula_cache.naive_sgp_line[..],
            )
            .color(palette.graph_edge)
            .style(LineStyle::Dashed { length: 6.0 }),
        );
        plot_ui.line(
            Line::new(
                "Simultaneous Tail Crash Probability",
                &self.copula_cache.crash_curve[..],
            )
            .color(palette.series_color(1))
            .width(2.2_f32),
        );
        plot_ui.line(
            Line::new(
                "Naive Independent Crash",
                &self.copula_cache.naive_crash_line[..],
            )
            .color(palette.series_color(4))
            .style(LineStyle::Dotted { spacing: 5.0 }),
        );
        plot_ui.points(
            Points::new("Active SGP", &self.copula_cache.active_sgp_marker[..])
                .color(palette.series_color(0))
                .radius(6.0_f32),
        );
        plot_ui.points(
            Points::new("Active Crash", &self.copula_cache.active_crash_marker[..])
                .color(palette.series_color(1))
                .radius(6.0_f32)
                .shape(MarkerShape::Diamond),
        );
    }

    fn show_ou_cone_view(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Ornstein-Uhlenbeck Mean-Reversion Forecast Cone (5% / 50% / 95%)");
        if let Some(ref err) = self.ou_cache.error {
            ui.colored_label(palette.error, err);
            return;
        }
        Plot::new("pw_ou_cone_plot")
            .height(320.0)
            .legend(Legend::default())
            .x_axis_label("Time Horizon t")
            .y_axis_label("State / Spread X(t)")
            .show(ui, |plot_ui| self.draw_ou_cone_series(plot_ui, &palette));
        ui.separator();
        if let Some(ref st) = self.ou_cache.forecast_stats {
            let regime = self
                .ou_cache
                .momentum
                .as_ref()
                .map_or("Mean-reverting", |m| m.description());
            ui.label(format!(
                "Half-Life: {:.3} | Terminal 5%: {:.4} | Median: {:.4} | Mean: {:.4} | 95%: {:.4} ({regime})",
                self.ou_cache.half_life,
                st.percentile_05,
                st.median_final,
                st.mean_final,
                st.percentile_95
            ));
        }
    }

    fn draw_ou_cone_series<'a>(
        &'a self,
        plot_ui: &mut egui_plot::PlotUi<'a>,
        palette: &crate::app::SemanticPalette,
    ) {
        plot_ui.line(
            Line::new("Equilibrium μ", &self.ou_cache.eq_line[..])
                .color(palette.graph_edge)
                .style(LineStyle::Dashed { length: 6.0 }),
        );
        plot_ui.line(
            Line::new("95th Percentile Cone", &self.ou_cache.p95_band[..])
                .color(palette.series_color(1))
                .style(LineStyle::Dashed { length: 5.0 }),
        );
        plot_ui.line(
            Line::new("Median (50th) Trajectory", &self.ou_cache.p50_band[..])
                .color(palette.series_color(2))
                .width(2.0_f32),
        );
        plot_ui.line(
            Line::new("5th Percentile Cone", &self.ou_cache.p05_band[..])
                .color(palette.series_color(1))
                .style(LineStyle::Dashed { length: 5.0 }),
        );
        plot_ui.line(
            Line::new(
                "Analytical Expectation E[X_t]",
                &self.ou_cache.mean_curve[..],
            )
            .color(palette.series_color(0))
            .width(2.4_f32),
        );
        for (idx, (label, pts)) in self.ou_cache.sample_paths.iter().enumerate() {
            plot_ui.line(
                Line::new(label, &pts[..])
                    .color(palette.series_color(3 + idx))
                    .width(1.1_f32),
            );
        }
    }

    fn show_bsm_greeks_view(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.heading(format!(
            "Black-Scholes-Merton European Option Profile — {}",
            self.state.bsm_metric.label()
        ));
        if let Some(ref err) = self.bsm_cache.error {
            ui.colored_label(palette.error, err);
            return;
        }
        Plot::new("pw_bsm_greeks_plot")
            .height(300.0)
            .legend(Legend::default())
            .x_axis_label("Underlying Spot Price S ($)")
            .y_axis_label(self.state.bsm_metric.label())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("European Call", &self.bsm_cache.call_curve[..])
                        .color(palette.series_color(0))
                        .width(2.2_f32),
                );
                plot_ui.line(
                    Line::new("European Put", &self.bsm_cache.put_curve[..])
                        .color(palette.series_color(1))
                        .width(2.2_f32),
                );
                plot_ui.points(
                    Points::new("Call @ Spot", &self.bsm_cache.call_marker[..])
                        .color(palette.series_color(0))
                        .radius(6.0_f32),
                );
                plot_ui.points(
                    Points::new("Put @ Spot", &self.bsm_cache.put_marker[..])
                        .color(palette.series_color(1))
                        .radius(6.0_f32)
                        .shape(MarkerShape::Diamond),
                );
            });
        ui.separator();
        self.show_bsm_greeks_table(ui);
    }

    fn show_bsm_greeks_table(&self, ui: &mut egui::Ui) {
        if let Some(ref pair) = self.bsm_cache.pair {
            egui::Grid::new("pw_bsm_table")
                .num_columns(7)
                .striped(true)
                .spacing([18.0, 5.0])
                .show(ui, |ui| {
                    for hdr in [
                        "Contract",
                        "Price ($)",
                        "Delta",
                        "Gamma",
                        "Vega",
                        "Theta (/yr)",
                        "Rho",
                    ] {
                        ui.strong(hdr);
                    }
                    ui.end_row();
                    for (name, g) in [("European Call", &pair.call), ("European Put", &pair.put)] {
                        ui.label(name);
                        ui.label(format!("{:.4}", g.price));
                        ui.label(format!("{:+.4}", g.delta));
                        ui.label(format!("{:.5}", g.gamma));
                        ui.label(format!("{:.4}", g.vega));
                        ui.label(format!("{:+.4}", g.theta));
                        ui.label(format!("{:+.4}", g.rho));
                        ui.end_row();
                    }
                });
        }
    }
}

// [cite:clinical_trials_statistics]
