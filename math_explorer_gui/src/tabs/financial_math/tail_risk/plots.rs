//! UI controls, tail-quantile spectrum plot, underwater drawdown plot, and exports for [`super::TailRiskDrawdownTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{TailRiskDrawdownTool, MAX_TAIL_HORIZON};
use crate::accessibility::{paint_canvas_focus_ring, register_canvas_accessibility};
use crate::app::SemanticPalette;
use eframe::egui;
use eframe::egui::accesskit::Role;
use egui_plot::{ExportPrimitive, HLine, Legend, Line, LineStyle, Plot, PlotPoint, VLine};

fn to_pairs(pts: &[PlotPoint]) -> Vec<[f64; 2]> {
    pts.iter().map(|p| [p.x, p.y]).collect()
}

impl TailRiskDrawdownTool {
    pub(super) fn render_workspace(&mut self, ui: &mut egui::Ui) {
        egui::SidePanel::left("tail_risk_controls")
            .resizable(true)
            .default_width(275.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if self.render_controls(ui) {
                        self.recompute();
                    }
                });
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.render_summary_banner(ui, &palette);
                ui.add_space(6.0);
                self.render_tail_spectrum_plot(ui, &palette);
                ui.add_space(8.0);
                self.render_wealth_underwater_plot(ui, &palette);
            });
        });
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("Return & Tail Regime");
        ui.small(format!("Evaluated in {:.3} ms", self.last_compute_ms));
        ui.separator();

        changed |= self.render_regime_presets(ui);
        ui.separator();

        ui.label("Confidence Level (α):");
        changed |= ui
            .add(egui::Slider::new(&mut self.confidence_level, 0.55..=0.99))
            .changed();
        ui.label("Annualized Volatility (σ):");
        changed |= ui
            .add(egui::Slider::new(&mut self.annual_volatility, 0.05..=0.80))
            .changed();
        ui.label("Annualized Drift (μ):");
        changed |= ui
            .add(egui::Slider::new(&mut self.annual_drift, -0.30..=0.50))
            .changed();
        ui.label("Downside Shock Probability:");
        changed |= ui
            .add(egui::Slider::new(&mut self.tail_shock_prob, 0.0..=0.30))
            .changed();
        ui.label("Downside Shock Magnitude:");
        changed |= ui
            .add(egui::Slider::new(&mut self.tail_shock_scale, 0.0..=0.35))
            .changed();
        ui.label("Annual Risk-Free Rate (r_f):");
        changed |= ui
            .add(egui::Slider::new(&mut self.risk_free_rate, 0.0..=0.15))
            .changed();
        ui.label("Horizon (Trading Periods):");
        changed |= ui
            .add(egui::Slider::new(
                &mut self.horizon_periods,
                30..=MAX_TAIL_HORIZON,
            ))
            .changed();
        ui.horizontal(|ui| {
            ui.label("RNG Seed:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.seed).speed(1))
                .changed();
        });
        changed
    }

    fn render_regime_presets(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            if ui.button("Gaussian Regime").clicked() {
                self.tail_shock_prob = 0.0;
                self.tail_shock_scale = 0.0;
                self.annual_volatility = 0.18;
                self.annual_drift = 0.09;
                changed = true;
            }
            if ui.button("Skewed Jump Tail").clicked() {
                self.tail_shock_prob = 0.08;
                self.tail_shock_scale = 0.11;
                self.annual_volatility = 0.24;
                self.annual_drift = 0.11;
                changed = true;
            }
        });
        changed
    }

    fn render_summary_banner(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let Some(sum) = self.summary else {
            ui.colored_label(
                palette.error,
                "Insufficient variance to summarize tail risk.",
            );
            return;
        };
        let pct = self.confidence_level * 100.0;
        egui::Grid::new("tail_risk_metrics_grid")
            .num_columns(3)
            .spacing([24.0, 6.0])
            .show(ui, |ui| {
                ui.colored_label(
                    palette.series_color(0),
                    format!(
                        "Historical VaR/CVaR ({pct:.1}%): {:.2}% / {:.2}%",
                        sum.historical_var * 100.0,
                        sum.historical_cvar * 100.0
                    ),
                );
                ui.colored_label(
                    palette.series_color(1),
                    format!(
                        "Gaussian VaR/CVaR ({pct:.1}%): {:.2}% / {:.2}%",
                        sum.gaussian_var * 100.0,
                        sum.gaussian_cvar * 100.0
                    ),
                );
                ui.colored_label(
                    palette.series_color(2),
                    format!(
                        "Cornish-Fisher VaR/CVaR: {:.2}% / {:.2}%",
                        sum.cornish_fisher_var * 100.0,
                        sum.cornish_fisher_cvar * 100.0
                    ),
                );
                ui.end_row();
                ui.label(format!(
                    "Sharpe: {:+.2} | Sortino: {:+.2}",
                    sum.sharpe, sum.sortino
                ));
                ui.label(format!("Calmar Ratio: {:+.2}", sum.calmar));
                ui.colored_label(
                    palette.warning,
                    format!(
                        "Max Drawdown: {:.2}% ({} underwater steps)",
                        sum.max_drawdown * 100.0,
                        sum.max_underwater_duration
                    ),
                );
                ui.end_row();
            });
    }

    fn render_tail_spectrum_plot(&mut self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.strong("One-Period Tail Loss Spectrum vs Confidence Level α");
        let resp = Plot::new("tail_risk_spectrum_plot")
            .height(235.0)
            .x_axis_label("Confidence Level α")
            .y_axis_label("One-Period Loss (% of Capital)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                self.draw_tail_spectrum_series(plot_ui, palette);
            });
        let desc = self.summary.map_or_else(
            || "Tail-risk loss spectrum".to_string(),
            |s| {
                format!(
                    "At alpha {:.2}: Historical CVaR {:.2}%, Gaussian VaR {:.2}%",
                    self.confidence_level,
                    s.historical_cvar * 100.0,
                    s.gaussian_var * 100.0
                )
            },
        );
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "Tail Loss VaR and CVaR Spectrum vs Confidence Level",
            Some(&desc),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }

    fn draw_tail_spectrum_series<'a>(
        &'a mut self,
        plot_ui: &mut egui_plot::PlotUi<'a>,
        palette: &SemanticPalette,
    ) {
        plot_ui.line(
            Line::new("Historical VaR", &self.hist_var_curve[..])
                .color(palette.series_color(0))
                .style(LineStyle::Dashed { length: 6.0 }),
        );
        plot_ui.line(
            Line::new("Historical CVaR", &self.hist_cvar_curve[..])
                .color(palette.series_color(0))
                .width(2.2_f32),
        );
        plot_ui.line(
            Line::new("Gaussian VaR", &self.gauss_var_curve[..])
                .color(palette.series_color(1))
                .style(LineStyle::Dashed { length: 6.0 }),
        );
        plot_ui.line(
            Line::new("Gaussian CVaR", &self.gauss_cvar_curve[..])
                .color(palette.series_color(1))
                .width(1.8_f32),
        );
        plot_ui.line(
            Line::new("Cornish-Fisher VaR", &self.cf_var_curve[..])
                .color(palette.series_color(2))
                .style(LineStyle::Dashed { length: 6.0 }),
        );
        plot_ui.line(
            Line::new("Cornish-Fisher CVaR", &self.cf_cvar_curve[..])
                .color(palette.series_color(2))
                .width(2.0_f32),
        );
        plot_ui.vline(
            VLine::new("Selected α", self.confidence_level)
                .color(palette.accent)
                .style(LineStyle::Dotted { spacing: 5.0 }),
        );
        self.callout_manager.render_on_plot(plot_ui);
    }

    fn render_wealth_underwater_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.strong("Compounded Capital Index & Underwater Drawdown (%)");
        let resp = Plot::new("tail_risk_underwater_plot")
            .height(210.0)
            .x_axis_label("Trading Period t")
            .y_axis_label("Capital Index / Underwater (%)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.hline(
                    HLine::new("Peak Reference (0% Drawdown)", 0.0)
                        .color(palette.graph_edge)
                        .style(LineStyle::Dashed { length: 5.0 }),
                );
                plot_ui.line(
                    Line::new("Capital Index (Base 100)", &self.wealth_points[..])
                        .color(palette.success)
                        .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new("Underwater Drawdown (%)", &self.underwater_points[..])
                        .color(palette.error)
                        .width(1.8_f32),
                );
            });
        let mdd = self.summary.map_or(0.0, |s| s.max_drawdown * 100.0);
        let desc = format!(
            "{} simulated periods; maximum drawdown {:.2}%",
            self.horizon_periods, mdd
        );
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "Compounded Capital Index and Underwater Drawdown Curve",
            Some(&desc),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }

    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            ("Historical VaR (%)".into(), to_pairs(&self.hist_var_curve)),
            (
                "Historical CVaR (%)".into(),
                to_pairs(&self.hist_cvar_curve),
            ),
            ("Gaussian VaR (%)".into(), to_pairs(&self.gauss_var_curve)),
            ("Gaussian CVaR (%)".into(), to_pairs(&self.gauss_cvar_curve)),
            (
                "Cornish-Fisher VaR (%)".into(),
                to_pairs(&self.cf_var_curve),
            ),
            (
                "Cornish-Fisher CVaR (%)".into(),
                to_pairs(&self.cf_cvar_curve),
            ),
            ("Capital Index".into(), to_pairs(&self.wealth_points)),
            (
                "Underwater Drawdown (%)".into(),
                to_pairs(&self.underwater_points),
            ),
        ]
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let pal = crate::app::palette::DARK_PALETTE;
        let series = [
            ("Historical VaR (%)", &self.hist_var_curve, pal.series[0]),
            ("Historical CVaR (%)", &self.hist_cvar_curve, pal.series[0]),
            ("Gaussian VaR (%)", &self.gauss_var_curve, pal.series[1]),
            ("Gaussian CVaR (%)", &self.gauss_cvar_curve, pal.series[1]),
            ("Cornish-Fisher VaR (%)", &self.cf_var_curve, pal.series[2]),
            (
                "Cornish-Fisher CVaR (%)",
                &self.cf_cvar_curve,
                pal.series[2],
            ),
        ];
        let mut prims = Vec::with_capacity(series.len());
        for (name, pts, color) in series {
            if pts.len() >= 2 {
                prims.push(ExportPrimitive::Polyline {
                    name: name.to_string(),
                    points: to_pairs(pts).into(),
                    color,
                    stroke_width: 2.0,
                    closed: false,
                    fill: None,
                });
            }
        }
        prims
    }
}

// [cite:clinical_trials_statistics]
