//! Sizing table readout and logarithmic growth-curve plot rendering for [`super::BetSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{BetSizingMode, OddsFormat, FRACTIONAL_MAX_DENOMINATOR};
use super::BetSizeCalculatorTool;
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, MarkerShape, Plot, Points};
use math_explorer::pure_math::statistics::kelly::Odds;

impl BetSizeCalculatorTool {
    pub(super) fn show_results_and_table(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref err) = self.calc_error {
            ui.colored_label(palette.error, err);
            return;
        }
        if matches!(
            self.mode,
            BetSizingMode::Binary | BetSizingMode::PushSpread | BetSizingMode::QuarterLine
        ) {
            if let Ok(odds) = self.resolve_active_odds() {
                self.render_converted_readout(ui, &odds);
                ui.add_space(4.0);
            }
        }
        if let Some(ref summary) = self.summary_line {
            ui.colored_label(palette.success, summary);
        } else if let Some(ev) = self.ev_per_unit {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Expected Value (per $1 staked): ${ev:+.4} ({:+.2}%)",
                    ev * 100.0
                ));
                if ev > 0.0 {
                    ui.colored_label(palette.success, "Positive Edge!");
                } else {
                    ui.colored_label(palette.error, "No Edge (Negative or Zero EV). Do not bet.");
                }
            });
        }
        self.render_strategy_grid(ui);
        ui.add_space(4.0);
        self.show_table_export_bar(ui, &palette);
    }

    fn render_converted_readout(&self, ui: &mut egui::Ui, odds: &Odds) {
        let (num, den) = odds.to_fractional(FRACTIONAL_MAX_DENOMINATOR);
        let american = if self.odds_format == OddsFormat::American {
            self.american_odds
        } else {
            odds.to_american()
        };
        ui.horizontal_wrapped(|ui| {
            ui.strong("Equivalent Odds:");
            ui.label(format!("Decimal {:.3}", odds.value()));
            ui.separator();
            ui.label(format!("American {:+.0}", american));
            ui.separator();
            ui.label(format!("Fractional {num}/{den}"));
            ui.separator();
            ui.label(format!(
                "Implied {:.2}%",
                odds.implied_probability() * 100.0
            ));
            ui.separator();
            ui.label(format!("Net b = {:.3}", odds.net_profit_multiplier()));
        });
    }

    fn render_strategy_grid(&self, ui: &mut egui::Ui) {
        egui::Grid::new("kelly_results")
            .num_columns(4)
            .striped(true)
            .spacing([28.0, 6.0])
            .show(ui, |ui| {
                ui.strong("Strategy");
                ui.strong("Fraction of Bankroll");
                ui.strong("Bet Amount");
                ui.strong("Growth Rate g(f)");
                ui.end_row();

                for row in &self.strategy_rows {
                    ui.label(&row.label);
                    ui.label(format!("{:.4}%", row.fraction * 100.0));
                    ui.label(format!("${:.2}", row.bet_amount));
                    let g_str = if row.growth_rate.is_finite() {
                        format!("{:+.6}", row.growth_rate)
                    } else {
                        String::from("-inf")
                    };
                    ui.label(g_str);
                    ui.end_row();
                }
            });
    }

    fn show_table_export_bar(&mut self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📋 Copy CSV")
                .accessible_hover_text("Copy formula-safe CSV Kelly sizing table to clipboard")
                .clicked()
            {
                self.copy_sizing_to_clipboard(ui.ctx(), false);
            }
            if ui
                .button("💾 Save CSV")
                .accessible_hover_text("Download or save CSV Kelly sizing table")
                .clicked()
            {
                self.save_sizing_file(false);
            }
            if ui
                .button("📋 Copy Markdown")
                .accessible_hover_text("Copy Markdown Kelly sizing table to clipboard")
                .clicked()
            {
                self.copy_sizing_to_clipboard(ui.ctx(), true);
            }
            if ui
                .button("💾 Save Markdown")
                .accessible_hover_text("Download or save Markdown Kelly sizing table")
                .clicked()
            {
                self.save_sizing_file(true);
            }
        });
        if let Some((ref msg, is_err)) = self.io_status {
            let color = if is_err { palette.error } else { palette.info };
            ui.colored_label(color, msg);
        }
    }

    pub(super) fn show_growth_curve_plot(&self, ui: &mut egui::Ui) {
        if self.growth_curve_points.is_empty() {
            return;
        }
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Expected Logarithmic Growth Curve g(f)");
        Plot::new("bet_size_growth_plot")
            .view_aspect(2.2)
            .legend(Legend::default())
            .x_axis_label("Wager Fraction (f)")
            .y_axis_label("Expected Log Growth Rate g(f)")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Break-Even (g = 0)", &self.zero_line_points[..])
                        .color(palette.graph_edge)
                        .style(LineStyle::Dashed { length: 6.0 }),
                );
                plot_ui.line(
                    Line::new("Expected Growth Rate g(f)", &self.growth_curve_points[..])
                        .color(palette.series_color(0))
                        .width(2.0_f32),
                );
                self.render_plot_markers(plot_ui, &palette);
            });
    }

    fn render_plot_markers<'a>(
        &'a self,
        plot_ui: &mut egui_plot::PlotUi<'a>,
        palette: &SemanticPalette,
    ) {
        if !self.quarter_marker.is_empty() {
            plot_ui.points(
                Points::new("Quarter Kelly (0.25 f*)", &self.quarter_marker[..])
                    .color(palette.series_color(2))
                    .radius(5.0_f32)
                    .shape(MarkerShape::Diamond),
            );
        }
        if !self.half_marker.is_empty() {
            plot_ui.points(
                Points::new("Half Kelly (0.50 f*)", &self.half_marker[..])
                    .color(palette.series_color(5))
                    .radius(5.5_f32)
                    .shape(MarkerShape::Square),
            );
        }
        if !self.full_marker.is_empty() {
            plot_ui.points(
                Points::new("Full Kelly (1.00 f*)", &self.full_marker[..])
                    .color(palette.series_color(3))
                    .radius(6.0_f32)
                    .shape(MarkerShape::Circle),
            );
        }
        if !self.drawdown_marker.is_empty() {
            plot_ui.points(
                Points::new("Drawdown-Constrained Kelly", &self.drawdown_marker[..])
                    .color(palette.series_color(1))
                    .radius(5.5_f32)
                    .shape(MarkerShape::Up),
            );
        }
    }
}

// [cite:clinical_trials_statistics]
