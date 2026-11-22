#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BetSizeCalculatorTool, OddsFormat};
use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, MarkerShape, Plot, Points};

impl BetSizeCalculatorTool {
    fn show_input_grid(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::Grid::new("bet_size_inputs")
            .num_columns(2)
            .spacing([32.0, 8.0])
            .show(ui, |ui| {
                ui.label("Win Probability (%):");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.probability_input)
                            .speed(0.1)
                            .range(0.0..=100.0)
                            .suffix("%"),
                    )
                    .accessible_hover_text(
                        "Estimated probability of winning the wager (0% to 100%)",
                    )
                    .changed();
                ui.end_row();

                ui.label("Odds Format:");
                ui.horizontal(|ui| {
                    changed |= ui
                        .radio_value(&mut self.odds_format, OddsFormat::Decimal, "Decimal")
                        .changed();
                    changed |= ui
                        .radio_value(&mut self.odds_format, OddsFormat::American, "American")
                        .changed();
                    changed |= ui
                        .radio_value(&mut self.odds_format, OddsFormat::Fractional, "Fractional")
                        .changed();
                });
                ui.end_row();

                ui.label("Odds:");
                changed |= self.show_odds_value_editor(ui);
                ui.end_row();

                ui.label("Total Bankroll ($):");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.bankroll)
                            .speed(10.0)
                            .range(1.0..=1_000_000.0)
                            .prefix("$"),
                    )
                    .accessible_hover_text("Available bankroll capital in dollars")
                    .changed();
                ui.end_row();
            });
        changed
    }

    fn show_odds_value_editor(&mut self, ui: &mut egui::Ui) -> bool {
        match self.odds_format {
            OddsFormat::Decimal => ui
                .add(
                    egui::DragValue::new(&mut self.decimal_odds)
                        .speed(0.01)
                        .range(1.01..=1000.0),
                )
                .accessible_hover_text("Decimal payout including returned stake (2.0 = even money)")
                .changed(),
            OddsFormat::American => ui
                .add(
                    egui::DragValue::new(&mut self.american_odds)
                        .speed(1.0)
                        .range(-10000.0..=10000.0),
                )
                .accessible_hover_text("American moneyline odds (+100 or <= -100)")
                .changed(),
            OddsFormat::Fractional => {
                ui.horizontal(|ui| {
                    let mut c = ui
                        .add(
                            egui::DragValue::new(&mut self.fractional_numerator)
                                .speed(1.0)
                                .range(1.0..=1000.0),
                        )
                        .changed();
                    ui.label("/");
                    c |= ui
                        .add(
                            egui::DragValue::new(&mut self.fractional_denominator)
                                .speed(1.0)
                                .range(1.0..=1000.0),
                        )
                        .changed();
                    c
                })
                .inner
            }
        }
    }

    fn show_results_and_table(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref err) = self.calc_error {
            ui.colored_label(palette.error, err);
            return;
        }

        let ev = self.ev_per_unit.unwrap_or(0.0);
        ui.horizontal(|ui| {
            ui.label(format!("Expected Value (per $1 staked): ${ev:.3}"));
            if ev > 0.0 {
                ui.colored_label(palette.success, "Positive Edge!");
            } else {
                ui.colored_label(palette.error, "No Edge (Negative or Zero EV). Do not bet.");
            }
        });

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
                    ui.label(format!("{:.2}%", row.fraction * 100.0));
                    ui.label(format!("${:.2}", row.bet_amount));
                    let g_str = if row.growth_rate.is_finite() {
                        format!("{:.5}", row.growth_rate)
                    } else {
                        String::from("-inf")
                    };
                    ui.label(g_str);
                    ui.end_row();
                }
            });

        ui.add_space(4.0);
        self.show_table_export_bar(ui);
    }

    fn show_table_export_bar(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
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

    fn show_growth_curve_plot(&self, ui: &mut egui::Ui) {
        if self.growth_curve_points.is_empty() {
            return;
        }
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
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
            });
    }

    pub(super) fn render_workbench_ui(&mut self, ui: &mut egui::Ui) {
        self.poll_save_channel(ui.ctx());
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Kelly Criterion Bet Size Calculator");
            ui.label("Calculate optimal wager sizing and inspect logarithmic growth g(f) across fractions.");
            ui.separator();

            if self.show_input_grid(ui) {
                self.recalculate();
            }
            ui.separator();
            ui.heading("Results & Sizing Table");
            self.show_results_and_table(ui);
            ui.separator();
            self.show_growth_curve_plot(ui);
        });
    }
}
