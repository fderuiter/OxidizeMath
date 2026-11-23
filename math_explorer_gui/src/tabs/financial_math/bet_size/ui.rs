//! Decomposed egui rendering helpers for [`super::BetSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    resolve_odds_for, skip_american_dead_zone, BetSizeCalculatorTool, OddsFormat,
    AMERICAN_ODDS_MAX, AMERICAN_ODDS_MIN, BANKROLL_MAX, BANKROLL_MIN, DECIMAL_ODDS_MAX,
    DECIMAL_ODDS_MIN, FRACTIONAL_MAX, FRACTIONAL_MAX_DENOMINATOR, FRACTIONAL_MIN, IMPLIED_PROB_MAX,
    IMPLIED_PROB_MIN, PROBABILITY_MAX, PROBABILITY_MIN,
};
use crate::accessibility::AccessibleHoverText;
use crate::framework::bounded_drag_value;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, MarkerShape, Plot, Points};
use math_explorer::pure_math::statistics::kelly::Odds;

impl BetSizeCalculatorTool {
    fn render_format_selector(&mut self, ui: &mut egui::Ui) -> bool {
        let prev = self.odds_format;
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.odds_format, OddsFormat::Decimal, "Decimal");
            ui.radio_value(&mut self.odds_format, OddsFormat::American, "American");
            ui.radio_value(&mut self.odds_format, OddsFormat::Fractional, "Fractional");
            ui.radio_value(
                &mut self.odds_format,
                OddsFormat::ImpliedProbability,
                "Implied Prob",
            );
        });
        if self.odds_format != prev {
            if let Ok(odds) = resolve_odds_for(
                prev,
                self.decimal_odds,
                self.american_odds,
                self.fractional_numerator,
                self.fractional_denominator,
                self.implied_probability_input,
            ) {
                self.sync_from_odds(odds, prev);
            }
            return true;
        }
        false
    }

    fn render_fractional_editor(&mut self, ui: &mut egui::Ui) -> bool {
        ui.horizontal(|ui| {
            let mut c = ui
                .add(bounded_drag_value(
                    &mut self.fractional_numerator,
                    FRACTIONAL_MIN..=FRACTIONAL_MAX,
                    1.0,
                ))
                .accessible_hover_text("Fractional profit numerator")
                .changed();
            ui.label("/");
            c |= ui
                .add(bounded_drag_value(
                    &mut self.fractional_denominator,
                    FRACTIONAL_MIN..=FRACTIONAL_MAX,
                    1.0,
                ))
                .accessible_hover_text("Fractional stake denominator")
                .changed();
            c
        })
        .inner
    }

    fn show_odds_value_editor(&mut self, ui: &mut egui::Ui) -> bool {
        let changed = match self.odds_format {
            OddsFormat::Decimal => ui
                .add(bounded_drag_value(
                    &mut self.decimal_odds,
                    DECIMAL_ODDS_MIN..=DECIMAL_ODDS_MAX,
                    0.01,
                ))
                .accessible_hover_text("Decimal payout including returned stake (2.0 = even money)")
                .changed(),
            OddsFormat::American => {
                let prev = self.american_odds;
                let edited = ui
                    .add(bounded_drag_value(
                        &mut self.american_odds,
                        AMERICAN_ODDS_MIN..=AMERICAN_ODDS_MAX,
                        1.0,
                    ))
                    .accessible_hover_text("American moneyline odds (>= +100 or <= -100)")
                    .changed();
                if edited {
                    self.american_odds = skip_american_dead_zone(prev, self.american_odds);
                }
                edited
            }
            OddsFormat::Fractional => self.render_fractional_editor(ui),
            OddsFormat::ImpliedProbability => ui
                .add(
                    bounded_drag_value(
                        &mut self.implied_probability_input,
                        IMPLIED_PROB_MIN..=IMPLIED_PROB_MAX,
                        0.1,
                    )
                    .suffix("%"),
                )
                .accessible_hover_text("Bookmaker implied win probability (0.1% to 99.0%)")
                .changed(),
        };
        if changed {
            if let Ok(odds) = self.resolve_active_odds() {
                self.sync_from_odds(odds, self.odds_format);
            }
        }
        changed
    }

    fn show_input_grid(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::Grid::new("bet_size_inputs")
            .num_columns(2)
            .spacing([32.0, 8.0])
            .show(ui, |ui| {
                ui.label("Win Probability (%):");
                changed |= ui
                    .add(
                        bounded_drag_value(
                            &mut self.probability_input,
                            PROBABILITY_MIN..=PROBABILITY_MAX,
                            0.1,
                        )
                        .suffix("%"),
                    )
                    .accessible_hover_text(
                        "Estimated probability of winning the wager (0% to 100%)",
                    )
                    .changed();
                ui.end_row();

                ui.label("Odds Format:");
                changed |= self.render_format_selector(ui);
                ui.end_row();

                ui.label("Placed Odds:");
                changed |= self.show_odds_value_editor(ui);
                ui.end_row();

                ui.label("Total Bankroll ($):");
                changed |= ui
                    .add(
                        bounded_drag_value(&mut self.bankroll, BANKROLL_MIN..=BANKROLL_MAX, 10.0)
                            .prefix("$"),
                    )
                    .accessible_hover_text("Available bankroll capital in dollars")
                    .changed();
                ui.end_row();

                ui.label("Drawdown Wealth Floor (% W₀):");
                changed |= ui
                    .add(egui::Slider::new(&mut self.drawdown_floor_pct, 5.0..=95.0).suffix("%"))
                    .accessible_hover_text(
                        "Wealth floor percentage used for proportional-Kelly barrier solvency bounds",
                    )
                    .changed();
                ui.end_row();
            });
        changed
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

    fn show_closing_line_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            changed |= ui
                .checkbox(
                    &mut self.closing_line_enabled,
                    "Audit Closing Line Value (Two-Way Closing Market)",
                )
                .accessible_hover_text(
                    "Remove bookmaker vigorish from closing decimal odds and compute CLV edge",
                )
                .changed();
            if self.closing_line_enabled {
                ui.label("Closing For:");
                changed |= ui
                    .add(bounded_drag_value(
                        &mut self.closing_for_decimal,
                        DECIMAL_ODDS_MIN..=DECIMAL_ODDS_MAX,
                        0.01,
                    ))
                    .accessible_hover_text("Closing decimal odds on your selected outcome")
                    .changed();
                ui.label("Closing Against:");
                changed |= ui
                    .add(bounded_drag_value(
                        &mut self.closing_against_decimal,
                        DECIMAL_ODDS_MIN..=DECIMAL_ODDS_MAX,
                        0.01,
                    ))
                    .accessible_hover_text("Closing decimal odds on the opposing outcome")
                    .changed();
            }
        });
        changed
    }

    fn show_results_and_table(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref err) = self.calc_error {
            ui.colored_label(palette.error, err);
            return;
        }

        if let Ok(odds) = self.resolve_active_odds() {
            self.render_converted_readout(ui, &odds);
            ui.add_space(4.0);
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

        self.show_strategy_grid(ui);
        ui.add_space(4.0);
        self.show_table_export_bar(ui);
        if ev > 0.0 {
            self.show_solvency_and_sample_size(ui, ev);
        }
        self.show_clv_panel(ui);
    }

    fn show_strategy_grid(&self, ui: &mut egui::Ui) {
        egui::Grid::new("kelly_results")
            .num_columns(5)
            .striped(true)
            .spacing([24.0, 6.0])
            .show(ui, |ui| {
                ui.strong("Strategy");
                ui.strong("Fraction of Bankroll");
                ui.strong("Bet Amount");
                ui.strong("Growth Rate g(f)");
                ui.strong(format!("Floor ≤ {:.0}% Bound", self.drawdown_floor_pct));
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
                    let bound_str = row
                        .barrier_bound
                        .map_or_else(|| String::from("N/A"), |b| format!("{:.1}%", b * 100.0));
                    ui.label(bound_str);
                    ui.end_row();
                }
            });
    }

    fn show_solvency_and_sample_size(&self, ui: &mut egui::Ui, ev: f64) {
        if self.lundberg_summary.is_none() && self.sample_size_95.is_none() {
            return;
        }
        let unit_stake = (0.01 * self.bankroll).max(1e-6);
        ui.add_space(6.0);
        ui.group(|ui| {
            ui.strong("Solvency & 95%-Significance Sample Size");
            if let Some(lundberg) = self.lundberg_summary {
                ui.label(format!(
                    "1%-Fixed-Stake (${unit_stake:.2}) Cramér-Lundberg Ruin Bound: {:.3}% (r* = {:.4}, Diffusion: {:.3}%)",
                    lundberg.cramer_lundberg_bound * 100.0,
                    lundberg.adjustment_coefficient,
                    lundberg.diffusion_approximation * 100.0
                ));
            }
            if let Some(n95) = self.sample_size_95 {
                ui.label(format!(
                    "Bets Needed to Confirm {:+.2}% Yield at 95% Significance: {} (1-sided) | {} (2-sided)",
                    ev * 100.0,
                    n95.one_sided_bets,
                    n95.two_sided_bets
                ));
            }
        });
    }

    fn show_clv_panel(&self, ui: &mut egui::Ui) {
        let Some(ref audit) = self.clv_audit else {
            return;
        };
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.add_space(6.0);
        ui.group(|ui| {
            ui.strong("Closing Line Value (CLV) Audit");
            match audit {
                Ok((p_close, clv)) => {
                    let color = if clv.ev_fraction > 0.0 {
                        palette.success
                    } else {
                        palette.warning
                    };
                    ui.colored_label(
                        color,
                        format!(
                            "Devigged Fair Close: {:.2}% | CLV EV: {:+.2}% | Prob Edge: {:+.2}% | Log-Odds Shift: {:+.4}",
                            p_close * 100.0,
                            clv.ev_pct,
                            clv.implied_prob_edge * 100.0,
                            clv.log_odds_shift
                        ),
                    );
                }
                Err(err) => {
                    ui.colored_label(palette.error, *err);
                }
            }
        });
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
            ui.heading("Kelly Criterion Bet Size, Solvency & CLV Calculator");
            ui.label(
                "Calculate optimal wager sizing across synchronized odds formats, growth g(f), drawdown bounds, and CLV.",
            );
            ui.separator();

            let mut changed = self.show_input_grid(ui);
            changed |= self.show_closing_line_inputs(ui);
            if changed {
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
