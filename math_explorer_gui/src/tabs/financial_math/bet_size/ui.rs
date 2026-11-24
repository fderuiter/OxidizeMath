//! Immediate-mode input controls, synchronized odds editors, and CLV audit UI for [`super::BetSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::BetSizeCalculatorTool;
use super::state::{
    AMERICAN_ODDS_MAX, AMERICAN_ODDS_MIN, BANKROLL_MAX, BANKROLL_MIN, BetSizingMode,
    DECIMAL_ODDS_MAX, DECIMAL_ODDS_MIN, FRACTIONAL_MAX, FRACTIONAL_MIN, IMPLIED_PROB_MAX,
    IMPLIED_PROB_MIN, OddsFormat, PROBABILITY_MAX, PROBABILITY_MIN, UI_MAX_MARKET_ROWS,
    resolve_odds_for, skip_american_dead_zone,
};
use crate::accessibility::AccessibleHoverText;
use crate::framework::bounded_drag_value;
use eframe::egui;
use math_explorer::pure_math::statistics::kelly::EXACT_CONVOLUTION_MAX_WAGERS;

impl BetSizeCalculatorTool {
    pub(super) fn render_calculator(&mut self, ui: &mut egui::Ui) {
        self.poll_save_channel(ui.ctx());
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Kelly Criterion Multi-State, Portfolio, Solvency & CLV Calculator");
                ui.label(
                    "Exact convex solvers, synchronized four-format odds conversions, growth curves g(f), solvency bounds, and CLV.",
                );
                let mut changed = self.render_mode_and_preset_bar(ui);
                ui.separator();
                changed |= self.render_active_inputs(ui);
                if matches!(
                    self.mode,
                    BetSizingMode::Binary | BetSizingMode::PushSpread | BetSizingMode::QuarterLine
                ) {
                    ui.add_space(4.0);
                    changed |= self.show_closing_line_inputs(ui);
                }
                if self.mode != BetSizingMode::MutuallyExclusive {
                    ui.add_space(4.0);
                    changed |= self.render_drawdown_collapsible(ui);
                }
                if changed {
                    self.recalculate();
                }
                ui.separator();
                ui.heading("Results & Sizing Table");
                self.show_results_and_table(ui);
                ui.separator();
                self.show_growth_curve_plot(ui);
            });
        });
    }

    fn render_mode_and_preset_bar(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Solver Mode:").strong());
            for mode in BetSizingMode::ALL {
                changed |= ui.radio_value(&mut self.mode, mode, mode.label()).changed();
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Canonical Presets:");
            if ui
                .button("-3 Spread @ -110")
                .accessible_hover_text("50% win, 10% push, 40% loss at -110 odds")
                .clicked()
            {
                self.load_push_spread_preset();
            }
            if ui
                .button("-2.75 Quarter-Line @ -110")
                .accessible_hover_text("44% full win, 12% half win, 44% full loss at -110 odds")
                .clicked()
            {
                self.load_quarter_line_preset();
            }
            if ui
                .button("3-Way Market")
                .accessible_hover_text("Smoczynski-Tomkins active-set allocation on 3 outcomes")
                .clicked()
            {
                self.load_three_way_market_preset();
            }
            if ui
                .button("10-Game Slate")
                .accessible_hover_text("10 simultaneous independent wagers (p=58%, b=1.0)")
                .clicked()
            {
                self.load_ten_game_slate_preset();
            }
        });
        changed
    }

    fn render_active_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        match self.mode {
            BetSizingMode::Binary => self.render_binary_grid(ui),
            BetSizingMode::PushSpread => {
                let mut changed = self.render_push_inputs(ui);
                ui.add_space(4.0);
                changed |= self.render_odds_and_bankroll_grid(ui, true);
                changed
            }
            BetSizingMode::QuarterLine => {
                let mut changed = self.render_quarter_line_inputs(ui);
                ui.add_space(4.0);
                changed |= self.render_odds_and_bankroll_grid(ui, true);
                changed
            }
            BetSizingMode::MutuallyExclusive => {
                let mut changed = self.render_odds_and_bankroll_grid(ui, false);
                ui.add_space(4.0);
                changed |= self.render_market_inputs(ui);
                changed
            }
            BetSizingMode::SimultaneousSlate => {
                let mut changed = self.render_odds_and_bankroll_grid(ui, false);
                ui.add_space(4.0);
                changed |= self.render_slate_inputs(ui);
                changed
            }
        }
    }

    fn render_binary_grid(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::Grid::new("bet_size_binary_inputs")
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

                changed |= self.render_odds_rows_in_grid(ui);
                changed |= self.render_bankroll_row_in_grid(ui);
            });
        changed
    }

    fn render_odds_and_bankroll_grid(&mut self, ui: &mut egui::Ui, include_odds: bool) -> bool {
        let mut changed = false;
        egui::Grid::new("bet_size_shared_inputs")
            .num_columns(2)
            .spacing([32.0, 8.0])
            .show(ui, |ui| {
                if include_odds {
                    changed |= self.render_odds_rows_in_grid(ui);
                }
                changed |= self.render_bankroll_row_in_grid(ui);
            });
        changed
    }

    fn render_odds_rows_in_grid(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Odds Format:");
        changed |= self.render_format_selector(ui);
        ui.end_row();

        ui.label("Placed Odds:");
        changed |= self.show_odds_value_editor(ui);
        ui.end_row();
        changed
    }

    fn render_bankroll_row_in_grid(&mut self, ui: &mut egui::Ui) -> bool {
        ui.label("Total Bankroll ($):");
        let mut changed = ui
            .add(
                bounded_drag_value(&mut self.bankroll, BANKROLL_MIN..=BANKROLL_MAX, 10.0)
                    .prefix("$"),
            )
            .accessible_hover_text("Available bankroll capital in dollars")
            .changed();
        ui.end_row();

        if self.mode == BetSizingMode::Binary {
            ui.label("Drawdown Wealth Floor (% W₀):");
            changed |= ui
                .add(egui::Slider::new(&mut self.drawdown_floor_pct, 5.0..=95.0).suffix("%"))
                .accessible_hover_text(
                    "Wealth floor percentage used for proportional-Kelly barrier solvency bounds",
                )
                .changed();
            ui.end_row();
        }
        changed
    }

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
        if changed && let Ok(odds) = self.resolve_active_odds() {
            self.sync_from_odds(odds, self.odds_format);
        }
        changed
    }

    fn render_push_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.label("Win (%):");
            changed |= ui
                .add(
                    bounded_drag_value(
                        &mut self.probability_input,
                        PROBABILITY_MIN..=PROBABILITY_MAX,
                        0.1,
                    )
                    .suffix("%"),
                )
                .changed();
            ui.label("Push (%):");
            changed |= ui
                .add(
                    bounded_drag_value(&mut self.push_probability_pct, 0.0..=99.9, 0.1).suffix("%"),
                )
                .changed();
            let loss_pct = 100.0 - self.probability_input - self.push_probability_pct;
            ui.label(format!("Implied Loss: {loss_pct:.2}%"));
        });
        changed
    }

    fn render_quarter_line_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            for (lbl, val) in [
                ("Full Win:", &mut self.ql_full_win_pct),
                ("Half Win:", &mut self.ql_half_win_pct),
                ("Push:", &mut self.ql_push_pct),
                ("Half Loss:", &mut self.ql_half_loss_pct),
                ("Full Loss:", &mut self.ql_full_loss_pct),
            ] {
                ui.label(lbl);
                changed |= ui
                    .add(
                        bounded_drag_value(val, PROBABILITY_MIN..=PROBABILITY_MAX, 0.1).suffix("%"),
                    )
                    .changed();
            }
        });
        changed
    }

    fn render_market_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Outcomes (N):");
            changed |= ui
                .add(egui::Slider::new(
                    &mut self.market_outcome_count,
                    2..=UI_MAX_MARKET_ROWS,
                ))
                .changed();
        });
        egui::Grid::new("market_outcome_inputs")
            .num_columns(3)
            .spacing([24.0, 6.0])
            .show(ui, |ui| {
                ui.strong("Outcome");
                ui.strong("True Prob (%)");
                ui.strong("Decimal Odds");
                ui.end_row();
                for idx in 0..self.market_outcome_count {
                    ui.label(format!("Outcome #{}", idx + 1));
                    changed |= ui
                        .add(
                            bounded_drag_value(
                                &mut self.market_rows[idx].probability_pct,
                                PROBABILITY_MIN..=PROBABILITY_MAX,
                                0.1,
                            )
                            .suffix("%"),
                        )
                        .changed();
                    changed |= ui
                        .add(bounded_drag_value(
                            &mut self.market_rows[idx].decimal_odds,
                            DECIMAL_ODDS_MIN..=DECIMAL_ODDS_MAX,
                            0.05,
                        ))
                        .changed();
                    ui.end_row();
                }
            });
        changed
    }

    fn render_slate_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let solver_badge = if self.slate_count <= EXACT_CONVOLUTION_MAX_WAGERS {
            "Exact State Convolution"
        } else {
            "128-Node Laplace Quadrature"
        };
        ui.horizontal_wrapped(|ui| {
            ui.label("Simultaneous Games (K):");
            changed |= ui
                .add(egui::Slider::new(&mut self.slate_count, 1..=64))
                .changed();
            ui.label(format!("({solver_badge})"));
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Per-Game Win (%):");
            changed |= ui
                .add(bounded_drag_value(&mut self.slate_win_pct, 0.1..=99.9, 0.1).suffix("%"))
                .changed();
            ui.label("Push (%):");
            changed |= ui
                .add(bounded_drag_value(&mut self.slate_push_pct, 0.0..=95.0, 0.1).suffix("%"))
                .changed();
            ui.label("Net Odds (b):");
            changed |= ui
                .add(bounded_drag_value(
                    &mut self.slate_net_odds,
                    0.05..=20.0,
                    0.05,
                ))
                .changed();
        });
        changed
    }
}

// [cite:clinical_trials_statistics]
// [cite:modular_polynomials_review]
