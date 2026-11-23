//! Decomposed egui rendering helpers for [`BetSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    resolve_odds_for, skip_american_dead_zone, BetSizeCalculatorTool, OddsFormat,
    AMERICAN_ODDS_MAX, AMERICAN_ODDS_MIN, BANKROLL_MAX, BANKROLL_MIN, DECIMAL_ODDS_MAX,
    DECIMAL_ODDS_MIN, FRACTIONAL_MAX, FRACTIONAL_MAX_DENOMINATOR, FRACTIONAL_MIN, IMPLIED_PROB_MAX,
    IMPLIED_PROB_MIN, PROBABILITY_MAX, PROBABILITY_MIN,
};
use crate::framework::bounded_drag_value;
use eframe::egui;
use math_explorer::pure_math::statistics::kelly::{
    expected_value, kelly_fraction, variants, BankrollFraction, Odds, UnitInterval,
};

impl BetSizeCalculatorTool {
    pub(super) fn render_format_selector(&mut self, ui: &mut egui::Ui) {
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
        }
    }

    fn render_fractional_input(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            let n = bounded_drag_value(
                &mut self.fractional_numerator,
                FRACTIONAL_MIN..=FRACTIONAL_MAX,
                1.0,
            );
            changed |= ui.add(n).changed();
            ui.label("/");
            let d = bounded_drag_value(
                &mut self.fractional_denominator,
                FRACTIONAL_MIN..=FRACTIONAL_MAX,
                1.0,
            );
            changed |= ui.add(d).changed();
        });
        changed
    }

    pub(super) fn render_active_odds_input(&mut self, ui: &mut egui::Ui) {
        let changed = match self.odds_format {
            OddsFormat::Decimal => ui
                .add(bounded_drag_value(
                    &mut self.decimal_odds,
                    DECIMAL_ODDS_MIN..=DECIMAL_ODDS_MAX,
                    0.01,
                ))
                .changed(),
            OddsFormat::American => {
                let prev = self.american_odds;
                let edited = ui
                    .add(bounded_drag_value(
                        &mut self.american_odds,
                        AMERICAN_ODDS_MIN..=AMERICAN_ODDS_MAX,
                        1.0,
                    ))
                    .changed();
                if edited {
                    self.american_odds = skip_american_dead_zone(prev, self.american_odds);
                }
                edited
            }
            OddsFormat::Fractional => self.render_fractional_input(ui),
            OddsFormat::ImpliedProbability => ui
                .add(
                    bounded_drag_value(
                        &mut self.implied_probability_input,
                        IMPLIED_PROB_MIN..=IMPLIED_PROB_MAX,
                        0.1,
                    )
                    .suffix("%"),
                )
                .changed(),
        };
        if changed {
            if let Ok(odds) = self.resolve_active_odds() {
                self.sync_from_odds(odds, self.odds_format);
            }
        }
    }

    pub(super) fn render_inputs_grid(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("bet_size_inputs")
            .num_columns(2)
            .spacing([40.0, 10.0])
            .show(ui, |ui| {
                ui.label("Win Probability (%):");
                ui.add(
                    bounded_drag_value(
                        &mut self.probability_input,
                        PROBABILITY_MIN..=PROBABILITY_MAX,
                        0.1,
                    )
                    .suffix("%"),
                );
                ui.end_row();

                ui.label("Odds Format:");
                self.render_format_selector(ui);
                ui.end_row();

                ui.label("Odds:");
                self.render_active_odds_input(ui);
                ui.end_row();

                ui.label("Total Bankroll ($):");
                ui.add(
                    bounded_drag_value(&mut self.bankroll, BANKROLL_MIN..=BANKROLL_MAX, 10.0)
                        .prefix("$"),
                );
                ui.end_row();
            });
    }

    pub(super) fn render_converted_readout(&self, ui: &mut egui::Ui, odds: &Odds) {
        let (num, den) = odds.to_fractional(FRACTIONAL_MAX_DENOMINATOR);
        let american = if self.odds_format == OddsFormat::American {
            self.american_odds
        } else {
            odds.to_american()
        };
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Equivalent Odds:").strong());
            ui.label(format!("Decimal {:.3}", odds.value()));
            ui.separator();
            ui.label(format!("American {:+.0}", american));
            ui.separator();
            ui.label(format!("Fractional {num}/{den}"));
            ui.separator();
            ui.label(format!("Implied {:.2}%", odds.implied_probability() * 100.0));
            ui.separator();
            ui.label(format!("Net b = {:.3}", odds.net_profit_multiplier()));
        });
    }

    fn render_kelly_row(ui: &mut egui::Ui, label: &str, frac: BankrollFraction, bankroll: f64) {
        ui.label(label);
        ui.label(format!("{:.2}%", frac.value() * 100.0));
        match frac.bet_amount(bankroll) {
            Ok(amount) => ui.label(format!("${amount:.2}")),
            Err(_) => ui.label("Invalid bankroll"),
        };
        ui.end_row();
    }

    fn render_kelly_table(
        &self,
        ui: &mut egui::Ui,
        prob: &UnitInterval,
        odds: &Odds,
        palette: &crate::app::SemanticPalette,
    ) {
        match kelly_fraction(prob, odds) {
            Ok(full_kelly) => {
                egui::Grid::new("kelly_results")
                    .num_columns(3)
                    .spacing([40.0, 10.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Strategy").strong());
                        ui.label(egui::RichText::new("Fraction of Bankroll").strong());
                        ui.label(egui::RichText::new("Bet Amount").strong());
                        ui.end_row();
                        Self::render_kelly_row(ui, "Full Kelly:", full_kelly, self.bankroll);
                        if let Ok(half) = variants::half_kelly(prob, odds) {
                            Self::render_kelly_row(ui, "Half Kelly:", half, self.bankroll);
                        }
                        if let Ok(quarter) = variants::quarter_kelly(prob, odds) {
                            Self::render_kelly_row(ui, "Quarter Kelly:", quarter, self.bankroll);
                        }
                    });
            }
            Err(e) => {
                ui.label(
                    egui::RichText::new(format!("Error calculating Kelly: {e}"))
                        .color(palette.error),
                );
            }
        }
    }

    pub(super) fn render_results_section(&self, ui: &mut egui::Ui) {
        let prob_result = UnitInterval::new(self.probability_input / 100.0);
        let odds_result = self.resolve_active_odds();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        match (prob_result, odds_result) {
            (Ok(prob), Ok(odds)) => {
                self.render_converted_readout(ui, &odds);
                ui.add_space(6.0);
                let ev = expected_value(&prob, &odds);
                ui.label(format!("Expected Value (per $1 staked): ${ev:.3}"));
                if ev > 0.0 {
                    ui.label(egui::RichText::new("Positive Edge!").color(palette.success));
                    self.render_kelly_table(ui, &prob, &odds, &palette);
                } else {
                    ui.label(
                        egui::RichText::new("No Edge (Negative or Zero EV). Do not bet.")
                            .color(palette.error),
                    );
                }
            }
            (Err(e), _) => {
                ui.label(
                    egui::RichText::new(format!("Invalid Probability: {e}")).color(palette.error),
                );
            }
            (_, Err(e)) => {
                ui.label(egui::RichText::new(format!("Invalid Odds: {e}")).color(palette.error));
            }
        }
    }
}

// [cite:modular_polynomials_review]
