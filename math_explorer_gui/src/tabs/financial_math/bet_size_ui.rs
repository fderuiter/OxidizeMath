//! UI rendering helpers for [`super::BetSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BetSizeCalculatorTool, OddsFormat};
use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use math_explorer::pure_math::statistics::kelly::{
    expected_growth_rate, kelly_fraction, variants, BankrollFraction, Odds, UnitInterval,
};
use math_explorer::pure_math::statistics::ruin::{kelly_barrier_risk, solve_single_bet_lundberg};
use math_explorer::pure_math::statistics::validation::{
    devig_two_way, required_bets_for_95_significance, single_bet_clv,
};

impl BetSizeCalculatorTool {
    pub(super) fn render_primary_inputs(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("bet_size_inputs")
            .num_columns(2)
            .spacing([36.0, 8.0])
            .show(ui, |ui| {
                ui.label("Win Probability (%):");
                ui.add(
                    egui::DragValue::new(&mut self.probability_input)
                        .speed(0.1)
                        .range(0.0..=100.0)
                        .suffix("%"),
                );
                ui.end_row();

                ui.label("Odds Format:");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.odds_format, OddsFormat::Decimal, "Decimal");
                    ui.radio_value(&mut self.odds_format, OddsFormat::American, "American");
                    ui.radio_value(&mut self.odds_format, OddsFormat::Fractional, "Fractional");
                });
                ui.end_row();

                ui.label("Placed Odds:");
                self.render_odds_value_input(ui);
                ui.end_row();

                ui.label("Total Bankroll ($):");
                ui.add(
                    egui::DragValue::new(&mut self.bankroll)
                        .speed(10.0)
                        .range(1.0..=1_000_000.0)
                        .prefix("$"),
                );
                ui.end_row();

                ui.label("Drawdown Wealth Floor (% W₀):");
                ui.add(
                    egui::DragValue::new(&mut self.drawdown_floor_pct)
                        .speed(1.0)
                        .range(5.0..=95.0)
                        .suffix("%"),
                )
                .accessible_hover_text(
                    "Wealth floor percentage used for proportional-Kelly barrier solvency bounds",
                );
                ui.end_row();
            });
    }

    fn render_odds_value_input(&mut self, ui: &mut egui::Ui) {
        match self.odds_format {
            OddsFormat::Decimal => {
                ui.add(
                    egui::DragValue::new(&mut self.decimal_odds)
                        .speed(0.01)
                        .range(1.01..=1000.0),
                );
            }
            OddsFormat::American => {
                ui.add(
                    egui::DragValue::new(&mut self.american_odds)
                        .speed(1.0)
                        .range(-10000.0..=10000.0),
                );
            }
            OddsFormat::Fractional => {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.fractional_numerator)
                            .speed(1.0)
                            .range(1.0..=1000.0),
                    );
                    ui.label("/");
                    ui.add(
                        egui::DragValue::new(&mut self.fractional_denominator)
                            .speed(1.0)
                            .range(1.0..=1000.0),
                    );
                });
            }
        }
    }

    pub(super) fn render_closing_line_inputs(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(
                &mut self.closing_line_enabled,
                "Audit Closing Line Value (Two-Way Closing Market)",
            )
            .accessible_hover_text(
                "Remove bookmaker vigorish from closing decimal odds and compute CLV edge",
            );
            if self.closing_line_enabled {
                ui.label("Closing For:");
                ui.add(
                    egui::DragValue::new(&mut self.closing_for_decimal)
                        .speed(0.01)
                        .range(1.01..=1000.0),
                );
                ui.label("Closing Against:");
                ui.add(
                    egui::DragValue::new(&mut self.closing_against_decimal)
                        .speed(0.01)
                        .range(1.01..=1000.0),
                );
            }
        });
    }

    pub(super) fn render_kelly_strategy_table(
        &self,
        ui: &mut egui::Ui,
        prob: &UnitInterval,
        odds: &Odds,
    ) {
        let floor_ratio = (self.drawdown_floor_pct / 100.0).clamp(0.05, 0.95);
        let Ok(full) = kelly_fraction(prob, odds) else {
            return;
        };
        egui::Grid::new("kelly_results")
            .num_columns(5)
            .spacing([24.0, 8.0])
            .show(ui, |ui| {
                ui.label(egui::RichText::new("Strategy").strong());
                ui.label(egui::RichText::new("Bankroll %").strong());
                ui.label(egui::RichText::new("Bet Amount").strong());
                ui.label(egui::RichText::new("Log Growth g(f)").strong());
                ui.label(
                    egui::RichText::new(format!("Floor ≤ {:.0}% Bound", self.drawdown_floor_pct))
                        .strong(),
                );
                ui.end_row();

                self.render_strategy_row(ui, "Full Kelly:", prob, odds, &full, floor_ratio);
                if let Ok(half) = variants::half_kelly(prob, odds) {
                    self.render_strategy_row(ui, "Half Kelly:", prob, odds, &half, floor_ratio);
                }
                if let Ok(quarter) = variants::quarter_kelly(prob, odds) {
                    self.render_strategy_row(
                        ui,
                        "Quarter Kelly:",
                        prob,
                        odds,
                        &quarter,
                        floor_ratio,
                    );
                }
            });
    }

    fn render_strategy_row(
        &self,
        ui: &mut egui::Ui,
        label: &str,
        prob: &UnitInterval,
        odds: &Odds,
        frac: &BankrollFraction,
        floor_ratio: f64,
    ) {
        ui.label(label);
        ui.label(format!("{:.2}%", frac.value() * 100.0));
        match frac.bet_amount(self.bankroll) {
            Ok(amount) => ui.label(format!("${amount:.2}")),
            Err(_) => ui.label("Invalid bankroll"),
        };
        ui.label(format!(
            "{:+.5}/bet",
            expected_growth_rate(prob, odds, frac)
        ));
        let bound_text = kelly_barrier_risk(prob, odds, frac, floor_ratio).map_or_else(
            |_| "N/A".into(),
            |r| format!("{:.1}%", r.lundberg_barrier_bound * 100.0),
        );
        ui.label(bound_text);
        ui.end_row();
    }

    pub(super) fn render_solvency_and_sample_size(
        &self,
        ui: &mut egui::Ui,
        prob: &UnitInterval,
        odds: &Odds,
        ev: f64,
    ) {
        ui.add_space(6.0);
        ui.group(|ui| {
            ui.label(egui::RichText::new("Solvency & 95%-Significance Sample Size").strong());
            let unit_stake = (0.01 * self.bankroll).max(1e-6);
            if let Ok(lundberg) = solve_single_bet_lundberg(prob, odds, unit_stake, self.bankroll) {
                ui.label(format!(
                    "1%-Fixed-Stake (${unit_stake:.2}) Cramér-Lundberg Ruin Bound: {:.3}% (r* = {:.4}, Diffusion: {:.3}%)",
                    lundberg.cramer_lundberg_bound * 100.0,
                    lundberg.adjustment_coefficient,
                    lundberg.diffusion_approximation * 100.0
                ));
            }
            let odds_arr = [*odds];
            let prob_arr = [*prob];
            if let Ok(n95) = required_bets_for_95_significance(ev, &odds_arr, Some(&prob_arr)) {
                ui.label(format!(
                    "Bets Needed to Confirm {:+.2}% Yield at 95% Significance: {} (1-sided) | {} (2-sided)",
                    ev * 100.0,
                    n95.one_sided_bets,
                    n95.two_sided_bets
                ));
            }
        });
    }

    pub(super) fn render_clv_panel(&self, ui: &mut egui::Ui, placed_odds: &Odds) {
        if !self.closing_line_enabled {
            return;
        }
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.add_space(6.0);
        ui.group(|ui| {
            ui.label(egui::RichText::new("Closing Line Value (CLV) Audit").strong());
            let pair = (
                Odds::new(self.closing_for_decimal),
                Odds::new(self.closing_against_decimal),
            );
            let (Ok(c_for), Ok(c_against)) = pair else {
                ui.colored_label(palette.error, "Invalid closing decimal odds.");
                return;
            };
            let Ok((p_close, _)) = devig_two_way(&c_for, &c_against) else {
                ui.colored_label(palette.error, "Unable to devig closing market.");
                return;
            };
            let Ok(clv) = single_bet_clv(placed_odds, &p_close) else {
                ui.colored_label(palette.error, "Closing probability lies on boundary.");
                return;
            };
            let color = if clv.ev_fraction > 0.0 {
                palette.success
            } else {
                palette.warning
            };
            ui.colored_label(
                color,
                format!(
                    "Devigged Fair Close: {:.2}% | CLV EV: {:+.2}% | Prob Edge: {:+.2}% | Log-Odds Shift: {:+.4}",
                    p_close.value() * 100.0,
                    clv.ev_pct,
                    clv.implied_prob_edge * 100.0,
                    clv.log_odds_shift
                ),
            );
        });
    }
}

// [cite:modular_polynomials_review]
