//! Immediate-mode input controls for [`super::BetSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{BetSizingMode, OddsFormat, UI_MAX_MARKET_ROWS};
use super::BetSizeCalculatorTool;
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use eframe::egui;
use math_explorer::pure_math::statistics::kelly::EXACT_CONVOLUTION_MAX_WAGERS;

impl BetSizeCalculatorTool {
    pub(super) fn render_calculator(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Kelly Criterion Multi-State & Portfolio Calculator");
                ui.label(
                    "Exact convex solvers for binary, push spread, Asian quarter-line, N-way markets, and simultaneous slates.",
                );
                self.render_mode_and_preset_bar(ui);
                ui.separator();
                self.render_shared_inputs(ui);
                ui.add_space(6.0);
                self.render_mode_inputs(ui);
                ui.separator();
                ui.heading("Optimal Sizing Results");
                let palette = SemanticPalette::from_visuals(ui.visuals());
                self.render_mode_results(ui, &palette);
            });
        });
    }

    fn render_mode_and_preset_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Solver Mode:").strong());
            for mode in BetSizingMode::ALL {
                ui.radio_value(&mut self.mode, mode, mode.label());
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
    }

    fn render_shared_inputs(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("bet_size_shared_inputs")
            .num_columns(2)
            .spacing([32.0, 8.0])
            .show(ui, |ui| {
                ui.label("Total Bankroll ($):");
                ui.add(
                    egui::DragValue::new(&mut self.bankroll)
                        .speed(10.0)
                        .range(1.0..=1_000_000.0)
                        .prefix("$"),
                );
                ui.end_row();

                if matches!(
                    self.mode,
                    BetSizingMode::Binary | BetSizingMode::PushSpread | BetSizingMode::QuarterLine
                ) {
                    self.render_odds_rows(ui);
                }

                if self.mode != BetSizingMode::MutuallyExclusive {
                    self.render_drawdown_row(ui);
                }
            });
    }

    fn render_odds_rows(&mut self, ui: &mut egui::Ui) {
        ui.label("Odds Format:");
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.odds_format, OddsFormat::Decimal, "Decimal");
            ui.radio_value(&mut self.odds_format, OddsFormat::American, "American");
            ui.radio_value(&mut self.odds_format, OddsFormat::Fractional, "Fractional");
        });
        ui.end_row();

        ui.label("Odds:");
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
                        .range(-10_000.0..=10_000.0),
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
        ui.end_row();
    }

    fn render_drawdown_row(&mut self, ui: &mut egui::Ui) {
        ui.label("Drawdown Bound (α, β):");
        ui.horizontal(|ui| {
            ui.label("Floor α:");
            ui.add(
                egui::DragValue::new(&mut self.drawdown_alpha)
                    .speed(0.01)
                    .range(0.05..=0.99),
            )
            .accessible_hover_text(
                "Wealth floor fraction alpha in P(min W_t <= alpha W_0) <= beta",
            );
            ui.label("Tail β:");
            ui.add(
                egui::DragValue::new(&mut self.drawdown_beta)
                    .speed(0.01)
                    .range(0.001..=0.99),
            )
            .accessible_hover_text("Maximum allowed drawdown tail probability beta");
            if let Some(dc) = self.active_drawdown_constraint() {
                ui.label(format!("(λ = {:.3})", dc.lambda()));
            }
        });
        ui.end_row();
    }

    fn render_mode_inputs(&mut self, ui: &mut egui::Ui) {
        match self.mode {
            BetSizingMode::Binary => {
                ui.horizontal(|ui| {
                    ui.label("Win Probability (%):");
                    ui.add(
                        egui::DragValue::new(&mut self.probability_input)
                            .speed(0.1)
                            .range(0.0..=100.0)
                            .suffix("%"),
                    );
                });
            }
            BetSizingMode::PushSpread => self.render_push_inputs(ui),
            BetSizingMode::QuarterLine => self.render_quarter_line_inputs(ui),
            BetSizingMode::MutuallyExclusive => self.render_market_inputs(ui),
            BetSizingMode::SimultaneousSlate => self.render_slate_inputs(ui),
        }
    }

    fn render_push_inputs(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Win (%):");
            ui.add(
                egui::DragValue::new(&mut self.probability_input)
                    .speed(0.1)
                    .range(0.0..=100.0)
                    .suffix("%"),
            );
            ui.label("Push (%):");
            ui.add(
                egui::DragValue::new(&mut self.push_probability_pct)
                    .speed(0.1)
                    .range(0.0..=99.9)
                    .suffix("%"),
            );
            let loss_pct = 100.0 - self.probability_input - self.push_probability_pct;
            ui.label(format!("Implied Loss: {loss_pct:.2}%"));
        });
    }

    fn render_quarter_line_inputs(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            for (lbl, val) in [
                ("Full Win:", &mut self.ql_full_win_pct),
                ("Half Win:", &mut self.ql_half_win_pct),
                ("Push:", &mut self.ql_push_pct),
                ("Half Loss:", &mut self.ql_half_loss_pct),
                ("Full Loss:", &mut self.ql_full_loss_pct),
            ] {
                ui.label(lbl);
                ui.add(
                    egui::DragValue::new(val)
                        .speed(0.1)
                        .range(0.0..=100.0)
                        .suffix("%"),
                );
            }
        });
    }

    fn render_market_inputs(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Outcomes (N):");
            ui.add(egui::Slider::new(
                &mut self.market_outcome_count,
                2..=UI_MAX_MARKET_ROWS,
            ));
        });
        egui::Grid::new("market_outcome_inputs")
            .num_columns(3)
            .spacing([24.0, 6.0])
            .show(ui, |ui| {
                ui.label(egui::RichText::new("Outcome").strong());
                ui.label(egui::RichText::new("True Prob (%)").strong());
                ui.label(egui::RichText::new("Decimal Odds").strong());
                ui.end_row();
                for idx in 0..self.market_outcome_count {
                    ui.label(format!("Outcome #{}", idx + 1));
                    ui.add(
                        egui::DragValue::new(&mut self.market_rows[idx].probability_pct)
                            .speed(0.1)
                            .range(0.0..=100.0)
                            .suffix("%"),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.market_rows[idx].decimal_odds)
                            .speed(0.05)
                            .range(1.01..=1000.0),
                    );
                    ui.end_row();
                }
            });
    }

    fn render_slate_inputs(&mut self, ui: &mut egui::Ui) {
        let solver_badge = if self.slate_count <= EXACT_CONVOLUTION_MAX_WAGERS {
            "Exact State Convolution"
        } else {
            "128-Node Laplace Quadrature"
        };
        ui.horizontal_wrapped(|ui| {
            ui.label("Simultaneous Games (K):");
            ui.add(egui::Slider::new(&mut self.slate_count, 1..=64));
            ui.label(format!("({solver_badge})"));
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Per-Game Win (%):");
            ui.add(
                egui::DragValue::new(&mut self.slate_win_pct)
                    .speed(0.1)
                    .range(0.1..=99.9)
                    .suffix("%"),
            );
            ui.label("Push (%):");
            ui.add(
                egui::DragValue::new(&mut self.slate_push_pct)
                    .speed(0.1)
                    .range(0.0..=95.0)
                    .suffix("%"),
            );
            ui.label("Net Odds (b):");
            ui.add(
                egui::DragValue::new(&mut self.slate_net_odds)
                    .speed(0.05)
                    .range(0.05..=20.0),
            );
        });
    }
}

// [cite:clinical_trials_statistics]
