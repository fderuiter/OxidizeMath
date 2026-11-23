//! Immediate-mode input controls, sizing table exports, and growth plot for [`super::BetSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{BetSizingMode, OddsFormat, UI_MAX_MARKET_ROWS};
use super::BetSizeCalculatorTool;
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use eframe::egui;
use math_explorer::pure_math::statistics::kelly::EXACT_CONVOLUTION_MAX_WAGERS;

impl BetSizeCalculatorTool {
    pub(super) fn render_calculator(&mut self, ui: &mut egui::Ui) {
        self.poll_save_channel(ui.ctx());
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Kelly Criterion Multi-State & Portfolio Calculator");
                ui.label(
                    "Exact convex solvers, logarithmic growth curves g(f), and formula-safe table exports across multi-state wagers, N-way markets, and simultaneous slates.",
                );
                let mut changed = self.render_mode_and_preset_bar(ui);
                ui.separator();
                changed |= self.render_shared_inputs(ui);
                ui.add_space(6.0);
                changed |= self.render_mode_inputs(ui);
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

    fn render_shared_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::Grid::new("bet_size_shared_inputs")
            .num_columns(2)
            .spacing([32.0, 8.0])
            .show(ui, |ui| {
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

                if matches!(
                    self.mode,
                    BetSizingMode::Binary | BetSizingMode::PushSpread | BetSizingMode::QuarterLine
                ) {
                    changed |= self.render_odds_rows(ui);
                }

                if self.mode != BetSizingMode::MutuallyExclusive {
                    changed |= self.render_drawdown_row(ui);
                }
            });
        changed
    }

    fn render_odds_rows(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
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
                        .range(-10_000.0..=10_000.0),
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

    fn render_drawdown_row(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Drawdown Bound (α, β):");
        ui.horizontal(|ui| {
            ui.label("Floor α:");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.drawdown_alpha)
                        .speed(0.01)
                        .range(0.05..=0.99),
                )
                .accessible_hover_text(
                    "Wealth floor fraction alpha in P(min W_t <= alpha W_0) <= beta",
                )
                .changed();
            ui.label("Tail β:");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.drawdown_beta)
                        .speed(0.01)
                        .range(0.001..=0.99),
                )
                .accessible_hover_text("Maximum allowed drawdown tail probability beta")
                .changed();
            if let Some(dc) = self.active_drawdown_constraint() {
                ui.label(format!("(λ = {:.3})", dc.lambda()));
            }
        });
        ui.end_row();
        changed
    }

    fn render_mode_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        match self.mode {
            BetSizingMode::Binary => {
                ui.horizontal(|ui| {
                    ui.label("Win Probability (%):");
                    ui.add(
                        egui::DragValue::new(&mut self.probability_input)
                            .speed(0.1)
                            .range(0.0..=100.0)
                            .suffix("%"),
                    )
                    .accessible_hover_text(
                        "Estimated probability of winning the wager (0% to 100%)",
                    )
                    .changed()
                })
                .inner
            }
            BetSizingMode::PushSpread => self.render_push_inputs(ui),
            BetSizingMode::QuarterLine => self.render_quarter_line_inputs(ui),
            BetSizingMode::MutuallyExclusive => self.render_market_inputs(ui),
            BetSizingMode::SimultaneousSlate => self.render_slate_inputs(ui),
        }
    }

    fn render_push_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.label("Win (%):");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.probability_input)
                        .speed(0.1)
                        .range(0.0..=100.0)
                        .suffix("%"),
                )
                .changed();
            ui.label("Push (%):");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.push_probability_pct)
                        .speed(0.1)
                        .range(0.0..=99.9)
                        .suffix("%"),
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
                        egui::DragValue::new(val)
                            .speed(0.1)
                            .range(0.0..=100.0)
                            .suffix("%"),
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
                            egui::DragValue::new(&mut self.market_rows[idx].probability_pct)
                                .speed(0.1)
                                .range(0.0..=100.0)
                                .suffix("%"),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.market_rows[idx].decimal_odds)
                                .speed(0.05)
                                .range(1.01..=1000.0),
                        )
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
                .add(
                    egui::DragValue::new(&mut self.slate_win_pct)
                        .speed(0.1)
                        .range(0.1..=99.9)
                        .suffix("%"),
                )
                .changed();
            ui.label("Push (%):");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.slate_push_pct)
                        .speed(0.1)
                        .range(0.0..=95.0)
                        .suffix("%"),
                )
                .changed();
            ui.label("Net Odds (b):");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.slate_net_odds)
                        .speed(0.05)
                        .range(0.05..=20.0),
                )
                .changed();
        });
        changed
    }

    fn show_results_and_table(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref err) = self.calc_error {
            ui.colored_label(palette.error, err);
            return;
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
}

// [cite:clinical_trials_statistics]
