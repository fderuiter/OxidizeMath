//! Solver evaluation and results table views for [`super::BetSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{BetSizingMode, UI_MAX_MARKET_ROWS};
use super::BetSizeCalculatorTool;
use crate::app::SemanticPalette;
use crate::web_io::PlatformInstant;
use eframe::egui;
use math_explorer::pure_math::statistics::kelly::{
    drawdown_constrained_kelly, drawdown_constrained_push_kelly,
    drawdown_constrained_quarter_line_kelly, expected_growth_rate, expected_value, kelly_fraction,
    push_expected_growth_rate, push_expected_value, push_kelly_fraction,
    quarter_line_expected_growth_rate, quarter_line_expected_value, quarter_line_kelly_fraction,
    solve_drawdown_constrained_slate, solve_mutually_exclusive_market, solve_simultaneous_slate,
    variants, BankrollFraction, MarketOutcome, Odds, PushWager, QuarterLineWager, SlateSolution,
    SlateWager, UnitInterval,
};

impl BetSizeCalculatorTool {
    pub(super) fn render_mode_results(&mut self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        match self.mode {
            BetSizingMode::Binary => self.render_binary_results(ui, palette),
            BetSizingMode::PushSpread => self.render_push_results(ui, palette),
            BetSizingMode::QuarterLine => self.render_quarter_line_results(ui, palette),
            BetSizingMode::MutuallyExclusive => self.render_market_results(ui, palette),
            BetSizingMode::SimultaneousSlate => self.render_slate_results(ui, palette),
        }
    }

    fn render_binary_results(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let prob_res = UnitInterval::new(self.probability_input / 100.0);
        let odds_res = self.active_odds();
        match (prob_res, odds_res) {
            (Ok(prob), Ok(odds)) => {
                let ev = expected_value(&prob, &odds);
                ui.label(format!(
                    "Expected Value (per $1 staked): ${ev:+.4} ({:+.2}%)",
                    ev * 100.0
                ));
                if ev <= 0.0 {
                    ui.colored_label(
                        palette.error,
                        "No Edge (Negative or Zero EV). Optimal stake is 0.00%.",
                    );
                    return;
                }
                if let Ok(fk) = kelly_fraction(&prob, &odds) {
                    let g = expected_growth_rate(&prob, &odds, &fk);
                    let hk = variants::half_kelly(&prob, &odds).ok();
                    let qk = variants::quarter_kelly(&prob, &odds).ok();
                    let rc = self
                        .active_drawdown_constraint()
                        .and_then(|dc| drawdown_constrained_kelly(&prob, &odds, &dc).ok());
                    ui.colored_label(
                        palette.success,
                        format!("Positive Edge! Log-growth g(f*) = {g:+.6} nats/bet"),
                    );
                    render_single_wager_grid(ui, self.bankroll, fk, hk, qk, rc);
                }
            }
            (Err(e), _) => {
                ui.colored_label(palette.error, format!("Invalid probability: {e}"));
            }
            (_, Err(e)) => {
                ui.colored_label(palette.error, format!("Invalid odds: {e}"));
            }
        }
    }

    fn render_push_results(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let odds = match self.active_odds() {
            Ok(o) => o,
            Err(e) => {
                ui.colored_label(palette.error, format!("Invalid odds: {e}"));
                return;
            }
        };
        let wager = match PushWager::from_win_and_push(
            self.probability_input / 100.0,
            self.push_probability_pct / 100.0,
            odds,
        ) {
            Ok(w) => w,
            Err(e) => {
                ui.colored_label(palette.error, format!("Invalid push distribution: {e}"));
                return;
            }
        };
        let ev = push_expected_value(&wager);
        ui.label(format!("Unit EV: ${ev:+.4} ({:+.4}%)", ev * 100.0));
        match push_kelly_fraction(&wager) {
            Ok(fk) => {
                let g = push_expected_growth_rate(&wager, &fk);
                let hk = BankrollFraction::new(0.5 * fk.value()).ok();
                let qk = BankrollFraction::new(0.25 * fk.value()).ok();
                let rc = self
                    .active_drawdown_constraint()
                    .and_then(|dc| drawdown_constrained_push_kelly(&wager, &dc).ok());
                ui.colored_label(
                    palette.success,
                    format!("Positive Edge! Log-growth g(f*) = {g:+.6} nats/wager"),
                );
                render_single_wager_grid(ui, self.bankroll, fk, hk, qk, rc);
            }
            Err(_) => {
                ui.colored_label(palette.error, "No positive edge: optimal stake is 0.00%.");
            }
        }
    }

    fn render_quarter_line_results(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let odds = match self.active_odds() {
            Ok(o) => o,
            Err(e) => {
                ui.colored_label(palette.error, format!("Invalid odds: {e}"));
                return;
            }
        };
        let wager = match QuarterLineWager::new(
            self.ql_full_win_pct / 100.0,
            self.ql_half_win_pct / 100.0,
            self.ql_push_pct / 100.0,
            self.ql_half_loss_pct / 100.0,
            self.ql_full_loss_pct / 100.0,
            odds,
        ) {
            Ok(w) => w,
            Err(e) => {
                ui.colored_label(
                    palette.error,
                    format!("Quarter-line probabilities must sum to 100%: {e}"),
                );
                return;
            }
        };
        let ev = quarter_line_expected_value(&wager);
        ui.label(format!(
            "Quarter-Line Unit EV: ${ev:+.4} ({:+.4}%)",
            ev * 100.0
        ));
        match quarter_line_kelly_fraction(&wager) {
            Ok(fk) => {
                let g = quarter_line_expected_growth_rate(&wager, &fk);
                let hk = BankrollFraction::new(0.5 * fk.value()).ok();
                let qk = BankrollFraction::new(0.25 * fk.value()).ok();
                let rc = self
                    .active_drawdown_constraint()
                    .and_then(|dc| drawdown_constrained_quarter_line_kelly(&wager, &dc).ok());
                ui.colored_label(
                    palette.success,
                    format!("Positive Edge! Log-growth g(f*) = {g:+.6} nats/wager"),
                );
                render_single_wager_grid(ui, self.bankroll, fk, hk, qk, rc);
            }
            Err(_) => {
                ui.colored_label(palette.error, "No positive edge: optimal stake is 0.00%.");
            }
        }
    }

    fn render_market_results(&mut self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let n = self.market_outcome_count.min(UI_MAX_MARKET_ROWS);
        let default_mo =
            MarketOutcome::new(UnitInterval::new(0.5).unwrap(), Odds::new(2.0).unwrap());
        let mut outcomes = [default_mo; UI_MAX_MARKET_ROWS];
        for (slot, row) in outcomes.iter_mut().zip(self.market_rows.iter()).take(n) {
            let p = match UnitInterval::new(row.probability_pct / 100.0) {
                Ok(p) => p,
                Err(e) => {
                    ui.colored_label(palette.error, format!("Invalid probability: {e}"));
                    return;
                }
            };
            let o = match Odds::new(row.decimal_odds) {
                Ok(o) => o,
                Err(e) => {
                    ui.colored_label(palette.error, format!("Invalid odds: {e}"));
                    return;
                }
            };
            *slot = MarketOutcome::new(p, o);
        }
        match solve_mutually_exclusive_market(&outcomes[..n], &mut self.market_scratch) {
            Ok(alloc) => {
                let cash_pct = (1.0 - alloc.total_fraction()) * 100.0;
                ui.label(format!(
                    "Reserve Rate r(S): {:.4} | Total Staked: {:.2}% | Cash Reserve: {cash_pct:.2}% | Log-Growth: {:+.6}",
                    alloc.reserve_rate(),
                    alloc.total_fraction() * 100.0,
                    alloc.expected_growth_rate()
                ));
                render_market_allocation_table(ui, self.bankroll, alloc.stakes());
            }
            Err(e) => {
                ui.colored_label(
                    palette.error,
                    format!("Market probabilities must sum to 100%: {e}"),
                );
            }
        }
    }

    fn render_slate_results(&mut self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let k = self.slate_count.clamp(1, 64);
        let wager = match Odds::from_net_odds(self.slate_net_odds).and_then(|o| {
            PushWager::from_win_and_push(self.slate_win_pct / 100.0, self.slate_push_pct / 100.0, o)
                .map(SlateWager::from_push)
        }) {
            Ok(w) => w,
            Err(e) => {
                ui.colored_label(palette.error, format!("Invalid slate wager: {e}"));
                return;
            }
        };
        let wagers = [wager; 64];
        let t0 = PlatformInstant::now();
        let sol_res = solve_simultaneous_slate(&wagers[..k], &mut self.slate_scratch);
        let rc_res = self.active_drawdown_constraint().and_then(|dc| {
            solve_drawdown_constrained_slate(&wagers[..k], &dc, &mut self.slate_scratch).ok()
        });
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        match sol_res {
            Ok(sol) => {
                let per_game = sol.stakes().first().copied().unwrap_or(0.0);
                ui.colored_label(
                    palette.success,
                    format!(
                        "Solved {k}-Game Slate in {elapsed_ms:.3} ms (KKT {:.2e}) | Joint g(f*) = {:+.6}",
                        sol.kkt_residual(),
                        sol.expected_growth_rate()
                    ),
                );
                render_slate_summary_grid(
                    ui,
                    self.bankroll,
                    k,
                    per_game,
                    sol.total_fraction(),
                    rc_res.as_ref(),
                );
            }
            Err(e) => {
                ui.colored_label(palette.error, format!("Slate solver error: {e}"));
            }
        }
    }
}

fn render_single_wager_grid(
    ui: &mut egui::Ui,
    bankroll: f64,
    fk: BankrollFraction,
    hk: Option<BankrollFraction>,
    qk: Option<BankrollFraction>,
    rc: Option<BankrollFraction>,
) {
    egui::Grid::new("kelly_results")
        .num_columns(3)
        .spacing([36.0, 8.0])
        .show(ui, |ui| {
            ui.label(egui::RichText::new("Strategy").strong());
            ui.label(egui::RichText::new("Fraction of Bankroll").strong());
            ui.label(egui::RichText::new("Bet Amount").strong());
            ui.end_row();
            render_fraction_row(ui, "Full Kelly:", fk, bankroll);
            if let Some(h) = hk {
                render_fraction_row(ui, "Half Kelly:", h, bankroll);
            }
            if let Some(q) = qk {
                render_fraction_row(ui, "Quarter Kelly:", q, bankroll);
            }
            if let Some(r) = rc {
                render_fraction_row(ui, "Drawdown-Constrained Kelly:", r, bankroll);
            }
        });
}

fn render_fraction_row(ui: &mut egui::Ui, label: &str, frac: BankrollFraction, bankroll: f64) {
    ui.label(label);
    ui.label(format!("{:.4}%", frac.value() * 100.0));
    if let Ok(amount) = frac.bet_amount(bankroll) {
        ui.label(format!("${amount:.2}"));
    } else {
        ui.label("Invalid bankroll");
    }
    ui.end_row();
}

fn render_market_allocation_table(ui: &mut egui::Ui, bankroll: f64, stakes: &[f64]) {
    egui::Grid::new("market_allocation_results")
        .num_columns(3)
        .spacing([36.0, 6.0])
        .show(ui, |ui| {
            ui.label(egui::RichText::new("Outcome").strong());
            ui.label(egui::RichText::new("Optimal Stake (%)").strong());
            ui.label(egui::RichText::new("Wager ($)").strong());
            ui.end_row();
            for (idx, &f) in stakes.iter().enumerate() {
                ui.label(format!("Outcome #{}", idx + 1));
                ui.label(format!("{:.4}%", f * 100.0));
                ui.label(format!("${:.2}", f * bankroll));
                ui.end_row();
            }
        });
}

fn render_slate_summary_grid(
    ui: &mut egui::Ui,
    bankroll: f64,
    k: usize,
    per_game: f64,
    total_frac: f64,
    rc: Option<&SlateSolution>,
) {
    egui::Grid::new("slate_summary_results")
        .num_columns(4)
        .spacing([28.0, 8.0])
        .show(ui, |ui| {
            ui.label(egui::RichText::new("Policy").strong());
            ui.label(egui::RichText::new("Per-Game Stake").strong());
            ui.label(egui::RichText::new(format!("Total ({k} Games)")).strong());
            ui.label(egui::RichText::new("Total Outlay ($)").strong());
            ui.end_row();

            ui.label("Unconstrained Joint Kelly:");
            ui.label(format!("{:.4}%", per_game * 100.0));
            ui.label(format!("{:.4}%", total_frac * 100.0));
            ui.label(format!("${:.2}", total_frac * bankroll));
            ui.end_row();

            if let Some(rc_sol) = rc {
                let rc_game = rc_sol.stakes().first().copied().unwrap_or(0.0);
                let rc_tot = rc_sol.total_fraction();
                ui.label("Drawdown-Constrained Slate:");
                ui.label(format!("{:.4}%", rc_game * 100.0));
                ui.label(format!("{:.4}%", rc_tot * 100.0));
                ui.label(format!("${:.2}", rc_tot * bankroll));
                ui.end_row();
            }
        });
}

// [cite:clinical_trials_statistics]
