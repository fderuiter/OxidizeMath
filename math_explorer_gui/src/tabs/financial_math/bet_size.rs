//! Kelly Criterion Bet Size, Growth Curve, Solvency Barrier, and Closing Line Value Calculator.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::kelly::{
    expected_growth_rate, expected_value, kelly_fraction, variants, BankrollFraction, Odds,
    UnitInterval,
};
use math_explorer::pure_math::statistics::ruin::{
    kelly_barrier_risk, solve_single_bet_lundberg, LundbergSolution,
};
use math_explorer::pure_math::statistics::validation::{
    devig_two_way, required_bets_for_95_significance, single_bet_clv, ClvMetrics, SampleSize95,
};
use std::sync::mpsc::Receiver;

#[path = "bet_size/export.rs"]
mod export;
#[path = "bet_size/state.rs"]
mod state;
#[path = "bet_size/ui.rs"]
mod ui;

#[derive(Debug, PartialEq, Eq, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(crate) enum OddsFormat {
    Decimal,
    American,
    Fractional,
}

pub(crate) const DEFAULT_CLOSING_DECIMAL: f64 = 1.91;
pub(crate) const DEFAULT_DRAWDOWN_FLOOR_PCT: f64 = 50.0;

#[derive(Debug, Clone)]
pub(crate) struct KellyStrategyRow {
    pub label: String,
    pub fraction: f64,
    pub bet_amount: f64,
    pub growth_rate: f64,
    pub ev_per_unit: f64,
    pub barrier_bound: Option<f64>,
}

pub struct BetSizeCalculatorTool {
    pub(crate) probability_input: f64,
    pub(crate) odds_format: OddsFormat,
    pub(crate) decimal_odds: f64,
    pub(crate) american_odds: f64,
    pub(crate) fractional_numerator: f64,
    pub(crate) fractional_denominator: f64,
    pub(crate) bankroll: f64,
    pub(crate) drawdown_floor_pct: f64,
    pub(crate) closing_line_enabled: bool,
    pub(crate) closing_for_decimal: f64,
    pub(crate) closing_against_decimal: f64,

    // Cached plot, table, solvency, sample-size & CLV results (recomputed only when inputs change)
    pub(crate) growth_curve_points: Vec<PlotPoint>,
    pub(crate) quarter_marker: Vec<PlotPoint>,
    pub(crate) half_marker: Vec<PlotPoint>,
    pub(crate) full_marker: Vec<PlotPoint>,
    pub(crate) zero_line_points: [PlotPoint; 2],
    pub(crate) strategy_rows: Vec<KellyStrategyRow>,
    pub(crate) ev_per_unit: Option<f64>,
    pub(crate) lundberg_summary: Option<LundbergSolution>,
    pub(crate) sample_size_95: Option<SampleSize95>,
    pub(crate) clv_audit: Option<Result<(f64, ClvMetrics), &'static str>>,
    pub(crate) calc_error: Option<String>,

    pub(crate) save_rx: Option<Receiver<Result<String, String>>>,
    pub(crate) io_status: Option<(String, bool)>,
}

impl Default for BetSizeCalculatorTool {
    fn default() -> Self {
        let mut tool = Self {
            probability_input: 55.0,
            odds_format: OddsFormat::Decimal,
            decimal_odds: 2.0,
            american_odds: 100.0,
            fractional_numerator: 1.0,
            fractional_denominator: 1.0,
            bankroll: 1000.0,
            drawdown_floor_pct: DEFAULT_DRAWDOWN_FLOOR_PCT,
            closing_line_enabled: false,
            closing_for_decimal: DEFAULT_CLOSING_DECIMAL,
            closing_against_decimal: DEFAULT_CLOSING_DECIMAL,
            growth_curve_points: Vec::new(),
            quarter_marker: Vec::new(),
            half_marker: Vec::new(),
            full_marker: Vec::new(),
            zero_line_points: [PlotPoint::new(0.0, 0.0), PlotPoint::new(0.25, 0.0)],
            strategy_rows: Vec::new(),
            ev_per_unit: None,
            lundberg_summary: None,
            sample_size_95: None,
            clv_audit: None,
            calc_error: None,
            save_rx: None,
            io_status: None,
        };
        tool.recalculate();
        tool
    }
}

impl BetSizeCalculatorTool {
    pub(crate) fn current_placed_odds(&self) -> Result<Odds, math_explorer::error::KellyError> {
        match self.odds_format {
            OddsFormat::Decimal => Odds::new(self.decimal_odds),
            OddsFormat::American => Odds::from_american(self.american_odds),
            OddsFormat::Fractional => {
                Odds::from_fractional(self.fractional_numerator, self.fractional_denominator)
            }
        }
    }

    pub(crate) fn active_odds(&self) -> Result<Odds, String> {
        self.current_placed_odds()
            .map_err(|e| format!("Invalid Odds: {e}"))
    }

    fn push_marker_and_row(
        &mut self,
        label: &str,
        prob: &UnitInterval,
        odds: &Odds,
        frac: &BankrollFraction,
        ev: f64,
        marker_slot: usize,
    ) {
        let g = expected_growth_rate(prob, odds, frac);
        let bet_amount = frac.bet_amount(self.bankroll).unwrap_or(0.0);
        let floor_ratio = (self.drawdown_floor_pct / 100.0).clamp(0.05, 0.95);
        let barrier_bound = kelly_barrier_risk(prob, odds, frac, floor_ratio)
            .ok()
            .map(|r| r.lundberg_barrier_bound);
        if g.is_finite() {
            let pt = PlotPoint::new(frac.value(), g);
            match marker_slot {
                0 => self.quarter_marker.push(pt),
                1 => self.half_marker.push(pt),
                _ => self.full_marker.push(pt),
            }
        }
        self.strategy_rows.push(KellyStrategyRow {
            label: label.to_string(),
            fraction: frac.value(),
            bet_amount,
            growth_rate: g,
            ev_per_unit: ev,
            barrier_bound,
        });
    }

    fn populate_growth_curve(&mut self, prob: &UnitInterval, odds: &Odds, f_max: f64) {
        const STEPS: usize = 120;
        for idx in 0..=STEPS {
            let f = f_max * (idx as f64 / STEPS as f64);
            if let Ok(frac) = BankrollFraction::new(f) {
                let g = expected_growth_rate(prob, odds, &frac);
                if g.is_finite() {
                    self.growth_curve_points.push(PlotPoint::new(f, g));
                }
            }
        }
        self.zero_line_points = [PlotPoint::new(0.0, 0.0), PlotPoint::new(f_max, 0.0)];
    }

    fn recompute_clv_audit(&mut self, placed_odds: &Odds) {
        if !self.closing_line_enabled {
            self.clv_audit = None;
            return;
        }
        let pair = (
            Odds::new(self.closing_for_decimal),
            Odds::new(self.closing_against_decimal),
        );
        let (Ok(c_for), Ok(c_against)) = pair else {
            self.clv_audit = Some(Err("Invalid closing decimal odds."));
            return;
        };
        let Ok((p_close, _)) = devig_two_way(&c_for, &c_against) else {
            self.clv_audit = Some(Err("Unable to devig closing market."));
            return;
        };
        let Ok(clv) = single_bet_clv(placed_odds, &p_close) else {
            self.clv_audit = Some(Err("Closing probability lies on boundary."));
            return;
        };
        self.clv_audit = Some(Ok((p_close.value(), clv)));
    }

    fn reset_cached_outputs(&mut self) {
        self.growth_curve_points.clear();
        self.quarter_marker.clear();
        self.half_marker.clear();
        self.full_marker.clear();
        self.strategy_rows.clear();
        self.ev_per_unit = None;
        self.lundberg_summary = None;
        self.sample_size_95 = None;
        self.clv_audit = None;
        self.calc_error = None;
    }

    fn populate_positive_edge_metrics(
        &mut self,
        prob: &UnitInterval,
        odds: &Odds,
        fk: &BankrollFraction,
        ev: f64,
        f_domain_cap: f64,
    ) {
        let f_max = (fk.value() * 2.25).clamp(0.05_f64.min(f_domain_cap), f_domain_cap);
        self.populate_growth_curve(prob, odds, f_max);
        if let Ok(qk) = variants::quarter_kelly(prob, odds) {
            self.push_marker_and_row("Quarter Kelly", prob, odds, &qk, ev, 0);
        }
        if let Ok(hk) = variants::half_kelly(prob, odds) {
            self.push_marker_and_row("Half Kelly", prob, odds, &hk, ev, 1);
        }
        self.push_marker_and_row("Full Kelly", prob, odds, fk, ev, 2);
        let unit_stake = (0.01 * self.bankroll).max(1e-6);
        self.lundberg_summary =
            solve_single_bet_lundberg(prob, odds, unit_stake, self.bankroll).ok();
        let odds_arr = [*odds];
        let prob_arr = [*prob];
        self.sample_size_95 =
            required_bets_for_95_significance(ev, &odds_arr, Some(&prob_arr)).ok();
    }

    pub(crate) fn recalculate(&mut self) {
        self.reset_cached_outputs();

        let prob = match UnitInterval::new(self.probability_input / 100.0) {
            Ok(p) => p,
            Err(e) => {
                self.calc_error = Some(format!("Invalid Probability: {e}"));
                return;
            }
        };
        let odds = match self.active_odds() {
            Ok(o) => o,
            Err(msg) => {
                self.calc_error = Some(msg);
                return;
            }
        };

        let ev = expected_value(&prob, &odds);
        self.ev_per_unit = Some(ev);
        self.recompute_clv_audit(&odds);
        let b = odds.net_profit_multiplier();
        let f_domain_cap = (0.98 / b).clamp(1e-4, 0.98);

        let full_kelly_opt = if ev > 0.0 {
            kelly_fraction(&prob, &odds).ok()
        } else {
            None
        };
        if let Some(fk) = full_kelly_opt {
            self.populate_positive_edge_metrics(&prob, &odds, &fk, ev, f_domain_cap);
        } else {
            let f_max = f_domain_cap.min(0.50);
            self.populate_growth_curve(&prob, &odds, f_max);
            self.strategy_rows.push(KellyStrategyRow {
                label: "No Wager (No Edge)".to_string(),
                fraction: 0.0,
                bet_amount: 0.0,
                growth_rate: 0.0,
                ev_per_unit: ev,
                barrier_bound: None,
            });
        }
    }
}

impl InteractiveTool for BetSizeCalculatorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Bet Size Calculator"
    }
    fn domain(&self) -> &'static str {
        "financial_math"
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn has_finite_parameters(&self) -> bool {
        [
            self.probability_input,
            self.decimal_odds,
            self.american_odds,
            self.fractional_numerator,
            self.fractional_denominator,
            self.bankroll,
            self.drawdown_floor_pct,
            self.closing_for_decimal,
            self.closing_against_decimal,
        ]
        .iter()
        .all(|v| v.is_finite())
    }

    fn save_state(&self) -> Option<String> {
        self.save_state_json()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.load_state_json(state)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.render_workbench_ui(ui);
        });
    }
}

// [cite:modular_polynomials_review]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BetSizeCalculatorTool",
        display_name: "Bet Size Calculator",
        domain: "financial_math",
        tags: &[],
        build: || Box::new(BetSizeCalculatorTool::default()),
    }
}

#[cfg(test)]
#[path = "bet_size/tests.rs"]
mod tests;
