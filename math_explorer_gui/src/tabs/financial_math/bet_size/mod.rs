//! Kelly Criterion Capital Sizing Calculator with Logarithmic Growth Curve & Exports.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod export;
mod state;
mod ui;

#[cfg(test)]
mod tests;

use crate::framework::{ExportCapabilities, InteractiveTool};
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::kelly::{
    expected_growth_rate, expected_value, kelly_fraction, variants, BankrollFraction, Odds,
    UnitInterval,
};
use std::sync::mpsc::Receiver;

#[derive(Debug, PartialEq, Eq, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(crate) enum OddsFormat {
    Decimal,
    American,
    Fractional,
}

#[derive(Debug, Clone)]
pub(crate) struct KellyStrategyRow {
    pub label: String,
    pub fraction: f64,
    pub bet_amount: f64,
    pub growth_rate: f64,
    pub ev_per_unit: f64,
}

/// Interactive Kelly capital-sizing calculator with logarithmic growth-versus-fraction curve.
pub struct BetSizeCalculatorTool {
    pub(crate) probability_input: f64,
    pub(crate) odds_format: OddsFormat,
    pub(crate) decimal_odds: f64,
    pub(crate) american_odds: f64,
    pub(crate) fractional_numerator: f64,
    pub(crate) fractional_denominator: f64,
    pub(crate) bankroll: f64,

    // Cached plot & table series (recomputed only when inputs change)
    pub(crate) growth_curve_points: Vec<PlotPoint>,
    pub(crate) quarter_marker: Vec<PlotPoint>,
    pub(crate) half_marker: Vec<PlotPoint>,
    pub(crate) full_marker: Vec<PlotPoint>,
    pub(crate) zero_line_points: [PlotPoint; 2],
    pub(crate) strategy_rows: Vec<KellyStrategyRow>,
    pub(crate) ev_per_unit: Option<f64>,
    pub(crate) optimal_f_star: Option<f64>,
    pub(crate) optimal_growth_rate: f64,
    pub(crate) last_compute_ms: f64,
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
            growth_curve_points: Vec::with_capacity(121),
            quarter_marker: Vec::with_capacity(1),
            half_marker: Vec::with_capacity(1),
            full_marker: Vec::with_capacity(1),
            zero_line_points: [PlotPoint::new(0.0, 0.0), PlotPoint::new(0.25, 0.0)],
            strategy_rows: Vec::with_capacity(3),
            ev_per_unit: None,
            optimal_f_star: None,
            optimal_growth_rate: 0.0,
            last_compute_ms: 0.0,
            calc_error: None,
            save_rx: None,
            io_status: None,
        };
        tool.recalculate();
        tool
    }
}

impl BetSizeCalculatorTool {
    pub(crate) fn active_odds(&self) -> Result<Odds, String> {
        let res = match self.odds_format {
            OddsFormat::Decimal => Odds::new(self.decimal_odds),
            OddsFormat::American => Odds::from_american(self.american_odds),
            OddsFormat::Fractional => {
                Odds::from_fractional(self.fractional_numerator, self.fractional_denominator)
            }
        };
        res.map_err(|e| format!("Invalid Odds: {e}"))
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
        if g.is_finite() {
            let pt = PlotPoint::new(frac.value(), g);
            match marker_slot {
                0 => self.quarter_marker.push(pt),
                1 => self.half_marker.push(pt),
                _ => {
                    self.optimal_growth_rate = g;
                    self.full_marker.push(pt);
                }
            }
        }
        self.strategy_rows.push(KellyStrategyRow {
            label: label.to_string(),
            fraction: frac.value(),
            bet_amount,
            growth_rate: g,
            ev_per_unit: ev,
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

    pub(crate) fn recalculate(&mut self) {
        let start = PlatformInstant::now();
        self.reset_cached_series();

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
        let f_domain_cap = (0.98 / odds.net_profit_multiplier()).clamp(1e-4, 0.98);
        let fk_opt = if ev > 0.0 {
            kelly_fraction(&prob, &odds).ok()
        } else {
            None
        };
        self.populate_tiers(&prob, &odds, ev, f_domain_cap, fk_opt);
        self.last_compute_ms = start.elapsed().as_secs_f64() * 1000.0;
    }

    fn reset_cached_series(&mut self) {
        self.growth_curve_points.clear();
        self.quarter_marker.clear();
        self.half_marker.clear();
        self.full_marker.clear();
        self.strategy_rows.clear();
        self.ev_per_unit = None;
        self.optimal_f_star = None;
        self.optimal_growth_rate = 0.0;
        self.calc_error = None;
    }

    fn populate_tiers(
        &mut self,
        prob: &UnitInterval,
        odds: &Odds,
        ev: f64,
        f_domain_cap: f64,
        fk_opt: Option<BankrollFraction>,
    ) {
        if let Some(fk) = fk_opt {
            self.optimal_f_star = Some(fk.value());
            let f_max = (fk.value() * 2.25).clamp(0.05_f64.min(f_domain_cap), f_domain_cap);
            self.populate_growth_curve(prob, odds, f_max);
            if let Ok(qk) = variants::quarter_kelly(prob, odds) {
                self.push_marker_and_row("Quarter Kelly", prob, odds, &qk, ev, 0);
            }
            if let Ok(hk) = variants::half_kelly(prob, odds) {
                self.push_marker_and_row("Half Kelly", prob, odds, &hk, ev, 1);
            }
            self.push_marker_and_row("Full Kelly", prob, odds, &fk, ev, 2);
        } else {
            self.populate_growth_curve(prob, odds, f_domain_cap.min(0.50));
            self.strategy_rows.push(KellyStrategyRow {
                label: "No Wager (No Edge)".to_string(),
                fraction: 0.0,
                bet_amount: 0.0,
                growth_rate: 0.0,
                ev_per_unit: ev,
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
        ]
        .iter()
        .all(|v| v.is_finite())
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("expected_value", self.ev_per_unit.unwrap_or(0.0)),
            ("optimal_fraction", self.optimal_f_star.unwrap_or(0.0)),
            ("optimal_growth_rate", self.optimal_growth_rate),
        ]
    }

    fn save_state(&self) -> Option<String> {
        self.save_state_json()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.load_state_json(state)
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.build_export_primitives()
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.render_workbench_ui(ui);
        });
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BetSizeCalculatorTool",
        display_name: "Bet Size Calculator",
        domain: "financial_math",
        tags: &["kelly", "kelly criterion", "wagering", "odds", "optimal fraction"],
        build: || Box::new(BetSizeCalculatorTool::default()),
    }
}

// [cite:modular_polynomials_review]
