#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::kelly::{
    expected_growth_rate, expected_value, kelly_fraction, variants, BankrollFraction, Odds,
    UnitInterval,
};
use std::sync::mpsc::Receiver;

#[path = "bet_size/export.rs"]
mod export;
#[path = "bet_size/ui.rs"]
mod ui;

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
            growth_curve_points: Vec::new(),
            quarter_marker: Vec::new(),
            half_marker: Vec::new(),
            full_marker: Vec::new(),
            zero_line_points: [PlotPoint::new(0.0, 0.0), PlotPoint::new(0.25, 0.0)],
            strategy_rows: Vec::new(),
            ev_per_unit: None,
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
                _ => self.full_marker.push(pt),
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
        self.growth_curve_points.clear();
        self.quarter_marker.clear();
        self.half_marker.clear();
        self.full_marker.clear();
        self.strategy_rows.clear();
        self.ev_per_unit = None;
        self.calc_error = None;

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
        let b = odds.net_profit_multiplier();
        let f_domain_cap = (0.98 / b).clamp(1e-4, 0.98);

        let full_kelly_opt = if ev > 0.0 {
            kelly_fraction(&prob, &odds).ok()
        } else {
            None
        };
        if let Some(fk) = full_kelly_opt {
            let f_max = (fk.value() * 2.25).clamp(0.05_f64.min(f_domain_cap), f_domain_cap);
            self.populate_growth_curve(&prob, &odds, f_max);
            if let Ok(qk) = variants::quarter_kelly(&prob, &odds) {
                self.push_marker_and_row("Quarter Kelly", &prob, &odds, &qk, ev, 0);
            }
            if let Ok(hk) = variants::half_kelly(&prob, &odds) {
                self.push_marker_and_row("Half Kelly", &prob, &odds, &hk, ev, 1);
            }
            self.push_marker_and_row("Full Kelly", &prob, &odds, &fk, ev, 2);
        } else {
            let f_max = f_domain_cap.min(0.50);
            self.populate_growth_curve(&prob, &odds, f_max);
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

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BetSizeState {
    probability_input: f64,
    odds_format: OddsFormat,
    decimal_odds: f64,
    american_odds: f64,
    fractional_numerator: f64,
    fractional_denominator: f64,
    bankroll: f64,
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

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&BetSizeState {
            probability_input: self.probability_input,
            odds_format: self.odds_format,
            decimal_odds: self.decimal_odds,
            american_odds: self.american_odds,
            fractional_numerator: self.fractional_numerator,
            fractional_denominator: self.fractional_denominator,
            bankroll: self.bankroll,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: BetSizeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bet size state: {e}"))?;
        if ![
            s.probability_input,
            s.decimal_odds,
            s.american_odds,
            s.fractional_numerator,
            s.fractional_denominator,
            s.bankroll,
        ]
        .iter()
        .all(|v| v.is_finite())
            || !(0.0..=100.0).contains(&s.probability_input)
            || !(1.01..=1000.0).contains(&s.decimal_odds)
            || !(-10_000.0..=10_000.0).contains(&s.american_odds)
            || !(1.0..=1000.0).contains(&s.fractional_numerator)
            || !(1.0..=1000.0).contains(&s.fractional_denominator)
            || !(1.0..=1_000_000.0).contains(&s.bankroll)
        {
            return Err("Bet size parameters out of bounds".into());
        }
        self.probability_input = s.probability_input;
        self.odds_format = s.odds_format;
        self.decimal_odds = s.decimal_odds;
        self.american_odds = s.american_odds;
        self.fractional_numerator = s.fractional_numerator;
        self.fractional_denominator = s.fractional_denominator;
        self.bankroll = s.bankroll;
        self.recalculate();
        Ok(())
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

impl scientific_metadata::theory::TheoryDescribable for BetSizeCalculatorTool {
    fn theory_description(&self) -> String {
        let net_b = self
            .active_odds()
            .map(|o| o.net_profit_multiplier())
            .unwrap_or(self.decimal_odds - 1.0);
        format!(
            "Kelly criterion optimal wagering fraction $f^* = \\frac{{b p - q}}{{b}}$ and logarithmic growth rate $g(f) = p \\ln(1 + b f) + q \\ln(1 - f)$ where $p = {:.1}\\%$, $q = 1 - p$, and net odds $b = {:.2}$.",
            self.probability_input, net_b
        )
    }
    fn phonetic_description(&self) -> String {
        "Kelly criterion optimal bet sizing and expected logarithmic growth rate theory".into()
    }
    fn theory_citation(&self) -> String {
        "A New Interpretation of Information Rate (Kelly, 1956)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Expected Growth Rate g(f)".into(),
            "Asymptotic logarithmic wealth growth per wager across stake fraction f".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Kelly Criterion Allocation",
    "domain": "financial_math",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "probability_input": 60.0,
    "odds_format": "Decimal",
    "decimal_odds": 2.0,
    "american_odds": 100.0,
    "fractional_numerator": 1.0,
    "fractional_denominator": 1.0,
    "bankroll": 2500.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "bet_size_starter",
        title: "Kelly Criterion Allocation",
        description: "f* = (bp - q)/b: Optimal leverage fraction maximizing long-term asymptotic capital growth rate.",
        category: "Risk Management",
        icon: "⚖️",
        domain: "financial_math",
        tool_name: Some("BetSizeCalculatorTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
#[path = "bet_size/tests.rs"]
mod tests;
