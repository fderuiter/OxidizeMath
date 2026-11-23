//! Kelly Criterion Capital Sizing Calculator with Logarithmic Growth Curve & Exports.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod plots;
mod state;

#[cfg(test)]
mod tests;

use crate::framework::{ExportCapabilities, InteractiveTool};
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::kelly::{
    expected_growth_rate, expected_value, kelly_fraction, BankrollFraction, Odds, UnitInterval,
};

const CURVE_SAMPLE_COUNT: usize = 161;

#[derive(PartialEq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(super) enum OddsFormat {
    Decimal,
    American,
    Fractional,
}

/// Interactive Kelly capital-sizing calculator with logarithmic growth-versus-fraction curve.
pub struct BetSizeCalculatorTool {
    pub(super) probability_input: f64,
    pub(super) odds_format: OddsFormat,
    pub(super) decimal_odds: f64,
    pub(super) american_odds: f64,
    pub(super) fractional_numerator: f64,
    pub(super) fractional_denominator: f64,
    pub(super) bankroll: f64,
    pub(super) growth_curve: Vec<PlotPoint>,
    pub(super) marker_points: Vec<PlotPoint>,
    pub(super) optimal_f_star: Option<f64>,
    pub(super) optimal_growth_rate: f64,
    pub(super) unit_ev: f64,
    pub(super) last_compute_ms: f64,
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
            growth_curve: Vec::with_capacity(CURVE_SAMPLE_COUNT),
            marker_points: Vec::with_capacity(4),
            optimal_f_star: None,
            optimal_growth_rate: 0.0,
            unit_ev: 0.0,
            last_compute_ms: 0.0,
        };
        tool.recompute_curve();
        tool
    }
}

impl BetSizeCalculatorTool {
    pub(super) fn resolve_prob_and_odds(
        &self,
    ) -> (Result<UnitInterval, String>, Result<Odds, String>) {
        let prob = UnitInterval::new(self.probability_input / 100.0).map_err(|e| e.to_string());
        let odds = match self.odds_format {
            OddsFormat::Decimal => Odds::new(self.decimal_odds),
            OddsFormat::American => Odds::from_american(self.american_odds),
            OddsFormat::Fractional => {
                Odds::from_fractional(self.fractional_numerator, self.fractional_denominator)
            }
        }
        .map_err(|e| e.to_string());
        (prob, odds)
    }

    pub(super) fn recompute_curve(&mut self) {
        let start = PlatformInstant::now();
        self.growth_curve.clear();
        self.marker_points.clear();
        self.optimal_f_star = None;
        self.optimal_growth_rate = 0.0;
        self.unit_ev = 0.0;

        let (Ok(prob), Ok(odds)) = self.resolve_prob_and_odds() else {
            self.last_compute_ms = start.elapsed().as_secs_f64() * 1000.0;
            return;
        };
        self.unit_ev = expected_value(&prob, &odds);
        let f_star = kelly_fraction(&prob, &odds).ok().map(|fk| fk.value());
        self.optimal_f_star = f_star;

        let f_max = f_star
            .map_or(0.50, |fs| (2.4 * fs).max(0.25))
            .clamp(0.15, 0.96);
        for idx in 0..CURVE_SAMPLE_COUNT {
            let f = f_max * (idx as f64) / ((CURVE_SAMPLE_COUNT - 1) as f64);
            if let Ok(bf) = BankrollFraction::new(f) {
                let g = expected_growth_rate(&prob, &odds, &bf);
                if g.is_finite() {
                    self.growth_curve.push(PlotPoint::new(f, g));
                }
            }
        }

        if let Some(fs) = f_star {
            for mult in [0.25, 0.5, 1.0, 2.0] {
                let f_m = (mult * fs).min(0.98);
                if let Ok(bf) = BankrollFraction::new(f_m) {
                    let g_m = expected_growth_rate(&prob, &odds, &bf);
                    if g_m.is_finite() {
                        if (mult - 1.0).abs() < 1e-12 {
                            self.optimal_growth_rate = g_m;
                        }
                        self.marker_points.push(PlotPoint::new(f_m, g_m));
                    }
                }
            }
        }
        self.last_compute_ms = start.elapsed().as_secs_f64() * 1000.0;
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

    fn observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("expected_value", self.unit_ev),
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
        self.render_workspace(ui);
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BetSizeCalculatorTool",
        display_name: "Bet Size Calculator",
        domain: "financial_math",
        tags: &[],
        build: || Box::new(BetSizeCalculatorTool::default()),
    }
}

// [cite:modular_polynomials_review]
