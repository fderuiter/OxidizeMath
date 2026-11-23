//! Kelly Criterion Bet Size, Solvency Barrier, and Closing Line Value Calculator.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::pure_math::statistics::kelly::{expected_value, Odds, UnitInterval};

#[path = "bet_size_ui.rs"]
mod ui;

#[derive(Debug, PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(super) enum OddsFormat {
    Decimal,
    American,
    Fractional,
}

const DEFAULT_CLOSING_DECIMAL: f64 = 1.91;
const DEFAULT_DRAWDOWN_FLOOR_PCT: f64 = 50.0;

fn default_closing_decimal() -> f64 {
    DEFAULT_CLOSING_DECIMAL
}

fn default_drawdown_floor_pct() -> f64 {
    DEFAULT_DRAWDOWN_FLOOR_PCT
}

fn is_false(val: &bool) -> bool {
    !*val
}

fn is_default_closing(val: &f64) -> bool {
    (*val - DEFAULT_CLOSING_DECIMAL).abs() < 1e-12
}

fn is_default_drawdown_floor(val: &f64) -> bool {
    (*val - DEFAULT_DRAWDOWN_FLOOR_PCT).abs() < 1e-12
}

pub struct BetSizeCalculatorTool {
    pub(super) probability_input: f64,
    pub(super) odds_format: OddsFormat,
    pub(super) decimal_odds: f64,
    pub(super) american_odds: f64,
    pub(super) fractional_numerator: f64,
    pub(super) fractional_denominator: f64,
    pub(super) bankroll: f64,
    pub(super) drawdown_floor_pct: f64,
    pub(super) closing_line_enabled: bool,
    pub(super) closing_for_decimal: f64,
    pub(super) closing_against_decimal: f64,
}

impl Default for BetSizeCalculatorTool {
    fn default() -> Self {
        Self {
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
    #[serde(
        default = "default_drawdown_floor_pct",
        skip_serializing_if = "is_default_drawdown_floor"
    )]
    drawdown_floor_pct: f64,
    #[serde(default, skip_serializing_if = "is_false")]
    closing_line_enabled: bool,
    #[serde(
        default = "default_closing_decimal",
        skip_serializing_if = "is_default_closing"
    )]
    closing_for_decimal: f64,
    #[serde(
        default = "default_closing_decimal",
        skip_serializing_if = "is_default_closing"
    )]
    closing_against_decimal: f64,
}

impl BetSizeCalculatorTool {
    pub(super) fn current_placed_odds(&self) -> Result<Odds, math_explorer::error::KellyError> {
        match self.odds_format {
            OddsFormat::Decimal => Odds::new(self.decimal_odds),
            OddsFormat::American => Odds::from_american(self.american_odds),
            OddsFormat::Fractional => {
                Odds::from_fractional(self.fractional_numerator, self.fractional_denominator)
            }
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
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::none()
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
            drawdown_floor_pct: self.drawdown_floor_pct,
            closing_line_enabled: self.closing_line_enabled,
            closing_for_decimal: self.closing_for_decimal,
            closing_against_decimal: self.closing_against_decimal,
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
            s.drawdown_floor_pct,
            s.closing_for_decimal,
            s.closing_against_decimal,
        ]
        .iter()
        .all(|v| v.is_finite())
            || !(0.0..=100.0).contains(&s.probability_input)
            || !(1.01..=1000.0).contains(&s.decimal_odds)
            || !(-10_000.0..=10_000.0).contains(&s.american_odds)
            || !(1.0..=1000.0).contains(&s.fractional_numerator)
            || !(1.0..=1000.0).contains(&s.fractional_denominator)
            || !(1.0..=1_000_000.0).contains(&s.bankroll)
            || !(5.0..=95.0).contains(&s.drawdown_floor_pct)
            || !(1.01..=1000.0).contains(&s.closing_for_decimal)
            || !(1.01..=1000.0).contains(&s.closing_against_decimal)
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
        self.drawdown_floor_pct = s.drawdown_floor_pct;
        self.closing_line_enabled = s.closing_line_enabled;
        self.closing_for_decimal = s.closing_for_decimal;
        self.closing_against_decimal = s.closing_against_decimal;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Kelly Criterion Bet Size, Solvency & CLV Calculator");
                ui.label(
                    "Calculate optimal sizing, drawdown barrier bounds, sample sizes, and CLV.",
                );
                ui.separator();
                self.render_primary_inputs(ui);
                self.render_closing_line_inputs(ui);
                ui.separator();
                ui.heading("Results");

                let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                let prob_res = UnitInterval::new(self.probability_input / 100.0);
                let odds_res = self.current_placed_odds();
                match (prob_res, odds_res) {
                    (Ok(prob), Ok(odds)) => {
                        let ev = expected_value(&prob, &odds);
                        ui.label(format!("Expected Value (per $1 staked): ${ev:.3}"));
                        if ev > 0.0 {
                            ui.colored_label(palette.success, "Positive Edge!");
                            self.render_kelly_strategy_table(ui, &prob, &odds);
                            self.render_solvency_and_sample_size(ui, &prob, &odds, ev);
                        } else {
                            ui.colored_label(
                                palette.error,
                                "No Edge (Negative or Zero EV). Do not bet.",
                            );
                        }
                        self.render_clv_panel(ui, &odds);
                    }
                    (Err(e), _) => {
                        ui.colored_label(palette.error, format!("Invalid Probability: {e}"));
                    }
                    (_, Err(e)) => {
                        ui.colored_label(palette.error, format!("Invalid Odds: {e}"));
                    }
                }
            });
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
        format!(
            "Kelly criterion optimal wagering fraction: $f^* = \\frac{{b p - q}}{{b}}$ where $p = {:.1}\\%$ is win probability, $q = 1 - p$, and decimal odds $b = {:.2}$, maximizing expected asymptotic logarithmic bankroll growth.",
            self.probability_input, self.decimal_odds
        )
    }
    fn phonetic_description(&self) -> String {
        "Kelly criterion optimal bet sizing theory".into()
    }
    fn theory_citation(&self) -> String {
        "A New Interpretation of Information Rate (Kelly, 1956); Beat the Bookmaker (Kaunitz et al., 2017)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

#[cfg(test)]
#[path = "bet_size_tests.rs"]
mod tests;

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
