//! Kelly Criterion Bet Size Calculator with synchronized four-format odds conversions.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use math_explorer::error::KellyError;
use math_explorer::pure_math::statistics::kelly::{
    expected_value, kelly_fraction, variants, Odds, UnitInterval,
};

#[path = "bet_size_ui.rs"]
mod ui;

const PROBABILITY_MIN: f64 = 0.0;
const PROBABILITY_MAX: f64 = 100.0;
const DECIMAL_ODDS_MIN: f64 = 1.01;
const DECIMAL_ODDS_MAX: f64 = 1000.0;
const AMERICAN_ODDS_MIN: f64 = -10_000.0;
const AMERICAN_ODDS_MAX: f64 = 10_000.0;
const FRACTIONAL_MIN: f64 = 1.0;
const FRACTIONAL_MAX: f64 = 1000.0;
const FRACTIONAL_MAX_DENOMINATOR: u64 = 1000;
const IMPLIED_PROB_MIN: f64 = 0.1;
const IMPLIED_PROB_MAX: f64 = 99.0;
const BANKROLL_MIN: f64 = 1.0;
const BANKROLL_MAX: f64 = 1_000_000.0;

#[derive(Debug, PartialEq, Eq, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(crate) enum OddsFormat {
    Decimal,
    American,
    Fractional,
    ImpliedProbability,
}

/// Jumps `american_odds` across the invalid `(-100.0, 100.0)` moneyline dead zone during scrubs.
pub(crate) fn skip_american_dead_zone(prev: f64, next: f64) -> f64 {
    let clamped = next.clamp(AMERICAN_ODDS_MIN, AMERICAN_ODDS_MAX);
    if clamped.abs() >= 100.0 {
        clamped
    } else if prev <= -100.0 || (prev < 100.0 && clamped >= 0.0) {
        100.0
    } else {
        -100.0
    }
}

fn resolve_odds_for(
    format: OddsFormat,
    dec: f64,
    am: f64,
    num: f64,
    den: f64,
    imp_pct: f64,
) -> Result<Odds, KellyError> {
    match format {
        OddsFormat::Decimal => Odds::new(dec),
        OddsFormat::American => Odds::from_american(am),
        OddsFormat::Fractional => Odds::from_fractional(num, den),
        OddsFormat::ImpliedProbability => Odds::from_implied_probability(imp_pct / 100.0),
    }
}

/// Interactive Kelly Criterion bet-sizing calculator supporting synchronized odds formats.
pub struct BetSizeCalculatorTool {
    pub(crate) probability_input: f64,
    pub(crate) odds_format: OddsFormat,
    pub(crate) decimal_odds: f64,
    pub(crate) american_odds: f64,
    pub(crate) fractional_numerator: f64,
    pub(crate) fractional_denominator: f64,
    pub(crate) implied_probability_input: f64,
    pub(crate) bankroll: f64,
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
            implied_probability_input: 50.0,
            bankroll: 1000.0,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    implied_probability_input: Option<f64>,
    bankroll: f64,
}

impl BetSizeState {
    fn from_tool(tool: &BetSizeCalculatorTool, include_imp: bool) -> Self {
        Self {
            probability_input: tool.probability_input,
            odds_format: tool.odds_format,
            decimal_odds: tool.decimal_odds,
            american_odds: tool.american_odds,
            fractional_numerator: tool.fractional_numerator,
            fractional_denominator: tool.fractional_denominator,
            implied_probability_input: include_imp.then_some(tool.implied_probability_input),
            bankroll: tool.bankroll,
        }
    }

    fn validate(&self) -> Result<Odds, String> {
        let imp_ok = self.implied_probability_input.is_none_or(|v| {
            v.is_finite() && (IMPLIED_PROB_MIN..=IMPLIED_PROB_MAX).contains(&v)
        });
        let finite = [
            self.probability_input,
            self.decimal_odds,
            self.american_odds,
            self.fractional_numerator,
            self.fractional_denominator,
            self.bankroll,
        ]
        .iter()
        .all(|v| v.is_finite());
        let in_bounds = (PROBABILITY_MIN..=PROBABILITY_MAX).contains(&self.probability_input)
            && (DECIMAL_ODDS_MIN..=DECIMAL_ODDS_MAX).contains(&self.decimal_odds)
            && (AMERICAN_ODDS_MIN..=AMERICAN_ODDS_MAX).contains(&self.american_odds)
            && self.american_odds.abs() >= 100.0
            && (FRACTIONAL_MIN..=FRACTIONAL_MAX).contains(&self.fractional_numerator)
            && (FRACTIONAL_MIN..=FRACTIONAL_MAX).contains(&self.fractional_denominator)
            && (BANKROLL_MIN..=BANKROLL_MAX).contains(&self.bankroll);
        if !finite || !imp_ok || !in_bounds {
            return Err("Bet size parameters out of bounds".into());
        }
        let imp = self
            .implied_probability_input
            .unwrap_or_else(|| (100.0 / self.decimal_odds).clamp(IMPLIED_PROB_MIN, IMPLIED_PROB_MAX));
        resolve_odds_for(
            self.odds_format,
            self.decimal_odds,
            self.american_odds,
            self.fractional_numerator,
            self.fractional_denominator,
            imp,
        )
        .map_err(|e| format!("Invalid odds state: {e}"))
    }
}

impl BetSizeCalculatorTool {
    pub(crate) fn resolve_active_odds(&self) -> Result<Odds, KellyError> {
        resolve_odds_for(
            self.odds_format,
            self.decimal_odds,
            self.american_odds,
            self.fractional_numerator,
            self.fractional_denominator,
            self.implied_probability_input,
        )
    }

    pub(crate) fn sync_from_odds(&mut self, odds: Odds, source: OddsFormat) {
        if source != OddsFormat::Decimal {
            self.decimal_odds = odds.value().clamp(DECIMAL_ODDS_MIN, DECIMAL_ODDS_MAX);
        }
        if source != OddsFormat::American {
            self.american_odds = skip_american_dead_zone(100.0, odds.to_american());
        }
        if source != OddsFormat::Fractional {
            let (num, den) = odds.to_fractional(FRACTIONAL_MAX_DENOMINATOR);
            self.fractional_numerator = (num as f64).clamp(FRACTIONAL_MIN, FRACTIONAL_MAX);
            self.fractional_denominator = (den as f64).clamp(FRACTIONAL_MIN, FRACTIONAL_MAX);
        }
        if source != OddsFormat::ImpliedProbability {
            self.implied_probability_input =
                (odds.implied_probability() * 100.0).clamp(IMPLIED_PROB_MIN, IMPLIED_PROB_MAX);
        }
    }

    fn apply_loaded_state(&mut self, s: BetSizeState, odds: Odds) {
        let default_imp = (odds.implied_probability() * 100.0).clamp(IMPLIED_PROB_MIN, IMPLIED_PROB_MAX);
        self.probability_input = s.probability_input;
        self.odds_format = s.odds_format;
        self.decimal_odds = s.decimal_odds;
        self.american_odds = s.american_odds;
        self.fractional_numerator = s.fractional_numerator;
        self.fractional_denominator = s.fractional_denominator;
        self.implied_probability_input = s.implied_probability_input.unwrap_or(default_imp);
        self.bankroll = s.bankroll;

        let expected_dec = odds.value().clamp(DECIMAL_ODDS_MIN, DECIMAL_ODDS_MAX);
        if (self.decimal_odds - expected_dec).abs() > 1e-6 {
            self.decimal_odds = expected_dec;
        }
        let expected_am = skip_american_dead_zone(self.american_odds, odds.to_american());
        if s.odds_format != OddsFormat::American && (self.american_odds - expected_am).abs() > 1e-6 {
            self.american_odds = expected_am;
        }
        let frac_val = (self.fractional_numerator / self.fractional_denominator) + 1.0;
        if s.odds_format != OddsFormat::Fractional && (frac_val - odds.value()).abs() > 1e-3 {
            let (num, den) = odds.to_fractional(FRACTIONAL_MAX_DENOMINATOR);
            self.fractional_numerator = (num as f64).clamp(FRACTIONAL_MIN, FRACTIONAL_MAX);
            self.fractional_denominator = (den as f64).clamp(FRACTIONAL_MIN, FRACTIONAL_MAX);
        }
        if s.odds_format != OddsFormat::ImpliedProbability
            && (self.implied_probability_input - default_imp).abs() > 1e-6
        {
            self.implied_probability_input = default_imp;
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
        ExportCapabilities::dataset_only()
    }

    fn has_finite_parameters(&self) -> bool {
        BetSizeState::from_tool(self, true).validate().is_ok()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let (Ok(prob), Ok(odds)) = (
            UnitInterval::new(self.probability_input / 100.0),
            self.resolve_active_odds(),
        ) else {
            return Vec::new();
        };
        if !self.bankroll.is_finite() || self.bankroll <= 0.0 {
            return Vec::new();
        }
        let full_f = kelly_fraction(&prob, &odds).map_or(0.0, |k| k.value());
        let half_f = variants::half_kelly(&prob, &odds).map_or(0.0, |k| k.value());
        let quarter_f = variants::quarter_kelly(&prob, &odds).map_or(0.0, |k| k.value());
        let ev = expected_value(&prob, &odds);
        if ![full_f, half_f, quarter_f, ev].iter().all(|v| v.is_finite()) {
            return Vec::new();
        }
        let b = self.bankroll;
        vec![
            ("Kelly Bet Fraction".into(), vec![[0.25, quarter_f], [0.5, half_f], [1.0, full_f]]),
            ("Kelly Bet Amount ($)".into(), vec![[0.25, quarter_f * b], [0.5, half_f * b], [1.0, full_f * b]]),
            (
                "Odds & Edge Summary".into(),
                vec![[0.0, odds.value()], [1.0, odds.net_profit_multiplier()], [2.0, odds.implied_probability()], [3.0, ev]],
            ),
        ]
    }

    fn save_state(&self) -> Option<String> {
        let include_imp = self.odds_format == OddsFormat::ImpliedProbability;
        serde_json::to_string(&BetSizeState::from_tool(self, include_imp)).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: BetSizeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bet size state: {e}"))?;
        let odds = s.validate()?;
        self.apply_loaded_state(s, odds);
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Kelly Criterion Bet Size Calculator");
            ui.label("Calculate optimal bet sizing based on your edge and odds.");
            ui.separator();
            self.render_inputs_grid(ui);
            ui.separator();
            ui.heading("Results");
            self.render_results_section(ui);
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
        let b = self.resolve_active_odds().map_or_else(
            |_| (self.decimal_odds - 1.0).max(0.01),
            |odds| odds.net_profit_multiplier(),
        );
        format!(
            "Kelly criterion optimal wagering fraction: $f^* = \\frac{{b p - q}}{{b}}$ where $p = {:.1}\\%$ is win probability, $q = 1 - p$, and net odds $b = {:.2}$, maximizing expected asymptotic logarithmic bankroll growth.",
            self.probability_input, b
        )
    }
    fn phonetic_description(&self) -> String {
        "Kelly criterion optimal bet sizing theory".into()
    }
    fn theory_citation(&self) -> String {
        "A New Interpretation of Information Rate (Kelly, 1956)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

#[cfg(test)]
#[path = "bet_size_tests.rs"]
mod tests;

const STARTER_PRESET_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Kelly Criterion Allocation","domain":"financial_math","timestamp":"2026-11-16T00:00:00Z"},"config":{"probability_input":60.0,"odds_format":"Decimal","decimal_odds":2.0,"american_odds":100.0,"fractional_numerator":1.0,"fractional_denominator":1.0,"bankroll":2500.0}}"#;

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
