//! State persistence, theory metadata, and starter preset for [`super::BetSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    resolve_odds_for, BetSizeCalculatorTool, OddsFormat, AMERICAN_ODDS_MAX, AMERICAN_ODDS_MIN,
    BANKROLL_MAX, BANKROLL_MIN, DECIMAL_ODDS_MAX, DECIMAL_ODDS_MIN, FRACTIONAL_MAX, FRACTIONAL_MIN,
    IMPLIED_PROB_MAX, IMPLIED_PROB_MIN, PROBABILITY_MAX, PROBABILITY_MIN,
};
use math_explorer::pure_math::statistics::kelly::Odds;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BetSizeState {
    pub(super) probability_input: f64,
    pub(super) odds_format: OddsFormat,
    pub(super) decimal_odds: f64,
    pub(super) american_odds: f64,
    pub(super) fractional_numerator: f64,
    pub(super) fractional_denominator: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) implied_probability_input: Option<f64>,
    pub(super) bankroll: f64,
}

impl BetSizeState {
    pub(super) fn from_tool(tool: &BetSizeCalculatorTool, include_imp: bool) -> Self {
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

    pub(super) fn validate(&self) -> Result<Odds, String> {
        let imp_ok = self
            .implied_probability_input
            .is_none_or(|v| v.is_finite() && (IMPLIED_PROB_MIN..=IMPLIED_PROB_MAX).contains(&v));
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
        let imp = self.implied_probability_input.unwrap_or_else(|| {
            (100.0 / self.decimal_odds).clamp(IMPLIED_PROB_MIN, IMPLIED_PROB_MAX)
        });
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
    pub(super) fn save_state_json(&self) -> Option<String> {
        let include_imp = self.odds_format == OddsFormat::ImpliedProbability;
        serde_json::to_string(&BetSizeState::from_tool(self, include_imp)).ok()
    }

    pub(super) fn load_state_json(&mut self, state: &str) -> Result<(), String> {
        let s: BetSizeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bet size state: {e}"))?;
        let odds = s.validate()?;
        self.apply_loaded_state(s, odds);
        Ok(())
    }
}

impl scientific_metadata::theory::TheoryDescribable for BetSizeCalculatorTool {
    fn theory_description(&self) -> String {
        let net_b = self.resolve_active_odds().map_or_else(
            |_| (self.decimal_odds - 1.0).max(0.01),
            |o| o.net_profit_multiplier(),
        );
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

// [cite:modular_polynomials_review]
