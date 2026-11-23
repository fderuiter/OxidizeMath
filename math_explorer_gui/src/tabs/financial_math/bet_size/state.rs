//! Persistence, theory metadata, and starter preset registration for [`super::BetSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    resolve_odds_for, skip_american_dead_zone, BetSizeCalculatorTool, OddsFormat,
    AMERICAN_ODDS_MAX, AMERICAN_ODDS_MIN, BANKROLL_MAX, BANKROLL_MIN, DECIMAL_ODDS_MAX,
    DECIMAL_ODDS_MIN, DEFAULT_CLOSING_DECIMAL, DEFAULT_DRAWDOWN_FLOOR_PCT, FRACTIONAL_MAX,
    FRACTIONAL_MAX_DENOMINATOR, FRACTIONAL_MIN, IMPLIED_PROB_MAX, IMPLIED_PROB_MIN,
    PROBABILITY_MAX, PROBABILITY_MIN,
};
use math_explorer::pure_math::statistics::kelly::Odds;

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
            drawdown_floor_pct: tool.drawdown_floor_pct,
            closing_line_enabled: tool.closing_line_enabled,
            closing_for_decimal: tool.closing_for_decimal,
            closing_against_decimal: tool.closing_against_decimal,
        }
    }

    fn validate(&self) -> Result<Odds, String> {
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
            self.drawdown_floor_pct,
            self.closing_for_decimal,
            self.closing_against_decimal,
        ]
        .iter()
        .all(|v| v.is_finite());
        let in_bounds = (PROBABILITY_MIN..=PROBABILITY_MAX).contains(&self.probability_input)
            && (DECIMAL_ODDS_MIN..=DECIMAL_ODDS_MAX).contains(&self.decimal_odds)
            && (AMERICAN_ODDS_MIN..=AMERICAN_ODDS_MAX).contains(&self.american_odds)
            && self.american_odds.abs() >= 100.0
            && (FRACTIONAL_MIN..=FRACTIONAL_MAX).contains(&self.fractional_numerator)
            && (FRACTIONAL_MIN..=FRACTIONAL_MAX).contains(&self.fractional_denominator)
            && (BANKROLL_MIN..=BANKROLL_MAX).contains(&self.bankroll)
            && (5.0..=95.0).contains(&self.drawdown_floor_pct)
            && (DECIMAL_ODDS_MIN..=DECIMAL_ODDS_MAX).contains(&self.closing_for_decimal)
            && (DECIMAL_ODDS_MIN..=DECIMAL_ODDS_MAX).contains(&self.closing_against_decimal);
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
    pub(super) fn has_valid_state(&self) -> bool {
        BetSizeState::from_tool(self, true).validate().is_ok()
    }

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

    fn apply_loaded_state(&mut self, s: BetSizeState, odds: Odds) {
        let default_imp =
            (odds.implied_probability() * 100.0).clamp(IMPLIED_PROB_MIN, IMPLIED_PROB_MAX);
        self.probability_input = s.probability_input;
        self.odds_format = s.odds_format;
        self.decimal_odds = s.decimal_odds;
        self.american_odds = s.american_odds;
        self.fractional_numerator = s.fractional_numerator;
        self.fractional_denominator = s.fractional_denominator;
        self.implied_probability_input = s.implied_probability_input.unwrap_or(default_imp);
        self.bankroll = s.bankroll;
        self.drawdown_floor_pct = s.drawdown_floor_pct;
        self.closing_line_enabled = s.closing_line_enabled;
        self.closing_for_decimal = s.closing_for_decimal;
        self.closing_against_decimal = s.closing_against_decimal;

        let expected_dec = odds.value().clamp(DECIMAL_ODDS_MIN, DECIMAL_ODDS_MAX);
        if (self.decimal_odds - expected_dec).abs() > 1e-6 {
            self.decimal_odds = expected_dec;
        }
        let expected_am = skip_american_dead_zone(self.american_odds, odds.to_american());
        if s.odds_format != OddsFormat::American && (self.american_odds - expected_am).abs() > 1e-6
        {
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
        self.recalculate();
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
        "A New Interpretation of Information Rate (Kelly, 1956); Beat the Bookmaker (Kaunitz et al., 2017)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Expected Growth Rate g(f)".into(),
            "Asymptotic logarithmic wealth growth per wager across stake fraction f".into(),
        );
        map.insert(
            "Drawdown Barrier & CLV".into(),
            "Exact log-return Lundberg drawdown bounds and devigged Closing Line Value edge".into(),
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

// [cite:modular_polynomials_review]
