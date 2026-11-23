//! Persistence, theory metadata, and starter preset registration for [`super::BetSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    BetSizeCalculatorTool, OddsFormat, DEFAULT_CLOSING_DECIMAL, DEFAULT_DRAWDOWN_FLOOR_PCT,
};

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

fn validate_bet_size_state(s: &BetSizeState) -> Result<(), String> {
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
    Ok(())
}

impl BetSizeCalculatorTool {
    pub(super) fn save_state_json(&self) -> Option<String> {
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

    pub(super) fn load_state_json(&mut self, state: &str) -> Result<(), String> {
        let s: BetSizeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bet size state: {e}"))?;
        validate_bet_size_state(&s)?;
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
        self.recalculate();
        Ok(())
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
