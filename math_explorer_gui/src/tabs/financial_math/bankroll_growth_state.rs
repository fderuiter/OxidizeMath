//! Persistence, theory metadata, and starter preset registration for [`super::BankrollGrowthTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::bankroll_ensemble::{
    BankrollLabMode, VolatilityHypothesis, DEFAULT_CUSTOM_FRACTION, MAX_CUSTOM_FRACTION,
};
use super::{is_default_stake_multiplier, BankrollGrowthTool};
use math_commons::primitives::{NonNegativeFloat, PositiveFloat};
use math_explorer::pure_math::statistics::kelly::{kelly_fraction, variants, Odds, UnitInterval};

fn default_stake_multiplier() -> NonNegativeFloat {
    NonNegativeFloat::new(1.0).expect("valid default stake multiplier")
}

fn is_default_stake_multiplier_prim(val: &NonNegativeFloat) -> bool {
    is_default_stake_multiplier(&val.value())
}

fn default_custom_fraction() -> NonNegativeFloat {
    NonNegativeFloat::new(DEFAULT_CUSTOM_FRACTION).expect("valid default custom fraction")
}

fn is_default_mode(mode: &BankrollLabMode) -> bool {
    *mode == BankrollLabMode::KellyStrategies
}

fn is_default_fraction(val: &NonNegativeFloat) -> bool {
    (val.value() - DEFAULT_CUSTOM_FRACTION).abs() < 1e-12
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BankrollState {
    initial_bankroll: PositiveFloat,
    probability: UnitInterval,
    odds: PositiveFloat,
    num_bets: usize,
    #[serde(
        default = "default_stake_multiplier",
        skip_serializing_if = "is_default_stake_multiplier_prim"
    )]
    stake_multiplier: NonNegativeFloat,
    #[serde(default, skip_serializing_if = "is_default_mode")]
    mode: BankrollLabMode,
    #[serde(
        default = "default_custom_fraction",
        skip_serializing_if = "is_default_fraction"
    )]
    custom_fraction: NonNegativeFloat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hypothesis: Option<VolatilityHypothesis>,
}

impl BankrollGrowthTool {
    pub(super) fn save_state_json(&self) -> Option<String> {
        serde_json::to_string(&BankrollState {
            initial_bankroll: PositiveFloat::new(self.initial_bankroll).ok()?,
            probability: UnitInterval::new(self.probability).ok()?,
            odds: PositiveFloat::new(self.odds).ok()?,
            num_bets: self.num_bets,
            stake_multiplier: NonNegativeFloat::new(self.stake_multiplier).ok()?,
            mode: self.mode,
            custom_fraction: NonNegativeFloat::new(self.ensemble_lab.custom_fraction).ok()?,
            hypothesis: self.ensemble_lab.hypothesis,
        })
        .ok()
    }

    pub(super) fn load_state_json(&mut self, state: &str) -> Result<(), String> {
        let s: BankrollState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bankroll state: {e}"))?;
        let initial_bankroll = s.initial_bankroll.value();
        let probability = s.probability.value();
        let odds = s.odds.value();
        let stake_multiplier = s.stake_multiplier.value();
        let custom_fraction = s.custom_fraction.value();
        if !(100.0..=10_000.0).contains(&initial_bankroll)
            || !(0.01..=0.99).contains(&probability)
            || !(0.1..=10.0).contains(&odds)
            || !(10..=1000).contains(&s.num_bets)
            || stake_multiplier > 3.0
            || custom_fraction > MAX_CUSTOM_FRACTION
        {
            return Err("Bankroll simulation parameters out of bounds".into());
        }
        let o = Odds::from_net_odds(odds).map_err(|e| format!("Error: {e}"))?;
        if s.mode == BankrollLabMode::KellyStrategies {
            kelly_fraction(&s.probability, &o).map_err(|e| format!("Error: {e}"))?;
            variants::half_kelly(&s.probability, &o).map_err(|e| format!("Error: {e}"))?;
            variants::quarter_kelly(&s.probability, &o).map_err(|e| format!("Error: {e}"))?;
        }
        self.initial_bankroll = initial_bankroll;
        self.probability = probability;
        self.odds = odds;
        self.num_bets = s.num_bets;
        self.stake_multiplier = stake_multiplier;
        self.seed_counter = 0;
        self.mode = s.mode;
        self.ensemble_lab.custom_fraction = custom_fraction.clamp(0.0, MAX_CUSTOM_FRACTION);
        self.ensemble_lab.hypothesis = s.hypothesis;
        self.recalculate();
        Ok(())
    }
}

impl scientific_metadata::theory::TheoryDescribable for BankrollGrowthTool {
    fn theory_description(&self) -> String {
        let b = self.odds.max(1e-6);
        let q = 1.0 - self.probability;
        let f_kelly = ((b * self.probability - q) / b).clamp(-10.0, 1.0);
        format!(
            "Capital compounding trajectory under Kelly criterion betting: $W_T = W_0 \\prod_{{t=1}}^T (1 + f R_t)$ with initial capital $W_0 = {:.1}$, win probability $p = {:.2}$, net odds $b = {:.2}$, and optimal fraction $f^* = {:.3}$.",
            self.initial_bankroll, self.probability, self.odds, f_kelly
        )
    }
    fn phonetic_description(&self) -> String {
        "Kelly criterion bankroll growth and geometric compounding theory".into()
    }
    fn theory_citation(&self) -> String {
        "A New Interpretation of Information Rate (Kelly, 1956); The Kelly Capital Growth Investment Criterion (MacLean et al., 2011)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Full Kelly".into(),
            "Maximizes expected logarithmic growth rate".into(),
        );
        map.insert(
            "Half Kelly".into(),
            "Reduces variance and drawdowns by half while retaining 75% of growth rate".into(),
        );
        map.insert(
            "Quarter Kelly".into(),
            "Conservative wagering for high volatility protection".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Geometric Bankroll Growth",
    "domain": "financial_math",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "initial_bankroll": 5000.0,
    "probability": 0.6,
    "odds": 2.5,
    "num_bets": 150
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "bankroll_growth_starter",
        title: "Geometric Bankroll Growth",
        description: "Log-wealth expectation maximizing capital trajectory under continuous compound yield.",
        category: "Portfolio Theory",
        icon: "💰",
        domain: "financial_math",
        tool_name: Some("BankrollGrowthTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:clinical_trials_statistics]
