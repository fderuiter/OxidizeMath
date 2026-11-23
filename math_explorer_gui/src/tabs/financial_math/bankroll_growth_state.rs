//! Persistence, theory metadata, and starter preset registration for [`super::BankrollGrowthTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::bankroll_ensemble::{
    BankrollLabMode, VolatilityHypothesis, DEFAULT_BARRIER_FLOOR_RATIO, DEFAULT_CUSTOM_FRACTION,
    MAX_BARRIER_FLOOR_RATIO, MAX_CUSTOM_FRACTION, MIN_BARRIER_FLOOR_RATIO,
};
use super::{is_default_stake_multiplier, BankrollGrowthTool};
use math_explorer::pure_math::statistics::kelly::{kelly_fraction, variants, Odds, UnitInterval};

const fn default_stake_multiplier() -> f64 {
    1.0
}

fn default_custom_fraction() -> f64 {
    DEFAULT_CUSTOM_FRACTION
}

fn default_barrier_floor_ratio() -> f64 {
    DEFAULT_BARRIER_FLOOR_RATIO
}

fn is_default_mode(mode: &BankrollLabMode) -> bool {
    *mode == BankrollLabMode::KellyStrategies
}

fn is_default_fraction(val: &f64) -> bool {
    (*val - DEFAULT_CUSTOM_FRACTION).abs() < 1e-12
}

fn is_default_barrier_floor_ratio(val: &f64) -> bool {
    (*val - DEFAULT_BARRIER_FLOOR_RATIO).abs() < 1e-12
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BankrollState {
    initial_bankroll: f64,
    probability: f64,
    odds: f64,
    num_bets: usize,
    #[serde(
        default = "default_stake_multiplier",
        skip_serializing_if = "is_default_stake_multiplier"
    )]
    stake_multiplier: f64,
    #[serde(default, skip_serializing_if = "is_default_mode")]
    mode: BankrollLabMode,
    #[serde(
        default = "default_custom_fraction",
        skip_serializing_if = "is_default_fraction"
    )]
    custom_fraction: f64,
    #[serde(
        default = "default_barrier_floor_ratio",
        skip_serializing_if = "is_default_barrier_floor_ratio"
    )]
    barrier_floor_ratio: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hypothesis: Option<VolatilityHypothesis>,
}

fn validate_bankroll_bounds(s: &BankrollState) -> Result<(), String> {
    if ![
        s.initial_bankroll,
        s.probability,
        s.odds,
        s.stake_multiplier,
        s.custom_fraction,
        s.barrier_floor_ratio,
    ]
    .iter()
    .all(|v| v.is_finite())
        || !(100.0..=10_000.0).contains(&s.initial_bankroll)
        || !(0.01..=0.99).contains(&s.probability)
        || !(0.1..=10.0).contains(&s.odds)
        || !(10..=1000).contains(&s.num_bets)
        || !(0.0..=3.0).contains(&s.stake_multiplier)
        || !(0.0..=MAX_CUSTOM_FRACTION).contains(&s.custom_fraction)
        || !(MIN_BARRIER_FLOOR_RATIO..=MAX_BARRIER_FLOOR_RATIO).contains(&s.barrier_floor_ratio)
    {
        return Err("Bankroll simulation parameters out of bounds".into());
    }
    Ok(())
}

impl BankrollGrowthTool {
    pub(super) fn save_state_json(&self) -> Option<String> {
        serde_json::to_string(&BankrollState {
            initial_bankroll: self.initial_bankroll,
            probability: self.probability,
            odds: self.odds,
            num_bets: self.num_bets,
            stake_multiplier: self.stake_multiplier,
            mode: self.mode,
            custom_fraction: self.ensemble_lab.custom_fraction,
            barrier_floor_ratio: self.ensemble_lab.barrier_floor_ratio,
            hypothesis: self.ensemble_lab.hypothesis,
        })
        .ok()
    }

    pub(super) fn load_state_json(&mut self, state: &str) -> Result<(), String> {
        let s: BankrollState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bankroll state: {e}"))?;
        validate_bankroll_bounds(&s)?;
        let p = UnitInterval::new(s.probability).map_err(|e| format!("Error: {e}"))?;
        let o = Odds::from_net_odds(s.odds).map_err(|e| format!("Error: {e}"))?;
        if s.mode == BankrollLabMode::KellyStrategies {
            kelly_fraction(&p, &o).map_err(|e| format!("Error: {e}"))?;
            variants::half_kelly(&p, &o).map_err(|e| format!("Error: {e}"))?;
            variants::quarter_kelly(&p, &o).map_err(|e| format!("Error: {e}"))?;
        }
        self.initial_bankroll = s.initial_bankroll;
        self.probability = s.probability;
        self.odds = s.odds;
        self.num_bets = s.num_bets;
        self.stake_multiplier = s.stake_multiplier;
        self.seed_counter = 0;
        self.mode = s.mode;
        self.ensemble_lab.custom_fraction = s.custom_fraction.clamp(0.0, MAX_CUSTOM_FRACTION);
        self.ensemble_lab.barrier_floor_ratio = s
            .barrier_floor_ratio
            .clamp(MIN_BARRIER_FLOOR_RATIO, MAX_BARRIER_FLOOR_RATIO);
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
            "Maximizes expected logarithmic growth rate (50% asymptotic probability of a 50% drawdown)".into(),
        );
        map.insert(
            "Half Kelly".into(),
            "Retains ~75% of growth rate while cutting 50%-drawdown risk to ~12.5%".into(),
        );
        map.insert(
            "Quarter Kelly".into(),
            "Conservative wagering with <2% asymptotic 50%-drawdown risk".into(),
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
