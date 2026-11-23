//! Starter Gallery presets for [`super::MarketDeviggingTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::starter_gallery::StarterPresetMetadata;

const PRESET_LONGSHOT_BIAS_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Shin vs Multiplicative Longshot Bias",
    "domain": "financial_math",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "bankroll": 2500.0,
    "quote_format": "American",
    "method": "Shin",
    "decimal_quotes": [1.2, 4.5],
    "target_outcome": 1,
    "offered_decimal_odds": 5.0
  }
}"#;

const PRESET_TOURNAMENT_POD_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "6-Way Pod Additive Simplex Guard",
    "domain": "financial_math",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "bankroll": 1500.0,
    "quote_format": "Decimal",
    "method": "Additive",
    "decimal_quotes": [1.9, 3.25, 6.5, 10.0, 21.0, 101.0],
    "target_outcome": 5,
    "offered_decimal_odds": 110.0
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "devig_longshot_bias_starter",
        title: "Shin vs Multiplicative Longshot Bias",
        description: "Filters phantom +5.26% Multiplicative EV on a +400 underdog against a -500 / +350 sharp market.",
        category: "Market Pricing",
        icon: "🎯",
        domain: "financial_math",
        tool_name: Some("MarketDeviggingTool"),
        preset_generator: || PRESET_LONGSHOT_BIAS_JSON,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "devig_tournament_pod_starter",
        title: "6-Way Pod Additive Simplex Guard",
        description: "Detects and clamps the -1.43% negative Additive probability on a 101.0 tournament longshot.",
        category: "Market Pricing",
        icon: "🛡️",
        domain: "financial_math",
        tool_name: Some("MarketDeviggingTool"),
        preset_generator: || PRESET_TOURNAMENT_POD_JSON,
    }
}

// [cite:clinical_trials_statistics]
