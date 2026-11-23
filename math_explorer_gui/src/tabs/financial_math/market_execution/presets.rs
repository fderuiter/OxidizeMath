//! Starter Gallery presets for [`super::MarketExecutionTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::starter_gallery::StarterPresetMetadata;

const PRESET_DUTCHING_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "3-Way Contender Dutching",
    "domain": "financial_math",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "budget": 100.0,
    "mode": "Dutching",
    "arb_odds": [2.1, 1.980392],
    "dutch_odds": [2.2, 3.5, 4.2, 6.0],
    "dutch_selected": [false, true, true, true],
    "dutch_target_profit_mode": false,
    "dutch_target_profit": 44.83,
    "middle_odds_a": 1.9090909,
    "middle_odds_b": 1.9090909,
    "middle_hit_prob_pct": 9.0,
    "hedge_orig_stake": 100.0,
    "hedge_orig_odds": 9.0,
    "hedge_live_odds": 1.6666667,
    "hedge_use_target_floor": false,
    "hedge_target_floor": 0.0
  }
}"#;

const PRESET_MIDDLE_HEDGE_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "+800 Futures Live Hedge",
    "domain": "financial_math",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "budget": 500.0,
    "mode": "Hedge",
    "arb_odds": [2.1, 1.980392],
    "dutch_odds": [2.2, 3.5, 4.2, 6.0],
    "dutch_selected": [false, true, true, true],
    "dutch_target_profit_mode": false,
    "dutch_target_profit": 44.83,
    "middle_odds_a": 1.9090909,
    "middle_odds_b": 1.9090909,
    "middle_hit_prob_pct": 9.0,
    "hedge_orig_stake": 100.0,
    "hedge_orig_odds": 9.0,
    "hedge_live_odds": 1.6666667,
    "hedge_use_target_floor": false,
    "hedge_target_floor": 0.0
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "market_arbitrage_dutching_starter",
        title: "3-Way Contender Dutching",
        description: "Equal-payout subset Dutching on 3.50 / 4.20 / 6.00 locking +$44.83 profit on a $100 budget.",
        category: "Market Execution",
        icon: "🧮",
        domain: "financial_math",
        tool_name: Some("MarketExecutionTool"),
        preset_generator: || PRESET_DUTCHING_JSON,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "market_middle_hedge_starter",
        title: "+800 Futures Live Hedge",
        description: "Sizes a $540.00 live hedge at -150 against a $100 +800 futures ticket to lock +$260.00 across both outcomes.",
        category: "Market Execution",
        icon: "🔒",
        domain: "financial_math",
        tool_name: Some("MarketExecutionTool"),
        preset_generator: || PRESET_MIDDLE_HEDGE_JSON,
    }
}

// [cite:clinical_trials_statistics]
