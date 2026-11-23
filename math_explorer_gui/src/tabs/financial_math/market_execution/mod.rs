//! Cross-bookmaker arbitrage, subset Dutching, spread/total middle, and live/futures hedge
//! calculator for `FinancialMathTab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod presets;
mod secondary_panels;
mod ui;

use crate::framework::{InteractiveTool, ToolMetadata};
use eframe::egui;
use std::collections::HashMap;

pub(super) const MIN_EXEC_LEGS: usize = 2;
pub(super) const MAX_EXEC_LEGS: usize = 8;

/// Active execution calculator sub-view inside [`MarketExecutionTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExecutionCalculatorMode {
    /// Cross-bookmaker surebet arbitrage (`sum(1 / O_i) < 1`).
    Arbitrage,
    /// Multi-contender equal-payout subset Dutching (Fixed Budget or Target Profit).
    Dutching,
    /// Overlapping spread or total middle break-even & EV sizing.
    Middle,
    /// Live or futures complement hedge (Equalized Profit or Target Floor).
    Hedge,
}

/// Interactive execution calculator covering Arbitrage, Dutching, Middling, and Hedging.
pub struct MarketExecutionTool {
    pub(super) budget: f64,
    pub(super) mode: ExecutionCalculatorMode,
    pub(super) arb_odds: Vec<f64>,
    pub(super) dutch_odds: Vec<f64>,
    pub(super) dutch_selected: Vec<bool>,
    pub(super) dutch_target_profit_mode: bool,
    pub(super) dutch_target_profit: f64,
    pub(super) middle_odds_a: f64,
    pub(super) middle_odds_b: f64,
    pub(super) middle_hit_prob_pct: f64,
    pub(super) hedge_orig_stake: f64,
    pub(super) hedge_orig_odds: f64,
    pub(super) hedge_live_odds: f64,
    pub(super) hedge_use_target_floor: bool,
    pub(super) hedge_target_floor: f64,
}

impl Default for MarketExecutionTool {
    fn default() -> Self {
        Self {
            budget: 1000.0,
            mode: ExecutionCalculatorMode::Arbitrage,
            arb_odds: vec![2.10, 1.980_392],
            dutch_odds: vec![2.20, 3.50, 4.20, 6.00],
            dutch_selected: vec![false, true, true, true],
            dutch_target_profit_mode: false,
            dutch_target_profit: 44.83,
            middle_odds_a: 1.909_090_9,
            middle_odds_b: 1.909_090_9,
            middle_hit_prob_pct: 9.0,
            hedge_orig_stake: 100.0,
            hedge_orig_odds: 9.0,
            hedge_live_odds: 1.666_666_7,
            hedge_use_target_floor: false,
            hedge_target_floor: 0.0,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MarketExecutionState {
    budget: f64,
    mode: ExecutionCalculatorMode,
    arb_odds: Vec<f64>,
    dutch_odds: Vec<f64>,
    dutch_selected: Vec<bool>,
    dutch_target_profit_mode: bool,
    dutch_target_profit: f64,
    middle_odds_a: f64,
    middle_odds_b: f64,
    middle_hit_prob_pct: f64,
    hedge_orig_stake: f64,
    hedge_orig_odds: f64,
    hedge_live_odds: f64,
    hedge_use_target_floor: bool,
    hedge_target_floor: f64,
}

fn validate_quotes(quotes: &[f64]) -> bool {
    (MIN_EXEC_LEGS..=MAX_EXEC_LEGS).contains(&quotes.len())
        && quotes
            .iter()
            .all(|q| q.is_finite() && (1.01..=1000.0).contains(q))
}

impl InteractiveTool for MarketExecutionTool {
    fn name(&self) -> &'static str {
        "Arbitrage, Dutching & Hedge"
    }

    fn domain(&self) -> &'static str {
        "financial_math"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn has_finite_parameters(&self) -> bool {
        [
            self.budget,
            self.dutch_target_profit,
            self.middle_odds_a,
            self.middle_odds_b,
            self.middle_hit_prob_pct,
            self.hedge_orig_stake,
            self.hedge_orig_odds,
            self.hedge_live_odds,
            self.hedge_target_floor,
        ]
        .iter()
        .all(|v| v.is_finite())
            && self.arb_odds.iter().all(|v| v.is_finite())
            && self.dutch_odds.iter().all(|v| v.is_finite())
    }

    fn clamp_parameters(&mut self) -> bool {
        self.budget = self.budget.clamp(1.0, 1_000_000.0);
        self.dutch_target_profit = self.dutch_target_profit.clamp(1.0, 1_000_000.0);
        self.middle_odds_a = self.middle_odds_a.clamp(1.01, 1000.0);
        self.middle_odds_b = self.middle_odds_b.clamp(1.01, 1000.0);
        self.middle_hit_prob_pct = self.middle_hit_prob_pct.clamp(0.0, 100.0);
        self.hedge_orig_stake = self.hedge_orig_stake.clamp(1.0, 1_000_000.0);
        self.hedge_orig_odds = self.hedge_orig_odds.clamp(1.01, 1000.0);
        self.hedge_live_odds = self.hedge_live_odds.clamp(1.01, 1000.0);
        self.hedge_target_floor = self.hedge_target_floor.clamp(-1_000_000.0, 1_000_000.0);
        for q in &mut self.arb_odds {
            *q = if q.is_finite() { q.clamp(1.01, 1000.0) } else { 2.0 };
        }
        for q in &mut self.dutch_odds {
            *q = if q.is_finite() { q.clamp(1.01, 1000.0) } else { 3.0 };
        }
        self.dutch_selected.resize(self.dutch_odds.len(), true);
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&MarketExecutionState {
            budget: self.budget,
            mode: self.mode,
            arb_odds: self.arb_odds.clone(),
            dutch_odds: self.dutch_odds.clone(),
            dutch_selected: self.dutch_selected.clone(),
            dutch_target_profit_mode: self.dutch_target_profit_mode,
            dutch_target_profit: self.dutch_target_profit,
            middle_odds_a: self.middle_odds_a,
            middle_odds_b: self.middle_odds_b,
            middle_hit_prob_pct: self.middle_hit_prob_pct,
            hedge_orig_stake: self.hedge_orig_stake,
            hedge_orig_odds: self.hedge_orig_odds,
            hedge_live_odds: self.hedge_live_odds,
            hedge_use_target_floor: self.hedge_use_target_floor,
            hedge_target_floor: self.hedge_target_floor,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: MarketExecutionState =
            serde_json::from_str(state).map_err(|e| format!("Invalid execution state: {e}"))?;
        if ![
            s.budget,
            s.dutch_target_profit,
            s.middle_odds_a,
            s.middle_odds_b,
            s.middle_hit_prob_pct,
            s.hedge_orig_stake,
            s.hedge_orig_odds,
            s.hedge_live_odds,
            s.hedge_target_floor,
        ]
        .iter()
        .all(|v| v.is_finite())
            || !(1.0..=1_000_000.0).contains(&s.budget)
            || !(1.0..=1_000_000.0).contains(&s.dutch_target_profit)
            || !(1.01..=1000.0).contains(&s.middle_odds_a)
            || !(1.01..=1000.0).contains(&s.middle_odds_b)
            || !(0.0..=100.0).contains(&s.middle_hit_prob_pct)
            || !(1.0..=1_000_000.0).contains(&s.hedge_orig_stake)
            || !(1.01..=1000.0).contains(&s.hedge_orig_odds)
            || !(1.01..=1000.0).contains(&s.hedge_live_odds)
            || !(-1_000_000.0..=1_000_000.0).contains(&s.hedge_target_floor)
            || !validate_quotes(&s.arb_odds)
            || !validate_quotes(&s.dutch_odds)
            || s.dutch_selected.len() != s.dutch_odds.len()
        {
            return Err("Market execution parameters out of bounds".into());
        }
        self.budget = s.budget;
        self.mode = s.mode;
        self.arb_odds = s.arb_odds;
        self.dutch_odds = s.dutch_odds;
        self.dutch_selected = s.dutch_selected;
        self.dutch_target_profit_mode = s.dutch_target_profit_mode;
        self.dutch_target_profit = s.dutch_target_profit;
        self.middle_odds_a = s.middle_odds_a;
        self.middle_odds_b = s.middle_odds_b;
        self.middle_hit_prob_pct = s.middle_hit_prob_pct;
        self.hedge_orig_stake = s.hedge_orig_stake;
        self.hedge_orig_odds = s.hedge_orig_odds;
        self.hedge_live_odds = s.hedge_live_odds;
        self.hedge_use_target_floor = s.hedge_use_target_floor;
        self.hedge_target_floor = s.hedge_target_floor;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        ui::render_execution_tool(self, ui);
    }
}

inventory::submit! {
    ToolMetadata {
        id: "MarketExecutionTool",
        display_name: "Arbitrage, Dutching & Hedge",
        domain: "financial_math",
        tags: &["arbitrage", "dutching", "middle", "hedge", "sports"],
        build: || Box::new(MarketExecutionTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for MarketExecutionTool {
    fn theory_description(&self) -> String {
        format!(
            "State-contingent equal-payout sizing across cross-book arbitrage ($R = \\sum 1/O_i < 1$, profit $B(1 - R)/R$), subset Dutching, spread/total middling (break-even $p_{{\\mathrm{{mid}}}} = R - 1$), and live hedging ($H^* = S_0 O_0 / O_h$) with active capital budget ${:.0}$.",
            self.budget
        )
    }

    fn phonetic_description(&self) -> String {
        "Cross-bookmaker arbitrage, subset Dutching, spread middling, and live futures hedge optimization".into()
    }

    fn theory_citation(&self) -> String {
        "Optimal Gambling Systems for Favorable Games (Thorp, 1969); Efficiency of Racetrack Betting Markets (Hausch, Ziemba & Rubinstein, 1981)".into()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "Arbitrage".into(),
            "Locks risk-free profit when cross-book reciprocal odds sum below 1.0".into(),
        );
        map.insert(
            "Dutching".into(),
            "Equalizes payout across a chosen subset of contenders".into(),
        );
        map.insert(
            "Middle".into(),
            "Balances two overlapping spread or total lines to capture double-payout middle windows"
                .into(),
        );
        map.insert(
            "Hedge".into(),
            "Sizes live or futures counter-stakes for equalized profit or downside floor protection"
                .into(),
        );
        map
    }
}

// [cite:clinical_trials_statistics]
