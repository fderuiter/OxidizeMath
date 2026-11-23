//! Multi-state, mutually exclusive market, and simultaneous slate Kelly bet size calculator.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod results;
mod state;
#[cfg(test)]
mod tests;
mod ui;

use state::{
    default_market_rows, BetSizeState, DEFAULT_DRAWDOWN_ALPHA, DEFAULT_DRAWDOWN_BETA,
    DEFAULT_MARKET_COUNT, DEFAULT_PUSH_PCT, DEFAULT_QL_FULL_LOSS_PCT, DEFAULT_QL_FULL_WIN_PCT,
    DEFAULT_QL_HALF_LOSS_PCT, DEFAULT_QL_HALF_WIN_PCT, DEFAULT_QL_PUSH_PCT, DEFAULT_SLATE_COUNT,
    DEFAULT_SLATE_NET_ODDS, DEFAULT_SLATE_PUSH_PCT, DEFAULT_SLATE_WIN_PCT,
};
pub use state::{BetSizingMode, MarketRowConfig, OddsFormat, UI_MAX_MARKET_ROWS};

use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::pure_math::statistics::kelly::{
    DrawdownConstraint, MarketScratch, Odds, SlateScratch,
};

/// Interactive multi-state and portfolio Kelly bet size calculator.
pub struct BetSizeCalculatorTool {
    pub(super) probability_input: f64,
    pub(super) odds_format: OddsFormat,
    pub(super) decimal_odds: f64,
    pub(super) american_odds: f64,
    pub(super) fractional_numerator: f64,
    pub(super) fractional_denominator: f64,
    pub(super) bankroll: f64,
    pub(super) mode: BetSizingMode,
    pub(super) push_probability_pct: f64,
    pub(super) ql_full_win_pct: f64,
    pub(super) ql_half_win_pct: f64,
    pub(super) ql_push_pct: f64,
    pub(super) ql_half_loss_pct: f64,
    pub(super) ql_full_loss_pct: f64,
    pub(super) market_outcome_count: usize,
    pub(super) market_rows: [MarketRowConfig; UI_MAX_MARKET_ROWS],
    pub(super) slate_count: usize,
    pub(super) slate_win_pct: f64,
    pub(super) slate_push_pct: f64,
    pub(super) slate_net_odds: f64,
    pub(super) drawdown_alpha: f64,
    pub(super) drawdown_beta: f64,
    pub(super) market_scratch: MarketScratch,
    pub(super) slate_scratch: SlateScratch,
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
            mode: BetSizingMode::Binary,
            push_probability_pct: DEFAULT_PUSH_PCT,
            ql_full_win_pct: DEFAULT_QL_FULL_WIN_PCT,
            ql_half_win_pct: DEFAULT_QL_HALF_WIN_PCT,
            ql_push_pct: DEFAULT_QL_PUSH_PCT,
            ql_half_loss_pct: DEFAULT_QL_HALF_LOSS_PCT,
            ql_full_loss_pct: DEFAULT_QL_FULL_LOSS_PCT,
            market_outcome_count: DEFAULT_MARKET_COUNT,
            market_rows: default_market_rows(),
            slate_count: DEFAULT_SLATE_COUNT,
            slate_win_pct: DEFAULT_SLATE_WIN_PCT,
            slate_push_pct: DEFAULT_SLATE_PUSH_PCT,
            slate_net_odds: DEFAULT_SLATE_NET_ODDS,
            drawdown_alpha: DEFAULT_DRAWDOWN_ALPHA,
            drawdown_beta: DEFAULT_DRAWDOWN_BETA,
            market_scratch: MarketScratch::new(),
            slate_scratch: SlateScratch::new(),
        }
    }
}

impl BetSizeCalculatorTool {
    /// Configures the canonical `-3` NFL spread at `-110` (`50%` win, `10%` push, `40%` loss).
    pub fn load_push_spread_preset(&mut self) {
        self.mode = BetSizingMode::PushSpread;
        self.probability_input = 50.0;
        self.push_probability_pct = 10.0;
        self.odds_format = OddsFormat::American;
        self.american_odds = -110.0;
    }

    /// Configures the canonical `-2.75` Asian quarter-line at `-110` (`44%` FW, `12%` HW, `44%` FL).
    pub fn load_quarter_line_preset(&mut self) {
        self.mode = BetSizingMode::QuarterLine;
        self.ql_full_win_pct = 44.0;
        self.ql_half_win_pct = 12.0;
        self.ql_push_pct = 0.0;
        self.ql_half_loss_pct = 0.0;
        self.ql_full_loss_pct = 44.0;
        self.odds_format = OddsFormat::American;
        self.american_odds = -110.0;
    }

    /// Configures the canonical 3-way market (`[50%, 28%, 22%]` at decimal `[2.20, 4.00, 3.20]`).
    pub fn load_three_way_market_preset(&mut self) {
        self.mode = BetSizingMode::MutuallyExclusive;
        self.market_outcome_count = 3;
        self.market_rows = default_market_rows();
    }

    /// Configures the canonical 10-game simultaneous slate (`p = 58%`, even-money `b = 1.0`).
    pub fn load_ten_game_slate_preset(&mut self) {
        self.mode = BetSizingMode::SimultaneousSlate;
        self.slate_count = 10;
        self.slate_win_pct = 58.0;
        self.slate_push_pct = 0.0;
        self.slate_net_odds = 1.0;
    }

    pub(super) fn active_odds(&self) -> Result<Odds, math_explorer::error::KellyError> {
        match self.odds_format {
            OddsFormat::Decimal => Odds::new(self.decimal_odds),
            OddsFormat::American => Odds::from_american(self.american_odds),
            OddsFormat::Fractional => {
                Odds::from_fractional(self.fractional_numerator, self.fractional_denominator)
            }
        }
    }

    pub(super) fn active_drawdown_constraint(&self) -> Option<DrawdownConstraint> {
        DrawdownConstraint::new(self.drawdown_alpha, self.drawdown_beta).ok()
    }

    fn to_serializable_state(&self) -> BetSizeState {
        BetSizeState {
            probability_input: self.probability_input,
            odds_format: self.odds_format,
            decimal_odds: self.decimal_odds,
            american_odds: self.american_odds,
            fractional_numerator: self.fractional_numerator,
            fractional_denominator: self.fractional_denominator,
            bankroll: self.bankroll,
            mode: self.mode,
            push_probability_pct: self.push_probability_pct,
            ql_full_win_pct: self.ql_full_win_pct,
            ql_half_win_pct: self.ql_half_win_pct,
            ql_push_pct: self.ql_push_pct,
            ql_half_loss_pct: self.ql_half_loss_pct,
            ql_full_loss_pct: self.ql_full_loss_pct,
            market_outcome_count: self.market_outcome_count,
            market_rows: self.market_rows,
            slate_count: self.slate_count,
            slate_win_pct: self.slate_win_pct,
            slate_push_pct: self.slate_push_pct,
            slate_net_odds: self.slate_net_odds,
            drawdown_alpha: self.drawdown_alpha,
            drawdown_beta: self.drawdown_beta,
        }
    }

    fn apply_validated_state(&mut self, s: BetSizeState) {
        self.probability_input = s.probability_input;
        self.odds_format = s.odds_format;
        self.decimal_odds = s.decimal_odds;
        self.american_odds = s.american_odds;
        self.fractional_numerator = s.fractional_numerator;
        self.fractional_denominator = s.fractional_denominator;
        self.bankroll = s.bankroll;
        self.mode = s.mode;
        self.push_probability_pct = s.push_probability_pct;
        self.ql_full_win_pct = s.ql_full_win_pct;
        self.ql_half_win_pct = s.ql_half_win_pct;
        self.ql_push_pct = s.ql_push_pct;
        self.ql_half_loss_pct = s.ql_half_loss_pct;
        self.ql_full_loss_pct = s.ql_full_loss_pct;
        self.market_outcome_count = s.market_outcome_count;
        self.market_rows = s.market_rows;
        self.slate_count = s.slate_count;
        self.slate_win_pct = s.slate_win_pct;
        self.slate_push_pct = s.slate_push_pct;
        self.slate_net_odds = s.slate_net_odds;
        self.drawdown_alpha = s.drawdown_alpha;
        self.drawdown_beta = s.drawdown_beta;
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
        serde_json::to_string(&self.to_serializable_state()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: BetSizeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bet size state: {e}"))?;
        s.validate()?;
        self.apply_validated_state(s);
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_calculator(ui);
    }
}

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
            "Multi-state convex Kelly criterion allocation maximizing expected logarithmic growth $\\mathbb{{E}}[\\ln(1 + f^\\top X)]$ subject to Busseti-Ryu-Boyd drawdown risk bounds $\\mathbb{{E}}[R^{{-\\lambda}}] \\le 1$ (win probability $p = {:.1}\\%$, decimal odds $d = {:.2}$).",
            self.probability_input, self.decimal_odds
        )
    }

    fn phonetic_description(&self) -> String {
        "Multi-state convex Kelly criterion optimal bet sizing theory".into()
    }

    fn theory_citation(&self) -> String {
        "Kelly (1956); Smoczynski & Tomkins (2010); Busseti, Ryu & Boyd (2016)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
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

// [cite:clinical_trials_statistics]
