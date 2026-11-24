//! Multi-state, mutually exclusive market, and simultaneous slate Kelly bet size calculator.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod export;
mod plot;
mod results;
mod state;
#[cfg(test)]
mod tests;
mod ui;

use state::{
    default_market_rows, resolve_odds_for, BetSizeState, DECIMAL_ODDS_MAX, DECIMAL_ODDS_MIN,
    DEFAULT_DRAWDOWN_ALPHA, DEFAULT_DRAWDOWN_BETA, DEFAULT_MARKET_COUNT, DEFAULT_PUSH_PCT,
    DEFAULT_QL_FULL_LOSS_PCT, DEFAULT_QL_FULL_WIN_PCT, DEFAULT_QL_HALF_LOSS_PCT,
    DEFAULT_QL_HALF_WIN_PCT, DEFAULT_QL_PUSH_PCT, DEFAULT_SLATE_COUNT, DEFAULT_SLATE_NET_ODDS,
    DEFAULT_SLATE_PUSH_PCT, DEFAULT_SLATE_WIN_PCT, FRACTIONAL_MAX, FRACTIONAL_MAX_DENOMINATOR,
    FRACTIONAL_MIN, IMPLIED_PROB_MAX, IMPLIED_PROB_MIN,
};
pub(crate) use state::{
    skip_american_dead_zone, DEFAULT_CLOSING_DECIMAL, DEFAULT_DRAWDOWN_FLOOR_PCT,
};
pub use state::{BetSizingMode, MarketRowConfig, OddsFormat, UI_MAX_MARKET_ROWS};

use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::error::KellyError;
use math_explorer::pure_math::statistics::kelly::{
    DrawdownConstraint, MarketScratch, Odds, SlateScratch,
};
use math_explorer::pure_math::statistics::ruin::LundbergSolution;
use math_explorer::pure_math::statistics::validation::{
    devig_two_way, single_bet_clv, ClvMetrics, SampleSize95,
};
use std::sync::mpsc::Receiver;

/// Computed strategy row in the Kelly sizing table and CSV/Markdown exports.
#[derive(Debug, Clone)]
pub(crate) struct KellyStrategyRow {
    pub label: String,
    pub fraction: f64,
    pub bet_amount: f64,
    pub growth_rate: f64,
    pub ev_per_unit: f64,
    pub barrier_bound: Option<f64>,
}

/// Interactive multi-state and portfolio Kelly bet size calculator.
pub struct BetSizeCalculatorTool {
    pub(crate) probability_input: f64,
    pub(crate) odds_format: OddsFormat,
    pub(crate) decimal_odds: f64,
    pub(crate) american_odds: f64,
    pub(crate) fractional_numerator: f64,
    pub(crate) fractional_denominator: f64,
    pub(crate) implied_probability_input: f64,
    pub(crate) bankroll: f64,
    pub(crate) drawdown_floor_pct: f64,
    pub(crate) closing_line_enabled: bool,
    pub(crate) closing_for_decimal: f64,
    pub(crate) closing_against_decimal: f64,
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
    pub(crate) growth_curve_points: Vec<PlotPoint>,
    pub(crate) quarter_marker: Vec<PlotPoint>,
    pub(crate) half_marker: Vec<PlotPoint>,
    pub(crate) full_marker: Vec<PlotPoint>,
    pub(crate) drawdown_marker: Vec<PlotPoint>,
    pub(crate) zero_line_points: [PlotPoint; 2],
    pub(crate) strategy_rows: Vec<KellyStrategyRow>,
    pub(crate) ev_per_unit: Option<f64>,
    pub(crate) lundberg_summary: Option<LundbergSolution>,
    pub(crate) sample_size_95: Option<SampleSize95>,
    pub(crate) clv_audit: Option<Result<(f64, ClvMetrics), &'static str>>,
    pub(super) summary_line: Option<String>,
    pub(crate) calc_error: Option<String>,
    pub(crate) save_rx: Option<Receiver<Result<String, String>>>,
    pub(crate) io_status: Option<(String, bool)>,
}

impl Default for BetSizeCalculatorTool {
    fn default() -> Self {
        let mut tool = Self {
            probability_input: 55.0,
            odds_format: OddsFormat::Decimal,
            decimal_odds: 2.0,
            american_odds: 100.0,
            fractional_numerator: 1.0,
            fractional_denominator: 1.0,
            implied_probability_input: 50.0,
            bankroll: 1000.0,
            drawdown_floor_pct: DEFAULT_DRAWDOWN_FLOOR_PCT,
            closing_line_enabled: false,
            closing_for_decimal: DEFAULT_CLOSING_DECIMAL,
            closing_against_decimal: DEFAULT_CLOSING_DECIMAL,
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
            growth_curve_points: Vec::with_capacity(121),
            quarter_marker: Vec::with_capacity(1),
            half_marker: Vec::with_capacity(1),
            full_marker: Vec::with_capacity(1),
            drawdown_marker: Vec::with_capacity(1),
            zero_line_points: [PlotPoint::new(0.0, 0.0), PlotPoint::new(0.25, 0.0)],
            strategy_rows: Vec::with_capacity(8),
            ev_per_unit: None,
            lundberg_summary: None,
            sample_size_95: None,
            clv_audit: None,
            summary_line: None,
            calc_error: None,
            save_rx: None,
            io_status: None,
        };
        tool.recalculate();
        tool
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
        if let Ok(odds) = Odds::from_american(-110.0) {
            self.sync_from_odds(odds, OddsFormat::American);
        }
        self.recalculate();
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
        if let Ok(odds) = Odds::from_american(-110.0) {
            self.sync_from_odds(odds, OddsFormat::American);
        }
        self.recalculate();
    }

    /// Configures the canonical 3-way market (`[50%, 28%, 22%]` at decimal `[2.20, 4.00, 3.20]`).
    pub fn load_three_way_market_preset(&mut self) {
        self.mode = BetSizingMode::MutuallyExclusive;
        self.market_outcome_count = 3;
        self.market_rows = default_market_rows();
        self.recalculate();
    }

    /// Configures the canonical 10-game simultaneous slate (`p = 58%`, even-money `b = 1.0`).
    pub fn load_ten_game_slate_preset(&mut self) {
        self.mode = BetSizingMode::SimultaneousSlate;
        self.slate_count = 10;
        self.slate_win_pct = 58.0;
        self.slate_push_pct = 0.0;
        self.slate_net_odds = 1.0;
        self.recalculate();
    }

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

    pub(crate) fn active_odds(&self) -> Result<Odds, KellyError> {
        self.resolve_active_odds()
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

    pub(super) fn active_drawdown_constraint(&self) -> Option<DrawdownConstraint> {
        DrawdownConstraint::new(self.drawdown_alpha, self.drawdown_beta).ok()
    }

    pub(super) fn recompute_clv_audit(&mut self, placed_odds: &Odds) {
        if !self.closing_line_enabled {
            self.clv_audit = None;
            return;
        }
        let (Ok(c_for), Ok(c_against)) = (
            Odds::new(self.closing_for_decimal),
            Odds::new(self.closing_against_decimal),
        ) else {
            self.clv_audit = Some(Err("Invalid closing decimal odds."));
            return;
        };
        let Ok((p_close, _)) = devig_two_way(&c_for, &c_against) else {
            self.clv_audit = Some(Err("Unable to devig closing market."));
            return;
        };
        self.clv_audit = Some(
            single_bet_clv(placed_odds, &p_close)
                .map(|clv| (p_close.value(), clv))
                .map_err(|_| "Closing probability lies on boundary."),
        );
    }

    fn to_serializable_state(&self, include_imp: bool) -> BetSizeState {
        BetSizeState {
            probability_input: self.probability_input,
            odds_format: self.odds_format,
            decimal_odds: self.decimal_odds,
            american_odds: self.american_odds,
            fractional_numerator: self.fractional_numerator,
            fractional_denominator: self.fractional_denominator,
            implied_probability_input: include_imp.then_some(self.implied_probability_input),
            bankroll: self.bankroll,
            drawdown_floor_pct: self.drawdown_floor_pct,
            closing_line_enabled: self.closing_line_enabled,
            closing_for_decimal: self.closing_for_decimal,
            closing_against_decimal: self.closing_against_decimal,
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

    fn apply_validated_state(&mut self, s: BetSizeState, odds: Odds) {
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
        self.reconcile_loaded_odds(s.odds_format, odds, default_imp);
        self.recalculate();
    }

    fn reconcile_loaded_odds(&mut self, fmt: OddsFormat, odds: Odds, default_imp: f64) {
        let expected_dec = odds.value().clamp(DECIMAL_ODDS_MIN, DECIMAL_ODDS_MAX);
        if (self.decimal_odds - expected_dec).abs() > 1e-6 {
            self.decimal_odds = expected_dec;
        }
        let expected_am = skip_american_dead_zone(self.american_odds, odds.to_american());
        if fmt != OddsFormat::American && (self.american_odds - expected_am).abs() > 1e-6 {
            self.american_odds = expected_am;
        }
        let frac_val = (self.fractional_numerator / self.fractional_denominator) + 1.0;
        if fmt != OddsFormat::Fractional && (frac_val - odds.value()).abs() > 1e-3 {
            let (num, den) = odds.to_fractional(FRACTIONAL_MAX_DENOMINATOR);
            self.fractional_numerator = (num as f64).clamp(FRACTIONAL_MIN, FRACTIONAL_MAX);
            self.fractional_denominator = (den as f64).clamp(FRACTIONAL_MIN, FRACTIONAL_MAX);
        }
        if fmt != OddsFormat::ImpliedProbability
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
        ExportCapabilities::plot()
    }

    fn has_finite_parameters(&self) -> bool {
        self.to_serializable_state(true).validate().is_ok()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }

    fn save_state(&self) -> Option<String> {
        let include_imp = self.odds_format == OddsFormat::ImpliedProbability;
        serde_json::to_string(&self.to_serializable_state(include_imp)).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: BetSizeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bet size state: {e}"))?;
        let odds = s.validate()?;
        self.apply_validated_state(s, odds);
        Ok(())
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
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
        tags: &["kelly", "kelly criterion", "wagering", "odds", "optimal fraction"],
        build: || Box::new(BetSizeCalculatorTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for BetSizeCalculatorTool {
    fn theory_description(&self) -> String {
        let net_b = self.resolve_active_odds().map_or_else(
            |_| (self.decimal_odds - 1.0).max(0.01),
            |o| o.net_profit_multiplier(),
        );
        format!(
            "Multi-state convex Kelly criterion allocation maximizing expected logarithmic growth $\\mathbb{{E}}[\\ln(1 + f^\\top X)]$ subject to Busseti-Ryu-Boyd drawdown risk bounds $\\mathbb{{E}}[R^{{-\\lambda}}] \\le 1$ (win probability $p = {:.1}\\%$, net odds $b = {:.2}$).",
            self.probability_input, net_b
        )
    }

    fn phonetic_description(&self) -> String {
        "Multi-state convex Kelly criterion optimal bet sizing and expected logarithmic growth rate theory".into()
    }

    fn theory_citation(&self) -> String {
        "Kelly (1956); Smoczynski & Tomkins (2010); Busseti, Ryu & Boyd (2016); Kaunitz et al. (2017)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Expected Growth Rate g(f)".into(),
            "Asymptotic logarithmic wealth growth per wager across stake fraction f".into(),
        );
        map.insert(
            "Drawdown-Constrained Kelly".into(),
            "Busseti-Ryu-Boyd martingale bound E[R^-lambda] <= 1".into(),
        );
        map.insert(
            "Drawdown Barrier & CLV".into(),
            "Exact log-return Lundberg drawdown bounds and devigged Closing Line Value edge".into(),
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

// [cite:clinical_trials_statistics]
