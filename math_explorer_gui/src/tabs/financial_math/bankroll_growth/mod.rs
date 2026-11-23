//! Kelly Criterion Multi-State Bankroll Growth and Volatility-Drag Paradox Tool.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod simulation;
mod state;
#[cfg(test)]
mod tests;
mod ui;

pub use state::BankrollWagerModel;
use state::{
    BankrollState, DEFAULT_DRAWDOWN_ALPHA, DEFAULT_DRAWDOWN_BETA, DEFAULT_HALF_LOSS_PROBABILITY,
    DEFAULT_HALF_WIN_PROBABILITY, DEFAULT_PUSH_PROBABILITY, DEFAULT_SLATE_SIZE,
    DEFAULT_STAKE_MULTIPLIER,
};

#[cfg(test)]
use math_explorer::pure_math::statistics::kelly::BankrollFraction;
#[cfg(test)]
use simulation::settle_bet;

use super::bankroll_ensemble::{BankrollEnsembleLab, BankrollLabMode, MAX_CUSTOM_FRACTION};
use crate::framework::{CalloutManager, CalloutSpec, InteractiveTool};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::kelly::{
    kelly_fraction, variants, Odds, SlateScratch, UnitInterval,
};

/// Interactive simulator for multi-state Kelly trajectories, drawdown bounds, and volatility drag.
pub struct BankrollGrowthTool {
    pub(super) initial_bankroll: f64,
    pub(super) probability: f64,
    pub(super) odds: f64,
    pub(super) num_bets: usize,
    pub(super) stake_multiplier: f64,
    pub(super) seed_counter: u64,
    pub(super) mode: BankrollLabMode,
    pub(super) wager_model: BankrollWagerModel,
    pub(super) push_probability: f64,
    pub(super) half_win_probability: f64,
    pub(super) half_loss_probability: f64,
    pub(super) slate_size: usize,
    pub(super) drawdown_alpha: f64,
    pub(super) drawdown_beta: f64,
    /// Hosted non-ergodic ensemble and volatility-drag laboratory.
    pub ensemble_lab: BankrollEnsembleLab,
    pub(super) slate_scratch: SlateScratch,
    pub(super) full_kelly_points: Vec<PlotPoint>,
    pub(super) half_kelly_points: Vec<PlotPoint>,
    pub(super) quarter_kelly_points: Vec<PlotPoint>,
    pub(super) risk_constrained_points: Vec<PlotPoint>,
    pub(super) active_stake_points: Vec<PlotPoint>,
    pub(super) kelly_f_star: f64,
    pub(super) active_fraction: f64,
    pub(super) active_growth_rate: f64,
    pub(super) unit_expected_value: f64,
    pub(super) callout_manager: CalloutManager,
    pub(super) error_msg: Option<String>,
}

impl Default for BankrollGrowthTool {
    fn default() -> Self {
        let mut tool = Self {
            initial_bankroll: 1000.0,
            probability: 0.55,
            odds: 2.0,
            num_bets: 100,
            stake_multiplier: DEFAULT_STAKE_MULTIPLIER,
            seed_counter: 0,
            mode: BankrollLabMode::KellyStrategies,
            wager_model: BankrollWagerModel::BinaryOrPush,
            push_probability: DEFAULT_PUSH_PROBABILITY,
            half_win_probability: DEFAULT_HALF_WIN_PROBABILITY,
            half_loss_probability: DEFAULT_HALF_LOSS_PROBABILITY,
            slate_size: DEFAULT_SLATE_SIZE,
            drawdown_alpha: DEFAULT_DRAWDOWN_ALPHA,
            drawdown_beta: DEFAULT_DRAWDOWN_BETA,
            ensemble_lab: BankrollEnsembleLab::new(),
            slate_scratch: SlateScratch::new(),
            full_kelly_points: Vec::with_capacity(1001),
            half_kelly_points: Vec::with_capacity(1001),
            quarter_kelly_points: Vec::with_capacity(1001),
            risk_constrained_points: Vec::with_capacity(1001),
            active_stake_points: Vec::with_capacity(1001),
            kelly_f_star: 0.0,
            active_fraction: 0.0,
            active_growth_rate: 0.0,
            unit_expected_value: 0.0,
            callout_manager: CalloutManager::new(),
            error_msg: None,
        };
        tool.recalculate();
        tool
    }
}

impl BankrollGrowthTool {
    /// Resets parameters, RNG rerun counter, and ensemble lab state to baseline defaults.
    pub fn reset_defaults(&mut self) {
        self.initial_bankroll = 1000.0;
        self.probability = 0.55;
        self.odds = 2.0;
        self.num_bets = 100;
        self.stake_multiplier = DEFAULT_STAKE_MULTIPLIER;
        self.seed_counter = 0;
        self.mode = BankrollLabMode::KellyStrategies;
        self.wager_model = BankrollWagerModel::BinaryOrPush;
        self.push_probability = DEFAULT_PUSH_PROBABILITY;
        self.half_win_probability = DEFAULT_HALF_WIN_PROBABILITY;
        self.half_loss_probability = DEFAULT_HALF_LOSS_PROBABILITY;
        self.slate_size = DEFAULT_SLATE_SIZE;
        self.drawdown_alpha = DEFAULT_DRAWDOWN_ALPHA;
        self.drawdown_beta = DEFAULT_DRAWDOWN_BETA;
        self.ensemble_lab.reset_defaults();
        self.recalculate();
    }

    /// Advances the deterministic `seed_counter` and recomputes stochastic sample paths.
    pub fn rerun_simulation(&mut self) {
        self.seed_counter = self.seed_counter.wrapping_add(1);
        self.recalculate();
    }

    fn apply_validated_state(&mut self, s: BankrollState) {
        self.initial_bankroll = s.initial_bankroll;
        self.probability = s.probability;
        self.odds = s.odds;
        self.num_bets = s.num_bets;
        self.stake_multiplier = s.stake_multiplier;
        self.seed_counter = 0;
        self.mode = s.mode;
        self.wager_model = s.wager_model;
        self.push_probability = s.push_probability;
        self.half_win_probability = s.half_win_probability;
        self.half_loss_probability = s.half_loss_probability;
        self.slate_size = s.slate_size;
        self.drawdown_alpha = s.drawdown_alpha;
        self.drawdown_beta = s.drawdown_beta;
        self.ensemble_lab.custom_fraction = s.custom_fraction.clamp(0.0, MAX_CUSTOM_FRACTION);
        self.ensemble_lab.hypothesis = s.hypothesis;
        self.recalculate();
    }
}

impl InteractiveTool for BankrollGrowthTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Bankroll Growth"
    }

    fn domain(&self) -> &'static str {
        "financial_math"
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        if self.error_msg.is_some() {
            return Vec::new();
        }
        let final_w = self
            .active_stake_points
            .last()
            .map_or(self.initial_bankroll, |pt| pt.y);
        vec![
            ("kelly_fraction", self.kelly_f_star),
            ("stake_multiplier", self.stake_multiplier),
            ("stake_fraction", self.active_fraction),
            ("expected_growth_rate", self.active_growth_rate),
            ("expected_value", self.unit_expected_value),
            ("final_bankroll", final_w),
        ]
    }

    fn sync_callouts(&mut self, callouts: &[CalloutSpec]) {
        self.callout_manager.sync_specs(callouts);
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&BankrollState {
            initial_bankroll: self.initial_bankroll,
            probability: self.probability,
            odds: self.odds,
            num_bets: self.num_bets,
            stake_multiplier: self.stake_multiplier,
            mode: self.mode,
            custom_fraction: self.ensemble_lab.custom_fraction,
            hypothesis: self.ensemble_lab.hypothesis,
            wager_model: self.wager_model,
            push_probability: self.push_probability,
            half_win_probability: self.half_win_probability,
            half_loss_probability: self.half_loss_probability,
            slate_size: self.slate_size,
            drawdown_alpha: self.drawdown_alpha,
            drawdown_beta: self.drawdown_beta,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: BankrollState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bankroll state: {e}"))?;
        s.validate_bounds()?;
        let p = UnitInterval::new(s.probability).map_err(|e| format!("Error: {e}"))?;
        let o = Odds::from_net_odds(s.odds).map_err(|e| format!("Error: {e}"))?;
        if s.mode == BankrollLabMode::KellyStrategies
            && s.wager_model == BankrollWagerModel::BinaryOrPush
            && s.push_probability <= 1e-12
        {
            kelly_fraction(&p, &o).map_err(|e| format!("Error: {e}"))?;
            variants::half_kelly(&p, &o).map_err(|e| format!("Error: {e}"))?;
            variants::quarter_kelly(&p, &o).map_err(|e| format!("Error: {e}"))?;
        }
        let check_kelly_mode = s.mode == BankrollLabMode::KellyStrategies;
        self.apply_validated_state(s);
        if check_kelly_mode {
            if let Some(ref err) = self.error_msg {
                return Err(err.clone());
            }
        }
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_workspace(ui);
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BankrollGrowthTool",
        display_name: "Bankroll Growth",
        domain: "financial_math",
        tags: &[],
        build: || Box::new(BankrollGrowthTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for BankrollGrowthTool {
    fn theory_description(&self) -> String {
        let b = self.odds.max(1e-6);
        let q = (1.0 - self.probability - self.push_probability).max(0.0);
        let denom = (b * (1.0 - self.push_probability)).max(1e-6);
        let f_kelly = ((b * self.probability - q) / denom).clamp(-10.0, 1.0);
        format!(
            "Capital compounding trajectory under multi-state and drawdown-constrained Kelly sizing: $W_T = W_0 \\prod_{{t=1}}^T (1 + f R_t)$ with initial capital $W_0 = {:.1}$, win probability $p = {:.2}$, net odds $b = {:.2}$, and optimal fraction $f^* = {:.3}$.",
            self.initial_bankroll, self.probability, self.odds, f_kelly
        )
    }

    fn phonetic_description(&self) -> String {
        "Kelly criterion bankroll growth and geometric compounding theory".into()
    }

    fn theory_citation(&self) -> String {
        "Kelly (1956); MacLean et al. (2011); Busseti, Ryu & Boyd (2016)".into()
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
        map.insert(
            "Drawdown-Constrained Kelly".into(),
            "Enforces Busseti-Ryu-Boyd martingale bound E[R^-lambda] <= 1".into(),
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
