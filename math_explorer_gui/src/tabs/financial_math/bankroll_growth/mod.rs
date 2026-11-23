//! Kelly Criterion Capital Growth and Volatility-Drag Paradox Tool.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod plots;
mod state;

#[cfg(test)]
mod tests;

use super::bankroll_ensemble::{BankrollEnsembleLab, BankrollLabMode};
use crate::framework::{CalloutManager, CalloutSpec, ExportCapabilities, InteractiveTool};
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::kelly::{
    expected_growth_rate, expected_value, kelly_fraction, variants, BankrollFraction, Odds,
    UnitInterval,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};
use rand::Rng;

pub(crate) fn is_default_stake_multiplier(val: &f64) -> bool {
    (*val - 1.0).abs() < 1e-12
}

fn finite_kelly_growth_rate(p: &UnitInterval, o: &Odds, stake_fraction: f64) -> f64 {
    let clamped = stake_fraction.clamp(0.0, 0.9999);
    if let Ok(bf) = BankrollFraction::new(clamped) {
        let g = expected_growth_rate(p, o, &bf);
        if g.is_finite() {
            return g;
        }
    }
    let b = o.net_profit_multiplier();
    if clamped <= 0.0 {
        return 0.0;
    }
    if clamped >= 1.0 || 1.0 + b * clamped <= 0.0 {
        return -20.0;
    }
    let fallback = p.value() * (1.0 + b * clamped).ln() + p.complement() * (1.0 - clamped).ln();
    if fallback.is_finite() {
        fallback
    } else {
        -20.0
    }
}

pub(crate) fn settle_bet(bankroll: f64, fraction: &BankrollFraction, net_b: f64, win: bool) -> f64 {
    settle_raw_fraction(bankroll, fraction.value(), net_b, win)
}

fn settle_raw_fraction(bankroll: f64, stake_fraction: f64, net_b: f64, win: bool) -> f64 {
    let allocation = (bankroll * stake_fraction).max(0.0);
    if win {
        (bankroll + allocation * net_b).max(0.0)
    } else {
        (bankroll - allocation).max(0.0)
    }
}

/// Interactive simulator for fractional Kelly trajectories and non-ergodic volatility drag.
pub struct BankrollGrowthTool {
    pub(super) initial_bankroll: f64,
    pub(super) probability: f64,
    pub(super) odds: f64,
    pub(super) num_bets: usize,
    pub(super) stake_multiplier: f64,
    pub(super) seed_counter: u64,
    pub(super) mode: BankrollLabMode,
    /// Hosted non-ergodic ensemble and percentile fan-band laboratory.
    pub ensemble_lab: BankrollEnsembleLab,
    pub(super) full_kelly_points: Vec<PlotPoint>,
    pub(super) half_kelly_points: Vec<PlotPoint>,
    pub(super) quarter_kelly_points: Vec<PlotPoint>,
    pub(super) active_stake_points: Vec<PlotPoint>,
    pub(super) kelly_f_star: f64,
    pub(super) active_fraction: f64,
    pub(super) active_growth_rate: f64,
    pub(super) unit_expected_value: f64,
    pub(super) last_compute_ms: f64,
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
            stake_multiplier: 1.0,
            seed_counter: 0,
            mode: BankrollLabMode::KellyStrategies,
            ensemble_lab: BankrollEnsembleLab::new(),
            full_kelly_points: Vec::with_capacity(1001),
            half_kelly_points: Vec::with_capacity(1001),
            quarter_kelly_points: Vec::with_capacity(1001),
            active_stake_points: Vec::with_capacity(1001),
            kelly_f_star: 0.0,
            active_fraction: 0.0,
            active_growth_rate: 0.0,
            unit_expected_value: 0.0,
            last_compute_ms: 0.0,
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
        self.stake_multiplier = 1.0;
        self.seed_counter = 0;
        self.mode = BankrollLabMode::KellyStrategies;
        self.ensemble_lab.reset_defaults();
        self.recalculate();
    }

    /// Advances the deterministic `seed_counter` and recomputes stochastic sample paths.
    pub fn rerun_simulation(&mut self) {
        self.seed_counter = self.seed_counter.wrapping_add(1);
        self.recalculate();
    }

    fn clear_trajectories(&mut self) {
        self.full_kelly_points.clear();
        self.half_kelly_points.clear();
        self.quarter_kelly_points.clear();
        self.active_stake_points.clear();
        self.error_msg = None;
    }

    pub(super) fn recalculate(&mut self) {
        let start = PlatformInstant::now();
        self.clear_trajectories();
        let p = match UnitInterval::new(self.probability) {
            Ok(p) => p,
            Err(e) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };
        let o = match Odds::from_net_odds(self.odds) {
            Ok(o) => o,
            Err(e) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };

        self.unit_expected_value = expected_value(&p, &o);
        self.ensemble_lab.evaluate_in_place(
            self.initial_bankroll,
            &p,
            &o,
            self.num_bets,
            self.seed_counter,
        );
        self.ensemble_lab.sync_callouts(self.num_bets);
        self.simulate_kelly_tiers(&p, &o);
        self.last_compute_ms = start.elapsed().as_secs_f64() * 1000.0;
    }

    fn simulate_kelly_tiers(&mut self, p: &UnitInterval, o: &Odds) {
        let tiers = (
            kelly_fraction(p, o),
            variants::half_kelly(p, o),
            variants::quarter_kelly(p, o),
        );
        let (fk, hk, qk) = match tiers {
            (Ok(fk), Ok(hk), Ok(qk)) => (fk, hk, qk),
            (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };

        let active_f = (fk.value() * self.stake_multiplier).clamp(0.0, 0.9999);
        self.kelly_f_star = fk.value();
        self.active_fraction = active_f;
        self.active_growth_rate = finite_kelly_growth_rate(p, o, active_f);

        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ self.seed_counter);
        let net_multiplier = o.net_profit_multiplier();
        let w0 = self.initial_bankroll.max(0.0);
        let (mut fk_w, mut hk_w, mut qk_w, mut act_w) = (w0, w0, w0, w0);

        self.full_kelly_points.push(PlotPoint::new(0.0, fk_w));
        self.half_kelly_points.push(PlotPoint::new(0.0, hk_w));
        self.quarter_kelly_points.push(PlotPoint::new(0.0, qk_w));
        self.active_stake_points.push(PlotPoint::new(0.0, act_w));

        for i in 1..=self.num_bets {
            let win = rng.r#gen::<f64>() < p.value();
            fk_w = settle_bet(fk_w, &fk, net_multiplier, win);
            hk_w = settle_bet(hk_w, &hk, net_multiplier, win);
            qk_w = settle_bet(qk_w, &qk, net_multiplier, win);
            act_w = settle_raw_fraction(act_w, active_f, net_multiplier, win);
            let x = i as f64;
            self.full_kelly_points.push(PlotPoint::new(x, fk_w));
            self.half_kelly_points.push(PlotPoint::new(x, hk_w));
            self.quarter_kelly_points.push(PlotPoint::new(x, qk_w));
            self.active_stake_points.push(PlotPoint::new(x, act_w));
        }
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
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
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
            ("ensemble_var_95", self.ensemble_lab.last_terminal_var_95),
            ("ensemble_cvar_95", self.ensemble_lab.last_terminal_cvar_95),
            (
                "ensemble_underwater_duration",
                self.ensemble_lab.last_underwater_duration as f64,
            ),
        ]
    }

    fn sync_callouts(&mut self, callouts: &[CalloutSpec]) {
        self.callout_manager.sync_specs(callouts);
    }

    fn save_state(&self) -> Option<String> {
        self.save_state_json()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.load_state_json(state)
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.build_export_primitives()
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

// [cite:clinical_trials_statistics]
