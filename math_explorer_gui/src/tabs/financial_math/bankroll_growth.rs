//! Kelly Criterion Bankroll Growth and Solvency Barrier Paradox Tool.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "bankroll_growth_state.rs"]
mod state;

use super::bankroll_ensemble::{BankrollEnsembleLab, BankrollLabMode};
use crate::accessibility::AccessibleHoverText;
use crate::framework::{CalloutManager, CalloutSpec, InteractiveTool};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoint};
use math_explorer::pure_math::statistics::kelly::{
    expected_growth_rate, expected_value, kelly_fraction, variants, BankrollFraction, Odds,
    UnitInterval,
};
use math_explorer::pure_math::statistics::ruin::kelly_barrier_risk;
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

fn settle_bet(bankroll: f64, fraction: &BankrollFraction, net_b: f64, win: bool) -> f64 {
    settle_raw_fraction(bankroll, fraction.value(), net_b, win)
}

fn settle_raw_fraction(bankroll: f64, stake_fraction: f64, net_b: f64, win: bool) -> f64 {
    let bet_amount = (bankroll * stake_fraction).max(0.0);
    if win {
        (bankroll + bet_amount * net_b).max(0.0)
    } else {
        (bankroll - bet_amount).max(0.0)
    }
}

/// Interactive simulator for fractional Kelly trajectories and non-ergodic volatility drag.
pub struct BankrollGrowthTool {
    initial_bankroll: f64,
    probability: f64,
    odds: f64,
    num_bets: usize,
    stake_multiplier: f64,
    seed_counter: u64,
    mode: BankrollLabMode,
    /// Hosted non-ergodic ensemble and drawdown barrier laboratory.
    pub ensemble_lab: BankrollEnsembleLab,
    full_kelly_points: Vec<PlotPoint>,
    half_kelly_points: Vec<PlotPoint>,
    quarter_kelly_points: Vec<PlotPoint>,
    active_stake_points: Vec<PlotPoint>,
    tier_50pct_drawdown_bounds: Option<[f64; 3]>,
    kelly_f_star: f64,
    active_fraction: f64,
    active_growth_rate: f64,
    unit_expected_value: f64,
    callout_manager: CalloutManager,
    error_msg: Option<String>,
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
            tier_50pct_drawdown_bounds: None,
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
        self.tier_50pct_drawdown_bounds = None;
        self.error_msg = None;
    }

    pub(super) fn recalculate(&mut self) {
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
    }

    fn update_tier_drawdown_bounds(
        &mut self,
        p: &UnitInterval,
        o: &Odds,
        fk: &BankrollFraction,
        hk: &BankrollFraction,
        qk: &BankrollFraction,
    ) {
        if let (Ok(b_fk), Ok(b_hk), Ok(b_qk)) = (
            kelly_barrier_risk(p, o, fk, 0.50),
            kelly_barrier_risk(p, o, hk, 0.50),
            kelly_barrier_risk(p, o, qk, 0.50),
        ) {
            self.tier_50pct_drawdown_bounds = Some([
                b_fk.lundberg_barrier_bound,
                b_hk.lundberg_barrier_bound,
                b_qk.lundberg_barrier_bound,
            ]);
        }
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

        self.update_tier_drawdown_bounds(p, o, &fk, &hk, &qk);
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

    fn render_parameter_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= ui
            .add(
                egui::Slider::new(&mut self.initial_bankroll, 100.0..=10_000.0)
                    .text("Initial Bankroll"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.probability, 0.01..=0.99).text("Win Probability (p)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.odds, 0.1..=10.0).text("Net Odds (b)"))
            .accessible_hover_text("Net profit multiplier. E.g., for +100 / even money, b = 1.0.")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.num_bets, 10..=1000).text("Number of Bets"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.stake_multiplier, 0.0..=3.0)
                    .text("Stake Multiplier (× f*)"),
            )
            .accessible_hover_text(
                "Multiple of Kelly optimal stake f*. Above 2.0f*, expected log-growth turns negative.",
            )
            .changed();
        changed
    }

    fn render_action_bar(&mut self, ui: &mut egui::Ui, changed: &mut bool) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Lab Mode:");
            *changed |= ui
                .radio_value(
                    &mut self.mode,
                    BankrollLabMode::KellyStrategies,
                    "Kelly Fractions",
                )
                .changed();
            *changed |= ui
                .radio_value(
                    &mut self.mode,
                    BankrollLabMode::VolatilityDrag,
                    "Volatility-Drag & Solvency Lab",
                )
                .changed();
            ui.separator();
            if ui
                .button("🎲 Rerun Simulation")
                .accessible_hover_text(
                    "Advance deterministic RNG seed and draw a fresh sample trajectory",
                )
                .clicked()
            {
                self.rerun_simulation();
                *changed = false;
            }
            if ui
                .button("↻ Reset to Defaults")
                .accessible_hover_text(
                    "Reset initial bankroll, win probability, odds, and bet count to default values",
                )
                .clicked()
            {
                self.reset_defaults();
                *changed = false;
            }
            ui.small(format!(
                "(Seed #{}) f* = {:.3} | Active f = {:.3} | EV = {:+.3} | Log-Growth g = {:+.4}",
                self.seed_counter,
                self.kelly_f_star,
                self.active_fraction,
                self.unit_expected_value,
                self.active_growth_rate
            ));
        });
    }

    fn render_kelly_plot(&mut self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        if let Some(ref err) = self.error_msg {
            ui.colored_label(palette.error, err);
            return;
        }
        ui.heading("Bankroll Simulation");
        if let Some([fk_b, hk_b, qk_b]) = self.tier_50pct_drawdown_bounds {
            ui.label(format!(
                "Asymptotic 50%-Drawdown Barrier Bound (W ≤ 0.5 W₀): Full Kelly {:.1}% | Half Kelly {:.1}% | Quarter Kelly {:.2}%",
                fk_b * 100.0,
                hk_b * 100.0,
                qk_b * 100.0
            ));
        }
        Plot::new("bankroll_growth_plot")
            .view_aspect(2.0)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Full Kelly", &self.full_kelly_points[..])
                        .color(palette.series_color(3)),
                );
                plot_ui.line(
                    Line::new("Half Kelly", &self.half_kelly_points[..])
                        .color(palette.series_color(5)),
                );
                plot_ui.line(
                    Line::new("Quarter Kelly", &self.quarter_kelly_points[..])
                        .color(palette.series_color(2)),
                );
                if !is_default_stake_multiplier(&self.stake_multiplier) {
                    let label = format!("Active Stake ({:.2}× f*)", self.stake_multiplier);
                    plot_ui.line(
                        Line::new(label, &self.active_stake_points[..])
                            .color(palette.series_color(0))
                            .width(2.2_f32),
                    );
                }
                self.callout_manager.render_on_plot(plot_ui);
            });
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
        self.save_state_json()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.load_state_json(state)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Parameters");
                let mut changed = self.render_parameter_sliders(ui);
                self.render_action_bar(ui, &mut changed);
                if changed {
                    self.recalculate();
                }
                ui.separator();

                let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                match self.mode {
                    BankrollLabMode::KellyStrategies => self.render_kelly_plot(ui, &palette),
                    BankrollLabMode::VolatilityDrag => {
                        let lab_changed = self.ensemble_lab.show_lab_ui(
                            ui,
                            &mut self.initial_bankroll,
                            &mut self.probability,
                            &mut self.odds,
                            &mut self.num_bets,
                        );
                        if lab_changed {
                            self.recalculate();
                        }
                    }
                }
            });
        });
    }
}

// [cite:clinical_trials_statistics]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BankrollGrowthTool",
        display_name: "Bankroll Growth",
        domain: "financial_math",
        tags: &[],
        build: || Box::new(BankrollGrowthTool::default()),
    }
}

#[cfg(test)]
#[path = "bankroll_growth_tests.rs"]
mod tests;
