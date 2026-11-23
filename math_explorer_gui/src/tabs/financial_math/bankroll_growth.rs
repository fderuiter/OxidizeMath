//! Kelly Criterion Bankroll Growth and Volatility-Drag Paradox Tool.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::bankroll_ensemble::{
    BankrollEnsembleLab, BankrollLabMode, VolatilityHypothesis, DEFAULT_CUSTOM_FRACTION,
    MAX_CUSTOM_FRACTION,
};
use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoint};
use math_explorer::pure_math::statistics::kelly::{
    kelly_fraction, variants, BankrollFraction, Odds, UnitInterval,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};
use rand::Rng;

fn default_custom_fraction() -> f64 {
    DEFAULT_CUSTOM_FRACTION
}

fn is_default_mode(mode: &BankrollLabMode) -> bool {
    *mode == BankrollLabMode::KellyStrategies
}

fn is_default_fraction(val: &f64) -> bool {
    (*val - DEFAULT_CUSTOM_FRACTION).abs() < 1e-12
}

/// Interactive simulator for fractional Kelly trajectories and non-ergodic volatility drag.
pub struct BankrollGrowthTool {
    initial_bankroll: f64,
    probability: f64,
    odds: f64,
    num_bets: usize,
    seed_counter: u64,
    mode: BankrollLabMode,
    /// Hosted non-ergodic ensemble and volatility-drag laboratory.
    pub ensemble_lab: BankrollEnsembleLab,
    full_kelly_points: Vec<PlotPoint>,
    half_kelly_points: Vec<PlotPoint>,
    quarter_kelly_points: Vec<PlotPoint>,
    error_msg: Option<String>,
}

impl Default for BankrollGrowthTool {
    fn default() -> Self {
        let mut tool = Self {
            initial_bankroll: 1000.0,
            probability: 0.55,
            odds: 2.0,
            num_bets: 100,
            seed_counter: 0,
            mode: BankrollLabMode::KellyStrategies,
            ensemble_lab: BankrollEnsembleLab::new(),
            full_kelly_points: Vec::with_capacity(1001),
            half_kelly_points: Vec::with_capacity(1001),
            quarter_kelly_points: Vec::with_capacity(1001),
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

    fn recalculate(&mut self) {
        self.full_kelly_points.clear();
        self.half_kelly_points.clear();
        self.quarter_kelly_points.clear();
        self.error_msg = None;

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

        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ self.seed_counter);
        let net_multiplier = o.net_profit_multiplier();
        let w0 = self.initial_bankroll.max(0.0);
        let (mut fk_w, mut hk_w, mut qk_w) = (w0, w0, w0);

        self.full_kelly_points.push(PlotPoint::new(0.0, fk_w));
        self.half_kelly_points.push(PlotPoint::new(0.0, hk_w));
        self.quarter_kelly_points.push(PlotPoint::new(0.0, qk_w));

        for i in 1..=self.num_bets {
            let win = rng.r#gen::<f64>() < p.value();
            fk_w = settle_bet(fk_w, &fk, net_multiplier, win);
            hk_w = settle_bet(hk_w, &hk, net_multiplier, win);
            qk_w = settle_bet(qk_w, &qk, net_multiplier, win);
            let x = i as f64;
            self.full_kelly_points.push(PlotPoint::new(x, fk_w));
            self.half_kelly_points.push(PlotPoint::new(x, hk_w));
            self.quarter_kelly_points.push(PlotPoint::new(x, qk_w));
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
                    "Volatility-Drag Paradox Lab",
                )
                .changed();
            ui.separator();
            if ui
                .button("🎲 Rerun Simulation")
                .accessible_hover_text("Advance deterministic RNG seed and draw a fresh sample trajectory")
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
            ui.label(format!("(Seed #{})", self.seed_counter));
        });
    }

    fn render_kelly_plot(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        if let Some(ref err) = self.error_msg {
            ui.colored_label(palette.error, err);
            return;
        }
        ui.heading("Bankroll Simulation");
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
            });
    }
}

fn settle_bet(bankroll: f64, fraction: &BankrollFraction, net_b: f64, win: bool) -> f64 {
    let bet_amount = bankroll * fraction.value();
    if win {
        (bankroll + bet_amount * net_b).max(0.0)
    } else {
        (bankroll - bet_amount).max(0.0)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BankrollState {
    initial_bankroll: f64,
    probability: f64,
    odds: f64,
    num_bets: usize,
    #[serde(default, skip_serializing_if = "is_default_mode")]
    mode: BankrollLabMode,
    #[serde(
        default = "default_custom_fraction",
        skip_serializing_if = "is_default_fraction"
    )]
    custom_fraction: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hypothesis: Option<VolatilityHypothesis>,
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

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&BankrollState {
            initial_bankroll: self.initial_bankroll,
            probability: self.probability,
            odds: self.odds,
            num_bets: self.num_bets,
            mode: self.mode,
            custom_fraction: self.ensemble_lab.custom_fraction,
            hypothesis: self.ensemble_lab.hypothesis,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: BankrollState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bankroll state: {e}"))?;
        if ![s.initial_bankroll, s.probability, s.odds, s.custom_fraction]
            .iter()
            .all(|v| v.is_finite())
            || !(100.0..=10_000.0).contains(&s.initial_bankroll)
            || !(0.01..=0.99).contains(&s.probability)
            || !(0.1..=10.0).contains(&s.odds)
            || !(10..=1000).contains(&s.num_bets)
            || !(0.0..=MAX_CUSTOM_FRACTION).contains(&s.custom_fraction)
        {
            return Err("Bankroll simulation parameters out of bounds".into());
        }
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
        self.mode = s.mode;
        self.ensemble_lab.custom_fraction = s.custom_fraction.clamp(0.0, MAX_CUSTOM_FRACTION);
        self.ensemble_lab.hypothesis = s.hypothesis;
        self.recalculate();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        eframe::egui::CentralPanel::default().show_inside(ui, |ui| {
            self.show_ui(ui);
        });
    }

    fn show_ui(&mut self, ui: &mut egui::Ui) {
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
    }
}

// [cite:clinical_trials_statistics]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BankrollGrowthTool",
        display_name: "Bankroll Growth",
        domain: "financial_math",
        tags: &["kelly", "kelly criterion", "compounding", "wealth", "drawdown"],
        build: || Box::new(BankrollGrowthTool::default()),
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

#[cfg(test)]
#[path = "bankroll_growth_tests.rs"]
mod tests;

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
