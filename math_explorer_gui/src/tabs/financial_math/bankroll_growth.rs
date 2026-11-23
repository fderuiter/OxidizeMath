use crate::accessibility::AccessibleHoverText;
use crate::framework::{CalloutManager, CalloutSpec, InteractiveTool};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoint};
use math_explorer::pure_math::statistics::kelly::{
    expected_growth_rate, expected_value, kelly_fraction, variants, BankrollFraction, Odds,
    UnitInterval,
};
use rand::Rng;

const BASE_KELLY_SEED: u64 = 0x0001_D1CE_CAFE_BABE;

const fn default_stake_multiplier() -> f64 {
    1.0
}

fn is_default_stake_multiplier(val: &f64) -> bool {
    (*val - 1.0).abs() < 1e-12
}

pub struct BankrollGrowthTool {
    initial_bankroll: f64,
    probability: f64,
    odds: f64,
    num_bets: usize,
    stake_multiplier: f64,
    seed_offset: u64,
    full_kelly_points: Vec<PlotPoint>,
    half_kelly_points: Vec<PlotPoint>,
    quarter_kelly_points: Vec<PlotPoint>,
    active_stake_points: Vec<PlotPoint>,
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
            seed_offset: 0,
            full_kelly_points: Vec::new(),
            half_kelly_points: Vec::new(),
            quarter_kelly_points: Vec::new(),
            active_stake_points: Vec::new(),
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

fn step_bankroll(bankroll: &mut f64, stake_fraction: f64, net_profit_mult: f64, win: bool) {
    let bet_amount = (*bankroll * stake_fraction).max(0.0);
    if win {
        *bankroll = (*bankroll + bet_amount * net_profit_mult).max(0.0);
    } else {
        *bankroll = (*bankroll - bet_amount).max(0.0);
    }
}

impl BankrollGrowthTool {
    pub fn reset_defaults(&mut self) {
        self.initial_bankroll = 1000.0;
        self.probability = 0.55;
        self.odds = 2.0;
        self.num_bets = 100;
        self.stake_multiplier = 1.0;
        self.seed_offset = 0;
        self.recalculate();
    }

    pub fn rerun_simulation(&mut self) {
        self.seed_offset = self.seed_offset.wrapping_add(1);
        self.recalculate();
    }

    fn clear_trajectories(&mut self) {
        self.full_kelly_points.clear();
        self.half_kelly_points.clear();
        self.quarter_kelly_points.clear();
        self.active_stake_points.clear();
        self.error_msg = None;
    }

    fn simulate_paths(&mut self, p: &UnitInterval, o: &Odds, fk: f64, hk: f64, qk: f64) {
        let mut rng =
            oxidize_core::rng::OxidizeRng::new(BASE_KELLY_SEED.wrapping_add(self.seed_offset));
        let net_b = o.net_profit_multiplier();
        let active_f = (fk * self.stake_multiplier).clamp(0.0, 0.9999);
        self.kelly_f_star = fk;
        self.active_fraction = active_f;
        self.active_growth_rate = finite_kelly_growth_rate(p, o, active_f);
        self.unit_expected_value = expected_value(p, o);

        let (mut w_fk, mut w_hk, mut w_qk, mut w_act) = (
            self.initial_bankroll,
            self.initial_bankroll,
            self.initial_bankroll,
            self.initial_bankroll,
        );
        self.full_kelly_points.push(PlotPoint::new(0.0, w_fk));
        self.half_kelly_points.push(PlotPoint::new(0.0, w_hk));
        self.quarter_kelly_points.push(PlotPoint::new(0.0, w_qk));
        self.active_stake_points.push(PlotPoint::new(0.0, w_act));

        for i in 1..=self.num_bets {
            let win = rng.r#gen::<f64>() < p.value();
            step_bankroll(&mut w_fk, fk, net_b, win);
            step_bankroll(&mut w_hk, hk, net_b, win);
            step_bankroll(&mut w_qk, qk, net_b, win);
            step_bankroll(&mut w_act, active_f, net_b, win);
            let x = i as f64;
            self.full_kelly_points.push(PlotPoint::new(x, w_fk));
            self.half_kelly_points.push(PlotPoint::new(x, w_hk));
            self.quarter_kelly_points.push(PlotPoint::new(x, w_qk));
            self.active_stake_points.push(PlotPoint::new(x, w_act));
        }
    }

    fn recalculate(&mut self) {
        self.clear_trajectories();
        let p = match UnitInterval::new(self.probability) {
            Ok(val) => val,
            Err(e) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };
        let o = match Odds::new(self.odds) {
            Ok(val) => val,
            Err(e) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };
        match (
            kelly_fraction(&p, &o),
            variants::half_kelly(&p, &o),
            variants::quarter_kelly(&p, &o),
        ) {
            (Ok(fk), Ok(hk), Ok(qk)) => {
                self.simulate_paths(&p, &o, fk.value(), hk.value(), qk.value());
            }
            (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
                self.error_msg = Some(format!("Error: {e}"));
            }
        }
    }

    fn draw_parameter_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Parameters");
        let mut changed = false;
        changed |= ui
            .add(
                egui::Slider::new(&mut self.initial_bankroll, 100.0..=10000.0)
                    .text("Initial Bankroll"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.probability, 0.01..=0.99).text("Win Probability (p)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.odds, 1.05..=10.0).text("Decimal Odds (b + 1)"))
            .accessible_hover_text(
                "Decimal odds: 2.0 = even money (+1.0 net profit per unit staked).",
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.num_bets, 10..=1000).text("Number of Bets"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.stake_multiplier, 0.0..=3.0).text("Stake Multiplier (× f*)"))
            .accessible_hover_text("Multiple of Kelly optimal stake f*. Above 2.0f*, expected log-growth turns negative.")
            .changed();

        ui.horizontal_wrapped(|ui| {
            if ui.button("Rerun Simulation").clicked() {
                self.rerun_simulation();
            }
            if ui
                .button("↻ Reset to Defaults")
                .accessible_hover_text("Reset initial bankroll, win probability, odds, and bet count to default values")
                .clicked()
            {
                self.reset_defaults();
            }
            ui.small(format!(
                "f* = {:.3} | Active f = {:.3} | EV = {:+.3} | Log-Growth g = {:+.4}",
                self.kelly_f_star, self.active_fraction, self.unit_expected_value, self.active_growth_rate
            ));
        });
        if changed {
            self.recalculate();
        }
    }

    fn draw_simulation_plot(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref err) = self.error_msg {
            ui.colored_label(palette.error, err);
            return;
        }
        ui.heading("Bankroll Simulation");
        let plot = Plot::new("bankroll_growth_plot")
            .view_aspect(2.0)
            .legend(egui_plot::Legend::default());
        plot.show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Full Kelly", &self.full_kelly_points[..]).color(palette.series_color(3)),
            );
            plot_ui.line(
                Line::new("Half Kelly", &self.half_kelly_points[..]).color(palette.series_color(5)),
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
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: BankrollState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bankroll state: {e}"))?;
        if ![
            s.initial_bankroll,
            s.probability,
            s.odds,
            s.stake_multiplier,
        ]
        .iter()
        .all(|v| v.is_finite())
            || !(100.0..=10_000.0).contains(&s.initial_bankroll)
            || !(0.01..=0.99).contains(&s.probability)
            || !(0.1..=10.0).contains(&s.odds)
            || !(10..=1000).contains(&s.num_bets)
            || !(0.0..=3.0).contains(&s.stake_multiplier)
        {
            return Err("Bankroll simulation parameters out of bounds".into());
        }
        let p = UnitInterval::new(s.probability).map_err(|e| format!("Error: {e}"))?;
        let o = Odds::new(s.odds).map_err(|e| format!("Error: {e}"))?;
        kelly_fraction(&p, &o).map_err(|e| format!("Error: {e}"))?;
        variants::half_kelly(&p, &o).map_err(|e| format!("Error: {e}"))?;
        variants::quarter_kelly(&p, &o).map_err(|e| format!("Error: {e}"))?;
        self.initial_bankroll = s.initial_bankroll;
        self.probability = s.probability;
        self.odds = s.odds;
        self.num_bets = s.num_bets;
        self.stake_multiplier = s.stake_multiplier;
        self.seed_offset = 0;
        self.recalculate();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        eframe::egui::CentralPanel::default().show_inside(ui, |ui| {
            self.show_ui(ui);
        });
    }

    fn show_ui(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            self.draw_parameter_controls(ui);
            ui.separator();
            self.draw_simulation_plot(ui);
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

impl scientific_metadata::theory::TheoryDescribable for BankrollGrowthTool {
    fn theory_description(&self) -> String {
        let f_kelly = (self.probability * self.odds - 1.0) / (self.odds - 1.0);
        format!(
            "Capital compounding trajectory under Kelly criterion betting: $W_T = W_0 \\prod_{{t=1}}^T (1 + f R_t)$ with initial capital $W_0 = {:.1}$, win probability $p = {:.2}$, decimal odds $b = {:.2}$, and optimal fraction $f^* = {:.3}$.",
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
#[path = "bankroll_growth/tests.rs"]
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
