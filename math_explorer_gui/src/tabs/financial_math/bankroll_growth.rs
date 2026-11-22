#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::{announce_live_status, AccessibleHoverText};
use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoint};
use math_explorer::pure_math::statistics::kelly::{
    kelly_fraction, variants, BankrollFraction, Odds, UnitInterval,
};
use rand::Rng;

pub struct BankrollGrowthTool {
    initial_bankroll: f64,
    probability: f64,
    odds: f64,
    num_bets: usize,

    // Cached plot data
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
            odds: 1.0,
            num_bets: 100,
            full_kelly_points: vec![],
            half_kelly_points: vec![],
            quarter_kelly_points: vec![],
            error_msg: None,
        };
        tool.recalculate();
        tool
    }
}

impl BankrollGrowthTool {
    pub fn reset_defaults(&mut self) {
        self.initial_bankroll = 1000.0;
        self.probability = 0.55;
        self.odds = 1.0;
        self.num_bets = 100;
        self.recalculate();
    }

    /// Updates a bankroll after a single wager settled with net odds `b = odds.net_profit_multiplier()`.
    pub(crate) fn apply_bet_outcome(
        bankroll: f64,
        fraction: &BankrollFraction,
        odds: &Odds,
        win: bool,
    ) -> f64 {
        let bet_amount = bankroll * fraction.value();
        if win {
            bankroll + bet_amount * odds.net_profit_multiplier()
        } else {
            bankroll - bet_amount
        }
    }

    /// Evaluates one Full-Kelly wager outcome from `bankroll` using the tool's active `(probability, odds)`.
    #[cfg(test)]
    pub fn evaluate_full_kelly_bet(&self, bankroll: f64, win: bool) -> Result<f64, String> {
        let p = UnitInterval::new(self.probability).map_err(|e| format!("Error: {e}"))?;
        let o = Odds::from_net_odds(self.odds).map_err(|e| format!("Error: {e}"))?;
        let fk = kelly_fraction(&p, &o).map_err(|e| format!("Error: {e}"))?;
        Ok(Self::apply_bet_outcome(bankroll, &fk, &o, win))
    }

    fn simulate_kelly_paths(&mut self, p: &UnitInterval, o: &Odds) {
        let full_res = kelly_fraction(p, o);
        let half_res = variants::half_kelly(p, o);
        let quarter_res = variants::quarter_kelly(p, o);

        let (fk, hk, qk) = match (full_res, half_res, quarter_res) {
            (Ok(fk), Ok(hk), Ok(qk)) => (fk, hk, qk),
            (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };

        let mut rng = oxidize_core::rng::OxidizeRng::default();
        let mut fk_bankroll = self.initial_bankroll;
        let mut hk_bankroll = self.initial_bankroll;
        let mut qk_bankroll = self.initial_bankroll;

        self.full_kelly_points
            .push(PlotPoint::new(0.0, fk_bankroll));
        self.half_kelly_points
            .push(PlotPoint::new(0.0, hk_bankroll));
        self.quarter_kelly_points
            .push(PlotPoint::new(0.0, qk_bankroll));

        for i in 1..=self.num_bets {
            let win = rng.r#gen::<f64>() < p.value();
            fk_bankroll = Self::apply_bet_outcome(fk_bankroll, &fk, o, win);
            hk_bankroll = Self::apply_bet_outcome(hk_bankroll, &hk, o, win);
            qk_bankroll = Self::apply_bet_outcome(qk_bankroll, &qk, o, win);

            let step = i as f64;
            self.full_kelly_points
                .push(PlotPoint::new(step, fk_bankroll));
            self.half_kelly_points
                .push(PlotPoint::new(step, hk_bankroll));
            self.quarter_kelly_points
                .push(PlotPoint::new(step, qk_bankroll));
        }
    }

    fn recalculate(&mut self) {
        self.full_kelly_points.clear();
        self.half_kelly_points.clear();
        self.quarter_kelly_points.clear();
        self.error_msg = None;

        let prob_result = UnitInterval::new(self.probability);
        let odds_result = Odds::from_net_odds(self.odds);

        match (prob_result, odds_result) {
            (Ok(p), Ok(o)) => self.simulate_kelly_paths(&p, &o),
            (Err(e), _) => self.error_msg = Some(format!("Error: {e}")),
            (_, Err(e)) => self.error_msg = Some(format!("Error: {e}")),
        }
    }

    fn show_parameter_controls(&mut self, ui: &mut egui::Ui) -> bool {
        ui.heading("Parameters");
        let mut changed = false;

        changed |= ui
            .add(
                egui::Slider::new(&mut self.initial_bankroll, 100.0..=10000.0)
                    .text("Initial Bankroll ($)"),
            )
            .accessible_hover_text("Starting capital before the first wager")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.probability, 0.01..=0.99).text("Win Probability (p)"))
            .accessible_hover_text("Estimated true win probability for each independent bet")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.odds, 0.1..=10.0).text("Net Odds (b)"))
            .accessible_hover_text(
                "Net profit multiplier per unit staked. For even money (+100 / decimal 2.0), b = 1.0.",
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.num_bets, 10..=1000).text("Number of Bets"))
            .accessible_hover_text("Number of sequential wagers in the simulated path")
            .changed();

        ui.horizontal(|ui| {
            if ui
                .button("Rerun Simulation")
                .accessible_hover_text(
                    "Recompute Full, Half, and Quarter Kelly bankroll trajectories",
                )
                .clicked()
            {
                changed = true;
                announce_live_status(ui.ctx(), "Recomputed Kelly bankroll growth paths");
            }
            if ui
                .button("↻ Reset to Defaults")
                .accessible_hover_text(
                    "Reset initial bankroll, win probability, net odds, and bet count to defaults",
                )
                .clicked()
            {
                self.reset_defaults();
                announce_live_status(ui.ctx(), "Reset bankroll growth parameters to defaults");
            }
        });
        changed
    }

    fn show_simulation_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref err) = self.error_msg {
            ui.colored_label(palette.error, err);
            return;
        }

        ui.heading("Bankroll Simulation");
        Plot::new("bankroll_growth_plot")
            .view_aspect(2.0)
            .legend(Legend::default())
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

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BankrollState {
    initial_bankroll: f64,
    probability: f64,
    odds: f64,
    num_bets: usize,
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
    fn has_finite_parameters(&self) -> bool {
        self.initial_bankroll.is_finite() && self.probability.is_finite() && self.odds.is_finite()
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&BankrollState {
            initial_bankroll: self.initial_bankroll,
            probability: self.probability,
            odds: self.odds,
            num_bets: self.num_bets,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: BankrollState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bankroll state: {e}"))?;
        if ![s.initial_bankroll, s.probability, s.odds]
            .iter()
            .all(|v| v.is_finite())
            || !(100.0..=10_000.0).contains(&s.initial_bankroll)
            || !(0.01..=0.99).contains(&s.probability)
            || !(0.1..=10.0).contains(&s.odds)
            || !(10..=1000).contains(&s.num_bets)
        {
            return Err("Bankroll simulation parameters out of bounds".into());
        }
        let p = UnitInterval::new(s.probability).map_err(|e| format!("Error: {e}"))?;
        let o = Odds::from_net_odds(s.odds).map_err(|e| format!("Error: {e}"))?;
        kelly_fraction(&p, &o).map_err(|e| format!("Error: {e}"))?;
        variants::half_kelly(&p, &o).map_err(|e| format!("Error: {e}"))?;
        variants::quarter_kelly(&p, &o).map_err(|e| format!("Error: {e}"))?;
        self.initial_bankroll = s.initial_bankroll;
        self.probability = s.probability;
        self.odds = s.odds;
        self.num_bets = s.num_bets;
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
            if self.show_parameter_controls(ui) {
                self.recalculate();
            }
            ui.separator();
            self.show_simulation_plot(ui);
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
        let q = 1.0 - self.probability;
        let f_kelly = if self.odds > 0.0 {
            (self.odds * self.probability - q) / self.odds
        } else {
            0.0
        };
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
mod tests {
    use super::*;

    #[test]
    fn test_bankroll_growth_reset_defaults() {
        let mut tool = BankrollGrowthTool {
            initial_bankroll: 5000.0,
            probability: 0.70,
            odds: 3.0,
            num_bets: 500,
            ..Default::default()
        };
        tool.recalculate();

        tool.reset_defaults();

        assert_eq!(tool.initial_bankroll, 1000.0);
        assert_eq!(tool.probability, 0.55);
        assert_eq!(tool.odds, 1.0);
        assert_eq!(tool.num_bets, 100);
        assert_eq!(tool.full_kelly_points.len(), 101);
        assert!(tool.error_msg.is_none());
    }

    #[test]
    fn test_bankroll_growth_even_money_net_odds_payout() {
        let mut tool = BankrollGrowthTool::default();
        tool.try_load_state(
            r#"{"initial_bankroll":100.0,"probability":0.6,"odds":1.0,"num_bets":10}"#,
        )
        .expect("b = 1.0 even-money net odds must be accepted");
        assert!(tool.error_msg.is_none());
        assert_eq!(tool.full_kelly_points.len(), 11);

        // Full Kelly for p = 0.6, b = 1.0 is f* = (1.0 * 0.6 - 0.4) / 1.0 = 0.20 ($20.00 bet on $100.00).
        // One win credits bet_amount * b = $20.00 * 1.0 => $120.00 total bankroll.
        let after_win = tool
            .evaluate_full_kelly_bet(100.0, true)
            .expect("single win evaluation must succeed");
        assert!((after_win - 120.0).abs() < 1e-12);

        let after_loss = tool
            .evaluate_full_kelly_bet(100.0, false)
            .expect("single loss evaluation must succeed");
        assert!((after_loss - 80.0).abs() < 1e-12);
    }

    #[test]
    fn test_bankroll_growth_borrowed_slice_points() {
        let tool = BankrollGrowthTool::default();
        let slice: &[PlotPoint] = &tool.full_kelly_points[..];
        let plot_points: egui_plot::PlotPoints<'_> = slice.into();
        assert!(matches!(plot_points, egui_plot::PlotPoints::Borrowed(_)));
        assert_eq!(plot_points.points().len(), tool.full_kelly_points.len());
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
    "odds": 1.5,
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
