use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::pure_math::statistics::kelly::{
    expected_value, kelly_fraction, variants, UnitInterval, Odds,
};

#[derive(PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum OddsFormat {
    Decimal,
    American,
    Fractional,
}

pub struct BetSizeCalculatorTool {
    probability_input: f64,
    odds_format: OddsFormat,
    decimal_odds: f64,
    american_odds: f64,
    fractional_numerator: f64,
    fractional_denominator: f64,
    bankroll: f64,
}

impl Default for BetSizeCalculatorTool {
    fn default() -> Self {
        Self {
            probability_input: 55.0, // 55%
            odds_format: OddsFormat::Decimal,
            decimal_odds: 2.0,
            american_odds: 100.0,
            fractional_numerator: 1.0,
            fractional_denominator: 1.0,
            bankroll: 1000.0,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct BetSizeState {
    probability_input: f64,
    odds_format: OddsFormat,
    decimal_odds: f64,
    american_odds: f64,
    fractional_numerator: f64,
    fractional_denominator: f64,
    bankroll: f64,
}

impl InteractiveTool for BetSizeCalculatorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Bet Size Calculator"
    }
    fn domain(&self) -> &'static str {
        "financial_math"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&BetSizeState {
            probability_input: self.probability_input,
            odds_format: self.odds_format,
            decimal_odds: self.decimal_odds,
            american_odds: self.american_odds,
            fractional_numerator: self.fractional_numerator,
            fractional_denominator: self.fractional_denominator,
            bankroll: self.bankroll,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: BetSizeState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid BetSizeCalculatorTool state: {e}"))?;
        self.probability_input = s.probability_input;
        self.odds_format = s.odds_format;
        self.decimal_odds = s.decimal_odds;
        self.american_odds = s.american_odds;
        self.fractional_numerator = s.fractional_numerator;
        self.fractional_denominator = s.fractional_denominator;
        self.bankroll = s.bankroll;
        Ok(())
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            self.show_ui(ui);
        });
    }
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Kelly Criterion Bet Size Calculator");
        ui.label("Calculate optimal bet sizing based on your edge and odds.");
        ui.separator();

        egui::Grid::new("bet_size_inputs")
            .num_columns(2)
            .spacing([40.0, 10.0])
            .show(ui, |ui| {
                // Probability
                ui.label("Win Probability (%):");
                ui.add(
                    egui::DragValue::new(&mut self.probability_input)
                        .speed(0.1)
                        .range(0.0..=100.0)
                        .suffix("%"),
                );
                ui.end_row();

                // Odds Format Selector
                ui.label("Odds Format:");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.odds_format, OddsFormat::Decimal, "Decimal");
                    ui.radio_value(&mut self.odds_format, OddsFormat::American, "American");
                    ui.radio_value(&mut self.odds_format, OddsFormat::Fractional, "Fractional");
                });
                ui.end_row();

                // Odds Input based on format
                ui.label("Odds:");
                match self.odds_format {
                    OddsFormat::Decimal => {
                        ui.add(
                            egui::DragValue::new(&mut self.decimal_odds)
                                .speed(0.01)
                                .range(1.01..=1000.0),
                        );
                    }
                    OddsFormat::American => {
                        ui.add(
                            egui::DragValue::new(&mut self.american_odds)
                                .speed(1.0)
                                .range(-10000.0..=10000.0),
                        );
                    }
                    OddsFormat::Fractional => {
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.fractional_numerator)
                                    .speed(1.0)
                                    .range(1.0..=1000.0),
                            );
                            ui.label("/");
                            ui.add(
                                egui::DragValue::new(&mut self.fractional_denominator)
                                    .speed(1.0)
                                    .range(1.0..=1000.0),
                            );
                        });
                    }
                }
                ui.end_row();

                // Bankroll
                ui.label("Total Bankroll ($):");
                ui.add(
                    egui::DragValue::new(&mut self.bankroll)
                        .speed(10.0)
                        .range(1.0..=1_000_000.0)
                        .prefix("$"),
                );
                ui.end_row();
            });

        ui.separator();
        ui.heading("Results");

        // Compute mathematical values safely
        let prob_result = UnitInterval::new(self.probability_input / 100.0);

        let odds_result = match self.odds_format {
            OddsFormat::Decimal => Odds::new(self.decimal_odds),
            OddsFormat::American => Odds::from_american(self.american_odds),
            OddsFormat::Fractional => {
                Odds::from_fractional(self.fractional_numerator, self.fractional_denominator)
            }
        };

        match (prob_result, odds_result) {
            (Ok(prob), Ok(odds)) => {
                let ev = expected_value(&prob, &odds);
                ui.label(format!("Expected Value (per $1 staked): ${:.3}", ev));

                if ev > 0.0 {
                    ui.label(egui::RichText::new("Positive Edge!").color(egui::Color32::GREEN));

                    match kelly_fraction(&prob, &odds) {
                        Ok(full_kelly) => {
                            egui::Grid::new("kelly_results")
                                .num_columns(3)
                                .spacing([40.0, 10.0])
                                .show(ui, |ui| {
                                    ui.label(egui::RichText::new("Strategy").strong());
                                    ui.label(egui::RichText::new("Fraction of Bankroll").strong());
                                    ui.label(egui::RichText::new("Bet Amount").strong());
                                    ui.end_row();

                                    // Full Kelly
                                    ui.label("Full Kelly:");
                                    ui.label(format!("{:.2}%", full_kelly.value() * 100.0));
                                    if let Ok(amount) = full_kelly.bet_amount(self.bankroll) {
                                        ui.label(format!("${:.2}", amount));
                                    } else {
                                        ui.label("Invalid bankroll");
                                    }
                                    ui.end_row();

                                    // Half Kelly
                                    if let Ok(half_kelly) = variants::half_kelly(&prob, &odds) {
                                        ui.label("Half Kelly:");
                                        ui.label(format!("{:.2}%", half_kelly.value() * 100.0));
                                        if let Ok(amount) = half_kelly.bet_amount(self.bankroll) {
                                            ui.label(format!("${:.2}", amount));
                                        } else {
                                            ui.label("Invalid bankroll");
                                        }
                                        ui.end_row();
                                    }

                                    // Quarter Kelly
                                    if let Ok(quarter_kelly) = variants::quarter_kelly(&prob, &odds)
                                    {
                                        ui.label("Quarter Kelly:");
                                        ui.label(format!("{:.2}%", quarter_kelly.value() * 100.0));
                                        if let Ok(amount) = quarter_kelly.bet_amount(self.bankroll)
                                        {
                                            ui.label(format!("${:.2}", amount));
                                        } else {
                                            ui.label("Invalid bankroll");
                                        }
                                        ui.end_row();
                                    }
                                });
                        }
                        Err(e) => {
                            ui.label(
                                egui::RichText::new(format!("Error calculating Kelly: {}", e))
                                    .color(egui::Color32::RED),
                            );
                        }
                    }
                } else {
                    ui.label(
                        egui::RichText::new("No Edge (Negative or Zero EV). Do not bet.")
                            .color(egui::Color32::RED),
                    );
                }
            }
            (Err(e), _) => {
                ui.label(
                    egui::RichText::new(format!("Invalid Probability: {}", e))
                        .color(egui::Color32::RED),
                );
            }
            (_, Err(e)) => {
                ui.label(
                    egui::RichText::new(format!("Invalid Odds: {}", e)).color(egui::Color32::RED),
                );
            }
        }
    }
}

// [cite:modular_polynomials_review]


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
            "Kelly criterion optimal wagering fraction: $f^* = \\frac{{b p - q}}{{b}}$ where $p = {:.1}\\%$ is win probability, $q = 1 - p$, and decimal odds $b = {:.2}$, maximizing expected asymptotic logarithmic bankroll growth.",
            self.probability_input, self.decimal_odds
        )
    }
    fn phonetic_description(&self) -> String { "Kelly criterion optimal bet sizing theory".into() }
    fn theory_citation(&self) -> String { "A New Interpretation of Information Rate (Kelly, 1956)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
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
