#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::{HLine, Legend, Line, LineStyle, Plot, PlotPoints, VLine};
use math_explorer::pure_math::statistics::kelly::{
    expected_growth_rate, expected_value, kelly_fraction, variants, BankrollFraction, Odds,
    UnitInterval,
};

#[derive(PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum OddsFormat {
    Decimal,
    American,
    Fractional,
}

/// Interactive Kelly Criterion bet-sizing calculator with log-growth curve and exports.
pub struct BetSizeCalculatorTool {
    probability_input: f64,
    odds_format: OddsFormat,
    decimal_odds: f64,
    american_odds: f64,
    fractional_numerator: f64,
    fractional_denominator: f64,
    bankroll: f64,
    revision: u64,
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
            revision: 1,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BetSizeState {
    probability_input: f64,
    odds_format: OddsFormat,
    decimal_odds: f64,
    american_odds: f64,
    fractional_numerator: f64,
    fractional_denominator: f64,
    bankroll: f64,
}

impl BetSizeCalculatorTool {
    fn resolve_prob_and_odds(&self) -> Result<(UnitInterval, Odds), String> {
        let prob = UnitInterval::new(self.probability_input / 100.0).map_err(|e| e.to_string())?;
        let odds = match self.odds_format {
            OddsFormat::Decimal => Odds::new(self.decimal_odds),
            OddsFormat::American => Odds::from_american(self.american_odds),
            OddsFormat::Fractional => {
                Odds::from_fractional(self.fractional_numerator, self.fractional_denominator)
            }
        }
        .map_err(|e| e.to_string())?;
        Ok((prob, odds))
    }

    fn compute_growth_curves(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>, f64) {
        let (prob, odds) = match self.resolve_prob_and_odds() {
            Ok(pair) => pair,
            Err(_) => {
                let fallback_p = UnitInterval::new_clamped(0.55);
                let fallback_o = Odds::new(2.0).expect("valid fallback odds");
                (fallback_p, fallback_o)
            }
        };
        let ev = expected_value(&prob, &odds);
        let f_star = kelly_fraction(&prob, &odds)
            .map(|bf| bf.value())
            .unwrap_or(0.0);
        let samples = 81usize;
        let mut growth_pts = Vec::with_capacity(samples);
        let mut linear_pts = Vec::with_capacity(samples);
        for idx in 0..samples {
            let f = 0.96 * (idx as f64) / ((samples - 1) as f64);
            if let Ok(frac) = BankrollFraction::new(f) {
                let g = expected_growth_rate(&prob, &odds, &frac);
                if g.is_finite() {
                    growth_pts.push([f, g]);
                }
            }
            linear_pts.push([f, f * ev]);
        }
        (growth_pts, linear_pts, f_star)
    }

    fn render_input_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("Kelly Bet Sizing");
        ui.label("Optimal capital allocation from edge and odds.");
        ui.separator();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.probability_input, 0.0..=100.0)
                    .text("Win Prob (%)")
                    .suffix("%"),
            )
            .accessible_hover_text("Estimated true win probability in percent")
            .changed();
        ui.label("Odds Format:");
        ui.horizontal(|ui| {
            changed |= ui
                .radio_value(&mut self.odds_format, OddsFormat::Decimal, "Decimal")
                .changed();
            changed |= ui
                .radio_value(&mut self.odds_format, OddsFormat::American, "American")
                .changed();
            changed |= ui
                .radio_value(&mut self.odds_format, OddsFormat::Fractional, "Fractional")
                .changed();
        });
        changed |= self.render_odds_format_editor(ui);
        changed |= ui
            .add(
                egui::DragValue::new(&mut self.bankroll)
                    .speed(10.0)
                    .range(1.0..=1_000_000.0)
                    .prefix("Bankroll: $"),
            )
            .changed();
        changed
    }

    fn render_odds_format_editor(&mut self, ui: &mut egui::Ui) -> bool {
        match self.odds_format {
            OddsFormat::Decimal => ui
                .add(egui::Slider::new(&mut self.decimal_odds, 1.01..=25.0).text("Decimal Odds"))
                .changed(),
            OddsFormat::American => {
                let changed = ui
                    .add(
                        egui::DragValue::new(&mut self.american_odds)
                            .speed(5.0)
                            .range(-10_000.0..=10_000.0)
                            .prefix("American: "),
                    )
                    .changed();
                if (-99.9..100.0).contains(&self.american_odds) {
                    ui.small("Note: Valid American odds satisfy |A| >= 100.");
                }
                changed
            }
            OddsFormat::Fractional => {
                let mut changed = false;
                ui.horizontal(|ui| {
                    ui.label("Fractional:");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.fractional_numerator)
                                .speed(1.0)
                                .range(1.0..=1000.0),
                        )
                        .changed();
                    ui.label("/");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.fractional_denominator)
                                .speed(1.0)
                                .range(1.0..=1000.0),
                        )
                        .changed();
                });
                changed
            }
        }
    }

    fn render_results_summary(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        match self.resolve_prob_and_odds() {
            Ok((prob, odds)) => {
                let ev = expected_value(&prob, &odds);
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!(
                        "Implied Prob: {:.2}% | EV per $1: ${:.3}",
                        odds.implied_probability_unit().value() * 100.0,
                        ev
                    ));
                    ui.separator();
                    if ev > 0.0 {
                        ui.colored_label(palette.success, "Positive Edge (+EV)");
                    } else {
                        ui.colored_label(palette.error, "No Edge (EV <= 0): Optimal Stake = $0.00");
                    }
                });
                if ev > 0.0 {
                    self.render_kelly_table(ui, &prob, &odds);
                }
            }
            Err(err) => {
                ui.colored_label(palette.error, format!("Input validation: {err}"));
            }
        }
    }

    fn render_kelly_table(&self, ui: &mut egui::Ui, prob: &UnitInterval, odds: &Odds) {
        let Ok(full) = kelly_fraction(prob, odds) else {
            return;
        };
        let half = variants::half_kelly(prob, odds).unwrap_or(full);
        let quarter = variants::quarter_kelly(prob, odds).unwrap_or(full);
        egui::Grid::new("kelly_results")
            .num_columns(4)
            .spacing([28.0, 6.0])
            .show(ui, |ui| {
                for hdr in ["Strategy", "Fraction", "Stake ($)", "Log Growth g(f)"] {
                    ui.label(egui::RichText::new(hdr).strong());
                }
                ui.end_row();
                for (label, frac) in [
                    ("Full Kelly (1.0x)", full),
                    ("Half Kelly (0.5x)", half),
                    ("Quarter Kelly (0.25x)", quarter),
                ] {
                    ui.label(label);
                    ui.label(format!("{:.2}%", frac.value() * 100.0));
                    let amt = frac.bet_amount(self.bankroll).unwrap_or(0.0);
                    ui.label(format!("${amt:.2}"));
                    let g = expected_growth_rate(prob, odds, &frac);
                    ui.label(format!("{g:.5}"));
                    ui.end_row();
                }
            });
    }

    fn render_growth_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let (growth_pts, linear_pts, f_star) = self.compute_growth_curves();
        Plot::new(ui.id().with("bet_size_kelly_growth_plot"))
            .legend(Legend::default())
            .x_axis_label("Bankroll Fraction Staked f")
            .y_axis_label("Asymptotic Log-Growth g(f) vs Linear Return f·EV")
            .show(ui, |plot_ui| {
                plot_ui.hline(HLine::new("Break-Even (g = 0)", 0.0).color(palette.muted_text));
                plot_ui.line(
                    Line::new("Asymptotic Log-Growth g(f)", PlotPoints::from(growth_pts))
                        .color(palette.series[0])
                        .width(2.4_f32),
                );
                plot_ui.line(
                    Line::new(
                        "Linear Expected Return f * EV",
                        PlotPoints::from(linear_pts),
                    )
                    .color(palette.series[1])
                    .style(LineStyle::Dashed { length: 7.0_f32 })
                    .width(1.8_f32),
                );
                if f_star > 0.0 {
                    plot_ui.vline(
                        VLine::new("Full Kelly f*", f_star)
                            .color(palette.success)
                            .width(2.0_f32),
                    );
                    plot_ui.vline(
                        VLine::new("Half Kelly 0.5 f*", 0.5 * f_star)
                            .color(palette.accent)
                            .style(LineStyle::Dashed { length: 6.0_f32 }),
                    );
                    if 2.0 * f_star < 0.98 {
                        plot_ui.vline(
                            VLine::new("Overbet Ruin Boundary 2 f*", 2.0 * f_star)
                                .color(palette.error)
                                .style(LineStyle::Dashed { length: 5.0_f32 }),
                        );
                    }
                }
            });
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
    fn state_revision(&self) -> u64 {
        self.revision
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let (growth_pts, linear_pts, _) = self.compute_growth_curves();
        vec![
            ("Asymptotic Log-Growth g(f)".into(), growth_pts),
            ("Linear Expected Return f * EV".into(), linear_pts),
        ]
    }

    fn has_finite_parameters(&self) -> bool {
        [
            self.probability_input,
            self.decimal_odds,
            self.american_odds,
            self.fractional_numerator,
            self.fractional_denominator,
            self.bankroll,
        ]
        .iter()
        .all(|v| v.is_finite())
    }

    fn clamp_parameters(&mut self) -> bool {
        self.probability_input = self.probability_input.clamp(0.0, 100.0);
        self.decimal_odds = self.decimal_odds.clamp(1.01, 1000.0);
        self.american_odds = self.american_odds.clamp(-10_000.0, 10_000.0);
        self.fractional_numerator = self.fractional_numerator.clamp(1.0, 1000.0);
        self.fractional_denominator = self.fractional_denominator.clamp(1.0, 1000.0);
        self.bankroll = self.bankroll.clamp(1.0, 1_000_000.0);
        self.revision = self.revision.wrapping_add(1);
        true
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
        self.state_bounds().validate_json(state)?;
        let val: serde_json::Value =
            serde_json::from_str(state).map_err(|e| format!("Invalid bet size state: {e}"))?;
        let s: BetSizeState =
            serde_json::from_value(val).map_err(|e| format!("Invalid bet size state: {e}"))?;
        if ![
            s.probability_input,
            s.decimal_odds,
            s.american_odds,
            s.fractional_numerator,
            s.fractional_denominator,
            s.bankroll,
        ]
        .iter()
        .all(|v| v.is_finite())
            || !(0.0..=100.0).contains(&s.probability_input)
            || !(1.01..=1000.0).contains(&s.decimal_odds)
            || !(-10_000.0..=10_000.0).contains(&s.american_odds)
            || !(1.0..=1000.0).contains(&s.fractional_numerator)
            || !(1.0..=1000.0).contains(&s.fractional_denominator)
            || !(1.0..=1_000_000.0).contains(&s.bankroll)
        {
            return Err("Bet size parameters out of bounds".into());
        }
        self.probability_input = s.probability_input;
        self.odds_format = s.odds_format;
        self.decimal_odds = s.decimal_odds;
        self.american_odds = s.american_odds;
        self.fractional_numerator = s.fractional_numerator;
        self.fractional_denominator = s.fractional_denominator;
        self.bankroll = s.bankroll;
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let side_id = crate::framework::scoped_panel_id(parent_ui, "bet_size_side");
        crate::framework::PaneSidePanel::left(side_id).show_inside(parent_ui, |ui| {
            if self.render_input_controls(ui) {
                self.revision = self.revision.wrapping_add(1);
            }
        });
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            self.render_results_summary(ui, &palette);
            ui.separator();
            self.render_growth_plot(ui, &palette);
        });
    }
}

// [cite:sports_quantitative_modeling]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BetSizeCalculatorTool",
        display_name: "Bet Size Calculator",
        domain: "financial_math",
        tags: &["kelly", "wagering", "bankroll", "sports_quant"],
        build: || Box::new(BetSizeCalculatorTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for BetSizeCalculatorTool {
    fn theory_description(&self) -> String {
        format!(
            "Kelly criterion optimal wagering fraction: $f^* = \\frac{{b p - q}}{{b}}$ where $p = {:.1}\\%$ is win probability, $q = 1 - p$, and decimal odds $d = {:.2}$ ($b = d - 1$), maximizing asymptotic logarithmic growth $g(f) = p \\ln(1 + b f) + q \\ln(1 - f)$.",
            self.probability_input, self.decimal_odds
        )
    }
    fn phonetic_description(&self) -> String {
        "Kelly criterion optimal bet sizing and logarithmic bankroll growth theory".into()
    }
    fn theory_citation(&self) -> String {
        "A New Interpretation of Information Rate (Kelly, 1956); Optimal Gambling Systems for Favorable Games (Thorp, 1969)".into()
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
