#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    evaluate_kelly_sizing, plot_line, plot_vline, render_sports_central_panel,
    render_sports_left_panel, render_sports_plot, slider_f64,
};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata};
use eframe::egui;
use math_explorer::sports_quant::{
    book_vigorish, booksum, devig_unit_into, fair_odds, DevigMethod, Odds, UnitInterval,
};
use serde::{Deserialize, Serialize};

/// Bounded URL/preset state for `MarketDeviggingTool`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarketDevigState {
    pub three_way_market: bool,
    pub odds_home: f64,
    pub odds_draw: f64,
    pub odds_away: f64,
    pub offered_bet_odds: f64,
    pub bankroll: f64,
    pub method: DevigMethod,
}

impl Default for MarketDevigState {
    fn default() -> Self {
        Self {
            three_way_market: true,
            odds_home: 1.95,
            odds_draw: 3.45,
            odds_away: 3.90,
            offered_bet_odds: 2.18,
            bankroll: 1000.0,
            method: DevigMethod::Shin,
        }
    }
}

impl MarketDevigState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let vals = [
            self.odds_home,
            self.odds_draw,
            self.odds_away,
            self.offered_bet_odds,
            self.bankroll,
        ];
        if !vals.iter().all(|v| v.is_finite()) {
            return Err("Market devigging parameters must be finite".into());
        }
        let odds_ok = (1.05..=25.0).contains(&self.odds_home)
            && (1.05..=25.0).contains(&self.odds_draw)
            && (1.05..=25.0).contains(&self.odds_away)
            && (1.05..=25.0).contains(&self.offered_bet_odds);
        if !odds_ok || !(10.0..=1_000_000.0).contains(&self.bankroll) {
            return Err("Market devigging parameters out of bounds".into());
        }
        Ok(())
    }
}

/// Interactive Market Devigging tool comparing Multiplicative, Additive, Power, and Shin models.
pub struct MarketDeviggingTool {
    pub(super) state: MarketDevigState,
    revision: u64,
    overround: f64,
    vigorish: f64,
    fair_probs: [UnitInterval; 3],
    curve_mult: Vec<[f64; 2]>,
    curve_power: Vec<[f64; 2]>,
    curve_shin: Vec<[f64; 2]>,
}

impl Default for MarketDeviggingTool {
    fn default() -> Self {
        let mut tool = Self {
            state: MarketDevigState::default(),
            revision: 1,
            overround: 1.0,
            vigorish: 0.0,
            fair_probs: [UnitInterval::new_clamped(0.5); 3],
            curve_mult: Vec::new(),
            curve_power: Vec::new(),
            curve_shin: Vec::new(),
        };
        tool.recompute();
        tool
    }
}

impl MarketDeviggingTool {
    fn active_odds_slice(&self) -> ([Odds; 3], usize) {
        let h = Odds::new(self.state.odds_home).unwrap_or_else(|_| Odds::new(1.95).unwrap());
        let d = Odds::new(self.state.odds_draw).unwrap_or_else(|_| Odds::new(3.45).unwrap());
        let a = Odds::new(self.state.odds_away).unwrap_or_else(|_| Odds::new(3.90).unwrap());
        if self.state.three_way_market {
            ([h, d, a], 3)
        } else {
            ([h, a, d], 2)
        }
    }

    pub(super) fn recompute(&mut self) {
        let (odds_arr, n) = self.active_odds_slice();
        let slice = &odds_arr[..n];
        self.overround = booksum(slice).unwrap_or(1.0);
        self.vigorish = book_vigorish(slice).map(|u| u.value()).unwrap_or(0.0);
        let mut out = [UnitInterval::new_clamped(0.5); 3];
        if devig_unit_into(slice, self.state.method, &mut out[..n]).is_ok() {
            self.fair_probs = out;
        }
        self.rebuild_sweep_curves();
        self.revision = self.revision.wrapping_add(1);
    }

    fn rebuild_sweep_curves(&mut self) {
        self.curve_mult.clear();
        self.curve_power.clear();
        self.curve_shin.clear();
        let away = Odds::new(self.state.odds_away).unwrap_or_else(|_| Odds::new(3.90).unwrap());
        let draw = Odds::new(self.state.odds_draw).unwrap_or_else(|_| Odds::new(3.45).unwrap());
        let n = if self.state.three_way_market { 3 } else { 2 };
        for idx in 0..49usize {
            let h_val = 1.15 + 4.85 * (idx as f64) / 48.0;
            let Ok(home) = Odds::new(h_val) else { continue };
            let arr = if n == 3 {
                [home, draw, away]
            } else {
                [home, away, draw]
            };
            if booksum(&arr[..n]).unwrap_or(0.0) <= 1.001 {
                continue;
            }
            let mut buf = [UnitInterval::new_clamped(0.5); 3];
            if devig_unit_into(&arr[..n], DevigMethod::Multiplicative, &mut buf[..n]).is_ok() {
                self.curve_mult.push([h_val, buf[0].value()]);
            }
            if devig_unit_into(&arr[..n], DevigMethod::Power, &mut buf[..n]).is_ok() {
                self.curve_power.push([h_val, buf[0].value()]);
            }
            if devig_unit_into(&arr[..n], DevigMethod::Shin, &mut buf[..n]).is_ok() {
                self.curve_shin.push([h_val, buf[0].value()]);
            }
        }
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        changed |= ui
            .checkbox(
                &mut self.state.three_way_market,
                "3-Way 1X2 Market (Include Draw)",
            )
            .accessible_hover_text("Toggle between 2-way moneyline and 3-way 1X2 soccer market")
            .changed();
        ui.label("Devigging Method:");
        for (method, label) in [
            (DevigMethod::Multiplicative, "Multiplicative (Proportional)"),
            (DevigMethod::Additive, "Additive (Equal Margin)"),
            (DevigMethod::Power, "Power (Logarithmic k*)"),
            (DevigMethod::Shin, "Shin (Insider Trading z*)"),
        ] {
            changed |= ui
                .radio_value(&mut self.state.method, method, label)
                .changed();
        }
        ui.separator();
        changed |=
            slider_f64(ui, &mut self.state.odds_home, 1.10..=12.0, "Book Home Odds").changed();
        if self.state.three_way_market {
            changed |=
                slider_f64(ui, &mut self.state.odds_draw, 1.50..=12.0, "Book Draw Odds").changed();
        }
        changed |=
            slider_f64(ui, &mut self.state.odds_away, 1.10..=12.0, "Book Away Odds").changed();
        ui.separator();
        changed |= slider_f64(
            ui,
            &mut self.state.offered_bet_odds,
            1.10..=12.0,
            "Soft Target Odds (Home)",
        )
        .accessible_hover_text("Target bookmaker decimal price on Home evaluated via Kelly")
        .changed();
        changed |= ui
            .add(
                egui::DragValue::new(&mut self.state.bankroll)
                    .speed(25.0)
                    .range(10.0..=1_000_000.0)
                    .prefix("Bankroll: $"),
            )
            .changed();
        if changed {
            self.recompute();
        }
    }

    fn render_summary(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let home_p = self.fair_probs[0];
        let fair_h = fair_odds(home_p).map(|o| o.value()).unwrap_or(0.0);
        let (ev, full_pct, half_stake) =
            evaluate_kelly_sizing(home_p, self.state.offered_bet_odds, self.state.bankroll);
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                palette.series[0],
                format!(
                    "Overround: {:.2}% | Vig: {:.2}%",
                    self.overround * 100.0,
                    self.vigorish * 100.0
                ),
            );
            ui.separator();
            ui.colored_label(
                palette.accent,
                format!(
                    "Fair Home: {:.2}% (Fair Odds {:.2})",
                    home_p.value() * 100.0,
                    fair_h
                ),
            );
            ui.separator();
            let edge_color = if ev > 0.0 {
                palette.success
            } else {
                palette.muted_text
            };
            ui.colored_label(
                edge_color,
                format!(
                    "EV @ {:.2}: ${:+.3} | Full Kelly: {:.2}% | Half-Kelly: ${:.2}",
                    self.state.offered_bet_odds, ev, full_pct, half_stake
                ),
            );
        });
    }

    fn render_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        render_sports_plot(
            ui,
            "market_devig_sweep_plot",
            "Bookmaker Home Decimal Odds d_H",
            "Devigged Fair Home Win Probability p_H",
            &mut |plot_ui| {
                plot_ui.vline(plot_vline(
                    "Active Book Home Odds",
                    self.state.odds_home,
                    palette.accent,
                    6.0,
                ));
                plot_ui.line(plot_line(
                    "Multiplicative Devig",
                    &self.curve_mult,
                    palette.series[0],
                    2.0,
                    None,
                ));
                plot_ui.line(plot_line(
                    "Power Devig (k*)",
                    &self.curve_power,
                    palette.series[1],
                    2.0,
                    Some(7.0),
                ));
                plot_ui.line(plot_line(
                    "Shin Insider Devig (z*)",
                    &self.curve_shin,
                    palette.success,
                    2.5,
                    None,
                ));
            },
        );
    }
}

impl InteractiveTool for MarketDeviggingTool {
    fn name(&self) -> &'static str {
        "Market Devigging"
    }
    fn domain(&self) -> &'static str {
        "sports_analytics"
    }
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn state_revision(&self) -> u64 {
        self.revision
    }
    fn has_finite_parameters(&self) -> bool {
        self.state.validate().is_ok()
    }
    fn clamp_parameters(&mut self) -> bool {
        let d = MarketDevigState::default();
        self.state.odds_home = super::clamp_finite(self.state.odds_home, 1.05, 25.0, d.odds_home);
        self.state.odds_draw = super::clamp_finite(self.state.odds_draw, 1.05, 25.0, d.odds_draw);
        self.state.odds_away = super::clamp_finite(self.state.odds_away, 1.05, 25.0, d.odds_away);
        self.state.offered_bet_odds =
            super::clamp_finite(self.state.offered_bet_odds, 1.05, 25.0, d.offered_bet_odds);
        self.state.bankroll =
            super::clamp_finite(self.state.bankroll, 10.0, 1_000_000.0, d.bankroll);
        self.recompute();
        true
    }
    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }
    fn try_load_state(&mut self, json: &str) -> Result<(), String> {
        self.state_bounds().validate_json(json)?;
        let parsed: MarketDevigState = serde_json::from_value(super::parse_state_value(json)?)
            .map_err(|e| format!("Invalid MarketDevigState: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.recompute();
        Ok(())
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            (
                "Multiplicative Fair Probability".into(),
                self.curve_mult.clone(),
            ),
            (
                "Power Devig Fair Probability".into(),
                self.curve_power.clone(),
            ),
            (
                "Shin Insider Fair Probability".into(),
                self.curve_shin.clone(),
            ),
        ]
    }
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        render_sports_left_panel(
            parent_ui,
            "market_devig_side",
            "Market Devigging",
            &mut |ui| {
                self.render_controls(ui);
            },
        );
        render_sports_central_panel(parent_ui, &mut |ui, palette| {
            self.render_summary(ui, palette);
            ui.separator();
            self.render_plot(ui, palette);
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for MarketDeviggingTool {
    fn theory_description(&self) -> String {
        "Bookmaker vigorish removal & favorite-longshot bias: recovers normalized probabilities $\\sum_i p_i = 1$ from overround quotes $S = \\sum_i d_i^{-1} > 1$ via Multiplicative, Additive, Power ($\\sum_i q_i^k = 1$), and Shin insider-trading equilibrium ($z \\in (0, 1)$) before Kelly stake sizing.".into()
    }
    fn phonetic_description(&self) -> String {
        "Sportsbook market devigging comparing multiplicative power and Shin insider probability models".into()
    }
    fn theory_citation(&self) -> String {
        "Measuring the Incidence of Insider Trading in a Market for State-Contingent Claims (Shin, 1993); A New Interpretation of Information Rate (Kelly, 1956)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    ToolMetadata {
        id: "SportsMarketDeviggingTool",
        display_name: "Market Devigging",
        domain: "sports_analytics",
        tags: &["devig", "shin", "vigorish", "odds", "kelly"],
        build: || Box::new(MarketDeviggingTool::default()),
    }
}

// [cite:sports_quantitative_modeling]
