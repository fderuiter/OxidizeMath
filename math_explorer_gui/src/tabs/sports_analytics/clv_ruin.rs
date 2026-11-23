#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    plot_hline, plot_line, plot_vline, render_sports_central_panel, render_sports_left_panel,
    render_sports_plot, slider_f64,
};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata};
use eframe::egui;
use math_explorer::sports_quant::{
    closing_line_edge_two_way, expected_growth_rate, fractional_kelly, kelly_drawdown_risk,
    kelly_fraction, unit_risk_of_ruin, DevigMethod, Odds, UnitInterval,
};
use serde::{Deserialize, Serialize};

/// Bounded URL/preset state for `ClvRuinTool`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClvRuinState {
    pub bet_odds: f64,
    pub close_home_odds: f64,
    pub close_away_odds: f64,
    pub kelly_multiplier: f64,
    pub drawdown_ratio: f64,
    pub bankroll_units: f64,
    pub devig_method: DevigMethod,
}

impl Default for ClvRuinState {
    fn default() -> Self {
        Self {
            bet_odds: 2.14,
            close_home_odds: 1.88,
            close_away_odds: 2.02,
            kelly_multiplier: 0.50,
            drawdown_ratio: 0.50,
            bankroll_units: 40.0,
            devig_method: DevigMethod::Shin,
        }
    }
}

impl ClvRuinState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let vals = [
            self.bet_odds,
            self.close_home_odds,
            self.close_away_odds,
            self.kelly_multiplier,
            self.drawdown_ratio,
            self.bankroll_units,
        ];
        if !vals.iter().all(|v| v.is_finite()) {
            return Err("CLV & Ruin parameters must be finite".into());
        }
        let ok = (1.05..=25.0).contains(&self.bet_odds)
            && (1.05..=25.0).contains(&self.close_home_odds)
            && (1.05..=25.0).contains(&self.close_away_odds)
            && (0.05..=2.20).contains(&self.kelly_multiplier)
            && (0.10..=0.90).contains(&self.drawdown_ratio)
            && (5.0..=250.0).contains(&self.bankroll_units);
        if !ok {
            return Err("CLV & Ruin parameters out of bounds".into());
        }
        Ok(())
    }
}

/// Interactive Closing Line Value (CLV), Fractional Kelly Growth, and Thorp Risk-of-Ruin tool.
pub struct ClvRuinTool {
    pub(super) state: ClvRuinState,
    revision: u64,
    clv_edge: f64,
    fair_close_p: UnitInterval,
    active_stake_pct: f64,
    drawdown_prob: UnitInterval,
    unit_ror: UnitInterval,
    curve_growth_ratio: Vec<[f64; 2]>,
    curve_drawdown: Vec<[f64; 2]>,
}

impl Default for ClvRuinTool {
    fn default() -> Self {
        let neutral = UnitInterval::new_clamped(0.5);
        let mut tool = Self {
            state: ClvRuinState::default(),
            revision: 1,
            clv_edge: 0.05,
            fair_close_p: neutral,
            active_stake_pct: 3.0,
            drawdown_prob: neutral,
            unit_ror: neutral,
            curve_growth_ratio: Vec::new(),
            curve_drawdown: Vec::new(),
        };
        tool.recompute();
        tool
    }
}

impl ClvRuinTool {
    pub(super) fn recompute(&mut self) {
        let bet = Odds::new(self.state.bet_odds).unwrap_or_else(|_| Odds::new(2.14).unwrap());
        let ch = Odds::new(self.state.close_home_odds).unwrap_or_else(|_| Odds::new(1.88).unwrap());
        let ca = Odds::new(self.state.close_away_odds).unwrap_or_else(|_| Odds::new(2.02).unwrap());
        (self.clv_edge, self.fair_close_p) =
            closing_line_edge_two_way(bet, ch, ca, self.state.devig_method)
                .unwrap_or((0.0, UnitInterval::new_clamped(0.5)));

        let full_f = kelly_fraction(&self.fair_close_p, &bet)
            .unwrap_or_else(|_| UnitInterval::new_clamped(0.0).into());
        let g_max = expected_growth_rate(&self.fair_close_p, &bet, &full_f).max(1e-9);
        if let Ok(frac) = fractional_kelly(&self.fair_close_p, &bet, self.state.kelly_multiplier) {
            self.active_stake_pct = frac.value() * 100.0;
            self.drawdown_prob =
                kelly_drawdown_risk(&self.fair_close_p, &bet, &frac, self.state.drawdown_ratio)
                    .unwrap_or_else(|_| UnitInterval::new_clamped(1.0));
        }
        self.unit_ror = unit_risk_of_ruin(&self.fair_close_p, &bet, self.state.bankroll_units)
            .unwrap_or_else(|_| UnitInterval::new_clamped(1.0));

        self.rebuild_curves(bet, g_max);
        self.revision = self.revision.wrapping_add(1);
    }

    fn rebuild_curves(&mut self, bet: Odds, g_max: f64) {
        self.curve_growth_ratio.clear();
        self.curve_drawdown.clear();
        for idx in 1..=44usize {
            let mult = 0.05 * (idx as f64);
            if let Ok(frac) = fractional_kelly(&self.fair_close_p, &bet, mult) {
                let g_ratio = if g_max > 1e-8 {
                    (expected_growth_rate(&self.fair_close_p, &bet, &frac) / g_max)
                        .clamp(-0.5, 1.05)
                } else {
                    0.0
                };
                let dd =
                    kelly_drawdown_risk(&self.fair_close_p, &bet, &frac, self.state.drawdown_ratio)
                        .map(|u| u.value())
                        .unwrap_or(1.0);
                self.curve_growth_ratio.push([mult, g_ratio]);
                self.curve_drawdown.push([mult, dd]);
            }
        }
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        changed |= slider_f64(
            ui,
            &mut self.state.bet_odds,
            1.10..=10.0,
            "Ticket Odds d_bet",
        )
        .accessible_hover_text("Decimal odds captured at bet placement")
        .changed();
        changed |= slider_f64(
            ui,
            &mut self.state.close_home_odds,
            1.10..=10.0,
            "Closing Side Odds",
        )
        .changed();
        changed |= slider_f64(
            ui,
            &mut self.state.close_away_odds,
            1.10..=10.0,
            "Closing Opp Odds",
        )
        .changed();
        ui.separator();
        changed |= slider_f64(
            ui,
            &mut self.state.kelly_multiplier,
            0.05..=2.20,
            "Kelly Multiplier c",
        )
        .accessible_hover_text("0.50 = Half Kelly, 1.00 = Full Kelly, >2.00 = negative log growth")
        .changed();
        changed |= slider_f64(
            ui,
            &mut self.state.drawdown_ratio,
            0.15..=0.85,
            "Drawdown Floor α",
        )
        .changed();
        changed |= slider_f64(
            ui,
            &mut self.state.bankroll_units,
            5.0..=150.0,
            "Flat Bankroll Units",
        )
        .changed();
        if changed {
            self.recompute();
        }
    }

    fn render_summary(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                if self.clv_edge > 0.0 {
                    palette.success
                } else {
                    palette.error
                },
                format!(
                    "CLV Edge: {:+.2}% (Fair Close {:.1}%)",
                    self.clv_edge * 100.0,
                    self.fair_close_p.value() * 100.0
                ),
            );
            ui.separator();
            ui.colored_label(
                palette.series[0],
                format!(
                    "Stake ({:.2}x Kelly): {:.2}% | P(Hit {:.0}% Floor): {:.1}%",
                    self.state.kelly_multiplier,
                    self.active_stake_pct,
                    self.state.drawdown_ratio * 100.0,
                    self.drawdown_prob.value() * 100.0
                ),
            );
            ui.separator();
            ui.colored_label(
                palette.accent,
                format!(
                    "Flat {:.0}u Risk of Ruin: {:.2}%",
                    self.state.bankroll_units,
                    self.unit_ror.value() * 100.0
                ),
            );
        });
    }

    fn render_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        render_sports_plot(
            ui,
            "clv_kelly_ruin_plot",
            "Fractional Kelly Multiplier c (f = c · f*)",
            "Normalized Growth g(cf*)/g(f*) & Drawdown Risk",
            &mut |plot_ui| {
                plot_ui.hline(plot_hline(
                    "Zero Growth Boundary",
                    0.0,
                    palette.muted_text,
                    None,
                ));
                plot_ui.vline(plot_vline(
                    "Active Kelly Multiplier",
                    self.state.kelly_multiplier,
                    palette.accent,
                    6.0,
                ));
                plot_ui.line(plot_line(
                    "Normalized Log-Growth g(cf*)/g(f*)",
                    &self.curve_growth_ratio,
                    palette.success,
                    2.5,
                    None,
                ));
                plot_ui.line(plot_line(
                    "Drawdown Probability P(Hit αW₀)",
                    &self.curve_drawdown,
                    palette.error,
                    2.2,
                    Some(7.0),
                ));
            },
        );
    }
}

impl InteractiveTool for ClvRuinTool {
    fn name(&self) -> &'static str {
        "CLV & Kelly Risk of Ruin"
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
        self.state.bet_odds = self.state.bet_odds.clamp(1.05, 25.0);
        self.state.close_home_odds = self.state.close_home_odds.clamp(1.05, 25.0);
        self.state.close_away_odds = self.state.close_away_odds.clamp(1.05, 25.0);
        self.state.kelly_multiplier = self.state.kelly_multiplier.clamp(0.05, 2.20);
        self.state.drawdown_ratio = self.state.drawdown_ratio.clamp(0.10, 0.90);
        self.state.bankroll_units = self.state.bankroll_units.clamp(5.0, 250.0);
        self.recompute();
        true
    }
    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }
    fn try_load_state(&mut self, json: &str) -> Result<(), String> {
        self.state_bounds().validate_json(json)?;
        let parsed: ClvRuinState = serde_json::from_value(super::parse_state_value(json)?)
            .map_err(|e| format!("Invalid ClvRuinState: {e}"))?;
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
                "Normalized Log-Growth Ratio".into(),
                self.curve_growth_ratio.clone(),
            ),
            (
                "Drawdown Probability Curve".into(),
                self.curve_drawdown.clone(),
            ),
        ]
    }
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        render_sports_left_panel(
            parent_ui,
            "clv_ruin_side",
            "CLV & Kelly Risk of Ruin",
            &mut |ui| self.render_controls(ui),
        );
        render_sports_central_panel(parent_ui, &mut |ui, palette| {
            self.render_summary(ui, palette);
            ui.separator();
            self.render_plot(ui, palette);
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for ClvRuinTool {
    fn theory_description(&self) -> String {
        "Closing Line Value (CLV) & fractional Kelly risk: evaluates $\\text{CLV} = d_{\\text{bet}} p_{\\text{close}} - 1$ against devigged closing lines, comparing normalized log-growth $g(c f^*) / g(f^*)$ against continuous drawdown risk $\\alpha^{2/c - 1}$ and Thorp discrete unit risk of ruin.".into()
    }
    fn phonetic_description(&self) -> String {
        "Closing line value fractional Kelly growth trade off and Thorp risk of ruin".into()
    }
    fn theory_citation(&self) -> String {
        "Optimal Gambling Systems for Favorable Games (Thorp, 1969); Measuring the Incidence of Insider Trading in a Market for State-Contingent Claims (Shin, 1993)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    ToolMetadata {
        id: "ClvRuinTool",
        display_name: "CLV & Kelly Risk of Ruin",
        domain: "sports_analytics",
        tags: &["clv", "kelly", "ruin", "drawdown", "bankroll"],
        build: || Box::new(ClvRuinTool::default()),
    }
}

// [cite:sports_quantitative_modeling]
