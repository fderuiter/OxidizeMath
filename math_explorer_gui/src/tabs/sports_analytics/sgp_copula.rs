#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    evaluate_kelly_sizing, plot_hline, plot_line, plot_vline, render_sports_central_panel,
    render_sports_left_panel, render_sports_plot, slider_f64,
};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata};
use eframe::egui;
use math_explorer::sports_quant::{
    sgp_joint_probability, Correlation, CorrelationMatrix, Probability, UnitInterval,
};
use serde::{Deserialize, Serialize};

/// Bounded URL/preset state for `SgpCopulaTool`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SgpCopulaState {
    pub three_legs: bool,
    pub leg1_prob: f64,
    pub leg2_prob: f64,
    pub leg3_prob: f64,
    pub rho_12: f64,
    pub rho_13: f64,
    pub rho_23: f64,
    pub offered_sgp_odds: f64,
    pub bankroll: f64,
}

impl Default for SgpCopulaState {
    fn default() -> Self {
        Self {
            three_legs: false,
            leg1_prob: 0.56,
            leg2_prob: 0.54,
            leg3_prob: 0.52,
            rho_12: 0.38,
            rho_13: 0.25,
            rho_23: 0.22,
            offered_sgp_odds: 3.15,
            bankroll: 1000.0,
        }
    }
}

impl SgpCopulaState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let vals = [
            self.leg1_prob,
            self.leg2_prob,
            self.leg3_prob,
            self.rho_12,
            self.rho_13,
            self.rho_23,
            self.offered_sgp_odds,
            self.bankroll,
        ];
        if !vals.iter().all(|v| v.is_finite()) {
            return Err("SGP Copula parameters must be finite".into());
        }
        let ok = (0.05..=0.95).contains(&self.leg1_prob)
            && (0.05..=0.95).contains(&self.leg2_prob)
            && (0.05..=0.95).contains(&self.leg3_prob)
            && (-0.85..=0.85).contains(&self.rho_12)
            && (-0.85..=0.85).contains(&self.rho_13)
            && (-0.85..=0.85).contains(&self.rho_23)
            && (1.10..=50.0).contains(&self.offered_sgp_odds)
            && (10.0..=1_000_000.0).contains(&self.bankroll);
        if !ok {
            return Err("SGP Copula parameters out of bounds".into());
        }
        Ok(())
    }
}

/// Interactive Gaussian Copula Same-Game Parlay (SGP) pricing and correlation tool.
pub struct SgpCopulaTool {
    pub(super) state: SgpCopulaState,
    revision: u64,
    joint_prob: UnitInterval,
    indep_prob: f64,
    spd_valid: bool,
    curve_copula: Vec<[f64; 2]>,
    curve_indep: Vec<[f64; 2]>,
}

impl Default for SgpCopulaTool {
    fn default() -> Self {
        let mut tool = Self {
            state: SgpCopulaState::default(),
            revision: 1,
            joint_prob: UnitInterval::new_clamped(0.30),
            indep_prob: 0.30,
            spd_valid: true,
            curve_copula: Vec::new(),
            curve_indep: Vec::new(),
        };
        tool.recompute();
        tool
    }
}

impl SgpCopulaTool {
    fn eval_for_rho12(&self, r12_val: f64) -> Option<UnitInterval> {
        let p1 = Probability::from(UnitInterval::new_clamped(self.state.leg1_prob));
        let p2 = Probability::from(UnitInterval::new_clamped(self.state.leg2_prob));
        let r12 = Correlation::new(r12_val).ok()?;
        if self.state.three_legs {
            let p3 = Probability::from(UnitInterval::new_clamped(self.state.leg3_prob));
            let r13 = Correlation::new(self.state.rho_13).ok()?.value();
            let r23 = Correlation::new(self.state.rho_23).ok()?.value();
            let r12v = r12.value();
            let mat = math_explorer::nalgebra::DMatrix::from_row_slice(
                3,
                3,
                &[1.0, r12v, r13, r12v, 1.0, r23, r13, r23, 1.0],
            );
            let corr = CorrelationMatrix::new(mat).ok()?;
            let jp = sgp_joint_probability(&[p1, p2, p3], &corr).ok()?;
            Some(UnitInterval::from(jp))
        } else {
            let corr = CorrelationMatrix::bivariate(r12).ok()?;
            let jp = sgp_joint_probability(&[p1, p2], &corr).ok()?;
            Some(UnitInterval::from(jp))
        }
    }

    pub(super) fn recompute(&mut self) {
        self.indep_prob = if self.state.three_legs {
            self.state.leg1_prob * self.state.leg2_prob * self.state.leg3_prob
        } else {
            self.state.leg1_prob * self.state.leg2_prob
        };
        if let Some(jp) = self.eval_for_rho12(self.state.rho_12) {
            self.joint_prob = jp;
            self.spd_valid = true;
        } else {
            self.joint_prob = UnitInterval::new_clamped(self.indep_prob);
            self.spd_valid = false;
        }
        self.curve_copula.clear();
        self.curve_indep.clear();
        for idx in 0..49usize {
            let r = -0.80 + 1.60 * (idx as f64) / 48.0;
            if let Some(jp) = self.eval_for_rho12(r) {
                self.curve_copula.push([r, jp.value()]);
                self.curve_indep.push([r, self.indep_prob]);
            }
        }
        self.revision = self.revision.wrapping_add(1);
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        changed |= ui
            .checkbox(
                &mut self.state.three_legs,
                "3-Leg SGP (Cholesky 3x3 Elliptope)",
            )
            .accessible_hover_text(
                "Switch between 2-leg bivariate normal and 3-leg Gaussian copula",
            )
            .changed();
        ui.separator();
        changed |=
            slider_f64(ui, &mut self.state.leg1_prob, 0.10..=0.90, "Leg 1 Fair p₁").changed();
        changed |=
            slider_f64(ui, &mut self.state.leg2_prob, 0.10..=0.90, "Leg 2 Fair p₂").changed();
        if self.state.three_legs {
            changed |=
                slider_f64(ui, &mut self.state.leg3_prob, 0.10..=0.90, "Leg 3 Fair p₃").changed();
        }
        ui.separator();
        changed |= slider_f64(ui, &mut self.state.rho_12, -0.80..=0.80, "Corr ρ₁₂").changed();
        if self.state.three_legs {
            changed |= slider_f64(ui, &mut self.state.rho_13, -0.75..=0.75, "Corr ρ₁₃").changed();
            changed |= slider_f64(ui, &mut self.state.rho_23, -0.75..=0.75, "Corr ρ₂₃").changed();
        }
        ui.separator();
        changed |= slider_f64(
            ui,
            &mut self.state.offered_sgp_odds,
            1.20..=25.0,
            "Offered SGP Odds",
        )
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
        if !self.spd_valid {
            ui.colored_label(
                palette.warning,
                "3x3 correlation triple outside SPD elliptope (det(R) <= 0); adjust off-diagonal correlations.",
            );
        }
        let mult = self.joint_prob.value() / self.indep_prob.max(1e-9);
        let (ev, full_pct, half_stake) = evaluate_kelly_sizing(
            self.joint_prob,
            self.state.offered_sgp_odds,
            self.state.bankroll,
        );
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                palette.series[0],
                format!(
                    "Copula Joint: {:.2}% vs Indep: {:.2}% ({:.2}x lift)",
                    self.joint_prob.value() * 100.0,
                    self.indep_prob * 100.0,
                    mult
                ),
            );
            ui.separator();
            ui.colored_label(
                if ev > 0.0 {
                    palette.success
                } else {
                    palette.muted_text
                },
                format!(
                    "SGP EV @ {:.2}: ${:+.3} | Full Kelly: {:.2}% | Half-Kelly: ${:.2}",
                    self.state.offered_sgp_odds, ev, full_pct, half_stake
                ),
            );
        });
    }

    fn render_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let breakeven = 1.0 / self.state.offered_sgp_odds.max(1.01);
        render_sports_plot(
            ui,
            "sgp_copula_corr_plot",
            "Latent Gaussian Leg Correlation ρ₁₂",
            "Joint Parlay Hit Probability P(Leg 1 ∩ Leg 2 ...)",
            &mut |plot_ui| {
                plot_ui.vline(plot_vline(
                    "Active ρ₁₂",
                    self.state.rho_12,
                    palette.accent,
                    6.0,
                ));
                plot_ui.hline(plot_hline(
                    "Bookmaker Break-Even 1/d",
                    breakeven,
                    palette.warning,
                    Some(5.0),
                ));
                plot_ui.line(plot_line(
                    "Gaussian Copula Joint Probability",
                    &self.curve_copula,
                    palette.series[0],
                    2.5,
                    None,
                ));
                plot_ui.line(plot_line(
                    "Naive Independence Product",
                    &self.curve_indep,
                    palette.muted_text,
                    1.8,
                    Some(7.0),
                ));
            },
        );
    }
}

impl InteractiveTool for SgpCopulaTool {
    fn name(&self) -> &'static str {
        "Same-Game Parlay Copula"
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
        let d = SgpCopulaState::default();
        self.state.leg1_prob = super::clamp_finite(self.state.leg1_prob, 0.05, 0.95, d.leg1_prob);
        self.state.leg2_prob = super::clamp_finite(self.state.leg2_prob, 0.05, 0.95, d.leg2_prob);
        self.state.leg3_prob = super::clamp_finite(self.state.leg3_prob, 0.05, 0.95, d.leg3_prob);
        self.state.rho_12 = super::clamp_finite(self.state.rho_12, -0.85, 0.85, d.rho_12);
        self.state.rho_13 = super::clamp_finite(self.state.rho_13, -0.85, 0.85, d.rho_13);
        self.state.rho_23 = super::clamp_finite(self.state.rho_23, -0.85, 0.85, d.rho_23);
        self.state.offered_sgp_odds =
            super::clamp_finite(self.state.offered_sgp_odds, 1.10, 50.0, d.offered_sgp_odds);
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
        let parsed: SgpCopulaState = serde_json::from_value(super::parse_state_value(json)?)
            .map_err(|e| format!("Invalid SgpCopulaState: {e}"))?;
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
                "Gaussian Copula Joint Probability".into(),
                self.curve_copula.clone(),
            ),
            (
                "Naive Independence Product".into(),
                self.curve_indep.clone(),
            ),
        ]
    }
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        render_sports_left_panel(
            parent_ui,
            "sgp_copula_side",
            "Same-Game Parlay Copula",
            &mut |ui| self.render_controls(ui),
        );
        render_sports_central_panel(parent_ui, &mut |ui, palette| {
            self.render_summary(ui, palette);
            ui.separator();
            self.render_plot(ui, palette);
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for SgpCopulaTool {
    fn theory_description(&self) -> String {
        "Gaussian Copula Same-Game Parlay (SGP) pricing: maps marginal leg probabilities $p_i \\in (0,1)$ via quantile $z_i = \\Phi^{-1}(p_i)$ and evaluates multivariate normal orthant probability $C_R^{\\text{Ga}}(u) = \\Phi_R(z_1, \\ldots, z_d)$ under SPD correlation matrix $R \\succ 0$.".into()
    }
    fn phonetic_description(&self) -> String {
        "Gaussian copula same game parlay joint probability and correlation pricing".into()
    }
    fn theory_citation(&self) -> String {
        "A New Interpretation of Information Rate (Kelly, 1956); Optimal Gambling Systems for Favorable Games (Thorp, 1969)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    ToolMetadata {
        id: "SgpCopulaTool",
        display_name: "Same-Game Parlay Copula",
        domain: "sports_analytics",
        tags: &["copula", "parlay", "sgp", "correlation", "kelly"],
        build: || Box::new(SgpCopulaTool::default()),
    }
}

// [cite:sports_quantitative_modeling]
