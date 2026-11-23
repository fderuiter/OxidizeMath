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
    elo_expected_score, elo_update_with_mov, head_to_head_win_probability_unit, update_rating,
    EloKFactor, EloRating, GlickoPlayer, HomeAdvantage, MatchResult, Rating, RatingDeviation,
    SystemConstant, UnitInterval, Volatility,
};
use serde::{Deserialize, Serialize};

/// Bounded URL/preset state for `SportsRatingsTool`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SportsRatingsState {
    pub home_rating: f64,
    pub away_rating: f64,
    pub home_rd: f64,
    pub away_rd: f64,
    pub volatility: f64,
    pub home_advantage: f64,
    pub elo_k: f64,
    pub margin_of_victory: u32,
    pub offered_home_odds: f64,
    pub bankroll: f64,
}

impl Default for SportsRatingsState {
    fn default() -> Self {
        Self {
            home_rating: 1615.0,
            away_rating: 1525.0,
            home_rd: 65.0,
            away_rd: 110.0,
            volatility: 0.06,
            home_advantage: 55.0,
            elo_k: 24.0,
            margin_of_victory: 2,
            offered_home_odds: 1.76,
            bankroll: 1000.0,
        }
    }
}

impl SportsRatingsState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let vals = [
            self.home_rating,
            self.away_rating,
            self.home_rd,
            self.away_rd,
            self.volatility,
            self.home_advantage,
            self.elo_k,
            self.offered_home_odds,
            self.bankroll,
        ];
        if !vals.iter().all(|v| v.is_finite()) {
            return Err("Ratings parameters must be finite".into());
        }
        let ok = (800.0..=2600.0).contains(&self.home_rating)
            && (800.0..=2600.0).contains(&self.away_rating)
            && (25.0..=350.0).contains(&self.home_rd)
            && (25.0..=350.0).contains(&self.away_rd)
            && (0.01..=0.25).contains(&self.volatility)
            && (0.0..=150.0).contains(&self.home_advantage)
            && (4.0..=64.0).contains(&self.elo_k)
            && self.margin_of_victory <= 15
            && (1.05..=25.0).contains(&self.offered_home_odds)
            && (10.0..=1_000_000.0).contains(&self.bankroll);
        if !ok {
            return Err("Ratings parameters out of bounds".into());
        }
        Ok(())
    }
}

/// Interactive Elo & Glicko-2 head-to-head matchup probability and rating update comparator.
pub struct SportsRatingsTool {
    pub(super) state: SportsRatingsState,
    revision: u64,
    elo_win: UnitInterval,
    glicko_win: UnitInterval,
    next_elo_home: f64,
    next_glicko_home: f64,
    next_glicko_rd: f64,
    curve_elo: Vec<[f64; 2]>,
    curve_glicko: Vec<[f64; 2]>,
}

impl Default for SportsRatingsTool {
    fn default() -> Self {
        let neutral = UnitInterval::new_clamped(0.5);
        let mut tool = Self {
            state: SportsRatingsState::default(),
            revision: 1,
            elo_win: neutral,
            glicko_win: neutral,
            next_elo_home: 1625.0,
            next_glicko_home: 1625.0,
            next_glicko_rd: 60.0,
            curve_elo: Vec::new(),
            curve_glicko: Vec::new(),
        };
        tool.recompute();
        tool
    }
}

impl SportsRatingsTool {
    pub(super) fn recompute(&mut self) {
        let ra = EloRating::new(self.state.home_rating).unwrap_or_default();
        let rb = EloRating::new(self.state.away_rating).unwrap_or_default();
        let hfa = HomeAdvantage::new(self.state.home_advantage).unwrap_or_default();
        let k = EloKFactor::new(self.state.elo_k).unwrap_or_default();
        self.elo_win = elo_expected_score(ra, rb, hfa);
        let (ra_next, _) = elo_update_with_mov(
            ra,
            rb,
            UnitInterval::new_clamped(1.0),
            self.state.margin_of_victory,
            k,
            hfa,
        );
        self.next_elo_home = ra_next.value();

        let vol = Volatility::new(self.state.volatility).unwrap_or_default();
        let home_g = GlickoPlayer::new(
            Rating::new(self.state.home_rating + self.state.home_advantage).unwrap_or_default(),
            RatingDeviation::new(self.state.home_rd).unwrap_or_default(),
            vol,
        );
        let away_g = GlickoPlayer::new(
            Rating::new(self.state.away_rating).unwrap_or_default(),
            RatingDeviation::new(self.state.away_rd).unwrap_or_default(),
            vol,
        );
        self.glicko_win = head_to_head_win_probability_unit(&home_g, &away_g);
        if let Ok(m) = MatchResult::new(away_g, 1.0) {
            if let Ok(tau) = SystemConstant::new(0.5) {
                if let Ok(updated) = update_rating(&home_g, &[m], &tau) {
                    self.next_glicko_home = updated.rating.value() - self.state.home_advantage;
                    self.next_glicko_rd = updated.rating_deviation.value();
                }
            }
        }
        self.rebuild_curves(vol, hfa);
        self.revision = self.revision.wrapping_add(1);
    }

    fn rebuild_curves(&mut self, vol: Volatility, hfa: HomeAdvantage) {
        self.curve_elo.clear();
        self.curve_glicko.clear();
        let away_r = self.state.away_rating;
        let away_elo = EloRating::new(away_r).unwrap_or_default();
        let away_g = GlickoPlayer::new(
            Rating::new(away_r).unwrap_or_default(),
            RatingDeviation::new(self.state.away_rd).unwrap_or_default(),
            vol,
        );
        for idx in 0..49usize {
            let diff = -400.0 + 800.0 * (idx as f64) / 48.0;
            let hr = (away_r + diff).clamp(400.0, 3200.0);
            let p_elo = elo_expected_score(EloRating::new(hr).unwrap_or_default(), away_elo, hfa);
            let hg = GlickoPlayer::new(
                Rating::new(hr + hfa.value()).unwrap_or_default(),
                RatingDeviation::new(self.state.home_rd).unwrap_or_default(),
                vol,
            );
            let p_gl = head_to_head_win_probability_unit(&hg, &away_g);
            self.curve_elo.push([diff, p_elo.value()]);
            self.curve_glicko.push([diff, p_gl.value()]);
        }
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        changed |= slider_f64(
            ui,
            &mut self.state.home_rating,
            1000.0..=2200.0,
            "Home Rating",
        )
        .changed();
        changed |= slider_f64(
            ui,
            &mut self.state.away_rating,
            1000.0..=2200.0,
            "Away Rating",
        )
        .changed();
        changed |= slider_f64(ui, &mut self.state.home_rd, 30.0..=350.0, "Home RD")
            .accessible_hover_text("Glicko-2 Rating Deviation uncertainty for Home team")
            .changed();
        changed |= slider_f64(ui, &mut self.state.away_rd, 30.0..=350.0, "Away RD").changed();
        changed |= slider_f64(
            ui,
            &mut self.state.home_advantage,
            0.0..=120.0,
            "Home Advantage H",
        )
        .changed();
        changed |= slider_f64(ui, &mut self.state.elo_k, 8.0..=48.0, "Elo K-Factor").changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.margin_of_victory, 0..=10)
                    .text("MoV Goal/Pt Diff"),
            )
            .changed();
        ui.separator();
        changed |= slider_f64(
            ui,
            &mut self.state.offered_home_odds,
            1.10..=10.0,
            "Book Home Odds",
        )
        .changed();
        if changed {
            self.recompute();
        }
    }

    fn render_summary(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let (ev, full_pct, half_stake) = evaluate_kelly_sizing(
            self.glicko_win,
            self.state.offered_home_odds,
            self.state.bankroll,
        );
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                palette.series[0],
                format!(
                    "Win Prob: Elo {:.1}% vs Glicko-2 {:.1}%",
                    self.elo_win.value() * 100.0,
                    self.glicko_win.value() * 100.0
                ),
            );
            ui.separator();
            ui.colored_label(
                palette.accent,
                format!(
                    "If Home Wins: Elo -> {:.0} | Glicko-2 -> {:.0} (RD {:.1})",
                    self.next_elo_home, self.next_glicko_home, self.next_glicko_rd
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
                    "Glicko EV: ${:+.3} | Full Kelly: {:.1}% (${:.2} Half)",
                    ev, full_pct, half_stake
                ),
            );
        });
    }

    fn render_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let active_diff = self.state.home_rating - self.state.away_rating;
        render_sports_plot(
            ui,
            "sports_ratings_win_curve_plot",
            "Raw Rating Differential R_Home - R_Away",
            "Home Matchup Win Probability",
            &mut |plot_ui| {
                plot_ui.vline(plot_vline(
                    "Active Rating Diff",
                    active_diff,
                    palette.accent,
                    6.0,
                ));
                plot_ui.line(plot_line(
                    "Elo Logistic Win Curve (Fixed Scale)",
                    &self.curve_elo,
                    palette.series[1],
                    2.0,
                    Some(7.0),
                ));
                plot_ui.line(plot_line(
                    "Glicko-2 RD-Attenuated Win Curve",
                    &self.curve_glicko,
                    palette.series[0],
                    2.5,
                    None,
                ));
            },
        );
    }
}

impl InteractiveTool for SportsRatingsTool {
    fn name(&self) -> &'static str {
        "Elo & Glicko-2 Ratings"
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
        self.state.home_rating = self.state.home_rating.clamp(800.0, 2600.0);
        self.state.away_rating = self.state.away_rating.clamp(800.0, 2600.0);
        self.state.home_rd = self.state.home_rd.clamp(25.0, 350.0);
        self.state.away_rd = self.state.away_rd.clamp(25.0, 350.0);
        self.state.volatility = self.state.volatility.clamp(0.01, 0.25);
        self.state.home_advantage = self.state.home_advantage.clamp(0.0, 150.0);
        self.state.elo_k = self.state.elo_k.clamp(4.0, 64.0);
        self.state.margin_of_victory = self.state.margin_of_victory.min(15);
        self.state.offered_home_odds = self.state.offered_home_odds.clamp(1.05, 25.0);
        self.state.bankroll = self.state.bankroll.clamp(10.0, 1_000_000.0);
        self.recompute();
        true
    }
    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }
    fn try_load_state(&mut self, json: &str) -> Result<(), String> {
        self.state_bounds().validate_json(json)?;
        let parsed: SportsRatingsState = serde_json::from_value(super::parse_state_value(json)?)
            .map_err(|e| format!("Invalid SportsRatingsState: {e}"))?;
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
                "Glicko-2 RD-Attenuated Win Curve".into(),
                self.curve_glicko.clone(),
            ),
            ("Elo Logistic Win Curve".into(), self.curve_elo.clone()),
        ]
    }
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        render_sports_left_panel(
            parent_ui,
            "sports_ratings_side",
            "Elo & Glicko-2 Ratings",
            &mut |ui| self.render_controls(ui),
        );
        render_sports_central_panel(parent_ui, &mut |ui, palette| {
            self.render_summary(ui, palette);
            ui.separator();
            self.render_plot(ui, palette);
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for SportsRatingsTool {
    fn theory_description(&self) -> String {
        "Elo & Glicko-2 paired-comparison ratings: contrasts Elo logistic expectation $E_A = (1 + 10^{-(R_A + H - R_B)/400})^{-1}$ with Margin-of-Victory autocorrelation scaling against Bayesian Glicko-2 uncertainty attenuation $g(\\sqrt{\\phi_A^2 + \\phi_B^2})$.".into()
    }
    fn phonetic_description(&self) -> String {
        "Elo margin of victory and Bayesian Glicko two sports rating comparison".into()
    }
    fn theory_citation(&self) -> String {
        "Parameter estimation in large dynamic paired comparison experiments (Glickman, 1999); The Rating of Chessplayers, Past and Present (Elo, 1978)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    ToolMetadata {
        id: "SportsRatingsTool",
        display_name: "Elo & Glicko-2 Ratings",
        domain: "sports_analytics",
        tags: &["elo", "glicko2", "ratings", "matchup", "kelly"],
        build: || Box::new(SportsRatingsTool::default()),
    }
}

// [cite:sports_quantitative_modeling]
