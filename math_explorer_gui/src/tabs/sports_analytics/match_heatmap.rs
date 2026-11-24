#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    evaluate_kelly_sizing, plot_line, plot_vline, render_sports_central_panel,
    render_sports_left_panel, render_sports_plot, slider_f64, slider_f64_step,
};
use crate::accessibility::{
    paint_canvas_focus_ring, register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::SemanticPalette;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata};
use eframe::egui;
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::sports_quant::{
    asian_handicap_home_cover, bivariate_poisson_matrix_into, both_teams_to_score_probability,
    dixon_coles_matrix_into, independent_poisson_matrix_into, match_1x2_probabilities,
    over_under_probabilities, DixonColesRho, MatchOutcomeProbs, PoissonRate, SharedGoalRate,
    UnitInterval,
};
use serde::{Deserialize, Serialize};

const MAX_GOALS: usize = 7;
const MATRIX_DIM: usize = MAX_GOALS + 1;
const MATRIX_LEN: usize = MATRIX_DIM * MATRIX_DIM;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreMatrixVariant {
    IndependentPoisson,
    DixonColes,
    BivariatePoisson,
}

/// Bounded URL/preset state for `MatchScoreHeatmapTool`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchHeatmapState {
    pub variant: ScoreMatrixVariant,
    pub lambda_home: f64,
    pub lambda_away: f64,
    pub dixon_coles_rho: f64,
    pub shared_lambda3: f64,
    pub goal_line: f64,
    pub handicap_line: f64,
    pub offered_home_odds: f64,
    pub bankroll: f64,
}

impl Default for MatchHeatmapState {
    fn default() -> Self {
        Self {
            variant: ScoreMatrixVariant::DixonColes,
            lambda_home: 1.62,
            lambda_away: 1.18,
            dixon_coles_rho: -0.12,
            shared_lambda3: 0.16,
            goal_line: 2.5,
            handicap_line: -0.5,
            offered_home_odds: 2.15,
            bankroll: 1000.0,
        }
    }
}

impl MatchHeatmapState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let vals = [
            self.lambda_home,
            self.lambda_away,
            self.dixon_coles_rho,
            self.shared_lambda3,
            self.goal_line,
            self.handicap_line,
            self.offered_home_odds,
            self.bankroll,
        ];
        if !vals.iter().all(|v| v.is_finite()) {
            return Err("Match score parameters must be finite".into());
        }
        let ok = (0.15..=4.5).contains(&self.lambda_home)
            && (0.15..=4.5).contains(&self.lambda_away)
            && (-0.22..=0.22).contains(&self.dixon_coles_rho)
            && (0.0..=1.2).contains(&self.shared_lambda3)
            && (0.5..=6.5).contains(&self.goal_line)
            && (-3.5..=3.5).contains(&self.handicap_line)
            && (1.05..=25.0).contains(&self.offered_home_odds)
            && (10.0..=1_000_000.0).contains(&self.bankroll);
        if !ok {
            return Err("Match score parameters out of bounds".into());
        }
        Ok(())
    }
}

/// Interactive Match Score Heatmap tool for Poisson, Dixon-Coles, and Karlis-Ntzoufras matrices.
pub struct MatchScoreHeatmapTool {
    pub(super) state: MatchHeatmapState,
    revision: u64,
    matrix: [f64; MATRIX_LEN],
    outcomes_1x2: MatchOutcomeProbs,
    over_prob: UnitInterval,
    btts_prob: UnitInterval,
    ah_cover: UnitInterval,
    total_goals_active: Vec<[f64; 2]>,
    total_goals_indep: Vec<[f64; 2]>,
}

impl Default for MatchScoreHeatmapTool {
    fn default() -> Self {
        let neutral = UnitInterval::new_clamped(0.0);
        let mut tool = Self {
            state: MatchHeatmapState::default(),
            revision: 1,
            matrix: [0.0; MATRIX_LEN],
            outcomes_1x2: MatchOutcomeProbs {
                home_win: neutral,
                draw: neutral,
                away_win: neutral,
            },
            over_prob: neutral,
            btts_prob: neutral,
            ah_cover: neutral,
            total_goals_active: Vec::new(),
            total_goals_indep: Vec::new(),
        };
        tool.recompute();
        tool
    }
}

impl MatchScoreHeatmapTool {
    pub(super) fn recompute(&mut self) {
        let lh = PoissonRate::new(self.state.lambda_home)
            .unwrap_or_else(|_| PoissonRate::new(1.5).unwrap());
        let la = PoissonRate::new(self.state.lambda_away)
            .unwrap_or_else(|_| PoissonRate::new(1.2).unwrap());
        let mut indep = [0.0_f64; MATRIX_LEN];
        let _ = independent_poisson_matrix_into(lh, la, MAX_GOALS, &mut indep);
        match self.state.variant {
            ScoreMatrixVariant::IndependentPoisson => self.matrix = indep,
            ScoreMatrixVariant::DixonColes => {
                let bound = (1.0 / (self.state.lambda_home * self.state.lambda_away)).min(0.22);
                let rho_v = self
                    .state
                    .dixon_coles_rho
                    .clamp(-bound + 1e-4, bound - 1e-4);
                if let Ok(rho) = DixonColesRho::new(rho_v) {
                    if dixon_coles_matrix_into(lh, la, rho, MAX_GOALS, &mut self.matrix).is_err() {
                        self.matrix = indep;
                    }
                }
            }
            ScoreMatrixVariant::BivariatePoisson => {
                let l3 = SharedGoalRate::new(self.state.shared_lambda3)
                    .unwrap_or_else(|_| SharedGoalRate::new(0.1).unwrap());
                if bivariate_poisson_matrix_into(lh, la, l3, MAX_GOALS, &mut self.matrix).is_err() {
                    self.matrix = indep;
                }
            }
        }
        self.update_derived_markets(&indep);
        self.revision = self.revision.wrapping_add(1);
    }

    fn update_derived_markets(&mut self, indep: &[f64; MATRIX_LEN]) {
        if let Ok(m1x2) = match_1x2_probabilities(&self.matrix, MAX_GOALS) {
            self.outcomes_1x2 = m1x2;
        }
        if let Ok((over, _)) =
            over_under_probabilities(&self.matrix, MAX_GOALS, self.state.goal_line)
        {
            self.over_prob = over;
        }
        if let Ok(btts) = both_teams_to_score_probability(&self.matrix, MAX_GOALS) {
            self.btts_prob = btts;
        }
        if let Ok(ah) = asian_handicap_home_cover(&self.matrix, MAX_GOALS, self.state.handicap_line)
        {
            self.ah_cover = ah;
        }
        self.total_goals_active.clear();
        self.total_goals_indep.clear();
        for total in 0..=(2 * MAX_GOALS) {
            let (mut s_act, mut s_ind) = (0.0, 0.0);
            for h in 0..=MAX_GOALS {
                for a in 0..=MAX_GOALS {
                    if h + a == total {
                        let idx = flatten_2d_index(h, a, MATRIX_DIM);
                        s_act += self.matrix[idx];
                        s_ind += indep[idx];
                    }
                }
            }
            self.total_goals_active.push([total as f64, s_act]);
            self.total_goals_indep.push([total as f64, s_ind]);
        }
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        ui.label("Goal Matrix Model:");
        for (v, lbl) in [
            (
                ScoreMatrixVariant::IndependentPoisson,
                "Independent Poisson",
            ),
            (ScoreMatrixVariant::DixonColes, "Dixon-Coles (Low-Score τ)"),
            (
                ScoreMatrixVariant::BivariatePoisson,
                "Karlis-Ntzoufras (Shared λ₃)",
            ),
        ] {
            changed |= ui.radio_value(&mut self.state.variant, v, lbl).changed();
        }
        ui.separator();
        changed |=
            slider_f64(ui, &mut self.state.lambda_home, 0.25..=4.0, "Home xG (λ₁)").changed();
        changed |=
            slider_f64(ui, &mut self.state.lambda_away, 0.25..=4.0, "Away xG (λ₂)").changed();
        if self.state.variant == ScoreMatrixVariant::DixonColes {
            changed |= slider_f64(
                ui,
                &mut self.state.dixon_coles_rho,
                -0.20..=0.20,
                "Dixon-Coles ρ",
            )
            .accessible_hover_text("Low-score correlation adjustment on (0,0), (1,0), (0,1), (1,1)")
            .changed();
        } else if self.state.variant == ScoreMatrixVariant::BivariatePoisson {
            changed |= slider_f64(
                ui,
                &mut self.state.shared_lambda3,
                0.0..=1.0,
                "Shared Tempo λ₃",
            )
            .changed();
        }
        ui.separator();
        changed |=
            slider_f64_step(ui, &mut self.state.goal_line, 0.5..=5.5, 0.5, "Total Line").changed();
        changed |= slider_f64_step(
            ui,
            &mut self.state.handicap_line,
            -2.5..=2.5,
            0.5,
            "Home AH",
        )
        .changed();
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
            self.outcomes_1x2.home_win,
            self.state.offered_home_odds,
            self.state.bankroll,
        );
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                palette.series[0],
                format!(
                    "1X2: H {:.1}% | D {:.1}% | A {:.1}%",
                    self.outcomes_1x2.home_win.value() * 100.0,
                    self.outcomes_1x2.draw.value() * 100.0,
                    self.outcomes_1x2.away_win.value() * 100.0
                ),
            );
            ui.separator();
            ui.colored_label(
                palette.accent,
                format!(
                    "Over {:.1}: {:.1}% | BTTS: {:.1}% | AH {:+.1}: {:.1}%",
                    self.state.goal_line,
                    self.over_prob.value() * 100.0,
                    self.btts_prob.value() * 100.0,
                    self.state.handicap_line,
                    self.ah_cover.value() * 100.0
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
                    "Home EV: ${:+.3} | Full Kelly: {:.1}% (${:.2} Half)",
                    ev, full_pct, half_stake
                ),
            );
        });
    }

    fn render_heatmap_grid(&mut self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let side = ui.available_width().clamp(150.0, 220.0);
        let (response, painter) = ui.allocate_painter(
            egui::Vec2::new(side, side),
            egui::Sense::focusable_noninteractive(),
        );
        let max_p = self.matrix.iter().copied().fold(1e-6_f64, f64::max);
        let cell = side / (MATRIX_DIM as f32);
        let dark = ui.visuals().dark_mode;
        for h in 0..MATRIX_DIM {
            for a in 0..MATRIX_DIM {
                let p = self.matrix[flatten_2d_index(h, a, MATRIX_DIM)];
                let t = (p / max_p).clamp(0.0, 1.0);
                let rgb = if dark {
                    heatmap_color_high_contrast(t)
                } else {
                    heatmap_color(t)
                };
                let min = egui::pos2(
                    response.rect.left() + (a as f32) * cell,
                    response.rect.top() + (h as f32) * cell,
                );
                let rect = egui::Rect::from_min_size(min, egui::vec2(cell - 1.0, cell - 1.0));
                painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
                if h <= 3 && a <= 3 && cell >= 20.0 {
                    painter.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        format!("{:.0}", p * 100.0),
                        egui::FontId::proportional(9.5),
                        palette.surface_fill,
                    );
                }
            }
        }
        let desc = format!(
            "8x8 score matrix (rows Home 0..7, cols Away 0..7), Draw {:.1}%",
            self.outcomes_1x2.draw.value() * 100.0
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "Match score probability heatmap",
            Some(&desc),
        );
        paint_canvas_focus_ring(ui, &response);
    }

    fn render_goals_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        render_sports_plot(
            ui,
            "match_total_goals_plot",
            "Total Match Goals (H + A)",
            "Probability Mass P(H + A = k)",
            &mut |plot_ui| {
                plot_ui.vline(plot_vline(
                    "Over/Under Line",
                    self.state.goal_line,
                    palette.accent,
                    6.0,
                ));
                plot_ui.line(plot_line(
                    "Selected Model Total Goals PMF",
                    &self.total_goals_active,
                    palette.series[0],
                    2.5,
                    None,
                ));
                plot_ui.line(plot_line(
                    "Independent Poisson Baseline",
                    &self.total_goals_indep,
                    palette.series[1],
                    1.9,
                    Some(6.0),
                ));
            },
        );
    }
}

impl InteractiveTool for MatchScoreHeatmapTool {
    fn name(&self) -> &'static str {
        "Match Score Heatmap"
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
        self.state.lambda_home = self.state.lambda_home.clamp(0.15, 4.5);
        self.state.lambda_away = self.state.lambda_away.clamp(0.15, 4.5);
        self.state.dixon_coles_rho = self.state.dixon_coles_rho.clamp(-0.22, 0.22);
        self.state.shared_lambda3 = self.state.shared_lambda3.clamp(0.0, 1.2);
        self.state.goal_line = self.state.goal_line.clamp(0.5, 6.5);
        self.state.handicap_line = self.state.handicap_line.clamp(-3.5, 3.5);
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
        let parsed: MatchHeatmapState = serde_json::from_value(super::parse_state_value(json)?)
            .map_err(|e| format!("Invalid MatchHeatmapState: {e}"))?;
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
                "Selected Model Total Goals PMF".into(),
                self.total_goals_active.clone(),
            ),
            (
                "Independent Poisson Baseline PMF".into(),
                self.total_goals_indep.clone(),
            ),
        ]
    }
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        render_sports_left_panel(
            parent_ui,
            "match_heatmap_side",
            "Match Score Heatmap",
            &mut |ui| self.render_controls(ui),
        );
        render_sports_central_panel(parent_ui, &mut |ui, palette| {
            self.render_summary(ui, palette);
            ui.separator();
            ui.columns(2, |cols| {
                cols[0].small("Joint Score Grid P(H = i, A = j) (% cells 0..3):");
                self.render_heatmap_grid(&mut cols[0], palette);
                self.render_goals_plot(&mut cols[1], palette);
            });
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for MatchScoreHeatmapTool {
    fn theory_description(&self) -> String {
        "Bivariate match score matrices: compares Independent Poisson against Dixon-Coles low-score dependence $\\tau_{\\lambda_1,\\lambda_2}(x,y;\\rho)$ on $\\{0,1\\}^2$ and Karlis-Ntzoufras shared-tempo Bivariate Poisson $(\\lambda_1, \\lambda_2, \\lambda_3)$ to price 1X2, Over/Under, BTTS, and Asian Handicap markets.".into()
    }
    fn phonetic_description(&self) -> String {
        "Dixon Coles and bivariate Poisson match score matrix heatmap and derivative market pricer"
            .into()
    }
    fn theory_citation(&self) -> String {
        "Modelling Association Football Scores and Inefficiencies in the Football Betting Market (Dixon & Coles, 1997); Analysis of sports data by using bivariate Poisson models (Karlis & Ntzoufras, 2003)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    ToolMetadata {
        id: "MatchScoreHeatmapTool",
        display_name: "Match Score Heatmap",
        domain: "sports_analytics",
        tags: &["poisson", "dixon_coles", "soccer", "heatmap", "kelly"],
        build: || Box::new(MatchScoreHeatmapTool::default()),
    }
}

// [cite:sports_quantitative_modeling]
