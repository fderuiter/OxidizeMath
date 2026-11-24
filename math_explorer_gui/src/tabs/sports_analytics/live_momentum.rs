#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    evaluate_kelly_sizing, plot_hline, plot_line, render_sports_central_panel,
    render_sports_left_panel, render_sports_plot, slider_f64,
};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata};
use eframe::egui;
use math_explorer::rng::OxidizeRng;
use math_explorer::sports_quant::{
    EulerMaruyama, MomentumType, OuAnalyzer, OuParams, TimeStep, UnitInterval,
};
use serde::{Deserialize, Serialize};

/// Bounded URL/preset state for `LiveMomentumTool`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveMomentumState {
    pub current_margin: f64,
    pub pregame_spread_mu: f64,
    pub reversion_theta: f64,
    pub volatility_sigma: f64,
    pub remaining_time: f64,
    pub offered_live_odds: f64,
    pub rng_seed: u64,
    pub bankroll: f64,
}

impl Default for LiveMomentumState {
    fn default() -> Self {
        Self {
            current_margin: -5.0,
            pregame_spread_mu: 3.5,
            reversion_theta: 1.6,
            volatility_sigma: 7.5,
            remaining_time: 0.50,
            offered_live_odds: 2.35,
            rng_seed: 2026,
            bankroll: 1000.0,
        }
    }
}

impl LiveMomentumState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let vals = [
            self.current_margin,
            self.pregame_spread_mu,
            self.reversion_theta,
            self.volatility_sigma,
            self.remaining_time,
            self.offered_live_odds,
            self.bankroll,
        ];
        if !vals.iter().all(|v| v.is_finite()) {
            return Err("Live momentum parameters must be finite".into());
        }
        let ok = (-30.0..=30.0).contains(&self.current_margin)
            && (-20.0..=20.0).contains(&self.pregame_spread_mu)
            && (0.10..=6.0).contains(&self.reversion_theta)
            && (1.0..=25.0).contains(&self.volatility_sigma)
            && (0.05..=1.0).contains(&self.remaining_time)
            && (1.05..=25.0).contains(&self.offered_live_odds)
            && (10.0..=1_000_000.0).contains(&self.bankroll);
        if !ok {
            return Err("Live momentum parameters out of bounds".into());
        }
        Ok(())
    }
}

/// Interactive Ornstein-Uhlenbeck in-play score-differential SDE and comeback win pricer.
pub struct LiveMomentumTool {
    pub(super) state: LiveMomentumState,
    revision: u64,
    comeback_prob: UnitInterval,
    regime: MomentumType,
    mean_curve: Vec<[f64; 2]>,
    sample_paths: [Vec<[f64; 2]>; 3],
}

impl Default for LiveMomentumTool {
    fn default() -> Self {
        let mut tool = Self {
            state: LiveMomentumState::default(),
            revision: 1,
            comeback_prob: UnitInterval::new_clamped(0.45),
            regime: MomentumType::Normal,
            mean_curve: Vec::new(),
            sample_paths: [Vec::new(), Vec::new(), Vec::new()],
        };
        tool.recompute();
        tool
    }
}

impl LiveMomentumTool {
    pub(super) fn recompute(&mut self) {
        let Ok(params) = OuParams::from_values(
            self.state.pregame_spread_mu,
            self.state.reversion_theta,
            self.state.volatility_sigma,
        ) else {
            return;
        };
        let n_steps = 60usize;
        let dt_val = (self.state.remaining_time / (n_steps as f64)).max(1e-4);
        let Ok(dt) = TimeStep::new(dt_val) else {
            return;
        };
        let analyzer = OuAnalyzer::new(params, dt);
        self.regime = MomentumType::classify(self.state.reversion_theta);
        let mut rng = OxidizeRng::new(self.state.rng_seed);
        self.comeback_prob = analyzer.comeback_probability_unit(
            self.state.current_margin,
            0.0,
            self.state.remaining_time,
            800,
            &mut rng,
        );

        let solver = EulerMaruyama::new(params, dt);
        self.mean_curve.clear();
        for k in 0..=n_steps {
            let t = (k as f64) * dt_val;
            let mean_t = self.state.pregame_spread_mu
                + (self.state.current_margin - self.state.pregame_spread_mu)
                    * (-self.state.reversion_theta * t).exp();
            self.mean_curve.push([t, mean_t]);
        }
        for path_buf in &mut self.sample_paths {
            path_buf.clear();
            let raw = solver.simulate(self.state.current_margin, n_steps, &mut rng);
            for (k, &val) in raw.iter().enumerate() {
                path_buf.push([(k as f64) * dt_val, val]);
            }
        }
        self.revision = self.revision.wrapping_add(1);
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        changed |= slider_f64(
            ui,
            &mut self.state.current_margin,
            -20.0..=20.0,
            "Live Score Diff X₀",
        )
        .accessible_hover_text("Current live score margin (negative = trailing underdog)")
        .changed();
        changed |= slider_f64(
            ui,
            &mut self.state.pregame_spread_mu,
            -15.0..=15.0,
            "Pre-Game Mean μ",
        )
        .changed();
        changed |= slider_f64(
            ui,
            &mut self.state.reversion_theta,
            0.15..=5.0,
            "Reversion Speed θ",
        )
        .changed();
        changed |= slider_f64(
            ui,
            &mut self.state.volatility_sigma,
            1.5..=20.0,
            "Score Volatility σ",
        )
        .changed();
        changed |= slider_f64(
            ui,
            &mut self.state.remaining_time,
            0.05..=1.0,
            "Clock Remaining τ",
        )
        .changed();
        ui.separator();
        changed |= slider_f64(
            ui,
            &mut self.state.offered_live_odds,
            1.10..=15.0,
            "Live Moneyline Odds",
        )
        .changed();
        if ui.button("Reseed Trajectory Paths").clicked() {
            self.state.rng_seed = self.state.rng_seed.wrapping_add(17);
            changed = true;
        }
        if changed {
            self.recompute();
        }
    }

    fn render_summary(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let (ev, full_pct, half_stake) = evaluate_kelly_sizing(
            self.comeback_prob,
            self.state.offered_live_odds,
            self.state.bankroll,
        );
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                palette.series[0],
                format!(
                    "Comeback P(X_T > 0): {:.1}% | Regime: {:?}",
                    self.comeback_prob.value() * 100.0,
                    self.regime
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
                    "Live EV @ {:.2}: ${:+.3} | Full Kelly: {:.1}% | Half-Kelly: ${:.2}",
                    self.state.offered_live_odds, ev, full_pct, half_stake
                ),
            );
        });
    }

    fn render_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        render_sports_plot(
            ui,
            "live_ou_momentum_plot",
            "Elapsed Remaining Regulation Time t",
            "Live Score Differential X(t)",
            &mut |plot_ui| {
                plot_ui.hline(plot_hline(
                    "Tie Game Threshold (X = 0)",
                    0.0,
                    palette.warning,
                    Some(5.0),
                ));
                plot_ui.line(plot_line(
                    "Analytical Mean Drift E[X(t)]",
                    &self.mean_curve,
                    palette.accent,
                    2.6,
                    None,
                ));
                for (idx, path) in self.sample_paths.iter().enumerate() {
                    let label = format!("Euler-Maruyama Sample Path {}", idx + 1);
                    plot_ui.line(plot_line(
                        &label,
                        path,
                        palette.series_color(idx),
                        1.5,
                        None,
                    ));
                }
            },
        );
    }
}

impl InteractiveTool for LiveMomentumTool {
    fn name(&self) -> &'static str {
        "Live Momentum (OU)"
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
        let d = LiveMomentumState::default();
        self.state.current_margin =
            super::clamp_finite(self.state.current_margin, -30.0, 30.0, d.current_margin);
        self.state.pregame_spread_mu = super::clamp_finite(
            self.state.pregame_spread_mu,
            -20.0,
            20.0,
            d.pregame_spread_mu,
        );
        self.state.reversion_theta =
            super::clamp_finite(self.state.reversion_theta, 0.10, 6.0, d.reversion_theta);
        self.state.volatility_sigma =
            super::clamp_finite(self.state.volatility_sigma, 1.0, 25.0, d.volatility_sigma);
        self.state.remaining_time =
            super::clamp_finite(self.state.remaining_time, 0.05, 1.0, d.remaining_time);
        self.state.offered_live_odds = super::clamp_finite(
            self.state.offered_live_odds,
            1.05,
            25.0,
            d.offered_live_odds,
        );
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
        let parsed: LiveMomentumState = serde_json::from_value(super::parse_state_value(json)?)
            .map_err(|e| format!("Invalid LiveMomentumState: {e}"))?;
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
                "Analytical Mean Drift E[X(t)]".into(),
                self.mean_curve.clone(),
            ),
            ("Sample Trajectory 1".into(), self.sample_paths[0].clone()),
            ("Sample Trajectory 2".into(), self.sample_paths[1].clone()),
        ]
    }
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        render_sports_left_panel(
            parent_ui,
            "live_momentum_side",
            "Live Momentum (OU)",
            &mut |ui| self.render_controls(ui),
        );
        render_sports_central_panel(parent_ui, &mut |ui, palette| {
            self.render_summary(ui, palette);
            ui.separator();
            self.render_plot(ui, palette);
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for LiveMomentumTool {
    fn theory_description(&self) -> String {
        "Ornstein-Uhlenbeck in-play momentum SDE: models live score differential $dX_t = \\theta(\\mu - X_t)\\,dt + \\sigma\\,dW_t$ reverting toward pre-game spread $\\mu$ at rate $\\theta$, estimating comeback win probability $P(X_T > 0 \\mid X_0)$ via Euler-Maruyama integration for live Kelly sizing.".into()
    }
    fn phonetic_description(&self) -> String {
        "Ornstein Uhlenbeck live sports momentum stochastic differential equation and comeback probability".into()
    }
    fn theory_citation(&self) -> String {
        "Optimal Gambling Systems for Favorable Games (Thorp, 1969); A New Interpretation of Information Rate (Kelly, 1956)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    ToolMetadata {
        id: "LiveMomentumTool",
        display_name: "Live Momentum (OU)",
        domain: "sports_analytics",
        tags: &["ou_process", "sde", "in_play", "momentum", "kelly"],
        build: || Box::new(LiveMomentumTool::default()),
    }
}

// [cite:sports_quantitative_modeling]
