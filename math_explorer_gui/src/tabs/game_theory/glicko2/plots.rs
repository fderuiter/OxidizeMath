//! Plot rendering, preset configuration, IO polling, and metadata for [`super::Glicko2TournamentTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CompetitorEntry, Glicko2PlotMode, Glicko2TournamentTool};
use crate::app::SemanticPalette;
use crate::presets::Glicko2TournamentPresetConfig;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints, Points};
use math_explorer::pure_math::statistics::glicko2::{
    head_to_head_win_probability, pairwise_win_matrix, GlickoPlayer, Rating, RatingDeviation,
    Volatility,
};
use std::sync::mpsc::Receiver;

impl Glicko2TournamentTool {
    pub(crate) fn apply_config(
        &mut self,
        config: Glicko2TournamentPresetConfig,
    ) -> Result<(), String> {
        config.validate()?;
        let mut competitors = Vec::with_capacity(config.players.len());
        for p in &config.players {
            let initial = GlickoPlayer::new(
                Rating::new(p.rating).map_err(|e| e.to_string())?,
                RatingDeviation::new(p.rd).map_err(|e| e.to_string())?,
                Volatility::new(p.volatility).map_err(|e| e.to_string())?,
            );
            competitors.push(CompetitorEntry::new(
                p.name.trim().to_string(),
                initial,
                p.inactive,
            ));
        }
        (self.tau, self.draw_probability, self.pairing_mode) =
            (config.tau, config.draw_probability, config.pairing_mode);
        (self.target_rounds, self.rng_seed) = (config.target_rounds, config.rng_seed);
        (self.competitors, self.completed_periods) = (competitors, config.periods);
        if self
            .focus_player
            .is_some_and(|i| i >= self.competitors.len())
        {
            self.focus_player = None;
        }
        self.rebuild_trajectory()
    }

    /// Formats upcoming round pairings as pretty-printed JSON.
    pub fn format_pairings_json(&self) -> Result<String, String> {
        let (next_period, mode) = (self.completed_periods.len() + 1, self.pairing_mode.label());
        let rows: Vec<serde_json::Value> = self
            .next_pairings
            .iter()
            .enumerate()
            .map(|(b_idx, row)| {
                let (ca, cb) = (
                    &self.competitors[row.pairing.player_a],
                    &self.competitors[row.pairing.player_b],
                );
                serde_json::json!({
                    "next_period": next_period,
                    "pairing_mode": mode,
                    "board": b_idx + 1,
                    "player_a": ca.name,
                    "rating_a": ca.current.rating.value(),
                    "player_b": cb.name,
                    "rating_b": cb.current.rating.value(),
                    "expected_score_a": head_to_head_win_probability(&ca.current, &cb.current),
                    "manual_score_a": row.manual_score_a,
                })
            })
            .collect();
        serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())
    }

    pub(crate) fn poll_io(&mut self, ctx: &egui::Context) {
        if let Some(Ok(outcome)) = self.io_state.open_rx.as_ref().map(Receiver::try_recv) {
            self.io_state.open_rx = None;
            let status = match outcome {
                Ok((name, raw)) => match self.import_match_log_str(&raw) {
                    Ok(count) => (false, format!("Imported {count} match(es) from {name}")),
                    Err(err) => (true, format!("Import rejected ({name}): {err}")),
                },
                Err(err) => (true, err),
            };
            self.io_state.status_message = Some(status);
            ctx.request_repaint();
        }
        if let Some(Ok(outcome)) = self.io_state.save_rx.as_ref().map(Receiver::try_recv) {
            self.io_state.save_rx = None;
            self.io_state.status_message = Some(match outcome {
                Ok(saved) => (false, format!("Exported tournament file: {saved}")),
                Err(err) => (true, err),
            });
            ctx.request_repaint();
        }
    }

    pub(super) fn show_central_plot(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            for (mode, label) in [
                (Glicko2PlotMode::ConfidenceBands, "r ± 2RD Confidence Bands"),
                (Glicko2PlotMode::Volatility, "σ Volatility Shifts"),
                (Glicko2PlotMode::WinMatrix, "Pairwise Win Matrix"),
                (
                    Glicko2PlotMode::SequentialDrift,
                    "Frozen vs. Sequential Drift",
                ),
            ] {
                ui.selectable_value(&mut self.plot_mode, mode, label);
            }
            ui.separator();
            egui::ComboBox::from_label("Focus")
                .selected_text(match self.focus_player {
                    Some(i) => self.competitors.get(i).map_or("All", |c| c.name.as_str()),
                    None => "All Competitors",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.focus_player, None, "All Competitors");
                    for (i, c) in self.competitors.iter().enumerate() {
                        ui.selectable_value(&mut self.focus_player, Some(i), &c.name);
                    }
                });
        });

        let indices: Vec<usize> = match self.focus_player {
            Some(idx) if idx < self.competitors.len() => vec![idx],
            _ => (0..self.competitors.len()).collect(),
        };

        let palette = SemanticPalette::from_visuals(ui.visuals());
        Plot::new("glicko2_primary_plot")
            .legend(Legend::default())
            .allow_drag(true)
            .allow_zoom(true)
            .show(ui, |plot_ui| match self.plot_mode {
                Glicko2PlotMode::ConfidenceBands => {
                    self.plot_confidence_bands(plot_ui, &indices, palette)
                }
                Glicko2PlotMode::Volatility => {
                    self.plot_volatility_series(plot_ui, &indices, palette)
                }
                Glicko2PlotMode::WinMatrix => {
                    self.plot_win_matrix_series(plot_ui, &indices, palette)
                }
                Glicko2PlotMode::SequentialDrift => {
                    self.plot_drift_series(plot_ui, &indices, palette)
                }
            });
    }

    fn plot_confidence_bands(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        indices: &[usize],
        palette: SemanticPalette,
    ) {
        for &i in indices {
            let (color, name) = (palette.series_color(i), &self.competitors[i].name);
            let mean: PlotPoints = self
                .snapshots
                .iter()
                .map(|s| [s.period as f64, s.frozen[i].rating.value()])
                .collect();
            let upper: PlotPoints = self
                .snapshots
                .iter()
                .map(|s| {
                    [
                        s.period as f64,
                        s.frozen[i].rating.value() + 2.0 * s.frozen[i].rating_deviation.value(),
                    ]
                })
                .collect();
            let lower: PlotPoints = self
                .snapshots
                .iter()
                .map(|s| {
                    [
                        s.period as f64,
                        s.frozen[i].rating.value() - 2.0 * s.frozen[i].rating_deviation.value(),
                    ]
                })
                .collect();
            plot_ui.line(
                Line::new(format!("{name} (r)"), mean)
                    .color(color)
                    .width(2.2_f32),
            );
            plot_ui.line(
                Line::new(format!("{name} (+2RD)"), upper)
                    .color(color)
                    .style(LineStyle::dashed_dense()),
            );
            plot_ui.line(
                Line::new(format!("{name} (-2RD)"), lower)
                    .color(color)
                    .style(LineStyle::dashed_dense()),
            );
        }
    }

    fn plot_volatility_series(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        indices: &[usize],
        palette: SemanticPalette,
    ) {
        for &i in indices {
            let pts: PlotPoints = self
                .snapshots
                .iter()
                .map(|s| [s.period as f64, s.frozen[i].volatility.value()])
                .collect();
            plot_ui.line(
                Line::new(format!("{} (σ)", self.competitors[i].name), pts)
                    .color(palette.series_color(i))
                    .width(2.0_f32),
            );
        }
    }

    fn plot_win_matrix_series(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        indices: &[usize],
        palette: SemanticPalette,
    ) {
        let players: Vec<GlickoPlayer> = self.competitors.iter().map(|c| c.current).collect();
        let mat = pairwise_win_matrix(&players);
        for &i in indices {
            let color = palette.series_color(i);
            let pts: Vec<[f64; 2]> = (0..self.competitors.len())
                .map(|j| [j as f64, mat[(i, j)]])
                .collect();
            plot_ui.line(
                Line::new(
                    format!("P({} beats j)", self.competitors[i].name),
                    pts.clone(),
                )
                .color(color)
                .width(2.0_f32),
            );
            plot_ui.points(
                Points::new(format!("{} vs Roster", self.competitors[i].name), pts)
                    .color(color)
                    .radius(4.5_f32),
            );
        }
    }

    fn plot_drift_series(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        indices: &[usize],
        palette: SemanticPalette,
    ) {
        for &i in indices {
            let (color, name) = (palette.series_color(i), &self.competitors[i].name);
            let frozen_pts: PlotPoints = self
                .snapshots
                .iter()
                .map(|s| [s.period as f64, s.frozen[i].rating.value()])
                .collect();
            let seq_pts: PlotPoints = self
                .snapshots
                .iter()
                .map(|s| [s.period as f64, s.sequential[i].rating.value()])
                .collect();
            plot_ui.line(
                Line::new(format!("{name} (Frozen)"), frozen_pts)
                    .color(color)
                    .width(2.2_f32),
            );
            plot_ui.line(
                Line::new(format!("{name} (Sequential)"), seq_pts)
                    .color(color)
                    .style(LineStyle::dashed_loose()),
            );
        }
    }
}

impl scientific_metadata::theory::TheoryDescribable for Glicko2TournamentTool {
    fn theory_description(&self) -> String {
        "Multi-competitor Bayesian skill estimation under Mark Glickman's Glicko-2 system: each rating period freezes pre-period ratings $(\\mu_j, \\phi_j)$, updates volatility $\\sigma'$ via Illinois root-finding, inflates deviation $\\phi^* = \\sqrt{\\phi^2 + \\sigma'^2}$, and computes posterior $(\\mu', \\phi')$ with 95% confidence interval $r \\pm 2\\,\\text{RD}$.".into()
    }
    fn phonetic_description(&self) -> String {
        "Glicko two Bayesian tournament rating simulator with rating deviation confidence bands and volatility".into()
    }
    fn theory_citation(&self) -> String {
        "Parameter estimation in large dynamic paired comparison experiments (Glickman, Applied Statistics, 1999); Example of the Glicko-2 system (Glickman, 2012)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::from([(
            "tau".to_string(),
            "System constant constraining log-volatility change per period (0.3 to 1.2)."
                .to_string(),
        )])
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "Glicko2TournamentTool",
        display_name: "Glicko-2 Tournament",
        domain: "game_theory",
        tags: &["glicko2", "ratings", "swiss", "tournament", "bayesian"],
        build: || Box::new(Glicko2TournamentTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "glicko2_swiss_upset",
        title: "Swiss Upset & Volatility Spike",
        description: "r ± 2RD & σ': Five-player Swiss tournament where an underdog prodigy triggers a volatility surge.",
        category: "Game Theory",
        icon: "♟️",
        domain: "game_theory",
        tool_name: Some("Glicko2TournamentTool"),
        preset_generator: || crate::starter_gallery::PRESET_GLICKO2_SWISS_UPSET,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "glicko2_round_robin_drift",
        title: "Round-Robin Frozen vs. Sequential Drift",
        description: "Δr = r_seq - r_frozen: Quantifies order-dependent rating drift vs. canonical frozen periods.",
        category: "Game Theory",
        icon: "🎯",
        domain: "game_theory",
        tool_name: Some("Glicko2TournamentTool"),
        preset_generator: || crate::starter_gallery::PRESET_GLICKO2_ROUND_ROBIN_DRIFT,
    }
}
