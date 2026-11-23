//! Central plot series, async file dialog state, and trait/inventory registration for [`super::Glicko2TournamentTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{Glicko2PlotMode, Glicko2TournamentTool};
use crate::app::SemanticPalette;
use crate::framework::InteractiveTool;
use crate::presets::{Glicko2TournamentPresetConfig, MAX_TOURNAMENT_PERIODS};
use crate::web_io::{
    spawn_open_file_dialog, spawn_save_as_file_dialog, OpenDialogConfig, SaveAsDialogConfig,
};
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints, Points};
use math_explorer::pure_math::statistics::glicko2::{pairwise_win_matrix, GlickoPlayer};
use std::sync::mpsc::{channel, Receiver};

const MATCH_LOG_OPEN_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "Match Log or Preset (.csv, .json)",
    extensions: &["csv", "json"],
    accept: ".csv,.json,text/csv,application/json",
    read_error_prefix: "Failed to read tournament file",
};

const CSV_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "CSV Spreadsheet (.csv)",
    extensions: &["csv"],
    fallback_filename: "glicko2_export.csv",
    write_error_prefix: "Failed to write CSV export",
};

const JSON_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "JSON Document (.json)",
    extensions: &["json"],
    fallback_filename: "glicko2_export.json",
    write_error_prefix: "Failed to write JSON export",
};

#[derive(Default)]
pub(crate) struct TournamentIoState {
    open_rx: Option<Receiver<Result<(String, String), String>>>,
    save_rx: Option<Receiver<Result<String, String>>>,
    pub status_message: Option<(bool, String)>,
}

impl TournamentIoState {
    pub(crate) fn trigger_import_dialog(&mut self) {
        let (tx, rx) = channel();
        self.open_rx = Some(rx);
        spawn_open_file_dialog(tx, MATCH_LOG_OPEN_CONFIG, |p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("match_log")
                .to_string()
        });
    }

    pub(crate) fn trigger_save_csv(&mut self, default_filename: &str, content: String) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        spawn_save_as_file_dialog(tx, CSV_SAVE_CONFIG, default_filename, content, |p, fb| {
            p.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(fb)
                .to_string()
        });
    }

    pub(crate) fn trigger_save_json(
        &mut self,
        default_filename: &str,
        content: Result<String, String>,
    ) {
        match content {
            Ok(payload) => {
                let (tx, rx) = channel();
                self.save_rx = Some(rx);
                spawn_save_as_file_dialog(
                    tx,
                    JSON_SAVE_CONFIG,
                    default_filename,
                    payload,
                    |p, fb| {
                        p.file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or(fb)
                            .to_string()
                    },
                );
            }
            Err(err) => self.status_message = Some((true, err)),
        }
    }
}

impl Glicko2TournamentTool {
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

impl InteractiveTool for Glicko2TournamentTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Glicko-2 Tournament"
    }
    fn domain(&self) -> &'static str {
        "game_theory"
    }
    fn state_revision(&self) -> u64 {
        self.state_rev
    }

    fn has_finite_parameters(&self) -> bool {
        self.tau.is_finite()
            && self.draw_probability.is_finite()
            && self.competitors.iter().all(|c| {
                c.initial.rating.value().is_finite()
                    && c.initial.rating_deviation.value().is_finite()
                    && c.initial.volatility.value().is_finite()
            })
    }

    fn clamp_parameters(&mut self) -> bool {
        let tau = if self.tau.is_finite() {
            self.tau.clamp(0.3, 1.2)
        } else {
            0.5
        };
        let draw = if self.draw_probability.is_finite() {
            self.draw_probability.clamp(0.0, 1.0)
        } else {
            0.15
        };
        let changed = (tau - self.tau).abs() > f64::EPSILON
            || (draw - self.draw_probability).abs() > f64::EPSILON;
        (self.tau, self.draw_probability) = (tau, draw);
        self.target_rounds = self.target_rounds.clamp(1, MAX_TOURNAMENT_PERIODS);
        if changed {
            let _ = self.rebuild_trajectory();
        }
        changed
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.export_config()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let cfg: Glicko2TournamentPresetConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Glicko2TournamentTool state: {e}"))?;
        self.apply_config(cfg)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let ctx = parent_ui.ctx().clone();
        self.poll_io(&ctx);
        let side_id = crate::framework::scoped_panel_id(parent_ui, "glicko2_control_panel");
        crate::framework::PaneSidePanel::left(side_id).show_inside(parent_ui, |ui| {
            self.show_side_controls(ui);
        });
        self.show_bottom_dock(parent_ui);
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.show_central_plot(ui);
        });
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
