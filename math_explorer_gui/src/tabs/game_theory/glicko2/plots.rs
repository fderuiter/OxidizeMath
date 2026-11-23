//! Central trajectory, volatility, pairwise win-matrix, and drift plots for [`super::Glicko2TournamentTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{Glicko2PlotMode, Glicko2TournamentTool};
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints, Points};
use math_explorer::pure_math::statistics::glicko2::{pairwise_win_matrix, GlickoPlayer};

impl Glicko2TournamentTool {
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
