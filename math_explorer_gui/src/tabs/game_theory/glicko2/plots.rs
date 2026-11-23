//! Docked standings/pairings/win-matrix tables and central `egui_plot` panels for [`super::Glicko2TournamentTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    BottomViewTab, CompetitorEntry, Glicko2PlotMode, Glicko2TournamentTool, ScheduledPairingRow,
};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints, Points};
use math_explorer::pure_math::statistics::glicko2::{
    head_to_head_win_probability, pairwise_win_matrix, GlickoPlayer,
};

fn show_pairing_board_row(
    ui: &mut egui::Ui,
    board_idx: usize,
    row: &mut ScheduledPairingRow,
    competitors: &[CompetitorEntry],
) {
    let (ca, cb) = (
        &competitors[row.pairing.player_a],
        &competitors[row.pairing.player_b],
    );
    let win_p = head_to_head_win_probability(&ca.current, &cb.current);
    ui.label(format!("#{}", board_idx + 1));
    ui.label(format!("{} ({:.0})", ca.name, ca.current.rating.value()));
    ui.label(format!("{} ({:.0})", cb.name, cb.current.rating.value()));
    ui.label(format!("{:.1}%", win_p * 100.0));
    ui.horizontal(|ui| {
        for (val, lbl) in [(1.0, "1-0"), (0.5, "½-½"), (0.0, "0-1")] {
            if ui
                .selectable_label((row.manual_score_a - val).abs() < 0.05, lbl)
                .clicked()
            {
                row.manual_score_a = val;
            }
        }
    });
    ui.end_row();
}

impl Glicko2TournamentTool {
    pub(super) fn show_bottom_dock(&mut self, parent_ui: &mut egui::Ui) {
        let dock_id = crate::framework::scoped_panel_id(parent_ui, "glicko2_bottom_dock");
        egui::TopBottomPanel::bottom(dock_id)
            .resizable(true)
            .min_height(185.0)
            .show_inside(parent_ui, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut self.bottom_tab,
                        BottomViewTab::Standings,
                        "🏆 Standings & Drift",
                    );
                    ui.selectable_value(
                        &mut self.bottom_tab,
                        BottomViewTab::NextPairings,
                        format!("⚔ Period {} Pairings", self.completed_periods.len() + 1),
                    );
                    ui.selectable_value(
                        &mut self.bottom_tab,
                        BottomViewTab::WinMatrixTable,
                        "🎯 Win Matrix",
                    );
                });
                ui.separator();
                egui::ScrollArea::both().show(ui, |ui| match self.bottom_tab {
                    BottomViewTab::Standings => self.show_standings_table(ui),
                    BottomViewTab::NextPairings => self.show_pairings_table(ui),
                    BottomViewTab::WinMatrixTable => self.show_win_matrix_table(ui),
                });
            });
    }

    fn show_standings_table(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        egui::Grid::new("glicko2_standings_grid")
            .striped(true)
            .min_col_width(56.0)
            .show(ui, |ui| {
                for h in [
                    "#",
                    "Competitor",
                    "Rating r",
                    "RD",
                    "95% Band [r ± 2RD]",
                    "σ (Vol)",
                    "Pts",
                    "W-D-L-B",
                    "Seq Drift Δr",
                ] {
                    ui.strong(h);
                }
                ui.end_row();
                for (rank_zero, idx) in self.standings_order().into_iter().enumerate() {
                    let c = &self.competitors[idx];
                    let (r, rd) = (c.current.rating.value(), c.current.rating_deviation.value());
                    let drift = c.sequential.rating.value() - r;
                    ui.label(format!("{}", rank_zero + 1));
                    ui.colored_label(palette.series_color(idx), &c.name);
                    ui.label(format!("{r:.1}"));
                    ui.label(format!("{rd:.1}"));
                    ui.label(format!("[{:.0}, {:.0}]", r - 2.0 * rd, r + 2.0 * rd));
                    ui.label(format!("{:.4}", c.current.volatility.value()));
                    ui.label(format!("{:.1}", c.points));
                    ui.label(format!("{}-{}-{}-{}", c.wins, c.draws, c.losses, c.byes));
                    ui.label(format!("{drift:+.2}"));
                    ui.end_row();
                }
            });
    }

    fn show_pairings_table(&mut self, ui: &mut egui::Ui) {
        let mut commit_clicked = false;
        ui.horizontal(|ui| {
            let can_commit = !self.next_pairings.is_empty() || self.next_bye_player.is_some();
            if ui
                .add_enabled(can_commit, egui::Button::new("✓ Commit Manual Round"))
                .accessible_hover_text("Evaluate Period with selected manual board outcomes")
                .clicked()
            {
                commit_clicked = true;
            }
            if let Some(bye_idx) = self.next_bye_player {
                ui.label(format!(
                    "Odd-Roster Bye (+1.0 pt): {}",
                    self.competitors[bye_idx].name
                ));
            }
        });
        let competitors = &self.competitors;
        egui::Grid::new("glicko2_pairings_grid")
            .striped(true)
            .show(ui, |ui| {
                for h in [
                    "Board",
                    "Player A",
                    "Player B",
                    "P(A Wins)",
                    "Manual Outcome",
                ] {
                    ui.strong(h);
                }
                ui.end_row();
                for (b_idx, row) in self.next_pairings.iter_mut().enumerate() {
                    show_pairing_board_row(ui, b_idx, row, competitors);
                }
            });
        if commit_clicked {
            if let Err(err) = self.commit_manual_period() {
                self.io_state.status_message = Some((true, err));
            }
        }
    }

    fn show_win_matrix_table(&self, ui: &mut egui::Ui) {
        let players: Vec<GlickoPlayer> = self.competitors.iter().map(|c| c.current).collect();
        let matrix = pairwise_win_matrix(&players);
        egui::Grid::new("glicko2_win_matrix_grid")
            .striped(true)
            .show(ui, |ui| {
                ui.strong("Row \\ Col");
                for c in &self.competitors {
                    ui.strong(&c.name);
                }
                ui.end_row();
                for (i, row_comp) in self.competitors.iter().enumerate() {
                    ui.strong(&row_comp.name);
                    for j in 0..self.competitors.len() {
                        if i == j {
                            ui.label("—");
                        } else {
                            ui.label(format!("{:.1}%", matrix[(i, j)] * 100.0));
                        }
                    }
                    ui.end_row();
                }
            });
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
