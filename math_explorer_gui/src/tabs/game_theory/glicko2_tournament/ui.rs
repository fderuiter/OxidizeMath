#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::io::{
    format_leaderboard_csv, format_leaderboard_markdown, format_matchup_csv,
    format_matchup_markdown,
};
use super::model::{
    RosterPlayer, ScheduledMatch, MAX_PLAYER_NAME_CHARS, MAX_TOURNAMENT_MATCHES,
    MAX_TOURNAMENT_PLAYERS,
};
use super::{Glicko2TournamentTool, TournamentSubView};
use crate::accessibility::{announce_live_status, AccessibleHoverText};
use eframe::egui;
use egui_plot::{Legend, Line, MarkerShape, Plot, Points};

impl Glicko2TournamentTool {
    pub(super) fn show_controls_panel(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Tournament Controls");
            ui.label(format!("Active Rating Period: #{}", self.period));

            let mut changed = ui
                .add(egui::Slider::new(&mut self.tau, 0.3..=1.2).text("System τ"))
                .accessible_hover_text(
                    "System constant tau (0.3 to 1.2) constraining rating volatility shifts",
                )
                .changed();

            ui.horizontal_wrapped(|ui| {
                if ui
                    .button("🏆 Glickman Period")
                    .accessible_hover_text("Load canonical Glickman (2012) rating period example")
                    .clicked()
                {
                    self.load_glickman_preset();
                    announce_live_status(ui.ctx(), "Loaded canonical Glickman rating period");
                }
                if ui
                    .button("⏭ Advance Period")
                    .accessible_hover_text(
                        "Commit post-period ratings to roster and advance to next rating period",
                    )
                    .clicked()
                {
                    self.advance_rating_period();
                    announce_live_status(
                        ui.ctx(),
                        format!("Advanced to rating period {}", self.period),
                    );
                }
                if ui
                    .button("📂 Import CSV")
                    .accessible_hover_text(
                        "Import local match log CSV (player_a,player_b,score; max 64 players, 1024 matches)",
                    )
                    .clicked()
                {
                    self.trigger_csv_import_dialog();
                }
            });

            ui.separator();
            changed |= self.show_roster_editor(ui);
            ui.separator();
            changed |= self.show_match_editor(ui);

            if changed {
                self.recalculate();
            }
        });
    }

    fn show_roster_editor(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.collapsing(format!("Roster ({} players)", self.roster.len()), |ui| {
            egui::Grid::new("glicko2_roster_grid")
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("Player");
                    ui.strong("r");
                    ui.strong("RD");
                    ui.strong("σ");
                    ui.end_row();
                    for p in &mut self.roster {
                        ui.label(&p.name);
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut p.rating)
                                    .speed(2.0)
                                    .range(100.0..=4000.0),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut p.rd)
                                    .speed(1.0)
                                    .range(1.0..=500.0),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut p.volatility)
                                    .speed(0.001)
                                    .range(0.001..=1.5),
                            )
                            .changed();
                        ui.end_row();
                    }
                });

            if self.roster.len() < MAX_TOURNAMENT_PLAYERS {
                ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut self.draft_player_name);
                    if ui
                        .button("+ Add Player")
                        .accessible_hover_text("Add player to pre-period tournament roster")
                        .clicked()
                    {
                        let trimmed = self.draft_player_name.trim();
                        if !trimmed.is_empty() && trimmed.chars().count() <= MAX_PLAYER_NAME_CHARS {
                            self.roster.push(RosterPlayer {
                                name: trimmed.to_string(),
                                rating: self.draft_player_rating,
                                rd: self.draft_player_rd,
                                volatility: 0.06,
                            });
                            changed = true;
                        }
                    }
                });
            }
        });
        changed
    }

    fn show_match_editor(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.collapsing(format!("Period Matches ({})", self.matches.len()), |ui| {
            let mut remove_idx = None;
            for (idx, m) in self.matches.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    let name_a = self.roster.get(m.player_a).map_or("?", |p| p.name.as_str());
                    let name_b = self.roster.get(m.player_b).map_or("?", |p| p.name.as_str());
                    ui.label(format!("{name_a} vs {name_b}:"));
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut m.score_a)
                                .speed(0.5)
                                .range(0.0..=1.0),
                        )
                        .changed();
                    if ui.small_button("✕").clicked() {
                        remove_idx = Some(idx);
                    }
                });
            }
            if let Some(idx) = remove_idx {
                self.matches.remove(idx);
                changed = true;
            }
            changed |= self.show_add_match_row(ui);
        });
        changed
    }

    fn show_add_match_row(&mut self, ui: &mut egui::Ui) -> bool {
        if self.matches.len() >= MAX_TOURNAMENT_MATCHES || self.roster.len() < 2 {
            return false;
        }
        let n = self.roster.len();
        self.draft_match_a = self.draft_match_a.min(n - 1);
        self.draft_match_b = self.draft_match_b.min(n - 1);
        let mut added = false;
        ui.horizontal_wrapped(|ui| {
            egui::ComboBox::from_id_salt("glicko_match_a")
                .selected_text(&self.roster[self.draft_match_a].name)
                .show_ui(ui, |ui| {
                    for (i, p) in self.roster.iter().enumerate() {
                        ui.selectable_value(&mut self.draft_match_a, i, &p.name);
                    }
                });
            ui.label("vs");
            egui::ComboBox::from_id_salt("glicko_match_b")
                .selected_text(&self.roster[self.draft_match_b].name)
                .show_ui(ui, |ui| {
                    for (i, p) in self.roster.iter().enumerate() {
                        ui.selectable_value(&mut self.draft_match_b, i, &p.name);
                    }
                });
            ui.add(
                egui::DragValue::new(&mut self.draft_score_a)
                    .speed(0.5)
                    .range(0.0..=1.0),
            );
            if ui
                .button("+ Match")
                .accessible_hover_text("Append match result to active rating period")
                .clicked()
                && self.draft_match_a != self.draft_match_b
            {
                self.matches.push(ScheduledMatch {
                    player_a: self.draft_match_a,
                    player_b: self.draft_match_b,
                    score_a: self.draft_score_a,
                });
                added = true;
            }
        });
        added
    }

    pub(super) fn show_main_workbench(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        egui::ScrollArea::vertical().show(ui, |ui| {
            if let Some(ref err) = self.eval_error {
                ui.colored_label(palette.error, err);
                return;
            }
            self.show_interval_plot(ui, &palette);
            ui.separator();
            self.show_export_and_tab_bar(ui, &palette);
            ui.add_space(6.0);
            match self.active_subview {
                TournamentSubView::Leaderboard => self.show_leaderboard_table(ui, &palette),
                TournamentSubView::MatchupMatrix => self.show_matchup_matrix_table(ui),
            }
        });
    }

    fn show_interval_plot(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        ui.heading("95% Skill Intervals (r ± 2·RD)");
        Plot::new("glicko2_skill_intervals_plot")
            .height(240.0)
            .legend(Legend::default())
            .x_axis_label("Leaderboard Rank")
            .y_axis_label("Glicko Rating (r)")
            .show(ui, |plot_ui| {
                for (idx, (label, bar_pts)) in self.error_bar_lines.iter().enumerate() {
                    plot_ui.line(
                        Line::new(label, &bar_pts[..])
                            .color(palette.series_color(idx))
                            .width(2.0_f32),
                    );
                }
                plot_ui.points(
                    Points::new("Post-Period Rating (r)", &self.post_rating_points[..])
                        .color(palette.accent)
                        .radius(5.5_f32)
                        .shape(MarkerShape::Circle),
                );
                plot_ui.points(
                    Points::new("Pre-Period Rating (r₀)", &self.pre_rating_points[..])
                        .color(palette.muted_text)
                        .radius(4.5_f32)
                        .shape(MarkerShape::Diamond),
                );
            });
    }

    fn show_export_and_tab_bar(
        &mut self,
        ui: &mut egui::Ui,
        palette: &crate::app::SemanticPalette,
    ) {
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.active_subview,
                TournamentSubView::Leaderboard,
                "📊 Standings Leaderboard",
            );
            ui.selectable_value(
                &mut self.active_subview,
                TournamentSubView::MatchupMatrix,
                "🎯 Pairwise Win Matrix",
            );
            ui.separator();
            if ui
                .button("📋 Copy CSV")
                .accessible_hover_text(
                    "Copy active standings or matchup matrix as formula-safe CSV",
                )
                .clicked()
            {
                let (csv, lbl) = self.active_export_csv();
                self.copy_text_artifact(ui.ctx(), &csv, lbl);
            }
            if ui
                .button("💾 Save CSV")
                .accessible_hover_text("Save active standings or matchup matrix as CSV")
                .clicked()
            {
                let (csv, _) = self.active_export_csv();
                let fname = match self.active_subview {
                    TournamentSubView::Leaderboard => "glicko2_standings.csv",
                    TournamentSubView::MatchupMatrix => "glicko2_matchup_matrix.csv",
                };
                self.save_text_artifact(fname, csv, false);
            }
            if ui
                .button("📋 Copy Markdown")
                .accessible_hover_text("Copy active standings or matchup matrix as Markdown")
                .clicked()
            {
                let (md, lbl) = self.active_export_markdown();
                self.copy_text_artifact(ui.ctx(), &md, lbl);
            }
            if ui
                .button("💾 Save Markdown")
                .accessible_hover_text("Save active standings or matchup matrix as Markdown")
                .clicked()
            {
                let (md, _) = self.active_export_markdown();
                let fname = match self.active_subview {
                    TournamentSubView::Leaderboard => "glicko2_standings.md",
                    TournamentSubView::MatchupMatrix => "glicko2_matchup_matrix.md",
                };
                self.save_text_artifact(fname, md, true);
            }
        });
        if let Some((ref msg, is_err)) = self.io_status {
            let color = if is_err { palette.error } else { palette.info };
            ui.colored_label(color, msg);
        }
    }

    fn active_export_csv(&self) -> (String, &'static str) {
        match self.active_subview {
            TournamentSubView::Leaderboard => {
                (format_leaderboard_csv(&self.standings), "CSV standings")
            }
            TournamentSubView::MatchupMatrix => (
                format_matchup_csv(&self.roster, &self.win_prob_matrix),
                "CSV matchup matrix",
            ),
        }
    }

    fn active_export_markdown(&self) -> (String, &'static str) {
        match self.active_subview {
            TournamentSubView::Leaderboard => (
                format_leaderboard_markdown(&self.standings, self.tau, self.period),
                "Markdown standings",
            ),
            TournamentSubView::MatchupMatrix => (
                format_matchup_markdown(&self.roster, &self.win_prob_matrix),
                "Markdown matchup matrix",
            ),
        }
    }

    fn show_leaderboard_table(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        egui::Grid::new("glicko2_standings_table")
            .striped(true)
            .spacing([18.0, 6.0])
            .show(ui, |ui| {
                ui.strong("Rank");
                ui.strong("Player");
                ui.strong("Post r (Δr)");
                ui.strong("Post RD (ΔRD)");
                ui.strong("Volatility σ'");
                ui.strong("95% Interval");
                ui.strong("Games (Pts)");
                ui.end_row();

                for s in &self.standings {
                    let dr = s.post_rating - s.pre_rating;
                    let drd = s.post_rd - s.pre_rd;
                    let delta_color = if dr > 1e-3 {
                        palette.success
                    } else if dr < -1e-3 {
                        palette.error
                    } else {
                        palette.muted_text
                    };
                    ui.label(format!("#{}", s.rank));
                    ui.label(&s.name);
                    ui.colored_label(delta_color, format!("{:.2} ({dr:+.2})", s.post_rating));
                    ui.label(format!("{:.2} ({drd:+.2})", s.post_rd));
                    ui.label(format!("{:.5}", s.post_volatility));
                    ui.label(format!("[{:.1}, {:.1}]", s.ci_low, s.ci_high));
                    if s.matches_played == 0 {
                        ui.colored_label(palette.warning, "0 (Inactive)");
                    } else {
                        ui.label(format!("{} ({:.1})", s.matches_played, s.score_sum));
                    }
                    ui.end_row();
                }
            });
    }

    fn show_matchup_matrix_table(&self, ui: &mut egui::Ui) {
        ui.label("Expected score E(μ_row, μ_col, φ_col) for row player vs. column opponent:");
        egui::Grid::new("glicko2_matchup_matrix_grid")
            .striped(true)
            .spacing([16.0, 6.0])
            .show(ui, |ui| {
                ui.strong("Player \\ Opp");
                for p in &self.roster {
                    ui.strong(&p.name);
                }
                ui.end_row();

                for (i, p) in self.roster.iter().enumerate() {
                    ui.strong(&p.name);
                    for j in 0..self.roster.len() {
                        if i == j {
                            ui.label("—");
                        } else {
                            let prob = self
                                .win_prob_matrix
                                .get(i)
                                .and_then(|r| r.get(j))
                                .copied()
                                .unwrap_or(0.5);
                            ui.label(format!("{:.1}%", prob * 100.0));
                        }
                    }
                    ui.end_row();
                }
            });
    }
}
