//! Interactive `egui` and `egui_plot` panels for [`super::Glicko2TournamentTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BottomViewTab, Glicko2TournamentTool};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::InteractiveTool;
use crate::presets::{Glicko2PairingMode, MAX_TOURNAMENT_PERIODS, MAX_TOURNAMENT_PLAYERS};
use eframe::egui;
use math_explorer::pure_math::statistics::glicko2::{
    head_to_head_win_probability, pairwise_win_matrix, GlickoPlayer, Rating, RatingDeviation,
    Volatility,
};

fn io_btn(ui: &mut egui::Ui, label: &str, tip: &str) -> bool {
    ui.button(label).accessible_hover_text(tip).clicked()
}

fn edit_initial_player(ui: &mut egui::Ui, initial: &mut GlickoPlayer) -> bool {
    let mut r0 = initial.rating.value();
    let mut rd0 = initial.rating_deviation.value();
    let mut sig0 = initial.volatility.value();
    let chg = ui
        .add(
            egui::DragValue::new(&mut r0)
                .speed(5.0)
                .range(100.0..=4000.0),
        )
        .changed()
        | ui.add(egui::DragValue::new(&mut rd0).speed(2.0).range(1.0..=600.0))
            .changed()
        | ui.add(
            egui::DragValue::new(&mut sig0)
                .speed(0.002)
                .range(0.001..=1.5),
        )
        .changed();
    if let (true, Ok(r), Ok(d), Ok(v)) = (
        chg,
        Rating::new(r0),
        RatingDeviation::new(rd0),
        Volatility::new(sig0),
    ) {
        *initial = GlickoPlayer::new(r, d, v);
        return true;
    }
    false
}

impl Glicko2TournamentTool {
    fn show_starter_scenarios(&mut self, ui: &mut egui::Ui) {
        ui.label("Starter Scenarios:");
        ui.horizontal_wrapped(|ui| {
            if io_btn(
                ui,
                "♟ Swiss Upset",
                "Load 5-player Swiss upset and volatility spike scenario",
            ) {
                let _ = self.import_preset_json(crate::starter_gallery::PRESET_GLICKO2_SWISS_UPSET);
            }
            if io_btn(
                ui,
                "🎯 Round-Robin Drift",
                "Load 4-player Round-Robin frozen vs sequential drift",
            ) {
                let _ = self
                    .import_preset_json(crate::starter_gallery::PRESET_GLICKO2_ROUND_ROBIN_DRIFT);
            }
        });
    }

    fn show_preset_and_schedule_controls(&mut self, ui: &mut egui::Ui) {
        self.show_starter_scenarios(ui);
        ui.separator();
        ui.heading("System & Schedule");
        let rebuild = ui
            .add(egui::Slider::new(&mut self.tau, 0.3..=1.2).text("τ (System Constant)"))
            .changed();
        if ui
            .add(egui::Slider::new(&mut self.draw_probability, 0.0..=0.8).text("Draw Rate"))
            .changed()
        {
            self.bump_revision();
        }
        ui.horizontal(|ui| {
            ui.label("Pairing:");
            for mode in [Glicko2PairingMode::Swiss, Glicko2PairingMode::RoundRobin] {
                if ui
                    .selectable_value(&mut self.pairing_mode, mode, mode.label())
                    .changed()
                {
                    self.refresh_next_pairings();
                    self.bump_revision();
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Rounds:");
            if ui
                .add(
                    egui::DragValue::new(&mut self.target_rounds).range(1..=MAX_TOURNAMENT_PERIODS),
                )
                .changed()
            {
                self.bump_revision();
            }
            ui.label("Seed:");
            if ui
                .add(egui::DragValue::new(&mut self.rng_seed).range(0..=9_999_999))
                .changed()
            {
                self.bump_revision();
            }
        });
        if rebuild {
            let _ = self.rebuild_trajectory();
        }
    }

    fn show_execution_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if io_btn(
                ui,
                "▶ Simulate Period",
                "Simulate next scheduled rating period with OxidizeRng",
            ) {
                if let Err(err) = self.simulate_next_period() {
                    self.io_state.status_message = Some((true, err));
                }
            }
            if io_btn(
                ui,
                "⏩ Run to Target",
                "Simulate remaining periods up to Target Rounds",
            ) {
                if let Err(err) = self.simulate_to_target_rounds() {
                    self.io_state.status_message = Some((true, err));
                }
            }
            if io_btn(
                ui,
                "↺ Reset Periods",
                "Clear completed match periods back to Period 0",
            ) {
                self.reset_periods();
            }
        });
        ui.small(format!(
            "Periods: {} / {} | URL Payload: {} / {} B",
            self.completed_periods.len(),
            self.target_rounds,
            self.estimated_url_state_bytes(),
            crate::url_state::MAX_DECODED_URL_STATE_BYTES
        ));
        if let Some(warning) = self.url_share_warning() {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            ui.colored_label(palette.warning, format!("⚠ {warning}"));
        }
    }

    fn show_io_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("Offline Match Logs & Exports");
        ui.horizontal_wrapped(|ui| {
            if io_btn(
                ui,
                "📂 Import CSV/JSON",
                "Import local CSV or JSON match log atomically",
            ) {
                self.io_state.trigger_import_dialog();
            }
            if io_btn(
                ui,
                "💾 Standings CSV",
                "Export standings and 95% confidence bands as CSV",
            ) {
                self.io_state
                    .trigger_save_csv("glicko2_standings.csv", self.format_standings_csv());
            }
            if io_btn(
                ui,
                "💾 Standings JSON",
                "Export standings and 95% confidence bands as JSON",
            ) {
                self.io_state
                    .trigger_save_json("glicko2_standings.json", self.format_standings_json());
            }
            if io_btn(
                ui,
                "💾 Pairings CSV",
                "Export upcoming pairings and win probabilities as CSV",
            ) {
                self.io_state
                    .trigger_save_csv("glicko2_pairings.csv", self.format_pairings_csv());
            }
            if io_btn(ui, "💾 Pairings JSON", "Export upcoming pairings as JSON") {
                self.io_state
                    .trigger_save_json("glicko2_pairings.json", self.format_pairings_json());
            }
        });
        if let Some((is_err, ref msg)) = self.io_state.status_message {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            ui.colored_label(
                if is_err {
                    palette.error
                } else {
                    palette.success
                },
                msg,
            );
        }
    }

    fn show_roster_editor(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading(format!(
            "Roster ({}/{MAX_TOURNAMENT_PLAYERS})",
            self.competitors.len()
        ));
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.new_player_name).desired_width(120.0));
            if ui
                .add_enabled(
                    self.competitors.len() < MAX_TOURNAMENT_PLAYERS,
                    egui::Button::new("+ Add"),
                )
                .accessible_hover_text("Add competitor to roster")
                .clicked()
            {
                let _ = self.add_competitor();
            }
        });

        let can_remove = self.competitors.len() > 2;
        let (mut remove_idx, mut rebuild, mut refresh_pairings) = (None, false, false);
        egui::Grid::new("glicko2_roster_grid")
            .striped(true)
            .show(ui, |ui| {
                for h in ["Name", "r₀", "RD₀", "σ₀", "Idle", ""] {
                    ui.strong(h);
                }
                ui.end_row();
                for (idx, comp) in self.competitors.iter_mut().enumerate() {
                    ui.label(&comp.name);
                    rebuild |= edit_initial_player(ui, &mut comp.initial);
                    if ui
                        .checkbox(&mut comp.inactive, "")
                        .accessible_hover_text("Sit out upcoming periods (inflates RD)")
                        .changed()
                    {
                        refresh_pairings = true;
                    }
                    if can_remove
                        && ui
                            .small_button("✖")
                            .accessible_hover_text("Remove competitor")
                            .clicked()
                    {
                        remove_idx = Some(idx);
                    }
                    ui.end_row();
                }
            });
        if let Some(idx) = remove_idx {
            let _ = self.remove_competitor(idx);
        } else if rebuild {
            let _ = self.rebuild_trajectory();
        } else if refresh_pairings {
            self.refresh_next_pairings();
            self.bump_revision();
        }
    }

    pub(super) fn show_side_controls(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            self.show_preset_and_schedule_controls(ui);
            self.show_execution_controls(ui);
            self.show_io_controls(ui);
            self.show_roster_editor(ui);
        });
    }

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
                    let (ca, cb) = (
                        &self.competitors[row.pairing.player_a],
                        &self.competitors[row.pairing.player_b],
                    );
                    let win_p = head_to_head_win_probability(&ca.current, &cb.current);
                    ui.label(format!("#{}", b_idx + 1));
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
}
