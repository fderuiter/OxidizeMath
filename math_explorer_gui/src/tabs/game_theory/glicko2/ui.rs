//! Side-panel controls and roster editor for [`super::Glicko2TournamentTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CompetitorEntry, Glicko2TournamentTool};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::InteractiveTool;
use crate::presets::{Glicko2PairingMode, MAX_TOURNAMENT_PERIODS, MAX_TOURNAMENT_PLAYERS};
use eframe::egui;
use math_explorer::pure_math::statistics::glicko2::{
    GlickoPlayer, Rating, RatingDeviation, Volatility,
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

fn show_roster_grid(
    ui: &mut egui::Ui,
    competitors: &mut [CompetitorEntry],
) -> (Option<usize>, bool, bool) {
    let can_remove = competitors.len() > 2;
    let (mut remove_idx, mut rebuild, mut refresh_pairings) = (None, false, false);
    egui::Grid::new("glicko2_roster_grid")
        .striped(true)
        .show(ui, |ui| {
            for h in ["Name", "r₀", "RD₀", "σ₀", "Idle", ""] {
                ui.strong(h);
            }
            ui.end_row();
            for (idx, comp) in competitors.iter_mut().enumerate() {
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
    (remove_idx, rebuild, refresh_pairings)
}

impl Glicko2TournamentTool {
    fn show_starter_scenario_buttons(&mut self, ui: &mut egui::Ui) {
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

    fn show_schedule_controls(&mut self, ui: &mut egui::Ui) {
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

        let (remove_idx, rebuild, refresh_pairings) = show_roster_grid(ui, &mut self.competitors);
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
            self.show_starter_scenario_buttons(ui);
            self.show_schedule_controls(ui);
            self.show_execution_controls(ui);
            self.show_io_controls(ui);
            self.show_roster_editor(ui);
        });
    }
}
