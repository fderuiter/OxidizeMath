//! Side control panel and bottom matrix/KPI inspector dock for [`super::MarkovSequenceWorkbenchTool`].
// [cite:clinical_trials_statistics]

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MarkovScenario, MarkovSequenceWorkbenchTool};
use crate::app::SemanticPalette;
use crate::presets::{
    MAX_BAUM_WELCH_ITERATIONS, MAX_OBSERVATION_SEQUENCE_LEN, MarkovWorkbenchMode,
};
use eframe::egui;

fn normalize_row_preserving_edit(row: &mut [f64], edited_col: usize) {
    let val = row[edited_col].clamp(0.0, 1.0);
    row[edited_col] = val;
    if row.len() == 1 {
        row[0] = 1.0;
        return;
    }
    let remainder = (1.0 - val).max(0.0);
    let others_sum: f64 = row
        .iter()
        .enumerate()
        .filter_map(|(j, &v)| (j != edited_col).then_some(v))
        .sum();
    if others_sum > 1e-12 {
        let scale = remainder / others_sum;
        for (j, slot) in row.iter_mut().enumerate() {
            if j != edited_col {
                *slot = (*slot * scale).clamp(0.0, 1.0);
            }
        }
    } else {
        let share = remainder / (row.len() - 1) as f64;
        for (j, slot) in row.iter_mut().enumerate() {
            if j != edited_col {
                *slot = share;
            }
        }
    }
}

impl MarkovSequenceWorkbenchTool {
    pub(super) fn show_side_controls(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Markov Sequence & Reliability");
            ui.label(
                egui::RichText::new(format!("Compute: {:.2} ms", self.last_compute_ms))
                    .small()
                    .color(palette.muted_text),
            );
            ui.separator();

            ui.label(egui::RichText::new("Workbench Mode").strong());
            for mode in [
                MarkovWorkbenchMode::Hmm,
                MarkovWorkbenchMode::Ctmc,
                MarkovWorkbenchMode::AbsorbingDtmc,
            ] {
                if ui
                    .selectable_label(self.config.mode == mode, mode.label())
                    .clicked()
                    && self.config.mode != mode
                {
                    self.config.mode = mode;
                    self.state_rev = self.state_rev.wrapping_add(1);
                }
            }

            ui.separator();
            ui.label(egui::RichText::new("Reference Scenarios").strong());
            let mut chosen = self.selected_scenario;
            egui::ComboBox::from_id_salt("markov_scenario_combo")
                .selected_text(chosen.label())
                .show_ui(ui, |ui| {
                    for sc in MarkovScenario::ALL {
                        ui.selectable_value(&mut chosen, sc, sc.label());
                    }
                });
            if chosen != self.selected_scenario || ui.button("Load Selected Scenario").clicked() {
                let _ = self.load_scenario(chosen);
            }

            ui.separator();
            self.show_io_controls(ui, palette);
            ui.separator();

            match self.config.mode {
                MarkovWorkbenchMode::Hmm => self.show_hmm_controls(ui, palette),
                MarkovWorkbenchMode::Ctmc => self.show_ctmc_controls(ui, palette),
                MarkovWorkbenchMode::AbsorbingDtmc => self.show_dtmc_controls(ui, palette),
            }
        });
    }

    fn show_io_controls(&mut self, ui: &mut egui::Ui, palette: SemanticPalette) {
        ui.label(egui::RichText::new("Local File Import & Exports").strong());
        if ui.button("📂 Import CSV / JSON…").clicked() {
            self.io_state.trigger_import_dialog();
        }
        match self.config.mode {
            MarkovWorkbenchMode::Hmm => {
                ui.horizontal_wrapped(|ui| {
                    if ui.button("⬇ Viterbi + Posterior (.csv)").clicked() {
                        let csv = self.format_hmm_timeline_csv();
                        self.io_state
                            .trigger_save_csv("hmm_viterbi_posteriors.csv", csv);
                    }
                    if ui.button("⬇ Fitted HMM (.json)").clicked() {
                        let json = self.format_hmm_model_json();
                        self.io_state
                            .trigger_save_json("fitted_hmm_model.json", json);
                    }
                });
            }
            MarkovWorkbenchMode::Ctmc => {
                ui.horizontal_wrapped(|ui| {
                    if ui.button("⬇ CTMC Reliability (.csv)").clicked() {
                        let csv = self.format_ctmc_report_csv();
                        self.io_state.trigger_save_csv("ctmc_reliability.csv", csv);
                    }
                    if ui.button("⬇ CTMC Report (.json)").clicked() {
                        let json = self.format_ctmc_report_json();
                        self.io_state
                            .trigger_save_json("ctmc_reliability.json", json);
                    }
                });
            }
            MarkovWorkbenchMode::AbsorbingDtmc => {
                ui.horizontal_wrapped(|ui| {
                    if ui.button("⬇ DTMC EPV & N (.csv)").clicked() {
                        let csv = self.format_dtmc_report_csv();
                        self.io_state.trigger_save_csv("dtmc_epv_report.csv", csv);
                    }
                    if ui.button("⬇ DTMC Report (.json)").clicked() {
                        let json = self.format_dtmc_report_json();
                        self.io_state
                            .trigger_save_json("dtmc_epv_report.json", json);
                    }
                });
            }
        }
        if let Some(warn) = self.url_share_warning() {
            ui.colored_label(palette.warning, warn);
        }
        if let Some((is_err, msg)) = &self.io_state.status_message {
            let color = if *is_err {
                palette.error
            } else {
                palette.success
            };
            ui.colored_label(color, msg);
        }
    }

    fn show_hmm_controls(&mut self, ui: &mut egui::Ui, palette: SemanticPalette) {
        ui.label(egui::RichText::new("Baum-Welch EM & Sequence").strong());
        let mut changed = false;
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut self.config.hmm.bw_max_iterations,
                    1..=MAX_BAUM_WELCH_ITERATIONS,
                )
                .text("Max EM Iterations"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.config.hmm.bw_tolerance, 1e-10..=1e-2)
                    .logarithmic(true)
                    .text("Tolerance"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.config.hmm.bw_dirichlet_alpha, 0.0..=1.0)
                    .text("Dirichlet α"),
            )
            .changed();
        if ui.button("▶ Run Baum-Welch EM").clicked()
            && let Err(err) = self.run_baum_welch()
        {
            self.io_state.status_message = Some((true, err));
        }
        ui.separator();
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut self.config.hmm.sim_length,
                    5..=MAX_OBSERVATION_SEQUENCE_LEN.min(2000),
                )
                .logarithmic(true)
                .text("Sim Steps"),
            )
            .changed();
        changed |= ui
            .add(egui::DragValue::new(&mut self.config.rng_seed).prefix("Seed: "))
            .changed();
        if ui.button("🎲 Simulate Sequence").clicked() {
            let _ = self.simulate_hmm_sequence();
        }
        if changed {
            self.state_rev = self.state_rev.wrapping_add(1);
        }
        ui.separator();
        self.show_hmm_kpi_summary(ui, palette);
    }

    fn show_hmm_kpi_summary(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        ui.label(format!(
            "Sequence length: {} | ln P(Y) = {:.4}",
            self.config.hmm.observations.len(),
            self.hmm_out.log_likelihood
        ));
        ui.label(format!("P(Y) = {:.3e}", self.hmm_out.forward_likelihood));
        if let Some((best_i, &best_p)) = self
            .hmm_out
            .filter_last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
        {
            let name = self
                .config
                .hmm
                .state_names
                .get(best_i)
                .map_or("?", String::as_str);
            ui.label(format!("Final filter: {name} ({best_p:.3})"));
        }
        if let Some(rep) = self.hmm_out.last_bw_report {
            ui.colored_label(
                palette.info,
                format!(
                    "EM: {} iter(s), converged={}, Δln P = {:+.4}",
                    rep.iterations,
                    rep.converged,
                    rep.final_log_likelihood - rep.initial_log_likelihood
                ),
            );
        }
    }

    fn show_ctmc_controls(&mut self, ui: &mut egui::Ui, palette: SemanticPalette) {
        ui.label(egui::RichText::new("Continuous-Time Parameters").strong());
        let n = self.config.ctmc.state_names.len();
        let mut changed = false;
        egui::ComboBox::from_label("Source State")
            .selected_text(&self.config.ctmc.state_names[self.config.ctmc.source_state.min(n - 1)])
            .show_ui(ui, |ui| {
                for (i, name) in self.config.ctmc.state_names.iter().enumerate() {
                    changed |= ui
                        .selectable_value(&mut self.config.ctmc.source_state, i, name)
                        .changed();
                }
            });
        changed |= ui
            .add(
                egui::Slider::new(&mut self.config.ctmc.horizon_time, 0.2..=100.0)
                    .logarithmic(true)
                    .text("Sweep Horizon T"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut self.config.ctmc.eval_time,
                    0.0..=self.config.ctmc.horizon_time,
                )
                .text("Eval Time t"),
            )
            .changed();
        if changed {
            let _ = self.recompute_all();
            self.state_rev = self.state_rev.wrapping_add(1);
        }
        if let Some(mttf) = &self.ctmc_out.mttf {
            for (idx, &val) in mttf.iter().enumerate() {
                let st = self.config.ctmc.transient_states[idx];
                ui.colored_label(
                    palette.info,
                    format!("MTTF ({}) = {val:.4}", self.config.ctmc.state_names[st]),
                );
            }
        }
    }

    fn show_dtmc_controls(&mut self, ui: &mut egui::Ui, palette: SemanticPalette) {
        ui.label(egui::RichText::new("Terminal Rewards & Valuation").strong());
        let mut changed = false;
        let abs_indices = self.dtmc_out.absorbing_indices.clone();
        for (k, abs_i) in abs_indices.into_iter().enumerate() {
            let label = format!("Reward ({})", self.config.dtmc.state_names[abs_i]);
            if let Some(slot) = self.config.dtmc.absorbing_rewards.get_mut(k) {
                changed |= ui
                    .add(egui::Slider::new(slot, -10.0..=10.0).text(label))
                    .changed();
            }
        }
        if changed {
            let _ = self.recompute_all();
            self.state_rev = self.state_rev.wrapping_add(1);
        }
        ui.separator();
        for (r, &st_i) in self.dtmc_out.transient_indices.iter().enumerate() {
            let epv = self.dtmc_out.epv.get(r).copied().unwrap_or(0.0);
            let steps = self.dtmc_out.expected_steps.get(r).copied().unwrap_or(0.0);
            ui.colored_label(
                palette.info,
                format!(
                    "{}: EPV = {epv:.3}, E[steps] = {steps:.2}",
                    self.config.dtmc.state_names[st_i]
                ),
            );
        }
    }

    pub(super) fn show_bottom_dock(&mut self, parent_ui: &mut egui::Ui) {
        let bottom_id = crate::framework::scoped_panel_id(parent_ui, "markov_workbench_bottom");
        egui::TopBottomPanel::bottom(bottom_id)
            .resizable(true)
            .default_height(175.0)
            .show_inside(parent_ui, |ui| {
                egui::ScrollArea::both().show(ui, |ui| match self.config.mode {
                    MarkovWorkbenchMode::Hmm => self.show_hmm_matrix_dock(ui),
                    MarkovWorkbenchMode::Ctmc => self.show_ctmc_matrix_dock(ui),
                    MarkovWorkbenchMode::AbsorbingDtmc => self.show_dtmc_matrix_dock(ui),
                });
            });
    }

    fn show_hmm_matrix_dock(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Transition Matrix A (auto-normalized rows)").strong());
        let mut changed = false;
        let n = self.config.hmm.state_names.len();
        egui::Grid::new("markov_hmm_trans_grid")
            .striped(true)
            .show(ui, |ui| {
                ui.label("From \\ To");
                for name in &self.config.hmm.state_names {
                    ui.label(name);
                }
                ui.end_row();
                for i in 0..n {
                    ui.label(&self.config.hmm.state_names[i]);
                    for j in 0..n {
                        if ui
                            .add(
                                egui::DragValue::new(&mut self.config.hmm.transitions[i][j])
                                    .speed(0.01)
                                    .range(0.0..=1.0),
                            )
                            .changed()
                        {
                            normalize_row_preserving_edit(&mut self.config.hmm.transitions[i], j);
                            changed = true;
                        }
                    }
                    ui.end_row();
                }
            });
        if changed {
            let _ = self.recompute_all();
            self.state_rev = self.state_rev.wrapping_add(1);
        }
    }

    fn show_ctmc_matrix_dock(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new("Generator G (off-diagonal rates >= 0; diagonal auto-balanced)")
                .strong(),
        );
        let mut changed = false;
        let n = self.config.ctmc.state_names.len();
        egui::Grid::new("markov_ctmc_gen_grid")
            .striped(true)
            .show(ui, |ui| {
                ui.label("From \\ To");
                for name in &self.config.ctmc.state_names {
                    ui.label(name);
                }
                ui.end_row();
                for i in 0..n {
                    ui.label(&self.config.ctmc.state_names[i]);
                    for j in 0..n {
                        if i == j {
                            ui.label(format!("{:.3}", self.config.ctmc.generator[i][i]));
                        } else if ui
                            .add(
                                egui::DragValue::new(&mut self.config.ctmc.generator[i][j])
                                    .speed(0.05)
                                    .range(0.0..=100.0),
                            )
                            .changed()
                        {
                            let off_sum: f64 = self.config.ctmc.generator[i]
                                .iter()
                                .enumerate()
                                .filter_map(|(c, &v)| (c != i).then_some(v.max(0.0)))
                                .sum();
                            self.config.ctmc.generator[i][i] = -off_sum;
                            changed = true;
                        }
                    }
                    ui.end_row();
                }
            });
        if changed {
            let _ = self.recompute_all();
            self.state_rev = self.state_rev.wrapping_add(1);
        }
    }

    fn show_dtmc_matrix_dock(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new("Transition Matrix P (transient rows auto-normalized)").strong(),
        );
        let mut changed = false;
        let n = self.config.dtmc.state_names.len();
        egui::Grid::new("markov_dtmc_trans_grid")
            .striped(true)
            .show(ui, |ui| {
                ui.label("From \\ To");
                for name in &self.config.dtmc.state_names {
                    ui.label(name);
                }
                ui.end_row();
                for i in 0..n {
                    ui.label(&self.config.dtmc.state_names[i]);
                    let is_abs =
                        self.config.dtmc.state_types[i] == crate::presets::DtmcStateKind::Absorbing;
                    for j in 0..n {
                        if is_abs {
                            ui.label(format!("{:.2}", self.config.dtmc.transition_matrix[i][j]));
                        } else if ui
                            .add(
                                egui::DragValue::new(&mut self.config.dtmc.transition_matrix[i][j])
                                    .speed(0.01)
                                    .range(0.0..=1.0),
                            )
                            .changed()
                        {
                            normalize_row_preserving_edit(
                                &mut self.config.dtmc.transition_matrix[i],
                                j,
                            );
                            changed = true;
                        }
                    }
                    ui.end_row();
                }
            });
        if changed {
            let _ = self.recompute_all();
            self.state_rev = self.state_rev.wrapping_add(1);
        }
    }
}
