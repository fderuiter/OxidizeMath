//! Interactive UI for CTMC generator rates, HMM Baum-Welch fitting, and regime plots.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    MarkovRegimeTool, RegimeWorkbenchMode, MAX_CTMC_STATES, MAX_HMM_STATES, MAX_HMM_SYMBOLS,
};
use crate::accessibility::AccessibleHoverText;
use crate::framework::bounded_drag_value;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, MarkerShape, Plot, Points};

impl MarkovRegimeTool {
    fn show_mode_selector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.label("Engine:");
            changed |= ui
                .selectable_value(
                    &mut self.mode,
                    RegimeWorkbenchMode::ContinuousTimeCtmc,
                    "Continuous-Time CTMC exp(Gt)",
                )
                .changed();
            changed |= ui
                .selectable_value(
                    &mut self.mode,
                    RegimeWorkbenchMode::HiddenMarkovHmm,
                    "Hidden Markov HMM (Baum-Welch & Viterbi)",
                )
                .changed();
            ui.separator();
            ui.label("RNG Seed:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.rng_seed).speed(1.0))
                .changed();
        });
        changed
    }

    fn show_ctmc_controls(&mut self, ui: &mut egui::Ui) -> bool {
        if self.mode != RegimeWorkbenchMode::ContinuousTimeCtmc {
            return false;
        }
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.label("States (2..=5):");
            let mut n = self.ctmc_states;
            if ui
                .add(egui::DragValue::new(&mut n).range(2..=MAX_CTMC_STATES))
                .changed()
            {
                self.resize_ctmc(n);
                changed = true;
            }
            ui.label("Initial State:");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.ctmc_source_state)
                        .range(0..=(self.ctmc_states - 1)),
                )
                .changed();
            ui.label("Horizon T:");
            changed |= ui
                .add(bounded_drag_value(&mut self.horizon_t, 0.2..=100.0, 0.1))
                .changed();
        });
        changed |= self.show_ctmc_rates_grid(ui);
        changed
    }

    fn show_ctmc_rates_grid(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.collapsing("Generator Off-Diagonal Transition Rates G[i, j]", |ui| {
            egui::Grid::new("ctmc_rates_grid")
                .striped(true)
                .spacing([10.0, 4.0])
                .show(ui, |ui| {
                    ui.label("From \\ To");
                    for j in 0..self.ctmc_states {
                        ui.strong(format!("S{j}"));
                    }
                    ui.end_row();
                    for i in 0..self.ctmc_states {
                        ui.strong(format!("S{i}"));
                        let diag = -self.ctmc_rates[i]
                            .iter()
                            .enumerate()
                            .filter(|&(c, _)| c != i)
                            .map(|(_, &v)| v.max(0.0))
                            .sum::<f64>();
                        for j in 0..self.ctmc_states {
                            if i == j {
                                ui.label(format!("{diag:.2}"));
                            } else {
                                changed |= ui
                                    .add(bounded_drag_value(
                                        &mut self.ctmc_rates[i][j],
                                        0.0..=25.0,
                                        0.02,
                                    ))
                                    .changed();
                            }
                        }
                        ui.end_row();
                    }
                });
        });
        changed
    }

    fn show_hmm_controls(&mut self, ui: &mut egui::Ui) -> bool {
        if self.mode != RegimeWorkbenchMode::HiddenMarkovHmm {
            return false;
        }
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            let mut st = self.hmm_states;
            let mut sym = self.hmm_symbols;
            ui.label("Hidden States:");
            let st_changed = ui
                .add(egui::DragValue::new(&mut st).range(2..=MAX_HMM_STATES))
                .changed();
            ui.label("Symbols:");
            let sym_changed = ui
                .add(egui::DragValue::new(&mut sym).range(2..=MAX_HMM_SYMBOLS))
                .changed();
            if st_changed || sym_changed {
                self.resize_hmm(st, sym);
                changed = true;
            }
            ui.label("Max EM Iters:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.bw_max_iters).range(1..=200))
                .changed();
            if ui
                .button("⚡ Fit Baum-Welch EM")
                .accessible_hover_text(
                    "Fit HMM transition and emission matrices via bounded Baum-Welch EM",
                )
                .clicked()
            {
                match self.run_baum_welch_fit() {
                    Ok(rep) => {
                        self.io_state.status = Some((
                            format!(
                                "Baum-Welch completed in {} iter(s): ln L {:.3} -> {:.3}",
                                rep.iterations,
                                rep.initial_log_likelihood,
                                rep.final_log_likelihood
                            ),
                            false,
                        ));
                    }
                    Err(err) => self.io_state.status = Some((err, true)),
                }
            }
            if ui
                .button("📂 Import Observations CSV")
                .accessible_hover_text("Import UTF-8 or BOM CSV integer observation log")
                .clicked()
            {
                self.trigger_open_observations();
            }
        });
        changed
    }

    fn show_regime_diagnostics(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        match self.mode {
            RegimeWorkbenchMode::ContinuousTimeCtmc => {
                if let Some(ref pi) = self.ctmc_steady_state {
                    let formatted = pi
                        .iter()
                        .enumerate()
                        .map(|(i, p)| format!("π_{i}={:.3}", p))
                        .collect::<Vec<_>>()
                        .join(", ");
                    ui.label(format!("Stationary Distribution: {formatted}"));
                }
                if !self.ctmc_absorption_times.is_empty() {
                    let abs_txt = self
                        .ctmc_absorption_times
                        .iter()
                        .map(|(s, t)| format!("E[T_abs | S{s}] = {t:.3}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    ui.label(format!("Expected Absorption Times: {abs_txt}"));
                }
            }
            RegimeWorkbenchMode::HiddenMarkovHmm => {
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!("Observations: {} steps", self.observations.len()));
                    if let Some(ref rep) = self.last_bw_report {
                        ui.separator();
                        ui.colored_label(
                            palette.success,
                            format!(
                                "Last Baum-Welch: {} iters, ln L = {:.3} (converged: {})",
                                rep.iterations, rep.final_log_likelihood, rep.converged
                            ),
                        );
                    }
                });
            }
        }
    }

    fn show_summary_and_export_bar(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if let Some(warn) = self.url_share_warning() {
            ui.colored_label(palette.warning, format!("⚠ {warn}"));
        }
        if let Some(ref err) = self.calc_error {
            ui.colored_label(palette.error, err);
            return;
        }
        self.show_regime_diagnostics(ui, &palette);
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📋 Copy CSV")
                .accessible_hover_text("Copy active CTMC or HMM regime schedule as CSV")
                .clicked()
            {
                let csv = self.export_regime_csv();
                self.copy_payload(ui.ctx(), "Markov regime CSV", csv);
            }
            if ui
                .button("💾 Save CSV")
                .accessible_hover_text("Download or save Markov regime CSV")
                .clicked()
            {
                self.trigger_save_csv();
            }
            if ui
                .button("💾 Save JSON")
                .accessible_hover_text("Download or save Markov regime JSON")
                .clicked()
            {
                self.trigger_save_json();
            }
            if ui
                .button("🦀 Copy Rust")
                .accessible_hover_text("Copy Rust CTMC or HMM snippet")
                .clicked()
            {
                let code = self.rust_snippet();
                self.copy_payload(ui.ctx(), "Rust Markov snippet", code);
            }
        });
        if let Some((ref msg, is_err)) = self.io_state.status {
            let color = if is_err { palette.error } else { palette.info };
            ui.colored_label(color, msg);
        }
    }

    fn show_ctmc_plot(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        Plot::new("markov_ctmc_plot")
            .view_aspect(2.15)
            .legend(Legend::default())
            .x_axis_label("Continuous Time (t)")
            .y_axis_label("Transition Probability P(t) / Scaled Jump Path")
            .show(ui, |plot_ui| {
                for (idx, (name, pts)) in self.ctmc_prob_series.iter().enumerate() {
                    plot_ui.line(
                        Line::new(name.as_str(), &pts[..])
                            .color(palette.series_color(idx))
                            .width(2.1_f32),
                    );
                }
                if !self.ctmc_gillespie_points.is_empty() {
                    plot_ui.line(
                        Line::new(
                            "Gillespie Sample Trajectory (scaled)",
                            &self.ctmc_gillespie_points[..],
                        )
                        .color(palette.graph_edge)
                        .style(LineStyle::Dashed { length: 5.0 }),
                    );
                }
                if !self.ctmc_absorption_points.is_empty() {
                    plot_ui.points(
                        Points::new(
                            "Expected Absorption Time E[T_abs] (scaled)",
                            &self.ctmc_absorption_points[..],
                        )
                        .color(palette.warning)
                        .radius(5.5_f32)
                        .shape(MarkerShape::Cross),
                    );
                }
            });
    }

    fn show_hmm_plot(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        Plot::new("markov_hmm_plot")
            .view_aspect(2.15)
            .legend(Legend::default())
            .x_axis_label("Observation Step (t)")
            .y_axis_label("Posterior Probability γ_t(i) / Viterbi Path")
            .show(ui, |plot_ui| {
                for (idx, (name, pts)) in self.hmm_posterior_series.iter().enumerate() {
                    plot_ui.line(
                        Line::new(name.as_str(), &pts[..])
                            .color(palette.series_color(idx))
                            .width(2.1_f32),
                    );
                }
                if !self.viterbi_points.is_empty() {
                    plot_ui.points(
                        Points::new(
                            "Viterbi Most-Likely State (scaled)",
                            &self.viterbi_points[..],
                        )
                        .color(palette.series_color(4))
                        .radius(5.0_f32)
                        .shape(MarkerShape::Diamond),
                    );
                }
            });
    }

    fn show_regime_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        match self.mode {
            RegimeWorkbenchMode::ContinuousTimeCtmc => self.show_ctmc_plot(ui, &palette),
            RegimeWorkbenchMode::HiddenMarkovHmm => self.show_hmm_plot(ui, &palette),
        }
    }

    pub(super) fn render_workbench_ui(&mut self, ui: &mut egui::Ui) {
        self.poll_io(ui.ctx());
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Markov Regime Switching (CTMC & HMM)");
            ui.label(
                "Explore continuous-time generator exponentials exp(Gt), steady-state/absorption metrics, and Hidden Markov Baum-Welch / Viterbi regime inference.",
            );
            ui.separator();

            let mut dirty = self.show_mode_selector(ui);
            dirty |= self.show_ctmc_controls(ui);
            dirty |= self.show_hmm_controls(ui);
            if dirty {
                self.bump_revision();
                self.recalculate();
            }
            ui.separator();
            self.show_summary_and_export_bar(ui);
            ui.separator();
            self.show_regime_plot(ui);
        });
    }
}

// [cite:clinical_trials_statistics]
