#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{SimpsonParadoxLab, TableCounts, TableEvaluation, WinnerPrediction};
use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use egui_plot::{Line, Plot, Points};
use math_explorer::applied::clinical_trials::analysis::RiskMetrics;

impl SimpsonParadoxLab {
    /// Renders the Simpson's Paradox lab controls, 2x2 summary grid, and RR forest plot.
    pub fn show_ui(&mut self, ui: &mut egui::Ui, seed_counter: &mut u64) {
        self.render_predict_winner_prompt(ui);
        ui.add_space(6.0);
        self.render_allocation_controls(ui, seed_counter);
        ui.add_space(6.0);
        self.render_cell_editors(ui);
        ui.separator();

        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref err) = self.validation_error {
            ui.colored_label(palette.error, err);
            return;
        }
        self.render_metrics_grid(ui, &palette);
        ui.add_space(6.0);
        self.render_rr_forest_plot(ui, &palette);
        let _ = self.callout_manager.render_ui(ui);
    }

    fn render_predict_winner_prompt(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.group(|ui| {
            ui.label(
                egui::RichText::new(
                    "Predict Winner: Treatment lowers adverse events in Mild (RR₁ < 1) AND Severe (RR₂ < 1) strata. Who wins the Pooled cohort?",
                )
                .strong(),
            );
            ui.horizontal_wrapped(|ui| {
                ui.radio_value(
                    &mut self.config.prediction,
                    Some(WinnerPrediction::TreatmentWins),
                    "Treatment Wins Pooled (RR < 1)",
                );
                ui.radio_value(
                    &mut self.config.prediction,
                    Some(WinnerPrediction::ControlWinsReversal),
                    "Control Appears to Win (RR > 1 — Simpson's Reversal!)",
                );
                ui.radio_value(
                    &mut self.config.prediction,
                    Some(WinnerPrediction::Tie),
                    "Tie (RR = 1)",
                );
            });
            if let Some(pred) = self.config.prediction {
                let matched = (pred == WinnerPrediction::ControlWinsReversal
                    && self.is_simpson_reversal)
                    || (pred == WinnerPrediction::TreatmentWins
                        && !self.is_simpson_reversal
                        && self
                            .pooled_eval
                            .as_ref()
                            .is_some_and(|p| p.risk.relative_risk < 1.0));
                let color = if matched {
                    palette.success
                } else {
                    palette.warning
                };
                let msg = if self.is_simpson_reversal {
                    "Confounded Severity! Assigning 80% of Severe patients to Treatment inflates Pooled RR > 1 despite RR < 1 in every stratum."
                } else {
                    "Balanced Allocation! Equalizing Treatment shares across strata aligns Pooled RR < 1 with the true protective effect."
                };
                ui.colored_label(color, msg);
            }
        });
    }

    fn render_allocation_controls(&mut self, ui: &mut egui::Ui, seed_counter: &mut u64) {
        let mut slider_changed = false;
        slider_changed |= ui
            .add(
                egui::Slider::new(&mut self.config.mild_treatment_share, 0.10..=0.90)
                    .text("Mild Stratum -> Treatment Share"),
            )
            .changed();
        slider_changed |= ui
            .add(
                egui::Slider::new(&mut self.config.severe_treatment_share, 0.10..=0.90)
                    .text("Severe Stratum -> Treatment Share"),
            )
            .changed();
        slider_changed |= ui
            .add(
                egui::Slider::new(&mut self.config.stratum_size, 20..=800)
                    .text("Patients per Stratum"),
            )
            .changed();
        if slider_changed {
            self.apply_allocation_sliders();
        }

        ui.horizontal_wrapped(|ui| {
            if ui
                .button("⚖ Balance via StratifiedRandomizer")
                .accessible_hover_text(
                    "Run StratifiedRandomizer<BlockRandomizer> with seeded OxidizeRng to balance Mild and Severe arms",
                )
                .clicked()
            {
                self.balance_via_stratified_randomizer(seed_counter);
            }
            if ui
                .button("🔄 Reset Confounded Simpson Preset")
                .accessible_hover_text("Restore 20% Mild / 80% Severe confounded allocation")
                .clicked()
            {
                self.reset_confounded_preset();
            }
        });
    }

    fn render_cell_editors(&mut self, ui: &mut egui::Ui) {
        let mut edited = false;
        egui::CollapsingHeader::new("Inspect / Edit 2×2 Cell Counts (min = 1)").show(ui, |ui| {
            edited |= render_stratum_row(ui, "Mild", &mut self.config.mild_counts);
            edited |= render_stratum_row(ui, "Severe", &mut self.config.severe_counts);
        });
        if edited {
            self.recalculate();
        }
    }

    fn render_metrics_grid(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        let (Some(mild), Some(sev), Some(pool)) =
            (&self.mild_eval, &self.severe_eval, &self.pooled_eval)
        else {
            return;
        };
        egui::Grid::new("simpson_metrics_grid")
            .striped(true)
            .spacing([18.0, 6.0])
            .show(ui, |ui| {
                ui.strong("Cohort");
                ui.strong("Treat Events");
                ui.strong("Ctrl Events");
                ui.strong("RR [95% CI]");
                ui.strong("Chi² (p-val)");
                ui.end_row();
                render_eval_row(ui, "Mild Stratum", mild, palette.success);
                render_eval_row(ui, "Severe Stratum", sev, palette.success);
                let pool_color = if self.is_simpson_reversal {
                    palette.error
                } else {
                    palette.success
                };
                render_eval_row(ui, "Pooled Total", pool, pool_color);
            });
    }

    fn render_rr_forest_plot(&mut self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        let (Some(mild), Some(sev), Some(pool)) = (
            self.mild_eval.clone(),
            self.severe_eval.clone(),
            self.pooled_eval.clone(),
        ) else {
            return;
        };
        Plot::new("simpson_rr_forest_plot")
            .height(210.0)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Null Effect (RR = 1.0)", vec![[1.0, 0.4], [1.0, 3.6]])
                        .color(palette.muted_text),
                );
                draw_ci_interval(
                    plot_ui,
                    "Mild 95% CI (y=3)",
                    3.0,
                    &mild.risk,
                    palette.series_color(2),
                );
                draw_ci_interval(
                    plot_ui,
                    "Severe 95% CI (y=2)",
                    2.0,
                    &sev.risk,
                    palette.series_color(5),
                );
                let pool_col = if self.is_simpson_reversal {
                    palette.error
                } else {
                    palette.series_color(0)
                };
                draw_ci_interval(plot_ui, "Pooled 95% CI (y=1)", 1.0, &pool.risk, pool_col);
                self.callout_manager.render_on_plot(plot_ui);
            });
    }
}

fn render_stratum_row(ui: &mut egui::Ui, label: &str, c: &mut TableCounts) -> bool {
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        ui.strong(format!("{label}:"));
        ui.label("T+:");
        changed |= ui
            .add(egui::DragValue::new(&mut c.treat_event).range(1..=1000))
            .changed();
        ui.label("T-:");
        changed |= ui
            .add(egui::DragValue::new(&mut c.treat_no_event).range(1..=1000))
            .changed();
        ui.label("C+:");
        changed |= ui
            .add(egui::DragValue::new(&mut c.ctrl_event).range(1..=1000))
            .changed();
        ui.label("C-:");
        changed |= ui
            .add(egui::DragValue::new(&mut c.ctrl_no_event).range(1..=1000))
            .changed();
    });
    changed
}

fn render_eval_row(ui: &mut egui::Ui, name: &str, ev: &TableEvaluation, rr_color: egui::Color32) {
    let t_tot = ev.table.treatment_event + ev.table.treatment_no_event;
    let c_tot = ev.table.control_event + ev.table.control_no_event;
    ui.label(name);
    ui.label(format!(
        "{}/{} ({:.1}%)",
        ev.table.treatment_event,
        t_tot,
        100.0 * ev.table.treatment_event as f64 / t_tot.max(1) as f64
    ));
    ui.label(format!(
        "{}/{} ({:.1}%)",
        ev.table.control_event,
        c_tot,
        100.0 * ev.table.control_event as f64 / c_tot.max(1) as f64
    ));
    ui.colored_label(
        rr_color,
        format!(
            "{:.2} [{:.2}, {:.2}]",
            ev.risk.relative_risk, ev.risk.rr_ci.0, ev.risk.rr_ci.1
        ),
    );
    ui.label(format!(
        "{:.2} ({:.3})",
        ev.chi_sq.statistic, ev.chi_sq.p_value
    ));
    ui.end_row();
}

fn draw_ci_interval(
    plot_ui: &mut egui_plot::PlotUi,
    name: &str,
    y: f64,
    risk: &RiskMetrics,
    color: egui::Color32,
) {
    plot_ui.line(
        Line::new(name, vec![[risk.rr_ci.0, y], [risk.rr_ci.1, y]])
            .color(color)
            .width(3.0_f32),
    );
    plot_ui.points(
        Points::new(format!("{name} Estimate"), vec![[risk.relative_risk, y]])
            .radius(5.0_f32)
            .color(color),
    );
}

// [cite:clinical_trials_statistics]
