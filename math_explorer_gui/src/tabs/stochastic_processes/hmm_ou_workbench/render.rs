#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! UI controls and accessible `egui_plot` visualizations for [`super::HmmOuWorkbenchTool`].

use super::state::CalibrationMode;
use super::{
    HmmOuWorkbenchTool, MAX_EM_ITERS, MAX_OU_STEPS, MAX_SEQ_LEN, MIN_EM_ITERS, MIN_OU_STEPS,
    MIN_SEQ_LEN,
};
use crate::accessibility::{
    paint_canvas_focus_ring, register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::SemanticPalette;
use crate::framework::{scoped_panel_id, PaneSidePanel};
use eframe::egui;
use eframe::egui::accesskit::Role;
use egui_plot::{Legend, Line, Plot, PlotPoint};

impl HmmOuWorkbenchTool {
    pub(super) fn render_workspace(&mut self, ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(ui, "hmm_ou_workbench_controls");
        PaneSidePanel::left(panel_id)
            .default_width(310.0)
            .show_inside(ui, |panel_ui| {
                self.render_controls_panel(panel_ui);
            });

        egui::CentralPanel::default().show_inside(ui, |central_ui| {
            self.render_plots_area(central_ui);
        });
    }

    pub(super) fn render_controls_panel(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.heading("HMM & OU Calibration");

        ui.label("Inference / Calibration Mode");
        ui.horizontal_wrapped(|ui| {
            for m in CalibrationMode::ALL {
                if ui
                    .selectable_label(self.mode == m, m.label())
                    .accessible_hover_text("Select HMM Viterbi + Baum-Welch or Exact OU + AR(1) calibration.")
                    .clicked()
                {
                    self.set_mode(m);
                }
            }
        });
        ui.separator();

        let mut changed = match self.mode {
            CalibrationMode::HmmViterbiBaumWelch => self.render_hmm_sliders(ui),
            CalibrationMode::OuExactAndCalibration => self.render_ou_sliders(ui),
        };
        ui.horizontal(|ui| {
            ui.label("Seed:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.seed).range(0..=1_000_000))
                .changed();
        });
        if changed {
            self.recompute_all();
        }

        ui.separator();
        self.render_diagnostics(ui, &palette);
    }

    fn render_hmm_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut self.seq_len, MIN_SEQ_LEN..=MAX_SEQ_LEN).text("Seq Len T"))
            .accessible_hover_text("Observation sequence length up to 600 steps (log-space safe).")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.stay_prob, 0.55..=0.97).text("Regime Stay A_ii"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.emission_clarity, 0.55..=0.95).text("Clarity B_ii"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.em_iters, MIN_EM_ITERS..=MAX_EM_ITERS).text("EM Iters"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.prob_floor, 0.001..=0.10).text("Simplex Floor"))
            .changed();
        changed
    }

    fn render_ou_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut self.ou_mu, -2.0..=2.0).text("Long-Term Mean μ"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.ou_theta, 0.10..=5.0).text("Reversion θ"))
            .accessible_hover_text("Mean reversion speed theta; fast theta highlights ExactOuSolver accuracy.")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.ou_sigma, 0.05..=1.50).text("Volatility σ"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.ou_x0, -2.0..=2.0).text("Initial X0"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.ou_dt, 0.01..=0.25).text("Time Step Δt"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.ou_steps, MIN_OU_STEPS..=MAX_OU_STEPS).text("Steps"))
            .changed();
        changed
    }

    fn render_diagnostics(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.heading("Calibration Readout");
        if let Some(ref err) = self.error_msg {
            ui.colored_label(palette.error, err);
            return;
        }
        match self.mode {
            CalibrationMode::HmmViterbiBaumWelch => {
                let final_ll = self.em_log_liks.last().map_or(f64::NAN, |p| p.y);
                ui.label(format!("True ln P(Y) = {:.2}", self.true_log_lik));
                ui.colored_label(
                    palette.success,
                    format!(
                        "EM Final ln P(Y) = {:.2} (converged: {})",
                        final_ll, self.em_converged
                    ),
                );
                ui.label(format!(
                    "Log-Viterbi Alignment: {:.1}% ({} steps)",
                    self.viterbi_accuracy * 100.0,
                    self.seq_len
                ));
            }
            CalibrationMode::OuExactAndCalibration => {
                ui.label(format!(
                    "True: μ={:.3}, θ={:.3}, σ={:.3}",
                    self.ou_mu, self.ou_theta, self.ou_sigma
                ));
                if let Some((m, th, sig)) = self.estimated_ou {
                    ui.colored_label(
                        palette.success,
                        format!("AR(1) Fit: μ̂={m:.3}, θ̂={th:.3}, σ̂={sig:.3}"),
                    );
                }
            }
        }
        ui.small(format!("Solved in {:.2} ms", self.last_compute_ms));
    }

    fn render_plots_area(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let avail_h = ui.available_height().max(260.0);
        match self.mode {
            CalibrationMode::HmmViterbiBaumWelch => {
                let top_h = (avail_h * 0.48).max(145.0);
                let bot_h = (avail_h - top_h - 18.0).max(135.0);
                self.render_baum_welch_plot(ui, &palette, top_h);
                ui.add_space(6.0);
                self.render_viterbi_plot(ui, &palette, bot_h);
            }
            CalibrationMode::OuExactAndCalibration => {
                self.render_ou_plot(ui, &palette, (avail_h - 24.0).max(220.0));
            }
        }
    }

    fn render_baum_welch_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette, height: f32) {
        ui.strong("Monotonic Baum-Welch EM Log-Likelihood Trace ln P(Y | θ^(k))");
        let true_ref = [
            PlotPoint::new(0.0, self.true_log_lik),
            PlotPoint::new(self.em_iters as f64, self.true_log_lik),
        ];
        let resp = Plot::new("hmm_baum_welch_em_plot")
            .height(height)
            .x_axis_label("EM Iteration k")
            .y_axis_label("Log-Likelihood ln P(Y)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Baum-Welch ln P(Y)", &self.em_log_liks[..])
                        .color(palette.series_color(0))
                        .width(2.3_f32),
                );
                plot_ui.line(
                    Line::new("Oracle Model ln P(Y)", &true_ref[..])
                        .color(palette.warning)
                        .width(1.6_f32),
                );
            });
        let summary = format!(
            "Baum-Welch EM over {} steps, accuracy {:.1}%",
            self.seq_len,
            self.viterbi_accuracy * 100.0
        );
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "Baum-Welch EM Log-Likelihood Convergence Plot",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }

    fn render_viterbi_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette, height: f32) {
        ui.strong("Log-Space Viterbi Decoded Hidden Regime vs True Hidden State");
        let resp = Plot::new("hmm_viterbi_alignment_plot")
            .height(height)
            .x_axis_label("Observation Step t")
            .y_axis_label("Hidden Regime {0, 1}")
            .include_y(-0.1)
            .include_y(1.2)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("True Hidden State", &self.true_state_curve[..])
                        .color(palette.series_color(2))
                        .width(1.6_f32),
                );
                plot_ui.line(
                    Line::new("Log-Space Viterbi", &self.viterbi_curve[..])
                        .color(palette.series_color(1))
                        .width(2.0_f32),
                );
            });
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "Log-Space Viterbi State Sequence Alignment Plot",
            Some("Comparison of true hidden regime and log-space Viterbi MAP path."),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }

    fn render_ou_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette, height: f32) {
        ui.strong("ExactOuSolver vs Euler-Maruyama Trajectory & Exact ±1σ Conditional Envelope");
        let resp = Plot::new("ou_exact_vs_euler_plot")
            .height(height)
            .x_axis_label("Time t")
            .y_axis_label("State X_t")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("ExactOuSolver", &self.ou_exact_curve[..])
                        .color(palette.series_color(0))
                        .width(2.1_f32),
                );
                plot_ui.line(
                    Line::new("Euler-Maruyama", &self.ou_euler_curve[..])
                        .color(palette.series_color(1))
                        .width(1.6_f32),
                );
                plot_ui.line(
                    Line::new("Conditional Mean E[X_t]", &self.ou_mean_curve[..])
                        .color(palette.success)
                        .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new("Mean + 1σ", &self.ou_upper_band[..])
                        .color(palette.warning)
                        .width(1.2_f32),
                );
                plot_ui.line(
                    Line::new("Mean - 1σ", &self.ou_lower_band[..])
                        .color(palette.warning)
                        .width(1.2_f32),
                );
            });
        let summary = format!(
            "Ornstein-Uhlenbeck mu={:.2}, theta={:.2}, sigma={:.2}, estimated {:?}",
            self.ou_mu, self.ou_theta, self.ou_sigma, self.estimated_ou
        );
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "Ornstein-Uhlenbeck Exact vs Euler-Maruyama Trajectory Plot",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }
}

// [cite:clinical_trials_statistics]
