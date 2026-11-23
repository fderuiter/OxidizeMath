#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    GrpoAlignmentLabTool, GrpoCalibrationSnapshot, GrpoScenarioPreset, ACCURACY_MAX, ACCURACY_MIN,
    BETA_MAX, BETA_MIN, EPSILON_MAX, EPSILON_MIN, GROUP_SIZE, LAMBDA_MAX, LAMBDA_MIN, SKEW_MAX,
    SKEW_MIN,
};
use crate::accessibility::AccessibleTheoryHover;
use crate::app::SemanticPalette;
use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Line, LineStyle, MarkerShape, Plot, PlotPoints, Points};

impl GrpoAlignmentLabTool {
    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "grpo_alignment_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("GRPO Alignment Lab");
            ui.separator();

            egui::ComboBox::from_label("Rollout Scenario")
                .selected_text(self.scenario.label())
                .show_ui(ui, |ui| {
                    for preset in GrpoScenarioPreset::ALL {
                        ui.selectable_value(&mut self.scenario, preset, preset.label());
                    }
                });

            ui.add(
                egui::Slider::new(&mut self.epsilon, EPSILON_MIN..=EPSILON_MAX)
                    .text("Clip Epsilon"),
            );
            ui.add(egui::Slider::new(&mut self.beta, BETA_MIN..=BETA_MAX).text("KL Weight Beta"));
            ui.add(
                egui::Slider::new(&mut self.solver_accuracy, ACCURACY_MIN..=ACCURACY_MAX)
                    .text("Solver Accuracy"),
            );
            ui.add(
                egui::Slider::new(&mut self.repetition_lambda, LAMBDA_MIN..=LAMBDA_MAX)
                    .text("Repetition Lambda"),
            );
            ui.add(egui::Slider::new(&mut self.cluster_size, 1..=GROUP_SIZE).text("Cluster Size"));
            ui.add(
                egui::Slider::new(&mut self.confidence_skew, SKEW_MIN..=SKEW_MAX)
                    .text("Confidence Skew"),
            );
            self.clamp_parameters();

            ui.separator();
            let snap = self.evaluate_snapshot();
            self.draw_diagnostic_readouts(ui, &snap);
        });
    }

    fn draw_diagnostic_readouts(&self, ui: &mut egui::Ui, snap: &GrpoCalibrationSnapshot) {
        ui.heading("Live Alignment Diagnostics");
        ui.label(format!("GRPO Surrogate Loss: {:.4}", snap.grpo_loss));
        ui.label(format!(
            "Uncertainty Reward: {:.3} | Penalty: {:.3}",
            snap.active_uncertainty_reward, snap.active_repetition_penalty
        ));
        ui.label(format!(
            "Challenger Composite Reward: {:.3}",
            snap.active_composite_reward
        ));
        ui.label(format!(
            "KL Divergence Lower Bound: {:.3}",
            snap.kl_lower_bound
        ));
        ui.separator();
        ui.heading("Self-Calibration Metrics");
        ui.label(format!(
            "Solver Consensus Accuracy: {:.1}%",
            snap.empirical_solver_acc * 100.0
        ));
        ui.label(format!(
            "Answer Entropy: {:.3} nats -> Temp: {:.3}",
            snap.answer_entropy, snap.dynamic_temperature
        ));
        ui.label(format!(
            "KL Distillation Loss: {:.5}",
            snap.calibration_kl_loss
        ));
    }

    pub(super) fn draw_central_plots(&mut self, parent_ui: &mut egui::Ui) {
        let snap = self.evaluate_snapshot();
        let series = self.compute_surrogate_and_reward_series();
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            let total_h = ui.available_height().max(260.0);
            let top_h = total_h * 0.52;
            let bot_h = (total_h - top_h - 14.0).max(110.0);

            self.draw_surrogate_plot(ui, &palette, &series, &snap, top_h);
            ui.separator();
            self.draw_calibration_bars_plot(ui, &palette, &snap, bot_h);
        });
    }

    fn draw_surrogate_plot(
        &self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        series: &[(String, Vec<[f64; 2]>)],
        snap: &GrpoCalibrationSnapshot,
        height: f32,
    ) {
        let clipped_pts = PlotPoints::from(series[0].1.clone());
        let unclipped_pts = PlotPoints::from(series[1].1.clone());
        let challenger_pts = PlotPoints::from(series[2].1.clone());

        let rollout_markers: Vec<[f64; 2]> = snap
            .advantages
            .iter()
            .map(|&a| {
                let r_t = (1.0 + 0.22 * a).clamp(0.25, 1.95);
                let c_r = r_t.clamp(1.0 - self.epsilon, 1.0 + self.epsilon);
                [r_t, (r_t * a).min(c_r * a)]
            })
            .collect();

        let resp = Plot::new("grpo_surrogate_and_challenger_plot")
            .height(height)
            .legend(Legend::default())
            .x_axis_label("Policy Ratio / Solver Accuracy")
            .y_axis_label("Objective / Challenger Reward")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Clipped Surrogate", clipped_pts)
                        .color(palette.accent)
                        .width(2.5_f32),
                );
                plot_ui.line(
                    Line::new("Unclipped Surrogate", unclipped_pts)
                        .color(palette.muted_text)
                        .style(LineStyle::Dashed { length: 6.0 })
                        .width(1.6_f32),
                );
                plot_ui.line(
                    Line::new("Challenger Composite Reward", challenger_pts)
                        .color(palette.success)
                        .width(2.2_f32),
                );
                plot_ui.points(
                    Points::new("Group Rollout Samples", rollout_markers)
                        .color(palette.warning)
                        .radius(4.5_f32)
                        .shape(MarkerShape::Diamond),
                );
                plot_ui.points(
                    Points::new(
                        "Active Challenger Point",
                        vec![[self.solver_accuracy, snap.active_composite_reward]],
                    )
                    .color(palette.error)
                    .radius(5.5_f32)
                    .shape(MarkerShape::Circle),
                );
            })
            .response;
        resp.accessible_theory_hover(self);
    }

    fn draw_calibration_bars_plot(
        &self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        snap: &GrpoCalibrationSnapshot,
        height: f32,
    ) {
        let mut pred_bars = Vec::new();
        let mut score_bars = Vec::new();
        let mut target_bars = Vec::new();
        for idx in 0..snap.target_distribution.len() {
            let base_x = (idx + 1) as f64;
            pred_bars.push(
                Bar::new(base_x - 0.22, snap.predicted_distribution[idx])
                    .width(0.20)
                    .fill(palette.warning),
            );
            score_bars.push(
                Bar::new(base_x, snap.consistency_scores[idx])
                    .width(0.20)
                    .fill(palette.accent),
            );
            target_bars.push(
                Bar::new(base_x + 0.22, snap.target_distribution[idx])
                    .width(0.20)
                    .fill(palette.success),
            );
        }

        let resp = Plot::new("grpo_self_calibration_bars_plot")
            .height(height)
            .legend(Legend::default())
            .x_axis_label("Candidate Response Index")
            .y_axis_label("Probability / Consistency Score")
            .show(ui, |plot_ui| {
                plot_ui.bar_chart(
                    BarChart::new("Predicted Confidence Q", pred_bars).color(palette.warning),
                );
                plot_ui.bar_chart(
                    BarChart::new("Soft Self-Consistency S", score_bars).color(palette.accent),
                );
                plot_ui.bar_chart(
                    BarChart::new("Entropy-Scaled Target P", target_bars).color(palette.success),
                );
            })
            .response;
        resp.accessible_theory_hover(self);
    }
}

impl scientific_metadata::theory::TheoryDescribable for GrpoAlignmentLabTool {
    fn theory_description(&self) -> String {
        format!(
            "Group Relative Policy Optimization standardizes rollout rewards $\\hat{{A}}_i = (r_i - \\mu_r)/(\\sigma_r + \\varepsilon)$ without a critic network, optimizing $L_{{\\mathrm{{GRPO}}}}(\\theta)$ with clipping $\\epsilon = {:.2}$ and KL penalty $\\beta = {:.2}$ alongside entropy-scaled self-calibration target $P_i \\propto \\exp(S_i / T(H))$.",
            self.epsilon, self.beta
        )
    }

    fn phonetic_description(&self) -> String {
        format!(
            "Group Relative Policy Optimization laboratory with clip epsilon {:.2}, KL penalty beta {:.2}, Challenger uncertainty curriculum, and entropy-scaled KL self-calibration.",
            self.epsilon, self.beta
        )
    }

    fn theory_citation(&self) -> String {
        "Shao, Z. et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models; Wang, X. et al. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models (arXiv:2203.11171).".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "epsilon".into(),
            "Symmetric policy ratio clipping radius limiting abrupt probability ratio updates."
                .into(),
        );
        map.insert(
            "beta".into(),
            "Regularization weight multiplying the reference-policy Kullback-Leibler divergence term.".into(),
        );
        map.insert(
            "solver_accuracy".into(),
            "Empirical Solver consensus rate peaking Challenger uncertainty reward at fifty percent accuracy.".into(),
        );
        map.insert(
            "calibration_target".into(),
            "Boltzmann probability distribution over soft self-consistency scores scaled by answer Shannon entropy.".into(),
        );
        map
    }
}
