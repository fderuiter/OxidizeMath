#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    LoraHubSimplexTool, LoraTaskPreset, ADAPTER_COUNT, ADAPTER_NAMES, ALPHA_MAX, ALPHA_MIN,
    BOUND_MAX, BOUND_MIN, STEP_MAX, STEP_MIN,
};
use crate::accessibility::AccessibleTheoryHover;
use crate::app::SemanticPalette;
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Line, LineStyle, MarkerShape, Plot, PlotPoints, Points};
use math_explorer::applied::lorahub::SimplexStepOutcome;

impl LoraHubSimplexTool {
    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "lorahub_simplex_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("LoraHub Simplex Composer");
            ui.separator();

            if let Some(action) = PlaybackTransport::in_thread(
                "lorahub_simplex_transport",
                "LoraHub Simplex Composer",
                self.is_playing,
            )
            .show(ui)
            {
                match action {
                    PlaybackAction::Play => self.is_playing = true,
                    PlaybackAction::Pause => self.is_playing = false,
                    PlaybackAction::Step => self.step_simplex_once(),
                    PlaybackAction::Reset => {
                        self.is_playing = false;
                        self.reset_simplex();
                    }
                }
            }

            ui.separator();
            self.draw_simplex_controls(ui);
            ui.separator();
            self.draw_simplex_metrics(ui);
            ui.separator();
            self.heatmap.show(ui, &self.merged_matrix);
        });
    }

    fn draw_simplex_controls(&mut self, ui: &mut egui::Ui) {
        let mut next_preset = self.task_preset;
        egui::ComboBox::from_label("Target Task")
            .selected_text(next_preset.label())
            .show_ui(ui, |ui| {
                for preset in LoraTaskPreset::ALL {
                    ui.selectable_value(&mut next_preset, preset, preset.label());
                }
            });
        let mut reset_needed = next_preset != self.task_preset;
        self.task_preset = next_preset;

        reset_needed |= ui
            .add(egui::Slider::new(&mut self.alpha, ALPHA_MIN..=ALPHA_MAX).text("L1 Penalty Alpha"))
            .changed();
        reset_needed |= ui
            .add(
                egui::Slider::new(&mut self.initial_step, STEP_MIN..=STEP_MAX)
                    .text("Simplex Step Size"),
            )
            .changed();
        reset_needed |= ui
            .add(
                egui::Slider::new(&mut self.weight_bound, BOUND_MIN..=BOUND_MAX)
                    .text("Weight Box Bound"),
            )
            .changed();
        self.clamp_parameters();
        if reset_needed {
            self.is_playing = false;
            self.reset_simplex();
        }
    }

    fn draw_simplex_metrics(&self, ui: &mut egui::Ui) {
        let best = self.simplex.best_weights();
        let active_count = best.iter().filter(|w| w.abs() >= 0.05).count();
        let l1_mean = best.iter().map(|w| w.abs()).sum::<f64>() / (best.len().max(1) as f64);
        let outcome_str = match self.last_outcome {
            SimplexStepOutcome::Converged => "Converged",
            SimplexStepOutcome::Reflected => "Reflection",
            SimplexStepOutcome::Expanded => "Expansion",
            SimplexStepOutcome::ContractedOutside => "Outside Contraction",
            SimplexStepOutcome::ContractedInside => "Inside Contraction",
            SimplexStepOutcome::Shrunk => "Simplex Shrink",
        };
        ui.label(format!(
            "Step: {} ({outcome_str})",
            self.simplex.steps_taken()
        ));
        ui.label(format!(
            "Best Objective: {:.5} (Spread: {:.2e})",
            self.simplex.best_value(),
            self.simplex.value_spread()
        ));
        ui.label(format!(
            "Active Adapters (|w| >= 0.05): {active_count}/{ADAPTER_COUNT} | Mean |w|: {l1_mean:.3}"
        ));
    }

    pub(super) fn draw_central_panel(&mut self, parent_ui: &mut egui::Ui) {
        let series = self.build_plot_series();
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            let total_h = ui.available_height().max(260.0);
            let top_h = total_h * 0.56;
            let bot_h = (total_h - top_h - 14.0).max(105.0);

            self.draw_contour_simplex_plot(ui, &palette, &series, top_h);
            ui.separator();
            self.draw_sparsity_bars_plot(ui, &palette, bot_h);
        });
    }

    fn draw_contour_simplex_plot(
        &self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        series: &[(String, Vec<[f64; 2]>)],
        height: f32,
    ) {
        let best = self.simplex.best_weights();
        let (target, _) = self.task_preset.target_and_stiffness();
        let resp = Plot::new("lorahub_contour_and_simplex_plot")
            .height(height)
            .data_aspect(1.0)
            .legend(Legend::default())
            .x_axis_label("Math LoRA Weight w1")
            .y_axis_label("Code LoRA Weight w2")
            .show(ui, |plot_ui| {
                for (name, pts) in series.iter().skip(2) {
                    plot_ui.line(
                        Line::new(name.clone(), PlotPoints::from(pts.clone()))
                            .color(palette.graph_edge)
                            .style(LineStyle::Dashed { length: 5.0 })
                            .width(1.3_f32),
                    );
                }
                plot_ui.line(
                    Line::new("Centroid Trajectory", PlotPoints::from(series[1].1.clone()))
                        .color(palette.success)
                        .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new("Simplex Polytope", PlotPoints::from(series[0].1.clone()))
                        .color(palette.accent)
                        .width(2.5_f32),
                );
                plot_ui.points(
                    Points::new("Task Optimum", vec![[target[0], target[1]]])
                        .color(palette.warning)
                        .radius(5.0_f32)
                        .shape(MarkerShape::Cross),
                );
                plot_ui.points(
                    Points::new("Best Simplex Vertex", vec![[best[0], best[1]]])
                        .color(palette.error)
                        .radius(5.5_f32)
                        .shape(MarkerShape::Diamond),
                );
            })
            .response;
        resp.accessible_theory_hover(self);
    }

    fn draw_sparsity_bars_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette, height: f32) {
        let best = self.simplex.best_weights();
        let mut signed_bars = Vec::with_capacity(ADAPTER_COUNT);
        let mut abs_bars = Vec::with_capacity(ADAPTER_COUNT);
        for (i, name) in ADAPTER_NAMES.iter().enumerate() {
            let x = (i + 1) as f64;
            let w = best[i];
            signed_bars.push(
                Bar::new(x - 0.16, w)
                    .name(format!("{name} Signed"))
                    .width(0.28)
                    .fill(palette.accent),
            );
            abs_bars.push(
                Bar::new(x + 0.16, w.abs())
                    .name(format!("{name} Magnitude"))
                    .width(0.28)
                    .fill(palette.warning),
            );
        }
        let resp = Plot::new("lorahub_adapter_sparsity_bars")
            .height(height)
            .legend(Legend::default())
            .x_axis_label("LoRA Module Index (1=Math, 2=Code, 3=Logic, 4=Style)")
            .y_axis_label("Adapter Weight / L1 Magnitude")
            .show(ui, |plot_ui| {
                plot_ui
                    .bar_chart(BarChart::new("Signed Weight w", signed_bars).color(palette.accent));
                plot_ui.bar_chart(
                    BarChart::new("Sparsity Magnitude |w|", abs_bars).color(palette.warning),
                );
            })
            .response;
        resp.accessible_theory_hover(self);
    }
}

impl scientific_metadata::theory::TheoryDescribable for LoraHubSimplexTool {
    fn theory_description(&self) -> String {
        format!(
            "LoraHub composes $N$ pre-trained LoRA modules $\\hat{{m}} = \\sum_{{i=1}}^N w_i m_i$ by minimizing few-shot loss plus L1 sparsity $L_{{\\mathrm{{task}}}}(w) + \\frac{{\\alpha}}{{N}}\\sum_i |w_i|$ ($\\alpha = {:.2}$) via gradient-free Nelder-Mead simplex reflections, expansions, and contractions.",
            self.alpha
        )
    }

    fn phonetic_description(&self) -> String {
        format!(
            "LoraHub gradient-free Nelder-Mead simplex optimizer blending four LoRA state dictionaries with L1 regularization alpha {:.2}.",
            self.alpha
        )
    }

    fn theory_citation(&self) -> String {
        "Huang, C., Liu, Q., Lin, B. Y., Pang, T., Du, C., & Lin, M. (2023). LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition (arXiv:2307.13269).".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "alpha".into(),
            "L1 regularization penalty encouraging sparse scalar combinations of task LoRA modules.".into(),
        );
        map.insert(
            "initial_step".into(),
            "Initial coordinate offset constructing the N plus 1 vertices of the Nelder-Mead simplex.".into(),
        );
        map.insert(
            "weight_bound".into(),
            "Symmetric box constraint preventing runaway adapter extrapolation during black-box search.".into(),
        );
        map
    }
}
