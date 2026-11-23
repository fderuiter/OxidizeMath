#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! UI controls and accessible `egui_plot` visualizations for [`super::DtmcTensorWorkbenchTool`].

use super::state::DtmcModelPreset;
use super::{DtmcTensorWorkbenchTool, MAX_MC_STEPS, MIN_MC_STEPS};
use crate::accessibility::{
    paint_canvas_focus_ring, register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::SemanticPalette;
use crate::framework::{scoped_panel_id, PaneSidePanel};
use eframe::egui;
use eframe::egui::accesskit::Role;
use egui_plot::{Bar, BarChart, Legend, Line, Plot};

impl DtmcTensorWorkbenchTool {
    pub(super) fn render_workspace(&mut self, ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(ui, "dtmc_tensor_workbench_controls");
        PaneSidePanel::left(panel_id)
            .default_width(305.0)
            .show_inside(ui, |panel_ui| {
                self.render_controls_panel(panel_ui);
            });

        egui::CentralPanel::default().show_inside(ui, |central_ui| {
            self.render_plots_area(central_ui);
        });
    }

    pub(super) fn render_controls_panel(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.heading("DTMC & Tensor Controls");

        ui.label("Model Regime");
        ui.horizontal_wrapped(|ui| {
            for mode in DtmcModelPreset::ALL {
                if ui
                    .selectable_label(self.preset == mode, mode.label())
                    .accessible_hover_text("Select canonical DTMC or TransitionTensor model.")
                    .clicked()
                {
                    self.apply_model_preset(mode);
                }
            }
        });
        ui.separator();

        let mut changed = self.render_regime_parameters(ui);
        ui.separator();
        changed |= self.render_simulation_controls(ui);
        if changed {
            self.recompute_all();
        }

        ui.separator();
        self.render_scc_and_diagnostics(ui, &palette);
    }

    fn render_regime_parameters(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        match self.preset {
            DtmcModelPreset::ErgodicChain => {
                changed |= ui
                    .add(egui::Slider::new(&mut self.row0_stay, 0.05..=0.85).text("P[0, 0]"))
                    .changed();
                let max_01 = (0.95 - self.row0_stay).max(0.02);
                self.row0_to1 = self.row0_to1.min(max_01);
                changed |= ui
                    .add(egui::Slider::new(&mut self.row0_to1, 0.02..=max_01).text("P[0, 1]"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.row1_to0, 0.05..=0.85).text("P[1, 0]"))
                    .changed();
                let max_11 = (0.95 - self.row1_to0).max(0.02);
                self.row1_stay = self.row1_stay.min(max_11);
                changed |= ui
                    .add(egui::Slider::new(&mut self.row1_stay, 0.02..=max_11).text("P[1, 1]"))
                    .changed();
            }
            DtmcModelPreset::PeriodTwoWalk => {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.row1_to0, 0.05..=0.95)
                            .text("Center Split P[1, 0]"),
                    )
                    .changed();
            }
            DtmcModelPreset::AsymmetricAbsorbing => {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.row0_stay, 0.0..=0.85)
                            .text("Transient Stay P[0, 0]"),
                    )
                    .changed();
                let max_abs1 = (1.0 - self.row0_stay).max(0.0);
                self.row0_to1 = self.row0_to1.min(max_abs1);
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.row0_to1, 0.0..=max_abs1)
                            .text("Absorb-1 Rate P[0, 1]"),
                    )
                    .changed();
            }
            DtmcModelPreset::ShotClockTensor => {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.shot_clock_time, 3.0..=24.0)
                            .text("Shot Clock t (s)"),
                    )
                    .accessible_hover_text(
                        "Interpolate non-stationary TransitionTensor slice at shot-clock time t.",
                    )
                    .changed();
            }
        }
        changed
    }

    fn render_simulation_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Initial State:");
            for s in 0..3 {
                changed |= ui
                    .selectable_value(&mut self.initial_state, s, format!("S{s}"))
                    .changed();
            }
        });
        ui.horizontal(|ui| {
            ui.label("Hitting Target:");
            for s in 0..3 {
                changed |= ui
                    .selectable_value(&mut self.target_state, s, format!("S{s}"))
                    .changed();
            }
        });
        changed |= ui
            .add(egui::Slider::new(&mut self.mc_steps, MIN_MC_STEPS..=MAX_MC_STEPS).text("Steps"))
            .changed();
        ui.horizontal(|ui| {
            ui.label("Seed:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.seed).range(0..=1_000_000))
                .changed();
        });
        changed
    }

    fn render_scc_and_diagnostics(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.heading("Tarjan SCC & Nullspace");
        if let Some(ref err) = self.error_msg {
            ui.colored_label(palette.error, err);
            return;
        }
        for (idx, c) in self.classes.iter().enumerate() {
            let role = if c.is_closed { "Closed/Recurrent" } else { "Transient" };
            let col = if c.is_closed { palette.success } else { palette.warning };
            ui.colored_label(col, format!("Class {idx}: {:?} ({role})", c.states));
        }
        let period_str = self
            .chain_period
            .map_or_else(|| "None (multi-class)".to_string(), |p| p.to_string());
        ui.label(format!(
            "Period: {period_str} | Aperiodic: {}",
            self.is_aperiodic
        ));
        match self.stationary {
            Some(pi) => ui.colored_label(
                palette.success,
                format!("QR Stationary π = [{:.3}, {:.3}, {:.3}]", pi[0], pi[1], pi[2]),
            ),
            None => ui.colored_label(
                palette.warning,
                "Stationary π: None (multiple closed classes)",
            ),
        };
        if let Some(lim) = self.limiting {
            ui.label(format!(
                "Limiting (from S{}): [{:.3}, {:.3}, {:.3}]",
                self.initial_state, lim[0], lim[1], lim[2]
            ));
        }
        ui.small(format!("Solved in {:.2} ms", self.last_compute_ms));
    }

    fn render_plots_area(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let avail_h = ui.available_height().max(260.0);
        let top_h = (avail_h * 0.50).max(150.0);
        let bot_h = (avail_h - top_h - 18.0).max(130.0);

        self.render_distribution_chart(ui, &palette, top_h);
        ui.add_space(6.0);
        if self.preset == DtmcModelPreset::ShotClockTensor {
            self.render_tensor_interpolation_chart(ui, &palette, bot_h);
        } else {
            self.render_hitting_chart(ui, &palette, bot_h);
        }
    }

    fn render_distribution_chart(&self, ui: &mut egui::Ui, palette: &SemanticPalette, height: f32) {
        ui.strong("QR Stationary vs B = N·R Limiting vs Empirical Visits");
        let resp = Plot::new("dtmc_distributions_plot")
            .height(height)
            .x_axis_label("State Index i")
            .y_axis_label("Probability Mass")
            .include_y(0.0)
            .include_y(1.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                if let Some(pi) = self.stationary {
                    let bars: Vec<Bar> = (0..3)
                        .map(|s| Bar::new(s as f64 - 0.22, pi[s]).width(0.18))
                        .collect();
                    plot_ui.bar_chart(
                        BarChart::new("QR Stationary π", bars).color(palette.series_color(0)),
                    );
                }
                if let Some(lim) = self.limiting {
                    let bars: Vec<Bar> = (0..3)
                        .map(|s| Bar::new(s as f64, lim[s]).width(0.18))
                        .collect();
                    plot_ui.bar_chart(
                        BarChart::new("Limiting (B = N·R)", bars).color(palette.series_color(1)),
                    );
                }
                let emp_bars: Vec<Bar> = (0..3)
                    .map(|s| Bar::new(s as f64 + 0.22, self.empirical_freq[s]).width(0.18))
                    .collect();
                plot_ui.bar_chart(
                    BarChart::new("OxidizeRng Visits", emp_bars).color(palette.series_color(2)),
                );
            });
        let summary = format!(
            "DTMC preset {} stationary {:?}, limiting {:?}",
            self.preset.label(),
            self.stationary,
            self.limiting
        );
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "DTMC Stationary, Limiting, and Empirical Probability Bar Chart",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }

    fn render_hitting_chart(&self, ui: &mut egui::Ui, palette: &SemanticPalette, height: f32) {
        ui.strong(format!("Mean First Hitting Time to State S{}", self.target_state));
        let resp = Plot::new("dtmc_hitting_times_plot")
            .height(height)
            .x_axis_label("Starting State i")
            .y_axis_label("Expected Steps E_i[T_target]")
            .include_y(0.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                if let Some(h) = self.hitting_times {
                    let bars: Vec<Bar> = (0..3)
                        .map(|s| Bar::new(s as f64, h[s]).width(0.32))
                        .collect();
                    plot_ui.bar_chart(
                        BarChart::new("Mean Hitting Steps", bars).color(palette.series_color(3)),
                    );
                }
            });
        let summary = match self.hitting_times {
            Some(h) => format!(
                "Mean hitting times to S{}: [{:.2}, {:.2}, {:.2}]",
                self.target_state, h[0], h[1], h[2]
            ),
            None => format!("Target S{} unreachable from another closed class", self.target_state),
        };
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "DTMC Mean First Hitting Time Bar Chart",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }

    fn render_tensor_interpolation_chart(
        &self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        height: f32,
    ) {
        ui.strong("TransitionTensor Interpolated Absorption Across Shot Clock t ∈ [3, 24] s");
        let resp = Plot::new("dtmc_shot_clock_tensor_plot")
            .height(height)
            .x_axis_label("Shot Clock Remaining t (seconds)")
            .y_axis_label("Absorption Probability from S0")
            .include_y(0.0)
            .include_y(1.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Absorb S1 (Score)", &self.tensor_curve_s1[..])
                        .color(palette.success)
                        .width(2.2_f32),
                );
                plot_ui.line(
                    Line::new("Absorb S2 (Turnover)", &self.tensor_curve_s2[..])
                        .color(palette.error)
                        .width(2.2_f32),
                );
            });
        let summary = format!(
            "Shot-clock time {:.1}s interpolated absorption probabilities {:?}",
            self.shot_clock_time, self.limiting
        );
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "Shot-Clock TransitionTensor Absorption Probability Curves",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }
}

// [cite:clinical_trials_statistics]
