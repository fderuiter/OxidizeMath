//! UI layout, plot rendering, and dataset/figure export builders for [`super::BankrollGrowthTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::bankroll_ensemble::BankrollLabMode;
use super::{is_default_stake_multiplier, BankrollGrowthTool};
use crate::accessibility::{
    paint_canvas_focus_ring, register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::SemanticPalette;
use eframe::egui;
use eframe::egui::accesskit::Role;
use egui_plot::{ExportPrimitive, Legend, Line, Plot, PlotPoint};

fn to_pairs(pts: &[PlotPoint]) -> Vec<[f64; 2]> {
    pts.iter().map(|p| [p.x, p.y]).collect()
}

fn polyline(name: &str, pts: &[PlotPoint], color: egui::Color32, width: f32) -> ExportPrimitive {
    ExportPrimitive::Polyline {
        name: name.to_string(),
        points: to_pairs(pts),
        color,
        stroke_width: width,
        closed: false,
        fill: None,
    }
}

impl BankrollGrowthTool {
    pub(super) fn render_workspace(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Parameters");
                let mut changed = self.render_parameter_sliders(ui);
                self.render_action_bar(ui, &mut changed);
                if changed {
                    self.recalculate();
                }
                ui.separator();

                let palette = SemanticPalette::from_visuals(ui.visuals());
                match self.mode {
                    BankrollLabMode::KellyStrategies => self.render_kelly_plot(ui, &palette),
                    BankrollLabMode::VolatilityDrag => {
                        let lab_changed = self.ensemble_lab.show_lab_ui(
                            ui,
                            &mut self.initial_bankroll,
                            &mut self.probability,
                            &mut self.odds,
                            &mut self.num_bets,
                        );
                        if lab_changed {
                            self.recalculate();
                        }
                    }
                }
            });
        });
    }

    fn render_parameter_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= ui
            .add(
                egui::Slider::new(&mut self.initial_bankroll, 100.0..=10_000.0)
                    .text("Initial Capital"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.probability, 0.01..=0.99)
                    .text("Success Probability (p)"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.odds, 0.1..=10.0).text("Net Payoff Ratio (b)"))
            .accessible_hover_text(
                "Net profit multiplier per unit allocated (e.g., even payoff b = 1.0).",
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.num_bets, 10..=1000).text("Horizon Steps"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.stake_multiplier, 0.0..=3.0)
                    .text("Allocation Multiplier (× f*)"),
            )
            .accessible_hover_text(
                "Multiple of optimal Kelly fraction f*. Above 2.0f*, expected log-growth turns negative.",
            )
            .changed();
        changed
    }

    fn render_action_bar(&mut self, ui: &mut egui::Ui, changed: &mut bool) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Lab Mode:");
            *changed |= ui
                .radio_value(
                    &mut self.mode,
                    BankrollLabMode::KellyStrategies,
                    "Kelly Fractions",
                )
                .changed();
            *changed |= ui
                .radio_value(
                    &mut self.mode,
                    BankrollLabMode::VolatilityDrag,
                    "Volatility-Drag & Fan Bands Lab",
                )
                .changed();
            ui.separator();
            if ui
                .button("🎲 Rerun Simulation")
                .accessible_hover_text(
                    "Advance deterministic RNG seed and draw a fresh sample trajectory",
                )
                .clicked()
            {
                self.rerun_simulation();
                *changed = false;
            }
            if ui
                .button("↻ Reset to Defaults")
                .accessible_hover_text(
                    "Reset initial capital, probability, payoff ratio, and horizon",
                )
                .clicked()
            {
                self.reset_defaults();
                *changed = false;
            }
            ui.small(format!(
                "(Seed #{}, {:.2} ms) f* = {:.3} | Active f = {:.3} | EV = {:+.3} | g = {:+.4}",
                self.seed_counter,
                self.last_compute_ms,
                self.kelly_f_star,
                self.active_fraction,
                self.unit_expected_value,
                self.active_growth_rate
            ));
        });
    }

    fn render_kelly_plot(&mut self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        if let Some(ref err) = self.error_msg {
            ui.colored_label(palette.error, err);
            return;
        }
        ui.heading("Capital Compounding Trajectories");
        let resp = Plot::new("bankroll_growth_plot")
            .view_aspect(2.0)
            .x_axis_label("Allocation Step t")
            .y_axis_label("Capital W(t)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Full Kelly", &self.full_kelly_points[..])
                        .color(palette.series_color(3)),
                );
                plot_ui.line(
                    Line::new("Half Kelly", &self.half_kelly_points[..])
                        .color(palette.series_color(5)),
                );
                plot_ui.line(
                    Line::new("Quarter Kelly", &self.quarter_kelly_points[..])
                        .color(palette.series_color(2)),
                );
                if !is_default_stake_multiplier(&self.stake_multiplier) {
                    let label = format!("Active Allocation ({:.2}× f*)", self.stake_multiplier);
                    plot_ui.line(
                        Line::new(label, &self.active_stake_points[..])
                            .color(palette.series_color(0))
                            .width(2.2_f32),
                    );
                }
                self.callout_manager.render_on_plot(plot_ui);
            });
        let summary = format!(
            "Full Kelly terminal {:.2}, Half Kelly {:.2}, Quarter Kelly {:.2}",
            self.full_kelly_points.last().map_or(0.0, |p| p.y),
            self.half_kelly_points.last().map_or(0.0, |p| p.y),
            self.quarter_kelly_points.last().map_or(0.0, |p| p.y)
        );
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "Fractional Kelly Capital Compounding Plot",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }

    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        match self.mode {
            BankrollLabMode::KellyStrategies => vec![
                ("Full Kelly".into(), to_pairs(&self.full_kelly_points)),
                ("Half Kelly".into(), to_pairs(&self.half_kelly_points)),
                ("Quarter Kelly".into(), to_pairs(&self.quarter_kelly_points)),
                (
                    "Active Allocation".into(),
                    to_pairs(&self.active_stake_points),
                ),
            ],
            BankrollLabMode::VolatilityDrag => vec![
                (
                    "Expected Wealth".into(),
                    to_pairs(&self.ensemble_lab.expected_points),
                ),
                (
                    "Analytical Median".into(),
                    to_pairs(&self.ensemble_lab.median_points),
                ),
                (
                    "Ensemble Median".into(),
                    to_pairs(&self.ensemble_lab.empirical_median_points),
                ),
                (
                    "5th Percentile".into(),
                    to_pairs(&self.ensemble_lab.p05_points),
                ),
                (
                    "25th Percentile".into(),
                    to_pairs(&self.ensemble_lab.p25_points),
                ),
                (
                    "75th Percentile".into(),
                    to_pairs(&self.ensemble_lab.p75_points),
                ),
                (
                    "95th Percentile".into(),
                    to_pairs(&self.ensemble_lab.p95_points),
                ),
                (
                    "Underwater Drawdown (%)".into(),
                    to_pairs(&self.ensemble_lab.underwater_points),
                ),
            ],
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let pal = crate::app::palette::DARK_PALETTE;
        match self.mode {
            BankrollLabMode::KellyStrategies => vec![
                polyline("Full Kelly", &self.full_kelly_points, pal.series[3], 2.0),
                polyline("Half Kelly", &self.half_kelly_points, pal.series[5], 2.0),
                polyline(
                    "Quarter Kelly",
                    &self.quarter_kelly_points,
                    pal.series[2],
                    2.0,
                ),
            ],
            BankrollLabMode::VolatilityDrag => vec![
                polyline(
                    "95th Percentile",
                    &self.ensemble_lab.p95_points,
                    pal.series[2],
                    1.5,
                ),
                polyline(
                    "5th Percentile",
                    &self.ensemble_lab.p05_points,
                    pal.warning,
                    1.5,
                ),
                polyline(
                    "Ensemble Median",
                    &self.ensemble_lab.empirical_median_points,
                    pal.series[5],
                    2.2,
                ),
                polyline(
                    "Analytical Median",
                    &self.ensemble_lab.median_points,
                    pal.series[3],
                    1.8,
                ),
            ],
        }
    }
}

// [cite:clinical_trials_statistics]
