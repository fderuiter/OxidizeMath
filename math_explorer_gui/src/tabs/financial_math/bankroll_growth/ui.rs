//! Immediate-mode controls and plot rendering for [`super::BankrollGrowthTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::bankroll_ensemble::BankrollLabMode;
use super::state::{is_default_stake_multiplier, BankrollWagerModel};
use super::BankrollGrowthTool;
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{Line, Plot};

impl BankrollGrowthTool {
    pub(super) fn render_workspace(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Parameters");
                let mut changed = self.render_parameter_sliders(ui);
                changed |= self.render_model_and_drawdown_sliders(ui);
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
                    .text("Initial Bankroll"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.probability, 0.01..=0.99).text("Win Probability (p)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.odds, 0.1..=10.0).text("Net Odds (b)"))
            .accessible_hover_text("Net profit multiplier. E.g., for +100 / even money, b = 1.0.")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.num_bets, 10..=1000).text("Number of Bets"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.stake_multiplier, 0.0..=3.0)
                    .text("Stake Multiplier (× f*)"),
            )
            .accessible_hover_text(
                "Multiple of Kelly optimal stake f*. Above 2.0f*, expected log-growth turns negative.",
            )
            .changed();
        changed
    }

    fn render_model_and_drawdown_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.label("Settlement Model:");
            for model in BankrollWagerModel::ALL {
                changed |= ui
                    .radio_value(&mut self.wager_model, model, model.label())
                    .changed();
            }
        });

        changed |= ui
            .add(
                egui::Slider::new(&mut self.push_probability, 0.0..=0.50)
                    .text("Push Probability (p_push)"),
            )
            .accessible_hover_text("Stake-refund push probability for integer spreads / totals")
            .changed();

        if self.wager_model == BankrollWagerModel::QuarterLine {
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.half_win_probability, 0.0..=0.40)
                        .text("Half-Win Probability (+b/2)"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.half_loss_probability, 0.0..=0.40)
                        .text("Half-Loss Probability (-1/2)"),
                )
                .changed();
        } else if self.wager_model == BankrollWagerModel::SimultaneousSlate {
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.slate_size, 2..=32)
                        .text("Simultaneous Games per Round (K)"),
                )
                .changed();
        }

        ui.horizontal_wrapped(|ui| {
            ui.label("Drawdown Risk Bound:");
            changed |= ui
                .add(egui::Slider::new(&mut self.drawdown_alpha, 0.20..=0.95).text("Floor α"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.drawdown_beta, 0.01..=0.50).text("Tail Prob β"))
                .changed();
        });
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
                    "Volatility-Drag Paradox Lab",
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
                    "Reset initial bankroll, win probability, odds, and bet count to default values",
                )
                .clicked()
            {
                self.reset_defaults();
                *changed = false;
            }
            ui.small(format!(
                "(Seed #{}) f* = {:.3} | Active f = {:.3} | EV = {:+.3} | Log-Growth g = {:+.4}",
                self.seed_counter,
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
        ui.heading("Bankroll Simulation");
        Plot::new("bankroll_growth_plot")
            .view_aspect(2.0)
            .legend(egui_plot::Legend::default())
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
                plot_ui.line(
                    Line::new(
                        "Drawdown-Constrained Kelly",
                        &self.risk_constrained_points[..],
                    )
                    .color(palette.series_color(1)),
                );
                if !is_default_stake_multiplier(&self.stake_multiplier) {
                    let label = format!("Active Stake ({:.2}× f*)", self.stake_multiplier);
                    plot_ui.line(
                        Line::new(label, &self.active_stake_points[..])
                            .color(palette.series_color(0))
                            .width(2.2_f32),
                    );
                }
                self.callout_manager.render_on_plot(plot_ui);
            });
    }
}

// [cite:clinical_trials_statistics]
