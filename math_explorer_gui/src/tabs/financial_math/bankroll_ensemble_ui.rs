//! UI rendering helpers for [`super::BankrollEnsembleLab`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    BankrollEnsembleLab, VolatilityHypothesis, ENSEMBLE_PATH_COUNT, MAX_BARRIER_FLOOR_RATIO,
    MAX_CUSTOM_FRACTION, MIN_BARRIER_FLOOR_RATIO,
};
use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use egui_plot::{Line, LineStyle, Plot};

impl BankrollEnsembleLab {
    /// Renders the hypothesis prompt, fraction/barrier sliders, solvency telemetry, and plot.
    pub fn show_lab_ui(
        &mut self,
        ui: &mut egui::Ui,
        initial_bankroll: &mut f64,
        probability: &mut f64,
        odds: &mut f64,
        num_bets: &mut usize,
    ) -> bool {
        let mut changed = self.render_hypothesis_card(ui);
        ui.add_space(4.0);
        changed |= self.render_fraction_and_preset_controls(
            ui,
            initial_bankroll,
            probability,
            odds,
            num_bets,
        );
        self.render_metrics_banner(ui, *num_bets);
        self.render_solvency_banner(ui, *num_bets);
        self.render_ensemble_plot(ui);
        let _ = self.callout_manager.render_ui(ui);
        changed
    }

    fn render_hypothesis_card(&mut self, ui: &mut egui::Ui) -> bool {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.group(|ui| {
            ui.label(
                egui::RichText::new(
                    "Predict Before Inspecting: At custom fraction f, how does typical wealth W_median(T) behave relative to E[W_T]?",
                )
                .strong(),
            );
            ui.horizontal_wrapped(|ui| {
                ui.radio_value(
                    &mut self.hypothesis,
                    Some(VolatilityHypothesis::TypicalGrows),
                    "Typical Wealth Compounds Upward",
                );
                ui.radio_value(
                    &mut self.hypothesis,
                    Some(VolatilityHypothesis::TypicalShrinksDiverges),
                    "Diverges: E[W_T] Grows, Typical Shrinks (Volatility Drag)",
                );
            });
            if let Some(choice) = self.hypothesis {
                let shrinks = self.last_growth_rate < 0.0;
                let correct = (choice == VolatilityHypothesis::TypicalShrinksDiverges) == shrinks;
                let (prefix, color) = if correct {
                    ("Hypothesis Confirmed:", palette.success)
                } else {
                    ("Counterintuitive Outcome:", palette.warning)
                };
                let detail = if shrinks {
                    "Overbetting past 2f* makes g(f) < 0: rare jackpots inflate E[W_T] while typical bettors decay toward ruin."
                } else {
                    "When 0 < f < 2f*, g(f) > 0 so typical wealth compounds upward — yet finite drawdowns remain substantial near Full Kelly."
                };
                ui.colored_label(color, format!("{prefix} {detail}"));
            }
        });
        false
    }

    fn render_fraction_and_preset_controls(
        &mut self,
        ui: &mut egui::Ui,
        initial_bankroll: &mut f64,
        probability: &mut f64,
        odds: &mut f64,
        num_bets: &mut usize,
    ) -> bool {
        let mut changed = ui
            .add(
                egui::Slider::new(&mut self.custom_fraction, 0.0..=MAX_CUSTOM_FRACTION)
                    .text("Custom Bet Fraction (f)"),
            )
            .accessible_hover_text("Fraction of bankroll wagered each round, clamped to 0.0..=0.95")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut self.barrier_floor_ratio,
                    MIN_BARRIER_FLOOR_RATIO..=MAX_BARRIER_FLOOR_RATIO,
                )
                .text("Drawdown Wealth Floor (B_floor / W₀)"),
            )
            .accessible_hover_text(
                "Relative wealth floor defining finite-drawdown barrier ruin (0.50 = 50% bankroll loss)",
            )
            .changed();
        changed |= self.render_benchmark_buttons(ui, initial_bankroll, probability, odds, num_bets);
        changed
    }

    fn render_benchmark_buttons(
        &mut self,
        ui: &mut egui::Ui,
        initial_bankroll: &mut f64,
        probability: &mut f64,
        odds: &mut f64,
        num_bets: &mut usize,
    ) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("⚡ Overbetting Paradox (f=0.30 > 2f*)")
                .accessible_hover_text(
                    "Load canonical W₀=100, p=0.55, b=1.0, T=200, f=0.30 non-ergodic divergence",
                )
                .clicked()
            {
                *initial_bankroll = 100.0;
                *probability = 0.55;
                *odds = 1.0;
                *num_bets = 200;
                self.custom_fraction = 0.30;
                self.barrier_floor_ratio = 0.50;
                changed = true;
            }
            if ui
                .button("🎯 Full Kelly 50%-Floor (f*=0.10)")
                .accessible_hover_text(
                    "Configure Full Kelly (p=0.55, b=1.0, T=200): ~38% finite-horizon 50%-drawdown risk and 50% asymptotic bound",
                )
                .clicked()
            {
                *initial_bankroll = 100.0;
                *probability = 0.55;
                *odds = 1.0;
                *num_bets = 200;
                self.custom_fraction = 0.10;
                self.barrier_floor_ratio = 0.50;
                changed = true;
            }
            if ui
                .button("🛡 Half Kelly 50%-Floor (0.5f*=0.05)")
                .accessible_hover_text(
                    "Configure Half Kelly (p=0.55, b=1.0, T=200): cuts asymptotic 50%-drawdown bound to ~12.5%",
                )
                .clicked()
            {
                *initial_bankroll = 100.0;
                *probability = 0.55;
                *odds = 1.0;
                *num_bets = 200;
                self.custom_fraction = 0.05;
                self.barrier_floor_ratio = 0.50;
                changed = true;
            }
        });
        changed
    }

    fn render_metrics_banner(&self, ui: &mut egui::Ui, num_bets: usize) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let f_star_text = self.last_f_star.map_or_else(
            || "No positive edge".to_string(),
            |fs| format!("{fs:.3} (2f* = {:.3})", 2.0 * fs),
        );
        ui.horizontal_wrapped(|ui| {
            ui.label(format!("Optimal f*: {f_star_text}"));
            ui.separator();
            ui.label(format!(
                "Growth Rate g(f): {:+.5}/bet",
                self.last_growth_rate
            ));
            ui.separator();
            ui.label(format!(
                "E[W_{num_bets}]: {:.2}",
                self.last_expected_terminal
            ));
            ui.separator();
            let med_color = if self.is_ruin_zone {
                palette.error
            } else {
                palette.success
            };
            ui.colored_label(
                med_color,
                format!("W_median({num_bets}): {:.4}", self.last_median_terminal),
            );
        });
    }

    fn render_solvency_banner(&self, ui: &mut egui::Ui, num_bets: usize) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let floor_pct = self.barrier_floor_ratio * 100.0;
        let dd_pct = (1.0 - self.barrier_floor_ratio) * 100.0;
        ui.group(|ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "Drawdown Solvency (Floor = {floor_pct:.0}% W₀ / {dd_pct:.0}% Drawdown):"
                    ))
                    .strong(),
                );
                if let Some(ref rep) = self.last_solvency_report {
                    let color = if rep.ruin_probability >= 0.25 {
                        palette.warning
                    } else {
                        palette.success
                    };
                    ui.colored_label(
                        color,
                        format!(
                            "{num_bets}-Bet Hit Rate: {:.1}% (95% Wilson [{:.1}%, {:.1}%])",
                            rep.ruin_probability * 100.0,
                            rep.wilson_95.lower * 100.0,
                            rep.wilson_95.upper * 100.0
                        ),
                    );
                    ui.separator();
                    ui.label(format!(
                        "Max DD p50 / p95: {:.1}% / {:.1}%",
                        rep.drawdown_quantiles.p50 * 100.0,
                        rep.drawdown_quantiles.p95 * 100.0
                    ));
                }
                if let Some(ref br) = self.last_barrier_risk {
                    ui.separator();
                    ui.label(format!(
                        "Asymptotic Lundberg Bound: {:.1}% (Diffusion: {:.1}%)",
                        br.lundberg_barrier_bound * 100.0,
                        br.diffusion_barrier_prob * 100.0
                    ));
                }
            });
        });
    }

    fn render_ensemble_plot(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let plot = Plot::new("bankroll_volatility_drag_plot")
            .view_aspect(2.1)
            .legend(egui_plot::Legend::default());
        plot.show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Expected Wealth E[W_t]", &self.expected_points[..])
                    .color(palette.series_color(1)),
            );
            plot_ui.line(
                Line::new("Analytical Median W_median(t)", &self.median_points[..])
                    .color(palette.series_color(3)),
            );
            plot_ui.line(
                Line::new(
                    format!("Ensemble Median ({ENSEMBLE_PATH_COUNT} Seeded Paths)"),
                    &self.empirical_median_points[..],
                )
                .color(palette.series_color(5)),
            );
            plot_ui.line(
                Line::new("Drawdown Barrier Floor B_floor", &self.barrier_points[..])
                    .color(palette.error)
                    .style(LineStyle::Dashed { length: 8.0 }),
            );
            self.callout_manager.render_on_plot(plot_ui);
        });
    }
}

// [cite:clinical_trials_statistics]
