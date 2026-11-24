//! UI controls, solvency banner, percentile fan-band plot, and underwater drawdown view for [`super::BankrollEnsembleLab`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    BankrollEnsembleLab, VolatilityHypothesis, ENSEMBLE_PATH_COUNT, MAX_BARRIER_FLOOR_RATIO,
    MAX_CUSTOM_FRACTION, MIN_BARRIER_FLOOR_RATIO,
};
use crate::accessibility::{
    paint_canvas_focus_ring, register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::SemanticPalette;
use eframe::egui;
use eframe::egui::accesskit::Role;
use egui_plot::{Legend, Line, LineStyle, Plot, Polygon};

impl BankrollEnsembleLab {
    /// Renders the hypothesis prompt, sliders, solvency/tail-risk banners, fan-band plot, and underwater plot.
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
        ui.add_space(4.0);
        self.render_underwater_plot(ui);
        let _ = self.callout_manager.render_ui(ui);
        changed
    }

    fn render_hypothesis_card(&mut self, ui: &mut egui::Ui) -> bool {
        let palette = SemanticPalette::from_visuals(ui.visuals());
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
                    "Over-allocating past 2f* makes g(f) < 0: rare outliers inflate E[W_T] while 5th–95th quantile paths decay."
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
                    .text("Allocation Fraction (f)"),
            )
            .accessible_hover_text(
                "Fraction of capital allocated each period, clamped to 0.0..=0.95",
            )
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
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let f_star_text = self.last_f_star.map_or_else(
            || "No positive edge".to_string(),
            |fs| format!("{fs:.3} (2f* = {:.3})", 2.0 * fs),
        );
        ui.horizontal_wrapped(|ui| {
            ui.label(format!("Optimal f*: {f_star_text}"));
            ui.separator();
            ui.label(format!("g(f): {:+.5}/step", self.last_growth_rate));
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
            if let Some(frc) = self.last_drawdown_f {
                ui.separator();
                ui.label(format!(
                    "Risk-Constrained (f={:.2}%): {:.2}",
                    frc * 100.0,
                    self.last_drawdown_terminal
                ));
            }
            ui.separator();
            ui.colored_label(
                palette.warning,
                format!(
                    "95% VaR: {:.1}% | 95% CVaR: {:.1}%",
                    self.last_terminal_var_95 * 100.0,
                    self.last_terminal_cvar_95 * 100.0
                ),
            );
            ui.separator();
            ui.label(format!(
                "Max DD: {:.1}% | Max Underwater: {} steps",
                self.last_max_drawdown * 100.0,
                self.last_underwater_duration
            ));
        });
    }

    fn render_solvency_banner(&self, ui: &mut egui::Ui, num_bets: usize) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
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
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let resp = Plot::new("bankroll_volatility_drag_plot")
            .height(260.0)
            .x_axis_label("Allocation Step t")
            .y_axis_label("Capital W(t)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                self.draw_ensemble_series(plot_ui, &palette);
            });
        let summary = format!(
            "Ensemble fan bands: W_median {:.4}, 95% VaR {:.1}%, 95% CVaR {:.1}%, underwater duration {} steps",
            self.last_median_terminal,
            self.last_terminal_var_95 * 100.0,
            self.last_terminal_cvar_95 * 100.0,
            self.last_underwater_duration
        );
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "Ensemble Percentile Fan Bands and Solvency Barrier Plot",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }

    fn draw_ensemble_series<'a>(
        &'a mut self,
        plot_ui: &mut egui_plot::PlotUi<'a>,
        palette: &SemanticPalette,
    ) {
        let outer_col = palette.series_color(2);
        let inner_col = palette.series_color(5);
        let outer_fill =
            egui::Color32::from_rgba_unmultiplied(outer_col.r(), outer_col.g(), outer_col.b(), 30);
        let inner_fill =
            egui::Color32::from_rgba_unmultiplied(inner_col.r(), inner_col.g(), inner_col.b(), 56);
        if self.fan_5_95_poly.len() >= 4 {
            plot_ui.polygon(
                Polygon::new("5th–95th Percentile Band", &self.fan_5_95_poly[..])
                    .fill_color(outer_fill),
            );
        }
        if self.fan_25_75_poly.len() >= 4 {
            plot_ui.polygon(
                Polygon::new("25th–75th Interquartile Band", &self.fan_25_75_poly[..])
                    .fill_color(inner_fill),
            );
        }
        plot_ui.line(
            Line::new("95th Percentile", &self.p95_points[..])
                .color(outer_col)
                .style(LineStyle::Dashed { length: 6.0 }),
        );
        plot_ui.line(
            Line::new("5th Percentile", &self.p05_points[..])
                .color(palette.warning)
                .style(LineStyle::Dashed { length: 6.0 }),
        );
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
            .color(inner_col)
            .width(2.0_f32),
        );
        if !self.drawdown_median_points.is_empty() {
            plot_ui.line(
                Line::new(
                    "Drawdown-Constrained Median",
                    &self.drawdown_median_points[..],
                )
                .color(palette.series_color(0)),
            );
        }
        plot_ui.line(
            Line::new("Drawdown Barrier Floor B_floor", &self.barrier_points[..])
                .color(palette.error)
                .style(LineStyle::Dashed { length: 8.0 }),
        );
        self.callout_manager.render_on_plot(plot_ui);
    }

    fn render_underwater_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let resp = Plot::new("bankroll_ensemble_underwater_plot")
            .height(150.0)
            .x_axis_label("Allocation Step t")
            .y_axis_label("Underwater Drawdown (%)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(
                        "Median Underwater Drawdown (%)",
                        &self.underwater_points[..],
                    )
                    .color(palette.error)
                    .width(1.8_f32),
                );
            });
        let summary = format!(
            "Peak-to-trough drawdown {:.1}% with maximum underwater recovery span {} steps",
            self.last_max_drawdown * 100.0,
            self.last_underwater_duration
        );
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "Ensemble Underwater Fractional Drawdown Curve",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }

    /// Synchronizes automated `CalloutManager` overlays for growth rate, ruin, and 95% `VaR`/`CVaR`.
    pub fn sync_callouts(&mut self, num_bets: usize) {
        let steps = num_bets.min(super::MAX_ENSEMBLE_BETS);
        if self.median_points.is_empty() {
            self.callout_manager.clear_auto();
            return;
        }
        let mid_idx = (steps / 2).min(self.median_points.len() - 1);
        let mid_pt = self.median_points[mid_idx];
        self.callout_manager.set_auto_callout(
            "growth_rate",
            [mid_pt.x, mid_pt.y],
            format!(
                "g(f={:.2}) = {:+.4}/step",
                self.custom_fraction, self.last_growth_rate
            ),
            [24.0, -26.0],
        );

        let end_x = steps as f64;
        let p05_end = self.p05_points.last().map_or(0.0, |pt| pt.y);
        let var_dollars = self.last_initial_bankroll * self.last_terminal_var_95;
        let cvar_dollars = self.last_initial_bankroll * self.last_terminal_cvar_95;
        self.callout_manager.set_auto_callout(
            "var_95",
            [end_x, p05_end],
            format!(
                "95% VaR: {:.1}% (${:.2})",
                self.last_terminal_var_95 * 100.0,
                var_dollars
            ),
            [-84.0, 18.0],
        );
        self.callout_manager.set_auto_callout(
            "cvar_95",
            [(0.78 * end_x).max(0.0), p05_end],
            format!(
                "95% CVaR: {:.1}% (${:.2})",
                self.last_terminal_cvar_95 * 100.0,
                cvar_dollars
            ),
            [-84.0, 38.0],
        );

        if let Some(fs) = self
            .last_f_star
            .filter(|&fs| self.custom_fraction > 2.0 * fs)
        {
            self.callout_manager.set_auto_callout(
                "ruin_warning",
                [end_x, self.last_median_terminal],
                format!(
                    "Ruin Trap (f={:.2} > 2f*={:.2}): W_med={:.4} vs E[W]={:.2}",
                    self.custom_fraction,
                    2.0 * fs,
                    self.last_median_terminal,
                    self.last_expected_terminal
                ),
                [-90.0, -34.0],
            );
        } else {
            self.remove_auto_tag("ruin_warning");
        }
    }

    fn remove_auto_tag(&mut self, tag: &str) {
        if let Some(id) = self
            .callout_manager
            .callouts()
            .iter()
            .find(|c| c.tag.as_deref() == Some(tag))
            .map(|c| c.id)
        {
            self.callout_manager.remove_callout(id);
        }
    }
}

// [cite:clinical_trials_statistics]
