//! UI controls, percentile fan-band plot, and underwater drawdown view for [`super::BankrollEnsembleLab`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BankrollEnsembleLab, VolatilityHypothesis, MAX_CUSTOM_FRACTION};
use crate::accessibility::{
    paint_canvas_focus_ring, register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::SemanticPalette;
use eframe::egui;
use eframe::egui::accesskit::Role;
use egui_plot::{Legend, Line, LineStyle, Plot, Polygon};

impl BankrollEnsembleLab {
    /// Renders the hypothesis prompt, allocation slider, tail-risk banner, fan-band plot, and underwater plot.
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
                    "When 0 < f < 2f*, g(f) > 0 so typical quantile wealth compounds upward alongside E[W_T]."
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

        if ui
            .button("⚡ Load Volatility-Drag Paradox (W₀=100, p=0.55, b=1.0, T=200, f=0.30)")
            .accessible_hover_text(
                "Configure canonical non-ergodic over-allocation divergence where E[W_200]=36,935.58 yet W_median(200)=3.9137",
            )
            .clicked()
        {
            *initial_bankroll = 100.0;
            *probability = 0.55;
            *odds = 1.0;
            *num_bets = 200;
            self.custom_fraction = 0.30;
            changed = true;
        }
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
            "Ensemble Percentile Fan Bands and Volatility-Drag Plot",
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
                "Ensemble Median (64 Seeded Paths)",
                &self.empirical_median_points[..],
            )
            .color(inner_col)
            .width(2.0_f32),
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
}

// [cite:clinical_trials_statistics]
