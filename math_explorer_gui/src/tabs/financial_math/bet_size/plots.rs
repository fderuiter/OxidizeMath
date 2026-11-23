//! UI panels, logarithmic growth curve `g(f)` plot, and export primitives for [`super::BetSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BetSizeCalculatorTool, OddsFormat};
use crate::accessibility::{paint_canvas_focus_ring, register_canvas_accessibility};
use crate::app::SemanticPalette;
use eframe::egui;
use eframe::egui::accesskit::Role;
use egui_plot::{
    ExportPrimitive, HLine, Legend, Line, LineStyle, MarkerShape, Plot, PlotPoint, Points, VLine,
};
use math_explorer::pure_math::statistics::kelly::{
    kelly_fraction, variants, BankrollFraction, Odds, UnitInterval,
};

fn to_pairs(pts: &[PlotPoint]) -> Vec<[f64; 2]> {
    pts.iter().map(|p| [p.x, p.y]).collect()
}

impl BetSizeCalculatorTool {
    pub(super) fn render_workspace(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Kelly Capital Allocation & Logarithmic Growth Curve");
                ui.label(
                    "Size single-opportunity capital commitments and inspect asymptotic logarithmic growth g(f).",
                );
                ui.separator();

                if self.render_input_grid(ui) {
                    self.recompute_curve();
                }

                ui.separator();
                let palette = SemanticPalette::from_visuals(ui.visuals());
                self.render_results_section(ui, &palette);
                ui.add_space(6.0);
                self.render_growth_curve_plot(ui, &palette);
            });
        });
    }

    fn render_input_grid(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::Grid::new("bet_size_inputs")
            .num_columns(2)
            .spacing([40.0, 10.0])
            .show(ui, |ui| {
                ui.label("Success Probability (%):");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.probability_input)
                            .speed(0.1)
                            .range(0.0..=100.0)
                            .suffix("%"),
                    )
                    .changed();
                ui.end_row();

                ui.label("Payoff Notation:");
                ui.horizontal(|ui| {
                    changed |= ui
                        .radio_value(&mut self.odds_format, OddsFormat::Decimal, "Decimal")
                        .changed();
                    changed |= ui
                        .radio_value(&mut self.odds_format, OddsFormat::American, "American")
                        .changed();
                    changed |= ui
                        .radio_value(&mut self.odds_format, OddsFormat::Fractional, "Ratio")
                        .changed();
                });
                ui.end_row();

                ui.label("Payoff Parameter:");
                changed |= self.render_payoff_input(ui);
                ui.end_row();

                ui.label("Available Capital ($):");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.bankroll)
                            .speed(10.0)
                            .range(1.0..=1_000_000.0)
                            .prefix("$"),
                    )
                    .changed();
                ui.end_row();
            });
        changed
    }

    fn render_payoff_input(&mut self, ui: &mut egui::Ui) -> bool {
        match self.odds_format {
            OddsFormat::Decimal => ui
                .add(
                    egui::DragValue::new(&mut self.decimal_odds)
                        .speed(0.01)
                        .range(1.01..=1000.0),
                )
                .changed(),
            OddsFormat::American => ui
                .add(
                    egui::DragValue::new(&mut self.american_odds)
                        .speed(1.0)
                        .range(-10_000.0..=10_000.0),
                )
                .changed(),
            OddsFormat::Fractional => {
                ui.horizontal(|ui| {
                    let c1 = ui
                        .add(
                            egui::DragValue::new(&mut self.fractional_numerator)
                                .speed(1.0)
                                .range(1.0..=1000.0),
                        )
                        .changed();
                    ui.label("/");
                    let c2 = ui
                        .add(
                            egui::DragValue::new(&mut self.fractional_denominator)
                                .speed(1.0)
                                .range(1.0..=1000.0),
                        )
                        .changed();
                    c1 || c2
                })
                .inner
            }
        }
    }

    fn render_results_section(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.horizontal_wrapped(|ui| {
            ui.heading("Allocation Tiers");
            ui.small(format!("({:.3} ms)", self.last_compute_ms));
        });
        let (prob_res, odds_res) = self.resolve_prob_and_odds();
        match (prob_res, odds_res) {
            (Ok(prob), Ok(odds)) => self.render_kelly_table(ui, palette, &prob, &odds),
            (Err(e), _) => {
                ui.colored_label(palette.error, format!("Invalid Probability: {e}"));
            }
            (_, Err(e)) => {
                ui.colored_label(palette.error, format!("Invalid Payoff: {e}"));
            }
        }
    }

    fn render_kelly_table(
        &self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        prob: &UnitInterval,
        odds: &Odds,
    ) {
        ui.label(format!(
            "Expected Return per $1 Allocated: {:+.3} | Peak Log-Growth g(f*): {:+.5}",
            self.unit_ev, self.optimal_growth_rate
        ));
        if self.unit_ev <= 0.0 {
            ui.colored_label(
                palette.error,
                "Non-positive expected return (EV <= 0): optimal allocation f* = 0%.",
            );
            return;
        }
        ui.colored_label(palette.success, "Positive Expected Edge Detected");
        let Ok(full) = kelly_fraction(prob, odds) else {
            return;
        };
        egui::Grid::new("kelly_results")
            .num_columns(3)
            .spacing([40.0, 8.0])
            .show(ui, |ui| {
                ui.strong("Allocation Tier");
                ui.strong("Capital Fraction");
                ui.strong("Dollar Commitment");
                ui.end_row();
                self.render_tier_row(ui, "Full Kelly (1.00 f*)", &full);
                if let Ok(half) = variants::half_kelly(prob, odds) {
                    self.render_tier_row(ui, "Half Kelly (0.50 f*)", &half);
                }
                if let Ok(quarter) = variants::quarter_kelly(prob, odds) {
                    self.render_tier_row(ui, "Quarter Kelly (0.25 f*)", &quarter);
                }
            });
    }

    fn render_tier_row(&self, ui: &mut egui::Ui, label: &str, frac: &BankrollFraction) {
        ui.label(label);
        ui.label(format!("{:.2}%", frac.value() * 100.0));
        if let Ok(amount) = frac.bet_amount(self.bankroll) {
            ui.label(format!("${amount:.2}"));
        } else {
            ui.label("Invalid capital");
        }
        ui.end_row();
    }

    fn render_growth_curve_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let resp = Plot::new("bet_size_log_growth_curve")
            .height(260.0)
            .x_axis_label("Allocation Fraction f")
            .y_axis_label("Expected Logarithmic Growth Rate g(f)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.hline(
                    HLine::new("Break-Even g(f) = 0", 0.0)
                        .color(palette.warning)
                        .style(LineStyle::Dashed { length: 6.0 }),
                );
                plot_ui.line(
                    Line::new("Logarithmic Growth g(f)", &self.growth_curve[..])
                        .color(palette.series_color(0))
                        .width(2.2_f32),
                );
                if let Some(fs) = self.optimal_f_star {
                    plot_ui.vline(
                        VLine::new("Optimal Kelly f*", fs)
                            .color(palette.success)
                            .style(LineStyle::Dashed { length: 6.0 }),
                    );
                    if 2.0 * fs < 0.98 {
                        plot_ui.vline(
                            VLine::new("Over-Allocation Boundary 2f*", 2.0 * fs)
                                .color(palette.error)
                                .style(LineStyle::Dotted { spacing: 5.0 }),
                        );
                    }
                }
                if !self.marker_points.is_empty() {
                    plot_ui.points(
                        Points::new(
                            "Fractional Kelly Tiers (0.25f*, 0.5f*, f*, 2f*)",
                            &self.marker_points[..],
                        )
                        .color(palette.accent)
                        .shape(MarkerShape::Diamond)
                        .radius(5.0_f32),
                    );
                }
            });
        let summary = format!(
            "Kelly growth curve g(f) with {} samples; optimal f* = {:.3}, g(f*) = {:+.5}",
            self.growth_curve.len(),
            self.optimal_f_star.unwrap_or(0.0),
            self.optimal_growth_rate
        );
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "Logarithmic Capital Growth Rate g(f) vs Allocation Fraction f",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }

    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut sets = vec![(
            "Logarithmic Growth Rate g(f)".to_string(),
            to_pairs(&self.growth_curve),
        )];
        if !self.marker_points.is_empty() {
            sets.push((
                "Kelly Fraction Tiers".to_string(),
                to_pairs(&self.marker_points),
            ));
        }
        sets
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let pal = crate::app::palette::DARK_PALETTE;
        let mut prims = Vec::new();
        if self.growth_curve.len() >= 2 {
            prims.push(ExportPrimitive::Polyline {
                name: "Logarithmic Growth Rate g(f)".to_string(),
                points: to_pairs(&self.growth_curve),
                color: pal.series[0],
                stroke_width: 2.2,
                closed: false,
                fill: None,
            });
        }
        if !self.marker_points.is_empty() {
            prims.push(ExportPrimitive::Points {
                name: "Kelly Fraction Tiers".to_string(),
                points: to_pairs(&self.marker_points),
                color: pal.accent,
                radius: 5.0,
            });
        }
        prims
    }
}

// [cite:modular_polynomials_review]
