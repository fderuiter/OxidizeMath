//! Logarithmic growth-curve plot rendering for [`super::BetSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::BetSizeCalculatorTool;
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, MarkerShape, Plot, Points};

impl BetSizeCalculatorTool {
    pub(super) fn show_growth_curve_plot(&self, ui: &mut egui::Ui) {
        if self.growth_curve_points.is_empty() {
            return;
        }
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Expected Logarithmic Growth Curve g(f)");
        Plot::new("bet_size_growth_plot")
            .view_aspect(2.2)
            .legend(Legend::default())
            .x_axis_label("Wager Fraction (f)")
            .y_axis_label("Expected Log Growth Rate g(f)")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Break-Even (g = 0)", &self.zero_line_points[..])
                        .color(palette.graph_edge)
                        .style(LineStyle::Dashed { length: 6.0 }),
                );
                plot_ui.line(
                    Line::new("Expected Growth Rate g(f)", &self.growth_curve_points[..])
                        .color(palette.series_color(0))
                        .width(2.0_f32),
                );
                self.render_plot_markers(plot_ui, &palette);
            });
    }

    fn render_plot_markers<'a>(
        &'a self,
        plot_ui: &mut egui_plot::PlotUi<'a>,
        palette: &SemanticPalette,
    ) {
        if !self.quarter_marker.is_empty() {
            plot_ui.points(
                Points::new("Quarter Kelly (0.25 f*)", &self.quarter_marker[..])
                    .color(palette.series_color(2))
                    .radius(5.0_f32)
                    .shape(MarkerShape::Diamond),
            );
        }
        if !self.half_marker.is_empty() {
            plot_ui.points(
                Points::new("Half Kelly (0.50 f*)", &self.half_marker[..])
                    .color(palette.series_color(5))
                    .radius(5.5_f32)
                    .shape(MarkerShape::Square),
            );
        }
        if !self.full_marker.is_empty() {
            plot_ui.points(
                Points::new("Full Kelly (1.00 f*)", &self.full_marker[..])
                    .color(palette.series_color(3))
                    .radius(6.0_f32)
                    .shape(MarkerShape::Circle),
            );
        }
        if !self.drawdown_marker.is_empty() {
            plot_ui.points(
                Points::new("Drawdown-Constrained Kelly", &self.drawdown_marker[..])
                    .color(palette.series_color(1))
                    .radius(5.5_f32)
                    .shape(MarkerShape::Up),
            );
        }
    }
}

// [cite:clinical_trials_statistics]
