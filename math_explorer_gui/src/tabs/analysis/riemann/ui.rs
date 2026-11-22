#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::framework::{bounded_drag_value, enforce_ordered_drag_range};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Polygon};

impl RiemannIntegrationTool {
    fn draw_preset_combo(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Function:");
        egui::ComboBox::from_id_salt("function_preset")
            .selected_text(self.preset.name())
            .show_ui(ui, |ui| {
                for (candidate, default_bounds) in [
                    (FunctionPreset::SinX, (0.0, std::f64::consts::PI)),
                    (FunctionPreset::XSquared, (-2.0, 2.0)),
                    (FunctionPreset::EToX, (-2.0, 2.0)),
                    (FunctionPreset::Custom, (-3.0, 3.0)),
                ] {
                    if ui
                        .selectable_value(&mut self.preset, candidate, candidate.name())
                        .changed()
                    {
                        self.min_x = default_bounds.0;
                        self.max_x = default_bounds.1;
                        changed = true;
                    }
                }
            });
        changed
    }

    pub(super) fn draw_controls(&mut self, parent_ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "riemann_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Integration Controls");
            ui.separator();
            changed |= self.draw_preset_combo(ui);

            if self.preset == FunctionPreset::Custom {
                ui.add_space(6.0);
                changed |= self.math_input.ui(ui);
            }

            ui.add_space(10.0);
            ui.label("Integration Interval [min, max]");
            let mut min_changed = false;
            let mut max_changed = false;
            ui.horizontal(|ui| {
                min_changed = ui
                    .add(bounded_drag_value(
                        &mut self.min_x,
                        RIEMANN_X_MIN_BOUND..=RIEMANN_X_MAX_BOUND,
                        0.1,
                    ))
                    .changed();
                ui.label("to");
                max_changed = ui
                    .add(bounded_drag_value(
                        &mut self.max_x,
                        RIEMANN_X_MIN_BOUND..=RIEMANN_X_MAX_BOUND,
                        0.1,
                    ))
                    .changed();
            });
            enforce_ordered_drag_range(
                &mut self.min_x,
                &mut self.max_x,
                RIEMANN_X_MIN_BOUND,
                RIEMANN_X_MAX_BOUND,
                RIEMANN_MIN_SPAN,
                min_changed,
                max_changed,
            );
            changed |= min_changed || max_changed;

            ui.add_space(10.0);
            if ui
                .add(
                    egui::Slider::new(&mut self.steps, 1..=200)
                        .logarithmic(true)
                        .text("Partitions (N)"),
                )
                .changed()
            {
                changed = true;
            }

            ui.separator();
            ui.heading("Result");
            render_copyable_metric(ui, "Area", &format!("{:.6}", self.result));
        });
        changed
    }

    pub(super) fn draw_plot(&self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("Visualization");
            let valid_bounds =
                self.min_x.is_finite() && self.max_x.is_finite() && self.max_x > self.min_x;
            let (plot_min, plot_max) = if valid_bounds {
                let span = self.max_x - self.min_x;
                (self.min_x - span * 0.1, self.max_x + span * 0.1)
            } else {
                (-1.0, 1.0)
            };

            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let fill_base = palette.series_color(1);
            let poly_fill = egui::Color32::from_rgba_unmultiplied(
                fill_base.r(),
                fill_base.g(),
                fill_base.b(),
                100,
            );
            let true_line = Line::new(
                self.preset.name(),
                PlotPoints::new(self.cached_true_curve.clone()),
            )
            .name(self.preset.name())
            .color(palette.series_color(0));

            Plot::new("riemann_plot")
                .view_aspect(2.0)
                .x_axis_label("x")
                .y_axis_label("f(x)")
                .show(ui, |plot_ui| {
                    if valid_bounds {
                        plot_ui.line(true_line);
                        for poly in &self.cached_trapezoid_polygons {
                            let polygon = Polygon::new("", PlotPoints::new(poly.clone()))
                                .fill_color(poly_fill);
                            plot_ui.polygon(polygon);
                        }
                    }
                    plot_ui.line(
                        Line::new("", PlotPoints::new(vec![[plot_min, 0.0], [plot_max, 0.0]]))
                            .color(palette.graph_edge),
                    );
                });
        });
    }
}
