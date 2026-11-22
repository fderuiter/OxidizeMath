#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::framework::{bounded_drag_value, enforce_ordered_drag_range};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::pure_math::analysis::complex::mapping::conformal_scale_factor;
use num_complex::Complex64;

const CONTOUR_RESOLUTION: usize = 161;

pub(super) fn split_contour_segments<F: Fn(usize) -> Complex64>(
    resolution: usize,
    grid_span: f64,
    sample: F,
) -> Vec<Vec<[f64; 2]>> {
    let max_radius = (grid_span.abs() * 12.0).max(25.0);
    let max_jump = (max_radius * 0.6).max(12.0);
    let mut segments = Vec::new();
    let mut current: Vec<[f64; 2]> = Vec::new();

    for j in 0..resolution {
        let w = sample(j);
        if !w.re.is_finite() || !w.im.is_finite() || w.norm() > max_radius {
            if current.len() >= 2 {
                segments.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
            continue;
        }
        if let Some(&prev) = current.last() {
            if (w.re - prev[0]).hypot(w.im - prev[1]) > max_jump {
                if current.len() >= 2 {
                    segments.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
        }
        current.push([w.re, w.im]);
    }
    if current.len() >= 2 {
        segments.push(current);
    }
    segments
}

impl ComplexMappingTool {
    pub(super) fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "complex_mapping_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Mapping Controls");
            ui.separator();
            ui.label("Function:");
            egui::ComboBox::from_id_salt("mapping_function")
                .selected_text(self.function.name())
                .show_ui(ui, |ui| {
                    for candidate in [
                        MappingFunction::ZSquared,
                        MappingFunction::Exponential,
                        MappingFunction::Sine,
                        MappingFunction::Inverse,
                        MappingFunction::Custom,
                    ] {
                        ui.selectable_value(&mut self.function, candidate, candidate.name());
                    }
                });

            if self.function == MappingFunction::Custom {
                ui.add_space(6.0);
                self.math_input.ui(ui);
            }
            self.draw_grid_range_controls(ui);
            self.draw_inspection_controls(ui);
        });
    }

    fn draw_grid_range_controls(&mut self, ui: &mut egui::Ui) {
        ui.add_space(10.0);
        ui.label("Grid Range [min, max]:");
        let mut min_changed = false;
        let mut max_changed = false;
        ui.horizontal(|ui| {
            min_changed = ui
                .add(bounded_drag_value(
                    &mut self.grid_min,
                    COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX,
                    0.1,
                ))
                .changed();
            ui.label("to");
            max_changed = ui
                .add(bounded_drag_value(
                    &mut self.grid_max,
                    COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX,
                    0.1,
                ))
                .changed();
        });
        enforce_ordered_drag_range(
            &mut self.grid_min,
            &mut self.grid_max,
            COMPLEX_COORD_MIN,
            COMPLEX_COORD_MAX,
            COMPLEX_MIN_SPAN,
            min_changed,
            max_changed,
        );

        ui.add_space(10.0);
        ui.add(
            egui::Slider::new(
                &mut self.grid_density,
                2..=math_commons::registry::MAX_GRID_SIZE,
            )
            .text("Grid Density:"),
        );
    }

    fn draw_inspection_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("Inspection Point (z)");
        ui.horizontal(|ui| {
            ui.label("Re:");
            ui.add(bounded_drag_value(
                &mut self.inspection_point[0],
                COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX,
                0.1,
            ));
            ui.label("Im:");
            ui.add(bounded_drag_value(
                &mut self.inspection_point[1],
                COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX,
                0.1,
            ));
        });

        let z0 = Complex64::new(self.inspection_point[0], self.inspection_point[1]);
        let w0 = self.eval_z(z0);
        let scale_factor = conformal_scale_factor(|z| self.deriv_z(z), z0);
        let safe_scale = if scale_factor.is_finite() {
            scale_factor
        } else {
            0.0
        };

        let w_text = if w0.re.is_finite() && w0.im.is_finite() {
            format!("{:.3} + {:.3}i", w0.re, w0.im)
        } else {
            "Pole / Undefined".to_string()
        };
        render_copyable_metric(ui, "w = f(z)", &w_text);
        render_copyable_metric(ui, "Scale factor |f'(z)|", &format!("{:.3}", safe_scale));
    }

    pub(super) fn draw_z_plane(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        ui.label("z-plane (Input)");
        let grid_ok = self.grid_min.is_finite()
            && self.grid_max.is_finite()
            && self.grid_max > self.grid_min
            && (2..=math_commons::registry::MAX_GRID_SIZE).contains(&self.grid_density);
        Plot::new("z_plane_plot")
            .view_aspect(1.0)
            .x_axis_label("Re(z)")
            .y_axis_label("Im(z)")
            .show(ui, |plot_ui| {
                if grid_ok {
                    let step = (self.grid_max - self.grid_min) / (self.grid_density as f64 - 1.0);
                    for i in 0..self.grid_density {
                        let x = self.grid_min + (i as f64) * step;
                        plot_ui.line(
                            Line::new(
                                "",
                                PlotPoints::new(vec![[x, self.grid_min], [x, self.grid_max]]),
                            )
                            .color(palette.series_color(0)),
                        );
                    }
                    for i in 0..self.grid_density {
                        let y = self.grid_min + (i as f64) * step;
                        plot_ui.line(
                            Line::new(
                                "",
                                PlotPoints::new(vec![[self.grid_min, y], [self.grid_max, y]]),
                            )
                            .color(palette.series_color(1)),
                        );
                    }
                }
                if self.inspection_point.iter().all(|v| v.is_finite()) {
                    plot_ui.points(
                        egui_plot::Points::new("", PlotPoints::new(vec![self.inspection_point]))
                            .color(palette.error)
                            .radius(5.0_f32),
                    );
                }
            });
    }

    pub(super) fn draw_w_plane(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        ui.label("w-plane (Output)");
        let grid_ok = self.grid_min.is_finite()
            && self.grid_max.is_finite()
            && self.grid_max > self.grid_min
            && (2..=math_commons::registry::MAX_GRID_SIZE).contains(&self.grid_density);
        Plot::new("w_plane_plot")
            .view_aspect(1.0)
            .x_axis_label("Re(w)")
            .y_axis_label("Im(w)")
            .show(ui, |plot_ui| {
                if grid_ok {
                    self.draw_w_grid_contours(plot_ui, palette);
                }
                if self.inspection_point.iter().all(|v| v.is_finite()) {
                    let w0 = self.eval_z(Complex64::new(
                        self.inspection_point[0],
                        self.inspection_point[1],
                    ));
                    if w0.re.is_finite() && w0.im.is_finite() {
                        plot_ui.points(
                            egui_plot::Points::new("", PlotPoints::new(vec![[w0.re, w0.im]]))
                                .color(palette.error)
                                .radius(5.0_f32),
                        );
                    }
                }
            });
    }

    fn draw_w_grid_contours(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &crate::app::SemanticPalette,
    ) {
        let span = self.grid_max - self.grid_min;
        let step = span / (self.grid_density as f64 - 1.0);
        let denom = (CONTOUR_RESOLUTION - 1) as f64;
        for i in 0..self.grid_density {
            let x = self.grid_min + (i as f64) * step;
            let segments = split_contour_segments(CONTOUR_RESOLUTION, span, |j| {
                let y = self.grid_min + span * (j as f64) / denom;
                self.eval_z(Complex64::new(x, y))
            });
            for seg in segments {
                plot_ui.line(Line::new("", PlotPoints::new(seg)).color(palette.series_color(0)));
            }
        }
        for i in 0..self.grid_density {
            let y = self.grid_min + (i as f64) * step;
            let segments = split_contour_segments(CONTOUR_RESOLUTION, span, |j| {
                let x = self.grid_min + span * (j as f64) / denom;
                self.eval_z(Complex64::new(x, y))
            });
            for seg in segments {
                plot_ui.line(Line::new("", PlotPoints::new(seg)).color(palette.series_color(1)));
            }
        }
    }
}
