#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::cache::ContourBatch;
use super::*;
use crate::framework::{bounded_drag_value, enforce_ordered_drag_range};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Plot, PlotPoint};
use num_complex::Complex64;

pub(super) const CONTOUR_RESOLUTION: usize = 161;

#[cfg(test)]
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
        let max_grid = crate::presets::MAX_INTERACTIVE_CONFORMAL_GRID_SIZE;
        self.grid_density = self.grid_density.clamp(2, max_grid);
        ui.add(egui::Slider::new(&mut self.grid_density, 2..=max_grid).text("Grid Density:"));
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

        let (w0, safe_scale) = self.ensure_cached_probe();
        let w_text = if w0.re.is_finite() && w0.im.is_finite() {
            format!("{:.3} + {:.3}i", w0.re, w0.im)
        } else {
            "Pole / Undefined".to_string()
        };
        render_copyable_metric(ui, "w = f(z)", &w_text);
        render_copyable_metric(ui, "Scale factor |f'(z)|", &format!("{:.3}", safe_scale));
    }

    pub(super) fn draw_z_plane(
        &self,
        ui: &mut egui::Ui,
        palette: &crate::app::SemanticPalette,
        grid_ok: bool,
    ) {
        ui.label("z-plane (Input)");
        let z_marker = [PlotPoint::new(
            self.inspection_point[0],
            self.inspection_point[1],
        )];
        let has_z_marker = self.inspection_point.iter().all(|v| v.is_finite());
        let generation = self.cache.generation;
        Plot::new("z_plane_plot")
            .view_aspect(1.0)
            .x_axis_label("Re(z)")
            .y_axis_label("Im(z)")
            .show(ui, |plot_ui| {
                if grid_ok {
                    if let Some(batch) = ContourBatch::from_pair(
                        "z-grid",
                        ("Re(z) grid", &self.cache.z_vertical, palette.series_color(0)),
                        ("Im(z) grid", &self.cache.z_horizontal, palette.series_color(1)),
                        generation,
                    ) {
                        plot_ui.add(batch);
                    }
                }
                if has_z_marker {
                    plot_ui.points(
                        egui_plot::Points::new("", z_marker.as_slice())
                            .color(palette.error)
                            .radius(5.0_f32),
                    );
                }
            });
    }

    pub(super) fn draw_w_plane(
        &self,
        ui: &mut egui::Ui,
        palette: &crate::app::SemanticPalette,
        grid_ok: bool,
    ) {
        ui.label("w-plane (Output)");
        let w0 = self
            .inspection_point
            .iter()
            .all(|v| v.is_finite())
            .then_some(self.cache.cached_w0);
        let has_w_marker = w0.is_some_and(|w| w.re.is_finite() && w.im.is_finite());
        let w_marker = [w0.map_or(PlotPoint::new(0.0, 0.0), |w| PlotPoint::new(w.re, w.im))];
        let generation = self.cache.generation;
        Plot::new("w_plane_plot")
            .view_aspect(1.0)
            .x_axis_label("Re(w)")
            .y_axis_label("Im(w)")
            .show(ui, |plot_ui| {
                if grid_ok {
                    if let Some(batch) = ContourBatch::from_pair(
                        "w-contours",
                        ("u-contours", &self.cache.w_vertical, palette.series_color(0)),
                        ("v-contours", &self.cache.w_horizontal, palette.series_color(1)),
                        generation,
                    ) {
                        plot_ui.add(batch);
                    }
                }
                if has_w_marker {
                    plot_ui.points(
                        egui_plot::Points::new("", w_marker.as_slice())
                            .color(palette.error)
                            .radius(5.0_f32),
                    );
                }
            });
    }

    pub(super) fn build_export_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        let theme = if self.high_contrast {
            crate::app::ThemeMode::HighContrast
        } else {
            crate::app::ThemeMode::Dark
        };
        let palette = crate::app::SemanticPalette::for_theme(theme);
        let mut cold_cache = super::cache::ConformalGridCache::default();
        let mut out = Vec::new();
        if let Some(cache) = self.resolve_export_cache(&mut cold_cache) {
            let total = cache.z_vertical.ranges.len()
                + cache.z_horizontal.ranges.len()
                + cache.w_vertical.ranges.len()
                + cache.w_horizontal.ranges.len()
                + 2;
            out.reserve(total);
            let (c0, c1) = (palette.series_color(0), palette.series_color(1));
            append_family_polylines("Re(z) grid", &cache.z_vertical, c0, &mut out);
            append_family_polylines("Im(z) grid", &cache.z_horizontal, c1, &mut out);
            append_family_polylines("u-contours", &cache.w_vertical, c0, &mut out);
            append_family_polylines("v-contours", &cache.w_horizontal, c1, &mut out);
        }
        if self.inspection_point.iter().all(|v| v.is_finite()) {
            out.push(egui_plot::ExportPrimitive::Points {
                name: "z".to_owned(),
                points: vec![self.inspection_point],
                color: palette.error,
                radius: 5.0,
            });
            let w0 = self.eval_z(Complex64::new(
                self.inspection_point[0],
                self.inspection_point[1],
            ));
            if w0.re.is_finite() && w0.im.is_finite() {
                out.push(egui_plot::ExportPrimitive::Points {
                    name: "w = f(z)".to_owned(),
                    points: vec![[w0.re, w0.im]],
                    color: palette.error,
                    radius: 5.0,
                });
            }
        }
        out
    }
}

fn append_family_polylines(
    name: &str,
    family: &super::cache::ContourFamily,
    color: egui::Color32,
    out: &mut Vec<egui_plot::ExportPrimitive>,
) {
    let pts = family.points.as_slice();
    for range in family.ranges.iter() {
        if let Some(slice) = pts.get(range.clone()) {
            if slice.len() >= 2 {
                out.push(egui_plot::ExportPrimitive::Polyline {
                    name: name.to_owned(),
                    points: slice.iter().map(|p| [p.x, p.y]).collect(),
                    color,
                    stroke_width: 1.5,
                    closed: false,
                    fill: None,
                });
            }
        }
    }
}
