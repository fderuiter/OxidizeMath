#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::accessibility::AccessibleTheoryHover;
use crate::framework::shader_foundry::{
    emit_domain_coloring_shader, render_shader_preview_section,
};
use crate::framework::{bounded_drag_value, enforce_ordered_drag_range};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_commons::math_kernel::colormap::ShaderPalette;
use math_explorer::pure_math::analysis::complex::mapping::conformal_scale_factor;
use num_complex::Complex64;
use scientific_metadata::theory::TheoryDescribable;

pub(super) const CONTOUR_RESOLUTION: usize = 161;

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
            egui::ScrollArea::vertical()
                .id_salt("complex_mapping_side_scroll")
                .show(ui, |ui| {
                    ui.heading("Mapping Controls");
                    ui.separator();
                    self.draw_function_and_coloring_controls(ui);
                    self.draw_grid_range_controls(ui);
                    self.draw_inspection_controls(ui);
                    self.draw_shader_preview(ui);
                });
        });
    }

    fn draw_function_and_coloring_controls(&mut self, ui: &mut egui::Ui) {
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

        ui.add_space(6.0);
        ui.checkbox(
            &mut self.domain_coloring,
            "Show phase-modulus domain coloring",
        );
        ui.label("Domain Palette:");
        egui::ComboBox::from_id_salt("complex_domain_palette")
            .selected_text(self.palette.label())
            .show_ui(ui, |ui| {
                for candidate in ShaderPalette::ALL {
                    ui.selectable_value(&mut self.palette, candidate, candidate.label());
                }
            });
    }

    fn draw_grid_range_controls(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
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

        ui.add_space(6.0);
        let max_grid = crate::presets::MAX_INTERACTIVE_GRID_SIZE;
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

    fn draw_shader_preview(&mut self, ui: &mut egui::Ui) {
        let rev = self.shader_revision();
        let expr_res = self.resolve_active_shader_expr();
        let defaults = self.shader_uniform_defaults();
        render_shader_preview_section(
            ui,
            "complex_mapping_shader_preview",
            &mut self.shader_preview,
            rev,
            move |fmt| {
                let expr = expr_res?;
                emit_domain_coloring_shader(fmt, &expr, &defaults)
            },
        );
    }

    pub(super) fn draw_central_canvas(&mut self, parent_ui: &mut egui::Ui, grid_ok: bool) {
        let ctx = parent_ui.ctx().clone();
        if self.domain_coloring {
            self.sync_domain_coloring_texture(&ctx);
        }
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("Conformal Mapping & Phase-Modulus Domain Coloring");
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            if self.domain_coloring && ui.available_width() >= 680.0 {
                ui.columns(3, |columns| {
                    self.draw_z_plane(&mut columns[0], &palette, grid_ok);
                    self.draw_w_plane(&mut columns[1], &palette, grid_ok);
                    self.draw_domain_coloring_panel(&mut columns[2]);
                });
            } else {
                ui.columns(2, |columns| {
                    self.draw_z_plane(&mut columns[0], &palette, grid_ok);
                    self.draw_w_plane(&mut columns[1], &palette, grid_ok);
                });
                if self.domain_coloring {
                    ui.add_space(6.0);
                    self.draw_domain_coloring_panel(ui);
                }
            }
        });
    }

    fn sync_domain_coloring_texture(&mut self, ctx: &egui::Context) {
        let key = self.shader_revision();
        if self.domain_texture_key == Some(key) && self.domain_texture.is_some() {
            return;
        }
        let img = self.render_domain_coloring_image(DOMAIN_COLOR_PREVIEW_SIDE);
        let opts = egui::TextureOptions::LINEAR;
        if let Some(tex) = &mut self.domain_texture {
            tex.set(img, opts);
        } else {
            self.domain_texture = Some(ctx.load_texture("complex_domain_coloring", img, opts));
        }
        self.domain_texture_key = Some(key);
    }

    fn draw_domain_coloring_panel(&self, ui: &mut egui::Ui) {
        ui.label("Domain Coloring arg(f(z)), |f(z)|");
        let Some(tex) = &self.domain_texture else {
            return;
        };
        let avail = ui.available_size();
        let side = avail.x.min(avail.y).max(64.0);
        let response = ui
            .add(
                egui::Image::new(tex)
                    .fit_to_exact_size(egui::vec2(side, side))
                    .alt_text(self.theory_description())
                    .sense(egui::Sense::focusable_noninteractive()),
            )
            .accessible_theory_hover(self);
        ui.painter().rect_stroke(
            response.rect,
            0.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Inside,
        );
        crate::accessibility::paint_canvas_focus_ring(ui, &response);
    }

    pub(super) fn draw_z_plane(
        &self,
        ui: &mut egui::Ui,
        palette: &crate::app::SemanticPalette,
        grid_ok: bool,
    ) {
        ui.label("z-plane (Input)");
        Plot::new("z_plane_plot")
            .view_aspect(1.0)
            .x_axis_label("Re(z)")
            .y_axis_label("Im(z)")
            .show(ui, |plot_ui| {
                if grid_ok {
                    let (z_v, b_v) = &self.cache.z_vertical;
                    if !z_v.is_empty() {
                        plot_ui.line(
                            Line::from_segments("Re(z) grid", std::sync::Arc::clone(z_v))
                                .with_bounds(*b_v)
                                .color(palette.series_color(0)),
                        );
                    }
                    let (z_h, b_h) = &self.cache.z_horizontal;
                    if !z_h.is_empty() {
                        plot_ui.line(
                            Line::from_segments("Im(z) grid", std::sync::Arc::clone(z_h))
                                .with_bounds(*b_h)
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

    pub(super) fn draw_w_plane(
        &self,
        ui: &mut egui::Ui,
        palette: &crate::app::SemanticPalette,
        grid_ok: bool,
    ) {
        ui.label("w-plane (Output)");
        Plot::new("w_plane_plot")
            .view_aspect(1.0)
            .x_axis_label("Re(w)")
            .y_axis_label("Im(w)")
            .show(ui, |plot_ui| {
                if grid_ok {
                    let (w_v, b_v) = &self.cache.w_vertical;
                    if !w_v.is_empty() {
                        plot_ui.line(
                            Line::from_segments("u-contours", std::sync::Arc::clone(w_v))
                                .with_bounds(*b_v)
                                .color(palette.series_color(0)),
                        );
                    }
                    let (w_h, b_h) = &self.cache.w_horizontal;
                    if !w_h.is_empty() {
                        plot_ui.line(
                            Line::from_segments("v-contours", std::sync::Arc::clone(w_h))
                                .with_bounds(*b_h)
                                .color(palette.series_color(1)),
                        );
                    }
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
}
