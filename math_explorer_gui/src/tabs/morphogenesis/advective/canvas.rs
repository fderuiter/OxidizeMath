//! Interactive `100 x 50` wind-tunnel canvas, mouse/keyboard obstacle sculpting, AccessKit node,
//! and paired wake-plume `egui_plot` inspector for [`super::AdvectiveMorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::AdvectiveViewMode;
use super::AdvectiveMorphogenesisTool;
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility,
};
use crate::app::{SemanticPalette, ThemeMode};
use crate::framework::{
    apply_stroke_to_fairing_mask, toggle_fairing_cell, BrushMode, BrushStroke, LBM_HEIGHT,
    LBM_INLET_COLS, LBM_WIDTH,
};
use eframe::egui;
use eframe::egui::ColorImage;
use egui_plot::{Line, Plot, PlotPoints};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;

impl AdvectiveMorphogenesisTool {
    pub(crate) fn refresh_pixels_in_place(&mut self) {
        let total = LBM_WIDTH * LBM_HEIGHT;
        if self.pixels.len() != total {
            self.pixels.resize(total, egui::Color32::BLACK);
        }
        let palette = SemanticPalette::for_theme(if self.high_contrast {
            ThemeMode::HighContrast
        } else {
            ThemeMode::Dark
        });
        let field = self.engine.active_morphogen(self.cfg.preset);
        let (f_min, f_span) = fluid_min_span(field, &self.engine.mask);
        let vort_scale = self.engine.metrics.max_vorticity.max(0.008);

        for y in 0..LBM_HEIGHT {
            let src_y = LBM_HEIGHT - 1 - y;
            for x in 0..LBM_WIDTH {
                let dst_idx = flatten_2d_index(x, y, LBM_WIDTH);
                let src_idx = flatten_2d_index(x, src_y, LBM_WIDTH);
                if self.engine.mask[src_idx] {
                    self.pixels[dst_idx] = palette.surface_fill;
                    continue;
                }
                let norm_c = ((field[src_idx] - f_min) / f_span).clamp(0.0, 1.0);
                let vort = self.engine.vorticity[src_idx];
                let scalar = match self.cfg.view_mode {
                    AdvectiveViewMode::MorphogenOnly => norm_c,
                    AdvectiveViewMode::VorticityOnly => {
                        (0.5 + 0.5 * (vort / vort_scale)).clamp(0.0, 1.0)
                    }
                    AdvectiveViewMode::MorphogenWithVorticity => {
                        let contour = ((vort.abs() / vort_scale) * 0.28).clamp(0.0, 0.28);
                        (0.82 * norm_c + contour).clamp(0.0, 1.0)
                    }
                };
                let rgb = if self.high_contrast {
                    heatmap_color_high_contrast(scalar)
                } else {
                    heatmap_color(scalar)
                };
                self.pixels[dst_idx] = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
            }
        }
        self.texture_dirty = true;
    }

    fn sync_gpu_texture(&mut self, ctx: &egui::Context) {
        if !self.texture_dirty && self.texture.is_some() {
            return;
        }
        let image = ColorImage {
            size: [LBM_WIDTH, LBM_HEIGHT],
            pixels: self.pixels.clone(),
            ..Default::default()
        };
        let opts = egui::TextureOptions::NEAREST;
        if let Some(tex) = &mut self.texture {
            tex.set(image, opts);
        } else {
            self.texture = Some(ctx.load_texture("advective_morphogenesis_canvas", image, opts));
        }
        self.texture_dirty = false;
    }

    pub(crate) fn draw_wind_tunnel_canvas(&mut self, ui: &mut egui::Ui, avail_width: f32) {
        let ctx = ui.ctx().clone();
        self.sync_gpu_texture(&ctx);

        let canvas_w = avail_width.clamp(280.0, 760.0);
        let canvas_h = (canvas_w * 0.5).clamp(140.0, 380.0);
        let (response, painter) = ui.allocate_painter(
            egui::vec2(canvas_w, canvas_h),
            egui::Sense::click_and_drag(),
        );

        if let Some(tex) = &self.texture {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            painter.image(tex.id(), response.rect, uv, egui::Color32::WHITE);
        }

        self.handle_pointer_sculpting(ui, &response);
        self.handle_keyboard_sculpting(ui, &response);
        self.paint_cursor_overlay(ui, &response, &painter);

        let m = &self.engine.metrics;
        let summary = format!(
            "{} on 100x50 lattice: Pe {:.1}, Re {:.1}, C_D {:.3}, solid cells {}, cursor ({}, {})",
            self.cfg.preset.label(),
            self.cfg.peclet,
            m.reynolds,
            m.c_d,
            m.solid_cells,
            self.cursor[0],
            self.cursor[1],
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "Advective Morphogenesis Wind-Tunnel Canvas",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &response);
        lock_canvas_arrow_keys(ui, &response);
        self.focused_canvas = response.has_focus().then_some(response.id);
    }

    fn handle_pointer_sculpting(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if response.clicked() || response.dragged() {
            response.request_focus();
        }
        let primary = response.clicked_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Primary);
        let secondary = response.clicked_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Secondary);
        if !primary && !secondary {
            self.last_drag_plot_pt = None;
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        if !response.rect.contains(pos) {
            return;
        }
        let rel_x = ((pos.x - response.rect.min.x) / response.rect.width()).clamp(0.0, 0.999_9);
        let rel_y = ((pos.y - response.rect.min.y) / response.rect.height()).clamp(0.0, 0.999_9);
        let plot_x = f64::from(rel_x) * (LBM_WIDTH as f64);
        let plot_y = (1.0 - f64::from(rel_y)) * (LBM_HEIGHT as f64);
        self.cursor = [
            (plot_x.floor() as usize).clamp(LBM_INLET_COLS, LBM_WIDTH - 1),
            (plot_y.floor() as usize).min(LBM_HEIGHT - 1),
        ];

        let mode = if secondary || ui.input(|i| i.modifiers.shift) {
            BrushMode::Erase
        } else {
            self.cfg.brush_mode
        };
        let start = self.last_drag_plot_pt.unwrap_or([plot_x, plot_y]);
        let stroke = BrushStroke {
            mode,
            start,
            end: [plot_x, plot_y],
            radius: self.cfg.brush_radius,
            drag_vector: [plot_x - start[0], plot_y - start[1]],
        };
        self.last_drag_plot_pt = Some([plot_x, plot_y]);
        self.apply_brush_stroke(&stroke);
    }

    /// Applies a spatial [`BrushStroke`] to the `100 x 50` obstacle mask and updates `mask_hex`.
    pub fn apply_brush_stroke(&mut self, stroke: &BrushStroke) -> bool {
        if apply_stroke_to_fairing_mask(&mut self.engine.mask, stroke, false) {
            self.sync_custom_mask_mutation();
            true
        } else {
            false
        }
    }

    fn handle_keyboard_sculpting(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if !response.has_focus() {
            return;
        }
        let step = if ui.input(|i| i.modifiers.shift) {
            4
        } else {
            1
        };
        let mut moved = false;
        let mut toggled = false;
        ui.input(|i| {
            if i.key_pressed(egui::Key::ArrowLeft) {
                self.cursor[0] = self.cursor[0].saturating_sub(step).max(LBM_INLET_COLS);
                moved = true;
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                self.cursor[0] = (self.cursor[0] + step).min(LBM_WIDTH - 1);
                moved = true;
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                self.cursor[1] = self.cursor[1].saturating_sub(step);
                moved = true;
            }
            if i.key_pressed(egui::Key::ArrowUp) {
                self.cursor[1] = (self.cursor[1] + step).min(LBM_HEIGHT - 1);
                moved = true;
            }
            if i.key_pressed(egui::Key::Space) || i.key_pressed(egui::Key::Enter) {
                toggled = true;
            }
        });
        if toggled {
            if let Some(is_solid) =
                toggle_fairing_cell(&mut self.engine.mask, self.cursor[0], self.cursor[1], false)
            {
                self.sync_custom_mask_mutation();
                let state_str = if is_solid { "solid" } else { "fluid" };
                announce_live_status(
                    ui.ctx(),
                    format!(
                        "Cell ({}, {}) set to {}",
                        self.cursor[0], self.cursor[1], state_str
                    ),
                );
            }
        } else if moved {
            announce_live_status(
                ui.ctx(),
                format!("Sculpt cursor at ({}, {})", self.cursor[0], self.cursor[1]),
            );
        }
    }

    fn paint_cursor_overlay(
        &self,
        ui: &egui::Ui,
        response: &egui::Response,
        painter: &egui::Painter,
    ) {
        if !response.has_focus() && !response.hovered() {
            return;
        }
        let palette = SemanticPalette::for_theme(if self.high_contrast {
            ThemeMode::HighContrast
        } else {
            ThemeMode::from_visuals(ui.visuals())
        });
        let cell_w = response.rect.width() / (LBM_WIDTH as f32);
        let cell_h = response.rect.height() / (LBM_HEIGHT as f32);
        let cx = response.rect.min.x + ((self.cursor[0] as f32) + 0.5) * cell_w;
        let cy = response.rect.min.y + (((LBM_HEIGHT - 1 - self.cursor[1]) as f32) + 0.5) * cell_h;
        let r_px = ((self.cfg.brush_radius as f32).max(0.75)) * cell_w;
        painter.circle_stroke(
            egui::pos2(cx, cy),
            r_px,
            egui::Stroke::new(1.5_f32, palette.accent),
        );
    }

    pub(crate) fn draw_wake_profile_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::for_theme(ThemeMode::from_visuals(ui.visuals()));
        let datasets = self.collect_wake_profile_datasets();
        Plot::new("advective_wake_plume_plot")
            .height(165.0)
            .allow_scroll(false)
            .x_axis_label("Streamwise Lattice X")
            .y_axis_label("Wake Profile")
            .show(ui, |plot_ui| {
                if let Some((_, morph)) = datasets.first() {
                    plot_ui.line(
                        Line::new("Centerline Morphogen", PlotPoints::from(morph.clone()))
                            .color(palette.series[0])
                            .width(2.0_f32),
                    );
                }
                if let Some((_, vort)) = datasets.get(1) {
                    plot_ui.line(
                        Line::new(
                            "Shear-Layer Vorticity (×10)",
                            PlotPoints::from(vort.clone()),
                        )
                        .color(palette.series[1])
                        .width(1.5_f32),
                    );
                }
            });
    }
}

fn fluid_min_span(field: &[f64], mask: &[bool]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (val, &is_obs) in field.iter().zip(mask.iter()) {
        if !is_obs && val.is_finite() {
            lo = lo.min(*val);
            hi = hi.max(*val);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, 1.0);
    }
    (lo, (hi - lo).max(1e-6))
}
