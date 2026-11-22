#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::accessibility::AccessibleTheoryHover;
use crate::framework::bounded_drag_value;
use crate::framework::shader_foundry::{emit_fractal_shader, render_shader_preview_section};
use eframe::egui;
use math_commons::math_kernel::colormap::ShaderPalette;
use num_complex::Complex;
use scientific_metadata::theory::TheoryDescribable;

impl FractalViewer {
    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "fractal_settings");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("fractal_side_scroll")
                .show(ui, |ui| {
                    ui.heading("Settings");
                    ui.separator();
                    self.draw_mode_controls(ui);
                    self.draw_coloring_controls(ui);
                    self.draw_view_controls(ui);
                    self.draw_shader_preview(ui);
                });
        });
    }

    fn draw_mode_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .radio_value(&mut self.mode, FractalMode::Mandelbrot, "Mandelbrot")
                .clicked()
            {
                self.center = Complex::new(-0.75, 0.0);
                self.zoom = 1.0;
                self.mark_dirty();
            }
            if ui
                .radio_value(&mut self.mode, FractalMode::Julia, "Julia")
                .clicked()
            {
                self.center = Complex::new(0.0, 0.0);
                self.zoom = 1.0;
                self.mark_dirty();
            }
            if ui
                .radio_value(&mut self.mode, FractalMode::Custom, "Custom")
                .clicked()
            {
                self.center = Complex::new(0.0, 0.0);
                self.zoom = 1.0;
                self.mark_dirty();
            }
        });

        if self.mode == FractalMode::Custom {
            ui.add_space(4.0);
            if self.math_input.ui(ui) {
                self.mark_dirty();
            }
            if ui
                .checkbox(
                    &mut self.custom_julia_seed,
                    "Julia plane (seed z₀ = pixel, fixed c)",
                )
                .changed()
            {
                self.mark_dirty();
            }
        }

        if self.mode == FractalMode::Julia
            || (self.mode == FractalMode::Custom && self.custom_julia_seed)
        {
            ui.separator();
            ui.label("Complex Constant (c):");
            let re_drag = bounded_drag_value(&mut self.julia_c.re, COORD_MIN..=COORD_MAX, 0.005)
                .prefix("Re: ");
            let im_drag = bounded_drag_value(&mut self.julia_c.im, COORD_MIN..=COORD_MAX, 0.005)
                .prefix("Im: ");
            if ui.add(re_drag).changed() | ui.add(im_drag).changed() {
                self.mark_dirty();
            }
        }
    }

    fn draw_coloring_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Coloring & Palette:");
        if ui
            .checkbox(&mut self.smooth_coloring, "Smooth escape renormalization")
            .changed()
        {
            self.mark_dirty();
        }
        egui::ComboBox::from_id_salt("fractal_shader_palette")
            .selected_text(self.palette.label())
            .show_ui(ui, |ui| {
                for candidate in ShaderPalette::ALL {
                    if ui
                        .selectable_value(&mut self.palette, candidate, candidate.label())
                        .changed()
                    {
                        self.mark_dirty();
                    }
                }
            });
    }

    fn draw_view_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("View Controls:");
        let zoom_slider = egui::Slider::new(&mut self.zoom, ZOOM_MIN..=ZOOM_MAX)
            .logarithmic(true)
            .text("Zoom");
        if ui.add(zoom_slider).changed() {
            self.mark_dirty();
        }

        ui.label("Center:");
        let safe_zoom = if self.zoom.is_finite() && self.zoom >= ZOOM_MIN {
            self.zoom
        } else {
            1.0
        };
        let speed = 0.1 / safe_zoom;
        let re_drag =
            bounded_drag_value(&mut self.center.re, COORD_MIN..=COORD_MAX, speed).prefix("Re: ");
        let im_drag =
            bounded_drag_value(&mut self.center.im, COORD_MIN..=COORD_MAX, speed).prefix("Im: ");
        if ui.add(re_drag).changed() | ui.add(im_drag).changed() {
            self.mark_dirty();
        }

        let iter_max = math_commons::registry::MAX_ITERATIONS as u32;
        let iter_slider = egui::Slider::new(&mut self.max_iter, 10..=iter_max)
            .logarithmic(true)
            .text("Max Iter");
        if ui.add(iter_slider).changed() {
            self.mark_dirty();
        }

        ui.separator();
        if ui.button("↻ Reset View").clicked() {
            self.center = match self.mode {
                FractalMode::Mandelbrot => Complex::new(-0.75, 0.0),
                FractalMode::Julia | FractalMode::Custom => Complex::new(0.0, 0.0),
            };
            self.zoom = 1.0;
            self.mark_dirty();
        }
        ui.label("Navigation:");
        ui.label("- Drag to Pan (Single Touch/Mouse)");
        ui.label("- Scroll/Pinch to Zoom");
        ui.label("- Two-Finger Pan (Touch)");
    }

    fn draw_shader_preview(&mut self, ui: &mut egui::Ui) {
        let rev = self.request_generation;
        let expr_res = self.resolve_active_shader_expr();
        let defaults = self.shader_uniform_defaults();
        render_shader_preview_section(
            ui,
            "fractal_shader_preview",
            &mut self.shader_preview,
            rev,
            move |fmt| {
                let (expr, use_julia_seed) = expr_res?;
                emit_fractal_shader(fmt, &expr, use_julia_seed, &defaults)
            },
        );
    }

    fn pan_by_pixels(&mut self, delta: egui::Vec2, view_w: f64, view_h: f64) {
        if delta == egui::Vec2::ZERO || view_w <= 0.0 || view_h <= 0.0 {
            return;
        }
        let scale_y = 3.0 / self.zoom.clamp(ZOOM_MIN, ZOOM_MAX);
        let scale_x = scale_y * (view_w / view_h);
        let dx = -f64::from(delta.x) / view_w * scale_x;
        let dy = f64::from(delta.y) / view_h * scale_y;
        self.center.re = (self.center.re + dx).clamp(COORD_MIN, COORD_MAX);
        self.center.im = (self.center.im + dy).clamp(COORD_MIN, COORD_MAX);
        self.mark_dirty();
    }

    fn handle_canvas_navigation(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        view_w: f64,
        view_h: f64,
    ) {
        let multi_touch = ui.input(|i| i.multi_touch());
        if let Some(touch) = multi_touch {
            if touch.zoom_delta != 1.0 && touch.zoom_delta.is_finite() {
                let dampened = (1.0 + (touch.zoom_delta - 1.0) * 0.5).max(0.01);
                self.zoom = (self.zoom * f64::from(dampened)).clamp(ZOOM_MIN, ZOOM_MAX);
                self.mark_dirty();
            }
            self.pan_by_pixels(touch.translation_delta, view_w, view_h);
        } else if response.dragged() {
            self.pan_by_pixels(response.drag_delta(), view_w, view_h);
        }

        if response.hovered() && multi_touch.is_none() {
            let scroll_delta = ui.input(|i| i.raw_scroll_delta.y);
            if scroll_delta != 0.0 && scroll_delta.is_finite() {
                let zoom_factor = if scroll_delta > 0.0 { 1.1 } else { 0.9 };
                self.zoom = (self.zoom * zoom_factor).clamp(ZOOM_MIN, ZOOM_MAX);
                self.mark_dirty();
            }
        }
    }

    pub(super) fn draw_central_panel(
        &mut self,
        parent_ui: &mut egui::Ui,
        ctx: &egui::Context,
        max_texture_side: usize,
    ) {
        let has_real_screen = ctx.input(|i| i.raw.screen_rect.is_some());
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let available_size = ui.available_size();
            let raw_w = available_size.x.max(0.0) as usize;
            let raw_h = available_size.y.max(0.0) as usize;
            if raw_w == 0 || raw_h == 0 {
                return;
            }
            let (eff_w, eff_h) = if has_real_screen {
                self.viewport_size = (raw_w, raw_h);
                (raw_w, raw_h)
            } else {
                DEFAULT_EXPORT_VIEWPORT_SIZE
            };
            let (render_w, render_h) = clamp_fractal_dimensions(eff_w, eff_h, max_texture_side);
            if self.texture.is_none() {
                let fallback = egui::ColorImage::new([1, 1], vec![egui::Color32::BLACK]);
                self.texture = Some(ctx.load_texture(
                    "fractal_texture",
                    fallback,
                    egui::TextureOptions::LINEAR,
                ));
            }
            if let Some(texture) = &self.texture {
                let alt_label = self.theory_description();
                let response = ui
                    .add(
                        egui::Image::new(texture)
                            .maintain_aspect_ratio(false)
                            .fit_to_exact_size(available_size)
                            .alt_text(alt_label)
                            .sense(egui::Sense::click_and_drag()),
                    )
                    .accessible_theory_hover(self);
                ui.painter().rect_stroke(
                    response.rect,
                    0.0,
                    ui.visuals().widgets.noninteractive.bg_stroke,
                    egui::StrokeKind::Inside,
                );
                crate::accessibility::paint_canvas_focus_ring(ui, &response);
                self.handle_canvas_navigation(
                    ui,
                    &response,
                    f64::from(response.rect.width()),
                    f64::from(response.rect.height()),
                );
            }
            self.dispatch_render_request(render_w, render_h, ctx);
        });
    }
}
