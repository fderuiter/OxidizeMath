#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::accessibility::AccessibleTheoryHover;
use crate::framework::bounded_drag_value;
use eframe::egui;
use num_complex::Complex;
use scientific_metadata::theory::TheoryDescribable;

impl FractalViewer {
    /// Synchronizes the active `(center, zoom)` into the per-mode viewport cache.
    pub(crate) fn sync_active_viewport_cache(&mut self) {
        match self.mode {
            FractalMode::Mandelbrot => self.mandelbrot_viewport = (self.center, self.zoom),
            FractalMode::Julia => self.julia_viewport = (self.center, self.zoom),
        }
    }

    /// Switches between Mandelbrot and Julia modes while preserving each mode's `(center, zoom)`.
    pub fn switch_mode(&mut self, next_mode: FractalMode) {
        if self.mode == next_mode {
            return;
        }
        self.sync_active_viewport_cache();
        self.mode = next_mode;
        let (cached_center, cached_zoom) = match next_mode {
            FractalMode::Mandelbrot => self.mandelbrot_viewport,
            FractalMode::Julia => self.julia_viewport,
        };
        self.center = cached_center;
        self.zoom = cached_zoom;
        self.hovered_c = None;
        self.dirty = true;
    }

    /// Teleports `julia_c` to `c` and switches into progressive Julia rendering while preserving viewports.
    pub fn teleport_to_julia(&mut self, c: Complex<f64>) {
        if !c.re.is_finite() || !c.im.is_finite() {
            return;
        }
        let clamped_c = Complex::new(
            c.re.clamp(COORD_MIN, COORD_MAX),
            c.im.clamp(COORD_MIN, COORD_MAX),
        );
        self.sync_active_viewport_cache();
        self.julia_c = clamped_c;
        self.mode = FractalMode::Julia;
        let (julia_center, julia_zoom) = self.julia_viewport;
        self.center = julia_center;
        self.zoom = julia_zoom;
        self.dirty = true;
    }

    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "fractal_settings");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Settings");
            ui.separator();
            let mut next_mode = self.mode;
            ui.horizontal(|ui| {
                ui.radio_value(&mut next_mode, FractalMode::Mandelbrot, "Mandelbrot");
                ui.radio_value(&mut next_mode, FractalMode::Julia, "Julia");
            });
            if next_mode != self.mode {
                self.switch_mode(next_mode);
            }

            ui.checkbox(&mut self.linked_explorer, "Linked Julia Explorer");

            if self.mode == FractalMode::Julia {
                ui.separator();
                ui.label("Julia Constant (c):");
                let re_drag =
                    bounded_drag_value(&mut self.julia_c.re, COORD_MIN..=COORD_MAX, 0.005)
                        .prefix("Re: ");
                let im_drag =
                    bounded_drag_value(&mut self.julia_c.im, COORD_MIN..=COORD_MAX, 0.005)
                        .prefix("Im: ");
                if ui.add(re_drag).changed() | ui.add(im_drag).changed() {
                    self.dirty = true;
                }
                if self.linked_explorer && ui.button("← Back to Mandelbrot Atlas").clicked() {
                    self.switch_mode(FractalMode::Mandelbrot);
                }
            } else if self.linked_explorer {
                ui.small("Hover Mandelbrot to preview Julia; click or press Enter to inspect.");
            }
            self.draw_view_controls(ui);
        });
    }

    fn draw_view_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("View Controls:");
        let zoom_slider = egui::Slider::new(&mut self.zoom, ZOOM_MIN..=ZOOM_MAX)
            .logarithmic(true)
            .text("Zoom");
        if ui.add(zoom_slider).changed() {
            self.sync_active_viewport_cache();
            self.dirty = true;
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
            self.sync_active_viewport_cache();
            self.dirty = true;
        }

        let iter_max = math_commons::registry::MAX_ITERATIONS as u32;
        let iter_slider = egui::Slider::new(&mut self.max_iter, 10..=iter_max)
            .logarithmic(true)
            .text("Max Iter");
        if ui.add(iter_slider).changed() {
            self.dirty = true;
        }

        ui.separator();
        if ui.button("↻ Reset View").clicked() {
            self.center = match self.mode {
                FractalMode::Mandelbrot => Complex::new(-0.75, 0.0),
                FractalMode::Julia => Complex::new(0.0, 0.0),
            };
            self.zoom = 1.0;
            self.sync_active_viewport_cache();
            self.dirty = true;
        }
        ui.label("Navigation:");
        ui.label("- Drag to Pan (Single Touch/Mouse)");
        ui.label("- Scroll/Pinch to Zoom");
        ui.label("- Click/Enter on Mandelbrot to Teleport Julia");
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
        self.sync_active_viewport_cache();
        self.dirty = true;
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
                self.sync_active_viewport_cache();
                self.dirty = true;
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
                self.sync_active_viewport_cache();
                self.dirty = true;
            }
        }
    }

    fn handle_mandelbrot_bridge(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if self.mode != FractalMode::Mandelbrot || !self.linked_explorer {
            self.hovered_c = None;
            return;
        }

        if response.clicked() && !response.dragged() {
            let click_pos = response
                .interact_pointer_pos()
                .unwrap_or_else(|| response.rect.center());
            let target_c =
                map_canvas_pos_to_complex(click_pos, response.rect, self.center, self.zoom);
            self.teleport_to_julia(target_c);
            return;
        }
        if response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            let target_c = self.hovered_c.unwrap_or(self.center);
            self.teleport_to_julia(target_c);
            return;
        }

        let pointer_down = ui.input(|i| i.pointer.any_down());
        if pointer_down || response.dragged() {
            return;
        }
        if let Some(pos) = response.hover_pos() {
            self.hovered_c = Some(map_canvas_pos_to_complex(
                pos,
                response.rect,
                self.center,
                self.zoom,
            ));
        } else if !response.has_focus() {
            self.hovered_c = None;
        } else if self.hovered_c.is_none() {
            self.hovered_c = Some(Complex::new(
                self.center.re.clamp(COORD_MIN, COORD_MAX),
                self.center.im.clamp(COORD_MIN, COORD_MAX),
            ));
        }
    }

    fn refresh_julia_hover_texture(&mut self, ctx: &egui::Context, hover_c: Complex<f64>) {
        let preview_iter = self.max_iter.clamp(10, 48);
        let key = (
            hover_c.re.to_bits(),
            hover_c.im.to_bits(),
            preview_iter,
            self.high_contrast,
        );
        if self.julia_inset_key == Some(key) && self.julia_inset_texture.is_some() {
            return;
        }
        let req = FractalRenderRequest {
            generation: 0,
            width: JULIA_INSET_SIZE.0,
            height: JULIA_INSET_SIZE.1,
            center: Complex::new(0.0, 0.0),
            zoom: 1.0,
            max_iter: preview_iter,
            mode: FractalMode::Julia,
            julia_c: hover_c,
            high_contrast: self.high_contrast,
        };
        let img = render_fractal_color_image(&req);
        let opts = egui::TextureOptions::LINEAR;
        if let Some(tex) = &mut self.julia_inset_texture {
            tex.set(img, opts);
        } else {
            self.julia_inset_texture = Some(ctx.load_texture("julia_hover_inset", img, opts));
        }
        self.julia_inset_key = Some(key);
    }

    fn paint_julia_hover_inset(
        &mut self,
        ui: &egui::Ui,
        ctx: &egui::Context,
        canvas_rect: egui::Rect,
    ) {
        if self.mode != FractalMode::Mandelbrot
            || !self.linked_explorer
            || ui.input(|i| i.pointer.any_down())
        {
            return;
        }
        let Some(hover_c) = self.hovered_c else {
            return;
        };
        self.refresh_julia_hover_texture(ctx, hover_c);

        let inset_size = egui::vec2(JULIA_INSET_SIZE.0 as f32, JULIA_INSET_SIZE.1 as f32);
        if canvas_rect.width() < inset_size.x + 24.0 || canvas_rect.height() < inset_size.y + 36.0 {
            return;
        }
        let box_rect = egui::Rect::from_min_size(
            egui::pos2(canvas_rect.max.x - inset_size.x - 12.0, canvas_rect.min.y + 12.0),
            egui::vec2(inset_size.x + 8.0, inset_size.y + 22.0),
        );
        let img_rect = egui::Rect::from_min_size(box_rect.min + egui::vec2(4.0, 4.0), inset_size);
        let painter = ui.painter();
        painter.rect_filled(box_rect, 4.0, egui::Color32::from_black_alpha(215));
        painter.rect_stroke(
            box_rect,
            4.0,
            egui::Stroke::new(1.0_f32, egui::Color32::from_gray(200)),
            egui::StrokeKind::Inside,
        );
        if let Some(tex) = &self.julia_inset_texture {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            painter.image(tex.id(), img_rect, uv, egui::Color32::WHITE);
        }
        painter.text(
            egui::pos2(box_rect.center().x, img_rect.max.y + 9.0),
            egui::Align2::CENTER_CENTER,
            format!("c={:.2}{:+.2}i", hover_c.re, hover_c.im),
            egui::FontId::proportional(10.5),
            egui::Color32::WHITE,
        );
    }

    pub(super) fn draw_central_panel(
        &mut self,
        parent_ui: &mut egui::Ui,
        ctx: &egui::Context,
        max_texture_side: usize,
    ) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let available_size = ui.available_size();
            let raw_w = available_size.x.max(0.0) as usize;
            let raw_h = available_size.y.max(0.0) as usize;
            if raw_w == 0 || raw_h == 0 {
                return;
            }
            let (render_w, render_h) = clamp_fractal_dimensions(raw_w, raw_h, max_texture_side);
            if self.texture.is_none() {
                let fallback = egui::ColorImage::new([1, 1], vec![egui::Color32::BLACK]);
                self.texture = Some(ctx.load_texture(
                    "fractal_texture",
                    fallback,
                    egui::TextureOptions::LINEAR,
                ));
            }
            let response_opt = self.texture.as_ref().map(|texture| {
                let alt_label = self.theory_description();
                ui.add(
                    egui::Image::new(texture)
                        .maintain_aspect_ratio(false)
                        .fit_to_exact_size(available_size)
                        .alt_text(alt_label)
                        .sense(egui::Sense::click_and_drag()),
                )
                .accessible_theory_hover(self)
            });
            if let Some(response) = response_opt {
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
                self.handle_mandelbrot_bridge(ui, &response);
                self.paint_julia_hover_inset(ui, ctx, response.rect);
            }
            self.dispatch_render_request(render_w, render_h, ctx);
        });
    }
}
