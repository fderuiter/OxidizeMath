#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility, AccessibleHoverText, AccessibleTheoryHover,
};
use crate::framework::bounded_drag_value;
use eframe::egui;
use num_complex::Complex;
use scientific_metadata::theory::TheoryDescribable;

impl FractalViewer {
    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "fractal_settings");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Settings");
            ui.separator();
            self.draw_mode_selector(ui);

            if self.mode.is_custom() {
                ui.separator();
                if self.math_input.ui(ui) {
                    self.dirty = true;
                }
            }

            ui.separator();
            if self.mode.uses_julia_c() {
                ui.label("Julia Constant (c):");
                let re_drag = bounded_drag_value(&mut self.julia_c.re, COORD_MIN..=COORD_MAX, 0.01)
                    .prefix("Re: ");
                let im_drag = bounded_drag_value(&mut self.julia_c.im, COORD_MIN..=COORD_MAX, 0.01)
                    .prefix("Im: ");
                if ui.add(re_drag).changed() | ui.add(im_drag).changed() {
                    self.dirty = true;
                }
            } else {
                ui.label(format!(
                    "Probe c: {:+.4} {:+.4}i",
                    self.julia_c.re, self.julia_c.im
                ));
                ui.checkbox(&mut self.c_pinned, "Pin c crosshair")
                    .accessible_hover_text(
                        "Freeze c against hover; click Mandelbrot canvas or press Enter to toggle",
                    );
            }
            self.draw_view_controls(ui);
        });
    }

    fn draw_mode_selector(&mut self, ui: &mut egui::Ui) {
        let options = [
            (FractalMode::Mandelbrot, "Mandelbrot"),
            (FractalMode::Julia, "Julia"),
            (FractalMode::CustomMandelbrot, "Custom (c-plane)"),
            (FractalMode::CustomJulia, "Custom (Julia)"),
        ];
        let mut selected = self.mode;
        ui.horizontal_wrapped(|ui| {
            for (mode, label) in options {
                if ui.radio_value(&mut selected, mode, label).clicked() {
                    self.switch_mode(mode);
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
            self.sync_active_viewport();
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
            self.sync_active_viewport();
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
                FractalMode::Mandelbrot => DEFAULT_MANDELBROT_CENTER,
                FractalMode::Julia | FractalMode::CustomMandelbrot | FractalMode::CustomJulia => {
                    DEFAULT_JULIA_CENTER
                }
            };
            self.zoom = 1.0;
            self.sync_active_viewport();
            self.dirty = true;
        }
        ui.label("Navigation:");
        ui.label("- Hover/Click: Probe/Pin c (Mandelbrot)");
        ui.label("- Drag to Pan | Scroll/Pinch to Zoom");
        ui.label("- Arrows + Enter: Step/Pin c");
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
        self.sync_active_viewport();
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
                self.sync_active_viewport();
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
                self.sync_active_viewport();
                self.dirty = true;
            }
        }
    }

    fn handle_mandelbrot_pointer(&mut self, ui: &egui::Ui, response: &egui::Response) -> bool {
        if self.mode != FractalMode::Mandelbrot {
            return false;
        }
        let rect = response.rect;
        if response.clicked() {
            response.request_focus();
            if let Some(pos) = response.interact_pointer_pos() {
                let current_px = complex_to_screen(self.julia_c, rect, self.center, self.zoom);
                if self.c_pinned && current_px.distance(pos) <= 5.0 {
                    self.c_pinned = false;
                } else {
                    self.julia_c = screen_to_complex(pos, rect, self.center, self.zoom);
                    self.c_pinned = true;
                }
                return true;
            }
        } else if response.hovered() && !response.dragged() && !self.c_pinned {
            let ptr = ui.input(|i| i.pointer.hover_pos().or_else(|| i.pointer.latest_pos()));
            if let Some(pos) = ptr.filter(|p| rect.contains(*p)) {
                self.julia_c = screen_to_complex(pos, rect, self.center, self.zoom);
            }
        }
        false
    }

    fn handle_focused_canvas_keys(&mut self, ui: &egui::Ui, response: &egui::Response) -> bool {
        if !response.has_focus() {
            return false;
        }
        let (left, right, up, down, enter) = ui.ctx().input_mut(|i| {
            (
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft),
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight),
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
                i.consume_key(egui::Modifiers::NONE, egui::Key::Enter),
            )
        });
        if left || right || up || down {
            ui.ctx()
                .memory_mut(|m| m.move_focus(egui::FocusDirection::None));
        }
        let step = (0.01 / self.zoom.clamp(ZOOM_MIN, ZOOM_MAX)).clamp(1e-6, 0.25);
        let mut c_changed = false;
        if left {
            self.julia_c = Complex::new(
                (self.julia_c.re - step).clamp(COORD_MIN, COORD_MAX),
                self.julia_c.im,
            );
            c_changed = true;
        }
        if right {
            self.julia_c = Complex::new(
                (self.julia_c.re + step).clamp(COORD_MIN, COORD_MAX),
                self.julia_c.im,
            );
            c_changed = true;
        }
        if up {
            self.julia_c = Complex::new(
                self.julia_c.re,
                (self.julia_c.im + step).clamp(COORD_MIN, COORD_MAX),
            );
            c_changed = true;
        }
        if down {
            self.julia_c = Complex::new(
                self.julia_c.re,
                (self.julia_c.im - step).clamp(COORD_MIN, COORD_MAX),
            );
            c_changed = true;
        }
        if enter {
            self.c_pinned = !self.c_pinned;
        }
        if c_changed && self.mode.uses_julia_c() {
            self.dirty = true;
        }
        c_changed || enter
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
            if let Some(texture) = self.texture.clone() {
                let alt_label = self.theory_description();
                let response = ui
                    .add(
                        egui::Image::new(&texture)
                            .maintain_aspect_ratio(false)
                            .fit_to_exact_size(available_size)
                            .alt_text(alt_label.clone())
                            .sense(egui::Sense::click_and_drag()),
                    )
                    .accessible_theory_hover(self);

                self.handle_canvas_navigation(
                    ui,
                    &response,
                    f64::from(response.rect.width()),
                    f64::from(response.rect.height()),
                );
                let mut announce = self.handle_mandelbrot_pointer(ui, &response);
                announce |= self.handle_focused_canvas_keys(ui, &response);
                self.draw_mandelbrot_overlays(ui, ctx, response.rect);

                ui.painter().rect_stroke(
                    response.rect,
                    0.0,
                    ui.visuals().widgets.noninteractive.bg_stroke,
                    egui::StrokeKind::Inside,
                );
                self.finalize_canvas_a11y(ui, ctx, &response, &alt_label, announce);
            }
            self.dispatch_render_request(render_w, render_h, ctx);
        });
    }

    fn finalize_canvas_a11y(
        &self,
        ui: &egui::Ui,
        ctx: &egui::Context,
        response: &egui::Response,
        alt_label: &str,
        announce: bool,
    ) {
        lock_canvas_arrow_keys(ui, response);
        paint_canvas_focus_ring(ui, response);
        let c_status = format!(
            "c = ({:+.4}, {:+.4}i) [{}]",
            self.julia_c.re,
            self.julia_c.im,
            if self.c_pinned { "pinned" } else { "unpinned" }
        );
        register_canvas_accessibility(
            ui,
            response,
            egui::accesskit::Role::Image,
            alt_label,
            Some(&c_status),
        );
        if announce {
            announce_live_status(ctx, format!("Fractal probe {c_status}"));
        }
    }
}
