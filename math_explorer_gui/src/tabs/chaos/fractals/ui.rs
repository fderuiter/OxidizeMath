#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::accessibility::AccessibleTheoryHover;
use crate::framework::bounded_drag_value;
use eframe::egui;
use num_complex::Complex;
use scientific_metadata::theory::TheoryDescribable;

impl FractalViewer {
    pub(crate) fn sync_active_plane_view(&mut self) {
        if self.mode.uses_julia_c() {
            self.z_plane_view = (self.center, self.zoom);
        } else {
            self.c_plane_view = (self.center, self.zoom);
        }
    }

    pub(crate) fn switch_mode(&mut self, next_mode: FractalMode) {
        if self.mode == next_mode {
            return;
        }
        self.sync_active_plane_view();
        self.mode = next_mode;
        let (next_center, next_zoom) = if next_mode.uses_julia_c() {
            self.z_plane_view
        } else {
            self.c_plane_view
        };
        self.center = next_center;
        self.zoom = next_zoom.clamp(ZOOM_MIN, ZOOM_MAX);
        self.hover_c = None;
        self.dirty = true;
    }

    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "fractal_settings");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Settings");
            ui.separator();
            self.draw_mode_selector(ui);
            let _ = self.atlas.show_toggle_button(ui);

            if self.mode.is_custom() {
                ui.separator();
                if self.math_input.ui(ui) {
                    self.dirty = true;
                }
            }

            ui.separator();
            ui.label("Julia Constant (c):");
            let re_drag = bounded_drag_value(&mut self.julia_c.re, COORD_MIN..=COORD_MAX, 0.01)
                .prefix("Re: ");
            let im_drag = bounded_drag_value(&mut self.julia_c.im, COORD_MIN..=COORD_MAX, 0.01)
                .prefix("Im: ");
            if (ui.add(re_drag).changed() | ui.add(im_drag).changed()) && self.mode.uses_julia_c() {
                self.dirty = true;
            }
            if let Some(hc) = self.hover_c.filter(|_| !self.mode.uses_julia_c()) {
                ui.small(format!(
                    "Hover c: ({:+.4}, {:+.4}) • Click c-plane to pin",
                    hc.re, hc.im
                ));
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
        let current = self.mode;
        ui.horizontal_wrapped(|ui| {
            for (mode, label) in options {
                if ui.selectable_label(current == mode, label).clicked() {
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
            self.sync_active_plane_view();
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
            self.sync_active_plane_view();
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
                FractalMode::Julia | FractalMode::CustomMandelbrot | FractalMode::CustomJulia => {
                    Complex::new(0.0, 0.0)
                }
            };
            self.zoom = 1.0;
            self.sync_active_plane_view();
            self.dirty = true;
        }
        ui.label("Navigation:");
        ui.label("- Drag to Pan / Scroll to Zoom");
        ui.label("- Click c-plane to set Julia Constant c");
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
        self.sync_active_plane_view();
        self.dirty = true;
    }

    pub(crate) fn screen_to_complex(&self, rect: egui::Rect, pos: egui::Pos2) -> Complex<f64> {
        let w = f64::from(rect.width()).max(1.0);
        let h = f64::from(rect.height()).max(1.0);
        let scale_y = 3.0 / self.zoom.clamp(ZOOM_MIN, ZOOM_MAX);
        let scale_x = scale_y * (w / h);
        let u = f64::from(pos.x - rect.left()) / w;
        let v = f64::from(pos.y - rect.top()) / h;
        Complex::new(
            (self.center.re + (u - 0.5) * scale_x).clamp(COORD_MIN, COORD_MAX),
            (self.center.im + (0.5 - v) * scale_y).clamp(COORD_MIN, COORD_MAX),
        )
    }

    fn complex_to_screen(&self, rect: egui::Rect, z: Complex<f64>) -> Option<egui::Pos2> {
        if !z.re.is_finite() || !z.im.is_finite() {
            return None;
        }
        let w = f64::from(rect.width()).max(1.0);
        let h = f64::from(rect.height()).max(1.0);
        let scale_y = 3.0 / self.zoom.clamp(ZOOM_MIN, ZOOM_MAX);
        let scale_x = scale_y * (w / h);
        let u = (z.re - self.center.re) / scale_x + 0.5;
        let v = 0.5 - (z.im - self.center.im) / scale_y;
        Some(egui::pos2(
            rect.left() + (u * w) as f32,
            rect.top() + (v * h) as f32,
        ))
    }

    fn paint_c_plane_orbit_overlay(&self, ui: &egui::Ui, rect: egui::Rect) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let painter = ui.painter().with_clip_rect(rect);
        let probe_c = self.hover_c.unwrap_or(self.julia_c);
        let mut orbit = [Complex::new(0.0, 0.0); 24];
        let count = if self.mode == FractalMode::CustomMandelbrot {
            let guard = self
                .shared_formula
                .read()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(job) = guard.as_deref() {
                job.trace_critical_orbit(probe_c, &mut orbit)
            } else {
                CompiledFractalJob::from_bar(&self.math_input)
                    .trace_critical_orbit(probe_c, &mut orbit)
            }
        } else {
            self.quadratic_job.trace_critical_orbit(probe_c, &mut orbit)
        };

        let mut prev_pos: Option<egui::Pos2> = None;
        for &z in &orbit[..count] {
            if let Some(screen_pt) = self.complex_to_screen(rect, z) {
                if let Some(prev) = prev_pos {
                    painter.line_segment(
                        [prev, screen_pt],
                        egui::Stroke::new(1.5_f32, palette.warning),
                    );
                }
                painter.circle_filled(screen_pt, 2.5, palette.info);
                prev_pos = Some(screen_pt);
            }
        }
        if let Some(seed_pt) = self.complex_to_screen(rect, self.julia_c) {
            painter.circle_stroke(seed_pt, 5.5, egui::Stroke::new(2.0_f32, palette.accent));
        }
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
                self.sync_active_plane_view();
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
                self.sync_active_plane_view();
                self.dirty = true;
            }
        }

        if !self.mode.uses_julia_c() {
            self.hover_c = response
                .hover_pos()
                .map(|pos| self.screen_to_complex(response.rect, pos));
            if response.clicked() && !response.dragged() {
                if let Some(pos) = response
                    .interact_pointer_pos()
                    .or_else(|| response.hover_pos())
                {
                    self.julia_c = self.screen_to_complex(response.rect, pos);
                }
            }
            self.paint_c_plane_orbit_overlay(ui, response.rect);
        } else {
            self.hover_c = None;
        }
    }

    fn draw_fractal_canvas(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        max_texture_side: usize,
    ) {
        let available_size = ui.available_size();
        let raw_w = available_size.x.max(0.0) as usize;
        let raw_h = available_size.y.max(0.0) as usize;
        if raw_w == 0 || raw_h == 0 {
            return;
        }
        let (render_w, render_h) = clamp_fractal_dimensions(raw_w, raw_h, max_texture_side);
        if self.texture.is_none() {
            let fallback = egui::ColorImage::new([1, 1], vec![egui::Color32::BLACK]);
            self.texture =
                Some(ctx.load_texture("fractal_texture", fallback, egui::TextureOptions::LINEAR));
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
    }

    pub(super) fn draw_central_panel(
        &mut self,
        parent_ui: &mut egui::Ui,
        ctx: &egui::Context,
        max_texture_side: usize,
    ) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            if self.atlas.is_visible() {
                if ui.available_width() >= 620.0 {
                    ui.columns(2, |cols| {
                        self.draw_fractal_canvas(&mut cols[0], ctx, max_texture_side);
                        self.render_fractal_atlas_panel(&mut cols[1]);
                    });
                } else {
                    self.render_fractal_atlas_panel(ui);
                    ui.separator();
                    self.draw_fractal_canvas(ui, ctx, max_texture_side);
                }
            } else {
                self.draw_fractal_canvas(ui, ctx, max_texture_side);
            }
        });
    }
}
