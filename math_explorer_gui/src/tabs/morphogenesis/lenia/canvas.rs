//! Interactive periodic torus canvas, mouse brushing, keyboard stamping, AccessKit registration,
//! and WCAG 2.2 AA colormap rendering for [`super::LeniaTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{LeniaBrushMode, LeniaTool};
use crate::accessibility::{
    lock_canvas_arrow_keys, paint_canvas_focus_ring, register_canvas_accessibility,
};
use eframe::egui;
use eframe::egui::ColorImage;
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_explorer::biology::lenia::presets::stamp_species_at;

impl LeniaTool {
    pub(crate) fn refresh_pixels_in_place(&mut self) {
        let total = self.engine.width() * self.engine.height();
        if self.pixels.len() != total {
            self.pixels.resize(total, egui::Color32::BLACK);
        }
        for (dst, &val) in self.pixels.iter_mut().zip(self.engine.state().iter()) {
            let clean = if val.is_finite() {
                val.clamp(0.0, 1.0)
            } else {
                0.0
            };
            let rgb = if self.high_contrast {
                heatmap_color_high_contrast(clean)
            } else {
                heatmap_color(clean)
            };
            *dst = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
        }
        self.texture_dirty = true;
    }

    fn sync_gpu_texture(&mut self, ctx: &egui::Context) {
        if !self.texture_dirty && self.texture.is_some() {
            return;
        }
        let image = ColorImage {
            size: [self.engine.width(), self.engine.height()],
            pixels: self.pixels.clone(),
            ..Default::default()
        };
        let opts = egui::TextureOptions::NEAREST;
        if let Some(tex) = &mut self.texture {
            tex.set(image, opts);
        } else {
            self.texture = Some(ctx.load_texture("lenia_studio_torus", image, opts));
        }
        self.texture_dirty = false;
    }

    pub(crate) fn draw_interactive_canvas(&mut self, ui: &mut egui::Ui, target_side: f32) {
        let ctx = ui.ctx().clone();
        self.sync_gpu_texture(&ctx);

        let side = target_side.clamp(140.0, 560.0);
        let (response, painter) =
            ui.allocate_painter(egui::vec2(side, side), egui::Sense::click_and_drag());

        if let Some(tex) = &self.texture {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            painter.image(tex.id(), response.rect, uv, egui::Color32::WHITE);
        }

        self.handle_pointer_brushing(ui, &response);
        self.handle_keyboard_stamping(ui, &response);
        self.paint_cursor_reticle(ui, &response, &painter);

        let tel = self.engine.telemetry();
        let value_summary = format!(
            "{} on {}x{} torus: mass {:.1}, active cells {:.0}, centroid speed {:.2}, cursor ({}, {})",
            self.species.label(),
            self.engine.width(),
            self.engine.height(),
            tel.total_mass,
            tel.active_area,
            tel.centroid_speed,
            self.cursor[0],
            self.cursor[1],
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "Lenia Studio Continuous Automaton Canvas",
            Some(&value_summary),
        );
        paint_canvas_focus_ring(ui, &response);
        lock_canvas_arrow_keys(ui, &response);
        self.focused_canvas = response.has_focus().then_some(response.id);
    }

    fn handle_pointer_brushing(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if response.clicked() || response.dragged() {
            response.request_focus();
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        if !response.rect.contains(pos) {
            return;
        }
        let primary_active = response.clicked_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Primary);
        let secondary_active = response.clicked_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Secondary);
        if !primary_active && !secondary_active {
            return;
        }

        let rel_x = ((pos.x - response.rect.min.x) / response.rect.width()).clamp(0.0, 0.999_9);
        let rel_y = ((pos.y - response.rect.min.y) / response.rect.height()).clamp(0.0, 0.999_9);
        let gx = (rel_x * (self.engine.width() as f32)) as usize;
        let gy = (rel_y * (self.engine.height() as f32)) as usize;
        self.cursor = [gx, gy];

        let shift_held = ui.input(|i| i.modifiers.shift);
        let delta = if secondary_active || shift_held || self.brush_mode == LeniaBrushMode::Slice {
            -0.55
        } else {
            0.38
        };
        self.engine.paint_disc(gx, gy, self.brush_radius, delta);
        self.record_telemetry_sample();
        self.refresh_pixels_in_place();
    }

    fn handle_keyboard_stamping(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if !response.has_focus() {
            return;
        }
        let w = self.engine.width();
        let h = self.engine.height();
        let step = if ui.input(|i| i.modifiers.shift) { 4 } else { 1 };
        let mut moved = false;

        ui.input(|i| {
            if i.key_pressed(egui::Key::ArrowLeft) {
                self.cursor[0] = (self.cursor[0] + w - (step % w)) % w;
                moved = true;
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                self.cursor[0] = (self.cursor[0] + step) % w;
                moved = true;
            }
            if i.key_pressed(egui::Key::ArrowUp) {
                self.cursor[1] = (self.cursor[1] + h - (step % h)) % h;
                moved = true;
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                self.cursor[1] = (self.cursor[1] + step) % h;
                moved = true;
            }
        });
        self.apply_keyboard_actions(ui, moved);
    }

    fn apply_keyboard_actions(&mut self, ui: &egui::Ui, moved: bool) {
        let (stamp, deposit, slice) = ui.input(|i| {
            let stamp = i.key_pressed(egui::Key::Space) || i.key_pressed(egui::Key::Enter);
            let deposit = i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals);
            let slice = i.key_pressed(egui::Key::Minus)
                || i.key_pressed(egui::Key::Backspace)
                || i.key_pressed(egui::Key::Delete);
            (stamp, deposit, slice)
        });
        if stamp {
            stamp_species_at(
                &mut self.engine,
                self.species.to_domain(),
                self.cursor[0],
                self.cursor[1],
                self.perturbation,
                &mut self.rng,
            );
            self.record_telemetry_sample();
            self.refresh_pixels_in_place();
        } else if deposit || slice {
            let delta = if slice { -0.60 } else { 0.45 };
            self.engine
                .paint_disc(self.cursor[0], self.cursor[1], self.brush_radius, delta);
            self.record_telemetry_sample();
            self.refresh_pixels_in_place();
        } else if moved {
            ui.ctx().request_repaint();
        }
    }

    fn paint_cursor_reticle(
        &self,
        ui: &egui::Ui,
        response: &egui::Response,
        painter: &egui::Painter,
    ) {
        if !response.has_focus() && !response.hovered() {
            return;
        }
        let w = self.engine.width().max(1) as f32;
        let h = self.engine.height().max(1) as f32;
        let cx = response.rect.min.x + ((self.cursor[0] as f32 + 0.5) / w) * response.rect.width();
        let cy = response.rect.min.y + ((self.cursor[1] as f32 + 0.5) / h) * response.rect.height();
        let rad_px = ((self.brush_radius as f32) / w) * response.rect.width();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let ring_color = if self.brush_mode == LeniaBrushMode::Slice {
            palette.warning
        } else {
            palette.accent
        };
        painter.circle_stroke(
            egui::pos2(cx, cy),
            rad_px.max(4.0),
            egui::Stroke::new(1.5_f32, ring_color),
        );
    }
}
