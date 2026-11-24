//! Interactive 1D profile & 2D heightfield canvas with mouse/keyboard mound sculpting,
//! AccessKit `Role::GraphicsDocument`, and WCAG 2.2 AA colormaps for [`KpzInterfaceTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::hash::{Hash, Hasher};

use eframe::egui;
use eframe::egui::ColorImage;
use math_commons::math_kernel::colormap::{
    ensure_dark_canvas_contrast, heatmap_color, heatmap_color_high_contrast,
};
use scientific_metadata::theory::TheoryDescribable;

use super::kpz_tool::{KpzDimensionMode, KpzInterfaceTool};
use super::presets::KpzToolState;
use crate::accessibility::{
    lock_canvas_arrow_keys, paint_canvas_focus_ring, register_canvas_accessibility,
};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};

impl KpzInterfaceTool {
    pub(crate) fn refresh_2d_pixels(&mut self) {
        let w = self.solver_2d.width();
        let h = self.solver_2d.height();
        let total = w * h;
        if self.pixels.len() != total {
            self.pixels.resize(total, egui::Color32::BLACK);
        }
        let heights = self.solver_2d.heights();
        let (min_h, max_h) = finite_min_max(heights);
        let span = (max_h - min_h).max(1e-6);

        for (dst, &val) in self.pixels.iter_mut().zip(heights.iter()) {
            let norm = if val.is_finite() {
                ((val - min_h) / span).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let rgb = if self.high_contrast {
                heatmap_color_high_contrast(norm)
            } else {
                heatmap_color(norm)
            };
            *dst = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
        }
        self.texture_dirty = true;
    }

    fn sync_2d_texture(&mut self, ctx: &egui::Context) {
        if !self.texture_dirty && self.texture.is_some() {
            return;
        }
        let image = ColorImage {
            size: [self.solver_2d.width(), self.solver_2d.height()],
            pixels: self.pixels.clone(),
            ..Default::default()
        };
        let opts = egui::TextureOptions::NEAREST;
        if let Some(tex) = &mut self.texture {
            tex.set(image, opts);
        } else {
            self.texture = Some(ctx.load_texture("kpz_heightfield_2d", image, opts));
        }
        self.texture_dirty = false;
    }

    pub(crate) fn draw_interface_canvas(&mut self, ui: &mut egui::Ui, desired_size: egui::Vec2) {
        let size = egui::vec2(
            desired_size.x.clamp(180.0, 640.0),
            desired_size.y.clamp(180.0, 480.0),
        );
        let (response, painter) = ui.allocate_painter(size, egui::Sense::click_and_drag());

        match self.dimension {
            KpzDimensionMode::Dim1D => self.paint_1d_profile(ui, &response, &painter),
            KpzDimensionMode::Dim2D => self.paint_2d_heightfield(ui, &response, &painter),
        }

        self.handle_canvas_pointer(ui, &response);
        self.handle_canvas_keyboard(ui, &response);

        let (t, w_rms) = match self.dimension {
            KpzDimensionMode::Dim1D => (self.solver_1d.time(), self.solver_1d.interface_width()),
            KpzDimensionMode::Dim2D => (self.solver_2d.time(), self.solver_2d.interface_width()),
        };
        let summary = format!(
            "{} (lambda = {:.2}): t = {:.2}, interface width W = {:.4}, cursor {:?}",
            self.dimension.label(),
            self.lambda,
            t,
            w_rms,
            self.cursor
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "KPZ Stochastic Interface Canvas",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &response);
        lock_canvas_arrow_keys(ui, &response);
        self.focused_canvas = response.has_focus().then_some(response.id);
    }

    fn paint_1d_profile(&self, ui: &egui::Ui, response: &egui::Response, painter: &egui::Painter) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        painter.rect_filled(response.rect, 4.0_f32, palette.surface_fill);
        painter.rect_stroke(
            response.rect,
            4.0_f32,
            egui::Stroke::new(1.0_f32, palette.graph_edge),
            egui::StrokeKind::Inside,
        );

        let heights = self.solver_1d.heights();
        if heights.len() < 2 {
            return;
        }
        let mean_h = self.solver_1d.mean_height();
        let (min_h, max_h) = finite_min_max(heights);
        let half_span = ((max_h - min_h) * 0.6).max(1.0);
        let rect = response.rect.shrink(10.0);

        let mid_y = rect.center().y;
        painter.line_segment(
            [egui::pos2(rect.min.x, mid_y), egui::pos2(rect.max.x, mid_y)],
            egui::Stroke::new(1.0_f32, palette.graph_edge),
        );

        let n_minus_1 = (heights.len() - 1) as f32;
        let mut pts = Vec::with_capacity(heights.len());
        for (i, &h) in heights.iter().enumerate() {
            let x = rect.min.x + (i as f32 / n_minus_1) * rect.width();
            let rel = (((h - mean_h) / half_span) as f32).clamp(-1.0, 1.0);
            let y = mid_y - rel * (rect.height() * 0.45);
            pts.push(egui::pos2(x, y));
        }

        let base_rgb = [palette.accent.r(), palette.accent.g(), palette.accent.b()];
        let safe_rgb = ensure_dark_canvas_contrast(base_rgb);
        let stroke_col = egui::Color32::from_rgb(safe_rgb[0], safe_rgb[1], safe_rgb[2]);
        painter.add(egui::Shape::line(
            pts,
            egui::Stroke::new(2.0_f32, stroke_col),
        ));

        if response.has_focus() || response.hovered() {
            let cx = rect.min.x
                + (self.cursor[0].min(heights.len() - 1) as f32 / n_minus_1) * rect.width();
            painter.line_segment(
                [egui::pos2(cx, rect.min.y), egui::pos2(cx, rect.max.y)],
                egui::Stroke::new(1.2_f32, palette.warning),
            );
        }
    }

    fn paint_2d_heightfield(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        painter: &egui::Painter,
    ) {
        let ctx = ui.ctx().clone();
        self.sync_2d_texture(&ctx);
        if let Some(tex) = &self.texture {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            painter.image(tex.id(), response.rect, uv, egui::Color32::WHITE);
        }
        if response.has_focus() || response.hovered() {
            let w = self.solver_2d.width().max(1) as f32;
            let h = self.solver_2d.height().max(1) as f32;
            let cx =
                response.rect.min.x + ((self.cursor[0] as f32 + 0.5) / w) * response.rect.width();
            let cy =
                response.rect.min.y + ((self.cursor[1] as f32 + 0.5) / h) * response.rect.height();
            let rad = ((self.brush_radius as f32) / w) * response.rect.width();
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            painter.circle_stroke(
                egui::pos2(cx, cy),
                rad.max(4.0_f32),
                egui::Stroke::new(1.5_f32, palette.accent),
            );
        }
    }

    fn handle_canvas_pointer(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if response.clicked() || response.dragged() {
            response.request_focus();
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        if !response.rect.contains(pos) {
            return;
        }
        let primary = response.clicked_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Primary);
        let secondary = response.clicked_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Secondary);
        if !primary && !secondary {
            return;
        }

        let rx = ((pos.x - response.rect.min.x) / response.rect.width()).clamp(0.0, 0.999_9);
        let ry = ((pos.y - response.rect.min.y) / response.rect.height()).clamp(0.0, 0.999_9);
        let invert = secondary || ui.input(|i| i.modifiers.shift);
        let amp = if invert { -1.2 } else { 1.2 };

        match self.dimension {
            KpzDimensionMode::Dim1D => {
                let gx = (rx * (self.solver_1d.len() as f32)) as usize;
                self.cursor[0] = gx;
                self.solver_1d.deposit_mound(gx, self.brush_radius, amp);
            }
            KpzDimensionMode::Dim2D => {
                let gx = (rx * (self.solver_2d.width() as f32)) as usize;
                let gy = (ry * (self.solver_2d.height() as f32)) as usize;
                self.cursor = [gx, gy];
                self.solver_2d.deposit_mound(gx, gy, self.brush_radius, amp);
                self.refresh_2d_pixels();
            }
        }
    }

    fn handle_canvas_keyboard(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if !response.has_focus() {
            return;
        }
        let max_x = match self.dimension {
            KpzDimensionMode::Dim1D => self.solver_1d.len(),
            KpzDimensionMode::Dim2D => self.solver_2d.width(),
        };
        let max_y = self.solver_2d.height().max(1);
        let (dx_key, dy_key, stamp_pos, stamp_neg) = ui.input(|i| {
            let step = if i.modifiers.shift { 4isize } else { 1isize };
            let dx = if i.key_pressed(egui::Key::ArrowRight) {
                step
            } else if i.key_pressed(egui::Key::ArrowLeft) {
                -step
            } else {
                0
            };
            let dy = if i.key_pressed(egui::Key::ArrowDown) {
                step
            } else if i.key_pressed(egui::Key::ArrowUp) {
                -step
            } else {
                0
            };
            let pos_hit = i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Equals);
            let neg_hit = i.key_pressed(egui::Key::Minus) || i.key_pressed(egui::Key::Backspace);
            (dx, dy, pos_hit, neg_hit)
        });

        if dx_key != 0 {
            self.cursor[0] = (self.cursor[0] as isize + dx_key).rem_euclid(max_x as isize) as usize;
        }
        if dy_key != 0 {
            self.cursor[1] = (self.cursor[1] as isize + dy_key).rem_euclid(max_y as isize) as usize;
        }
        if stamp_pos || stamp_neg {
            let amp = if stamp_neg { -1.5 } else { 1.5 };
            match self.dimension {
                KpzDimensionMode::Dim1D => {
                    self.solver_1d
                        .deposit_mound(self.cursor[0], self.brush_radius, amp);
                }
                KpzDimensionMode::Dim2D => {
                    self.solver_2d.deposit_mound(
                        self.cursor[0],
                        self.cursor[1],
                        self.brush_radius,
                        amp,
                    );
                    self.refresh_2d_pixels();
                }
            }
        }
    }
}

fn finite_min_max(values: &[f64]) -> (f64, f64) {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    if !min_v.is_finite() || !max_v.is_finite() {
        (-1.0, 1.0)
    } else {
        (min_v, max_v)
    }
}

impl InteractiveTool for KpzInterfaceTool {
    fn name(&self) -> &'static str {
        "KPZ Interface Growth"
    }

    fn domain(&self) -> &'static str {
        "stochastic_pde"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        (self.dimension == KpzDimensionMode::Dim2D
            && self.pixels.len() == self.solver_2d.width() * self.solver_2d.height())
        .then(|| ColorImage {
            size: [self.solver_2d.width(), self.solver_2d.height()],
            pixels: self.pixels.clone(),
            ..Default::default()
        })
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_export_datasets()
    }

    fn step_simulation(&mut self) {
        self.step_once();
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_workspace(ui);
    }

    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    fn state_bounds(&self) -> ToolStateBounds {
        crate::url_state::URL_JSON_BOUNDS
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.dimension.hash(&mut h);
        self.nu.to_bits().hash(&mut h);
        self.lambda.to_bits().hash(&mut h);
        self.noise_sigma.to_bits().hash(&mut h);
        self.dx.to_bits().hash(&mut h);
        self.dt.to_bits().hash(&mut h);
        self.system_size.hash(&mut h);
        self.steps_per_frame.hash(&mut h);
        self.rng_seed.hash(&mut h);
        self.plot_view.hash(&mut h);
        self.brush_radius.hash(&mut h);
        self.playing.hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.snapshot_state()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: KpzToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid KPZ state JSON: {e}"))?;
        parsed.validate()?;
        self.dimension = parsed.dimension;
        self.nu = parsed.nu;
        self.lambda = parsed.lambda;
        self.noise_sigma = parsed.noise_sigma;
        self.dx = parsed.dx;
        self.dt = parsed.dt;
        self.system_size = parsed.system_size;
        self.steps_per_frame = parsed.steps_per_frame;
        self.rng_seed = parsed.rng_seed;
        self.plot_view = parsed.plot_view;
        self.brush_radius = parsed.brush_radius;
        self.playing = parsed.playing;
        self.rebuild_and_seed();
        Ok(())
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        let (t, w_rms) = match self.dimension {
            KpzDimensionMode::Dim1D => (self.solver_1d.time(), self.solver_1d.interface_width()),
            KpzDimensionMode::Dim2D => (self.solver_2d.time(), self.solver_2d.interface_width()),
        };
        let mut obs = vec![("t", t), ("interface_width", w_rms)];
        if let Some((beta, _)) = self.fitted_beta() {
            obs.push(("fitted_beta", beta));
        }
        obs
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "KpzInterfaceTool",
        display_name: "KPZ Interface Growth",
        domain: "stochastic_pde",
        tags: &[
            "kpz",
            "edwards-wilkinson",
            "spde",
            "interface-growth",
            "family-vicsek",
            "scaling",
            "roughness",
        ],
        build: || Box::new(KpzInterfaceTool::default()),
    }
}
