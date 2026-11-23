#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility,
};
use crate::app::SemanticPalette;
use eframe::egui;
use math_explorer::math_kernel::colormap::heatmap_color_high_contrast;
use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::nalgebra::{DMatrix, Vector3};

#[inline]
fn unit_rgb_to_color32(r: f64, g: f64, b: f64) -> egui::Color32 {
    egui::Color32::from(egui::Rgba::from_rgb(
        r.clamp(0.0, 1.0) as f32,
        g.clamp(0.0, 1.0) as f32,
        b.clamp(0.0, 1.0) as f32,
    ))
}

/// Converts a rendered `(height, width)` RGB [`DMatrix<Vector3<f64>>`] into row-major pixels.
pub(super) fn rgb_matrix_to_colors(image: &DMatrix<Vector3<f64>>, out: &mut Vec<egui::Color32>) {
    let h = image.nrows();
    let w = image.ncols();
    out.clear();
    out.reserve(w * h);
    for y in 0..h {
        for x in 0..w {
            let c = image[(y, x)];
            out.push(unit_rgb_to_color32(c.x, c.y, c.z));
        }
    }
}

/// Converts a `(width * height, 3)` `compute_sds_gradient` residual tensor into heatmap pixels.
pub(super) fn latent_grad_to_colors(
    sds_grad: &DMatrix<f64>,
    width: usize,
    height: usize,
    out: &mut Vec<egui::Color32>,
) {
    out.clear();
    out.reserve(width * height);
    let mut max_mag = 1e-6_f64;
    for r in 0..sds_grad.nrows() {
        let mag =
            (sds_grad[(r, 0)].powi(2) + sds_grad[(r, 1)].powi(2) + sds_grad[(r, 2)].powi(2)).sqrt();
        if mag.is_finite() && mag > max_mag {
            max_mag = mag;
        }
    }
    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(y, x, height);
            let mag = if idx < sds_grad.nrows() {
                (sds_grad[(idx, 0)].powi(2)
                    + sds_grad[(idx, 1)].powi(2)
                    + sds_grad[(idx, 2)].powi(2))
                .sqrt()
            } else {
                0.0
            };
            let norm = (mag / max_mag).clamp(0.0, 1.0);
            let rgb = heatmap_color_high_contrast(norm);
            out.push(egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
        }
    }
}

/// Builds a side-by-side `[Rendered NeRF | SDS Gradient Residual]` image for deferred figure export.
pub(super) fn build_composite_export_image(
    grid: usize,
    render_pixels: &[egui::Color32],
    grad_pixels: &[egui::Color32],
) -> egui::ColorImage {
    let gap = 2_usize;
    let total_w = grid * 2 + gap;
    let total_h = grid;
    let sep = unit_rgb_to_color32(0.25, 0.28, 0.34);
    let mut pixels = vec![sep; total_w * total_h];
    for y in 0..grid {
        for x in 0..grid {
            let src = flatten_2d_index(x, y, grid);
            let dst_left = flatten_2d_index(x, y, total_w);
            let dst_right = flatten_2d_index(grid + gap + x, y, total_w);
            if let Some(&px) = render_pixels.get(src) {
                pixels[dst_left] = px;
            }
            if let Some(&gpx) = grad_pixels.get(src) {
                pixels[dst_right] = gpx;
            }
        }
    }
    egui::ColorImage {
        size: [total_w, total_h],
        source_size: egui::vec2(total_w as f32, total_h as f32),
        pixels,
    }
}

/// Focusable 2D diagnostic raster preview tile with keyboard inspection and AccessKit semantics.
pub struct SdsDiagnosticTile {
    tex_name: &'static str,
    texture: Option<egui::TextureHandle>,
    dirty: bool,
    cursor: (usize, usize),
}

impl SdsDiagnosticTile {
    /// Creates a named diagnostic tile.
    pub fn new(tex_name: &'static str) -> Self {
        Self {
            tex_name,
            texture: None,
            dirty: true,
            cursor: (0, 0),
        }
    }

    /// Marks the texture dirty so the next frame uploads fresh pixel data.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn upload_if_dirty(&mut self, ctx: &egui::Context, grid: usize, pixels: &[egui::Color32]) {
        if self.dirty || self.texture.is_none() {
            let img = egui::ColorImage {
                size: [grid, grid],
                pixels: pixels[..grid * grid].to_vec(),
                ..Default::default()
            };
            if let Some(tex) = &mut self.texture {
                tex.set(img, egui::TextureOptions::NEAREST);
            } else {
                self.texture =
                    Some(ctx.load_texture(self.tex_name, img, egui::TextureOptions::NEAREST));
            }
            self.dirty = false;
        }
    }

    /// Renders the focusable preview tile with keyboard cursor navigation and ARIA announcements.
    pub fn show(&mut self, ui: &mut egui::Ui, label: &str, grid: usize, pixels: &[egui::Color32]) {
        ui.small(label);
        if grid == 0 || pixels.len() < grid * grid {
            return;
        }
        self.upload_if_dirty(ui.ctx(), grid, pixels);

        let side = 96.0_f32;
        let (rect, response) = ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::click());
        if let Some(tex) = &self.texture {
            ui.painter().image(
                tex.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.painter().rect_stroke(
            rect,
            2.0,
            egui::Stroke::new(1.1_f32, palette.graph_edge),
            egui::StrokeKind::Middle,
        );
        self.handle_tile_interaction(ui, rect, &response, label, grid, pixels);
    }

    fn handle_tile_interaction(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        response: &egui::Response,
        label: &str,
        grid: usize,
        pixels: &[egui::Color32],
    ) {
        self.cursor.0 = self.cursor.0.min(grid - 1);
        self.cursor.1 = self.cursor.1.min(grid - 1);
        let mut announce = false;
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                if rect.contains(pos) {
                    let cx = (((pos.x - rect.min.x) / rect.width()) * grid as f32)
                        .floor()
                        .clamp(0.0, (grid - 1) as f32) as usize;
                    let cy = (((pos.y - rect.min.y) / rect.height()) * grid as f32)
                        .floor()
                        .clamp(0.0, (grid - 1) as f32) as usize;
                    self.cursor = (cy, cx);
                    response.request_focus();
                    announce = true;
                }
            }
        }
        if response.has_focus() {
            lock_canvas_arrow_keys(ui, response);
            if self.step_cursor(ui, grid) || response.gained_focus() {
                announce = true;
            }
            paint_canvas_focus_ring(ui, response);
        }

        let c = pixels[flatten_2d_index(self.cursor.1, self.cursor.0, grid)];
        let summary = format!(
            "Pixel ({}, {}) RGB ({}, {}, {})",
            self.cursor.1,
            self.cursor.0,
            c.r(),
            c.g(),
            c.b()
        );
        register_canvas_accessibility(
            ui,
            response,
            egui::accesskit::Role::GraphicsDocument,
            label,
            Some(&summary),
        );
        if announce {
            announce_live_status(ui.ctx(), format!("{label} — {summary}"));
        }
    }

    fn step_cursor(&mut self, ui: &egui::Ui, grid: usize) -> bool {
        let (up, down, left, right) = ui.ctx().input_mut(|i| {
            (
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft),
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight),
            )
        });
        if !(up || down || left || right) {
            return false;
        }
        ui.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
        if up {
            self.cursor.0 = self.cursor.0.saturating_sub(1);
        }
        if down {
            self.cursor.0 = (self.cursor.0 + 1).min(grid - 1);
        }
        if left {
            self.cursor.1 = self.cursor.1.saturating_sub(1);
        }
        if right {
            self.cursor.1 = (self.cursor.1 + 1).min(grid - 1);
        }
        true
    }
}
