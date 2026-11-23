#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility,
};
use crate::app::{SemanticPalette, ThemeMode};
use eframe::egui;
use math_explorer::math_kernel::colormap::{
    ensure_dark_canvas_contrast, heatmap_color, heatmap_color_high_contrast,
};
use math_explorer::nalgebra::DMatrix;

/// Accessible keyboard-focusable heatmap for merged
/// [`math_explorer::applied::lorahub::LoraStateDict`] weight matrices.
#[derive(Default)]
pub struct MergedTensorHeatmap {
    texture: Option<egui::TextureHandle>,
    pixels: Vec<egui::Color32>,
    dirty: bool,
    last_theme: Option<ThemeMode>,
    cursor: (usize, usize),
}

impl MergedTensorHeatmap {
    /// Creates an uninitialized dirty tensor heatmap.
    pub fn new() -> Self {
        Self {
            texture: None,
            pixels: Vec::new(),
            dirty: true,
            last_theme: None,
            cursor: (0, 0),
        }
    }

    /// Marks the GPU texture dirty so the next frame re-uploads colors.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn map_cell_color(norm: f64, theme: ThemeMode) -> egui::Color32 {
        let clamped = norm.clamp(0.0, 1.0);
        let adjusted = match theme {
            ThemeMode::Dark | ThemeMode::HighContrast => heatmap_color_high_contrast(clamped),
            ThemeMode::Light => ensure_dark_canvas_contrast(heatmap_color(clamped)),
        };
        egui::Color32::from_rgb(adjusted[0], adjusted[1], adjusted[2])
    }

    fn refresh_texture(&mut self, ctx: &egui::Context, matrix: &DMatrix<f64>) {
        if matrix.nrows() == 0 || matrix.ncols() == 0 {
            return;
        }
        let theme = ThemeMode::from_visuals(&ctx.style().visuals);
        if !self.dirty && self.texture.is_some() && self.last_theme == Some(theme) {
            return;
        }
        let max_abs = matrix
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .map(f64::abs)
            .fold(1e-6_f64, f64::max);

        self.pixels.clear();
        self.pixels.reserve(matrix.nrows() * matrix.ncols());
        for r in 0..matrix.nrows() {
            for c in 0..matrix.ncols() {
                let norm = 0.5 + 0.5 * (matrix[(r, c)] / max_abs).clamp(-1.0, 1.0);
                self.pixels.push(Self::map_cell_color(norm, theme));
            }
        }
        let img = egui::ColorImage {
            size: [matrix.ncols(), matrix.nrows()],
            pixels: self.pixels.clone(),
            ..Default::default()
        };
        if let Some(tex) = &mut self.texture {
            tex.set(img, egui::TextureOptions::NEAREST);
        } else {
            self.texture = Some(ctx.load_texture(
                "lorahub_merged_state_dict",
                img,
                egui::TextureOptions::NEAREST,
            ));
        }
        self.dirty = false;
        self.last_theme = Some(theme);
    }

    /// Renders the merged tensor heatmap with keyboard cell navigation and ARIA readout.
    pub fn show(&mut self, ui: &mut egui::Ui, matrix: &DMatrix<f64>) {
        ui.label("Merged LoRA Projection Tensor");
        self.refresh_texture(ui.ctx(), matrix);
        let side = ui.available_width().clamp(132.0, 204.0);
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
            egui::Stroke::new(1.2_f32, palette.graph_edge),
            egui::StrokeKind::Middle,
        );

        let nrows = matrix.nrows().max(1);
        let ncols = matrix.ncols().max(1);
        self.cursor.0 = self.cursor.0.min(nrows - 1);
        self.cursor.1 = self.cursor.1.min(ncols - 1);
        self.handle_cursor_and_accessibility(ui, rect, &response, &palette, matrix);
    }

    fn handle_cursor_and_accessibility(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        response: &egui::Response,
        palette: &SemanticPalette,
        matrix: &DMatrix<f64>,
    ) {
        let (nrows, ncols) = (matrix.nrows().max(1), matrix.ncols().max(1));
        let mut announce = false;
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                if rect.contains(pos) {
                    let c = (((pos.x - rect.min.x) / rect.width()) * ncols as f32)
                        .floor()
                        .clamp(0.0, (ncols - 1) as f32) as usize;
                    let r = (((pos.y - rect.min.y) / rect.height()) * nrows as f32)
                        .floor()
                        .clamp(0.0, (nrows - 1) as f32) as usize;
                    self.cursor = (r, c);
                    response.request_focus();
                    announce = true;
                }
            }
        }
        if response.has_focus() {
            lock_canvas_arrow_keys(ui, response);
            if self.consume_arrows(ui, nrows, ncols) || response.gained_focus() {
                announce = true;
            }
            paint_canvas_focus_ring(ui, response);
            let (cw, ch) = (rect.width() / ncols as f32, rect.height() / nrows as f32);
            let cell = egui::Rect::from_min_size(
                egui::pos2(
                    rect.min.x + self.cursor.1 as f32 * cw,
                    rect.min.y + self.cursor.0 as f32 * ch,
                ),
                egui::vec2(cw, ch),
            );
            ui.painter().rect_stroke(
                cell,
                1.0,
                egui::Stroke::new(2.2_f32, palette.accent),
                egui::StrokeKind::Inside,
            );
        }
        let val = if matrix.nrows() > 0 && matrix.ncols() > 0 {
            matrix[self.cursor]
        } else {
            0.0
        };
        let desc = format!(
            "Row {}, Column {}, Merged Delta {:.4}",
            self.cursor.0, self.cursor.1, val
        );
        register_canvas_accessibility(
            ui,
            response,
            egui::accesskit::Role::GraphicsDocument,
            "Merged LoRA State Dict Heatmap",
            Some(&desc),
        );
        if announce {
            announce_live_status(ui.ctx(), format!("Merged LoRA Tensor — {desc}"));
        }
        ui.small(desc);
    }

    fn consume_arrows(&mut self, ui: &egui::Ui, nrows: usize, ncols: usize) -> bool {
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
            self.cursor.0 = (self.cursor.0 + 1).min(nrows.saturating_sub(1));
        }
        if left {
            self.cursor.1 = self.cursor.1.saturating_sub(1);
        }
        if right {
            self.cursor.1 = (self.cursor.1 + 1).min(ncols.saturating_sub(1));
        }
        true
    }
}
