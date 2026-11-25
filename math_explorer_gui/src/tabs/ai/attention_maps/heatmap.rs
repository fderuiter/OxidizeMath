#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::{
    announce_live_status, draw_legend_swatch, lock_canvas_arrow_keys, paint_canvas_border,
    paint_canvas_focus_ring, paint_two_tone_rect_frame, paint_two_tone_reticle,
    register_canvas_accessibility,
};
use crate::app::{SemanticPalette, ThemeMode as ThemeContrastMode};
use crate::framework::InputMode;
use eframe::egui;
use nalgebra::DMatrix;

/// Maps a normalized `[0.0, 1.0]` weight to a theme-aware heatmap cell color.
pub fn compute_attention_cell_color(
    normalized: f64,
    theme_mode: ThemeContrastMode,
) -> egui::Color32 {
    theme_mode.palette().attention_ramp_color(normalized)
}

/// Computes the matrix `(row, col)` cell coordinates under cursor position `pos`.
pub fn cell_at_pos(
    ncols: usize,
    nrows: usize,
    rect: egui::Rect,
    pos: egui::Pos2,
) -> Option<(usize, usize)> {
    if !rect.contains(pos) || ncols == 0 || nrows == 0 {
        return None;
    }
    let cell_w = rect.width() / ncols as f32;
    let cell_h = rect.height() / nrows as f32;
    let col = ((pos.x - rect.min.x) / cell_w)
        .floor()
        .clamp(0.0, (ncols - 1) as f32) as usize;
    let row = ((pos.y - rect.min.y) / cell_h)
        .floor()
        .clamp(0.0, (nrows - 1) as f32) as usize;
    Some((row, col))
}

fn cell_rect(rect: egui::Rect, nrows: usize, ncols: usize, row: usize, col: usize) -> egui::Rect {
    let cell_w = rect.width() / ncols.max(1) as f32;
    let cell_h = rect.height() / nrows.max(1) as f32;
    let cell_min = egui::pos2(
        rect.min.x + col as f32 * cell_w,
        rect.min.y + row as f32 * cell_h,
    );
    egui::Rect::from_min_size(cell_min, egui::vec2(cell_w, cell_h))
}

fn format_cell_readout(row: usize, col: usize, weight: f64) -> String {
    format!("Token: [Row {row}, Col {col}], Weight: {weight:.4}")
}

/// Cached GPU texture and keyboard-navigable cell cursor for a 2D attention matrix.
#[derive(Default)]
pub struct HeatmapTexture {
    texture: Option<egui::TextureHandle>,
    pub(super) pixels_buffer: Vec<egui::Color32>,
    dirty: bool,
    pub(super) last_theme: Option<ThemeContrastMode>,
    cursor: (usize, usize),
}

impl HeatmapTexture {
    /// Creates an empty dirty heatmap texture with cursor at `(0, 0)`.
    pub fn new() -> Self {
        Self {
            texture: None,
            pixels_buffer: Vec::new(),
            dirty: true,
            last_theme: None,
            cursor: (0, 0),
        }
    }

    /// Marks the cached texture dirty so the next frame regenerates pixels.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    #[cfg(test)]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    #[cfg(test)]
    pub fn texture(&self) -> Option<&egui::TextureHandle> {
        self.texture.as_ref()
    }

    #[cfg(test)]
    pub fn buffer_capacity(&self) -> usize {
        self.pixels_buffer.capacity()
    }

    #[cfg(test)]
    pub fn cursor(&self) -> (usize, usize) {
        self.cursor
    }

    /// Uploads matrix colors to the GPU texture when marked dirty or when theme changes.
    pub fn update_if_needed(
        &mut self,
        ctx: &egui::Context,
        matrix: &DMatrix<f64>,
        texture_name: &str,
    ) {
        if matrix.nrows() == 0 || matrix.ncols() == 0 {
            return;
        }

        let current_theme = ThemeContrastMode::from_visuals(&ctx.style().visuals);
        if !self.dirty && self.texture.is_some() && self.last_theme == Some(current_theme) {
            return;
        }

        let (min_val, max_val) = matrix
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), val| {
                (min.min(val), max.max(val))
            });
        let (min_val, max_val) = if min_val.is_finite() && max_val.is_finite() {
            (min_val, max_val)
        } else {
            (0.0, 1.0)
        };

        self.pixels_buffer.clear();
        self.pixels_buffer.reserve(matrix.nrows() * matrix.ncols());
        for r in 0..matrix.nrows() {
            for c in 0..matrix.ncols() {
                let val = matrix[(r, c)];
                let normalized = if (max_val - min_val).abs() > 1e-6 {
                    (val - min_val) / (max_val - min_val)
                } else {
                    0.5
                };
                self.pixels_buffer
                    .push(compute_attention_cell_color(normalized, current_theme));
            }
        }

        let image = egui::ColorImage {
            size: [matrix.ncols(), matrix.nrows()],
            pixels: self.pixels_buffer.clone(),
            ..Default::default()
        };
        let texture_opts = egui::TextureOptions::NEAREST;
        if let Some(texture) = &mut self.texture {
            texture.set(image, texture_opts);
        } else {
            self.texture = Some(ctx.load_texture(texture_name, image, texture_opts));
        }

        self.dirty = false;
        self.last_theme = Some(current_theme);
    }

    /// Renders the focusable heatmap, processes pointer/arrow-key cell selection,
    /// and announces `Token` and `Weight` updates via the accessibility live region.
    pub fn draw(
        &mut self,
        ui: &mut egui::Ui,
        name: &str,
        matrix: &DMatrix<f64>,
        input_mode: InputMode,
    ) {
        ui.label(name);
        self.update_if_needed(ui.ctx(), matrix, name);

        let cell_size = match input_mode {
            InputMode::Touch => 44.0_f32,
            InputMode::Mouse => 40.0_f32,
        };
        let desired_size = egui::vec2(
            matrix.ncols() as f32 * cell_size,
            matrix.nrows() as f32 * cell_size,
        );
        let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

        if let Some(texture) = &self.texture {
            ui.painter().image(
                texture.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }

        let current_theme = ThemeContrastMode::from_visuals(ui.visuals());
        let palette = current_theme.palette();
        paint_canvas_border(ui, rect);

        self.handle_cell_interaction(ui, &response, rect, name, matrix, &palette);
        Self::draw_hover_overlay(ui, response, rect, matrix, current_theme, &palette);
        draw_attention_ramp_legend(ui, &palette);
    }

    fn handle_cell_interaction(
        &mut self,
        ui: &mut egui::Ui,
        response: &egui::Response,
        rect: egui::Rect,
        name: &str,
        matrix: &DMatrix<f64>,
        palette: &SemanticPalette,
    ) {
        if matrix.nrows() == 0 || matrix.ncols() == 0 {
            register_canvas_accessibility(
                ui,
                response,
                egui::accesskit::Role::GraphicsDocument,
                name,
                None,
            );
            paint_canvas_focus_ring(ui, response);
            return;
        }

        self.cursor.0 = self.cursor.0.min(matrix.nrows() - 1);
        self.cursor.1 = self.cursor.1.min(matrix.ncols() - 1);
        let mut announce = false;

        if response.clicked() {
            if let Some(rc) = response
                .interact_pointer_pos()
                .and_then(|pos| cell_at_pos(matrix.ncols(), matrix.nrows(), rect, pos))
            {
                self.cursor = rc;
                response.request_focus();
                announce = true;
            }
        }

        if response.has_focus() {
            lock_canvas_arrow_keys(ui, response);
            if self.apply_arrow_navigation(ui, matrix.nrows(), matrix.ncols())
                || response.gained_focus()
            {
                announce = true;
            }
            paint_canvas_focus_ring(ui, response);
            let active_rect = cell_rect(
                rect,
                matrix.nrows(),
                matrix.ncols(),
                self.cursor.0,
                self.cursor.1,
            );
            paint_two_tone_rect_frame(ui.painter(), active_rect, 1.0_f32, palette.accent, 2.0_f32);
        }

        let (row, col) = self.cursor;
        let readout = format_cell_readout(row, col, matrix[(row, col)]);
        let description = format!(
            "{readout}; Color key: 3-stop luminance ramp from low to mid to high attention weight"
        );
        register_canvas_accessibility(
            ui,
            response,
            egui::accesskit::Role::GraphicsDocument,
            &format!("{name} Heatmap"),
            Some(&description),
        );
        if announce {
            announce_live_status(ui.ctx(), format!("{name} — {readout}"));
        }
    }

    fn apply_arrow_navigation(&mut self, ui: &egui::Ui, nrows: usize, ncols: usize) -> bool {
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

    fn draw_hover_overlay(
        ui: &mut egui::Ui,
        response: egui::Response,
        rect: egui::Rect,
        matrix: &DMatrix<f64>,
        current_theme: ThemeContrastMode,
        palette: &SemanticPalette,
    ) {
        if !response.hovered() {
            return;
        }
        let Some(pos) = response.hover_pos() else {
            return;
        };
        let Some((row, col)) = cell_at_pos(matrix.ncols(), matrix.nrows(), rect, pos) else {
            return;
        };
        let val = matrix[(row, col)];
        let target_rect = cell_rect(rect, matrix.nrows(), matrix.ncols(), row, col);

        let hover_primary = if current_theme == ThemeContrastMode::HighContrast {
            palette.accent
        } else {
            palette.node_stroke
        };
        paint_two_tone_reticle(ui.painter(), target_rect, 0.0_f32, hover_primary);

        response.on_hover_ui(|ui| {
            ui.label(format!("Token: [Row {}, Col {}]", row, col));
            ui.label(format!("Weight: {}", val));
        });
    }
}

fn draw_attention_ramp_legend(ui: &mut egui::Ui, palette: &SemanticPalette) {
    ui.horizontal_wrapped(|ui| {
        draw_legend_swatch(ui, palette.attention_ramp_color(0.0), None, "Low (0.0)");
        draw_legend_swatch(ui, palette.attention_ramp_color(0.5), None, "Mid (0.5)");
        draw_legend_swatch(ui, palette.attention_ramp_color(1.0), None, "High (1.0)");
    });
}
