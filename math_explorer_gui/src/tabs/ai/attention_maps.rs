use crate::framework::{InputMode, InteractiveTool};
use eframe::egui;
use math_explorer::ai::transformer::{Decoder, Encoder, Transformer};
use nalgebra::DMatrix;
use scientific_metadata::theory::TheoryDescribable;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeContrastMode {
    Dark,
    Light,
    HighContrast,
}

impl ThemeContrastMode {
    pub fn from_visuals(visuals: &egui::Visuals) -> Self {
        if visuals.dark_mode
            && visuals.extreme_bg_color == egui::Color32::BLACK
            && visuals.selection.bg_fill == egui::Color32::YELLOW
        {
            ThemeContrastMode::HighContrast
        } else if visuals.dark_mode {
            ThemeContrastMode::Dark
        } else {
            ThemeContrastMode::Light
        }
    }
}

fn color_high_contrast(normalized: f64) -> egui::Color32 {
    egui::Color32::from_rgb(
        (normalized * 255.0) as u8,
        (normalized * 255.0) as u8,
        ((1.0 - normalized) * 40.0) as u8,
    )
}

fn color_dark(normalized: f64) -> egui::Color32 {
    egui::Color32::from_rgb(
        (normalized * 255.0) as u8,
        (normalized * 200.0) as u8,
        ((1.0 - normalized) * 200.0 + 55.0) as u8,
    )
}

fn color_light(normalized: f64) -> egui::Color32 {
    egui::Color32::from_rgb(
        ((1.0 - normalized * 0.9) * 240.0) as u8,
        ((1.0 - normalized * 0.8) * 245.0) as u8,
        ((1.0 - normalized * 0.5) * 255.0) as u8,
    )
}

pub fn compute_attention_cell_color(
    normalized: f64,
    theme_mode: ThemeContrastMode,
) -> egui::Color32 {
    let normalized = normalized.clamp(0.0, 1.0);
    match theme_mode {
        ThemeContrastMode::HighContrast => color_high_contrast(normalized),
        ThemeContrastMode::Dark => color_dark(normalized),
        ThemeContrastMode::Light => color_light(normalized),
    }
}

/// Computes the matrix (row, col) cell coordinates under cursor position `pos`.
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

#[derive(Default)]
pub struct HeatmapTexture {
    texture: Option<egui::TextureHandle>,
    pixels_buffer: Vec<egui::Color32>,
    dirty: bool,
    last_theme: Option<ThemeContrastMode>,
}

impl HeatmapTexture {
    pub fn new() -> Self {
        Self {
            texture: None,
            pixels_buffer: Vec::new(),
            dirty: true,
            last_theme: None,
        }
    }

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

        let num_pixels = matrix.nrows() * matrix.ncols();
        self.pixels_buffer.clear();
        self.pixels_buffer.reserve(num_pixels);

        for r in 0..matrix.nrows() {
            for c in 0..matrix.ncols() {
                let val = matrix[(r, c)];
                let normalized = if (max_val - min_val).abs() > 1e-6 {
                    (val - min_val) / (max_val - min_val)
                } else {
                    0.5
                };
                let color = compute_attention_cell_color(normalized, current_theme);
                self.pixels_buffer.push(color);
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

        let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::hover());

        if let Some(texture) = &self.texture {
            ui.painter().image(
                texture.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }

        let current_theme = ThemeContrastMode::from_visuals(&ui.ctx().style().visuals);
        let border_stroke = match current_theme {
            ThemeContrastMode::HighContrast => egui::Stroke::new(2.0_f32, egui::Color32::WHITE),
            ThemeContrastMode::Dark => egui::Stroke::new(1.0_f32, egui::Color32::from_gray(70)),
            ThemeContrastMode::Light => egui::Stroke::new(1.0_f32, egui::Color32::from_gray(180)),
        };
        ui.painter()
            .rect_stroke(rect, 0.0_f32, border_stroke, egui::StrokeKind::Middle);

        Self::draw_hover_overlay(ui, response, rect, matrix, current_theme);
    }

    fn draw_hover_overlay(
        ui: &mut egui::Ui,
        response: egui::Response,
        rect: egui::Rect,
        matrix: &DMatrix<f64>,
        current_theme: ThemeContrastMode,
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

        let cell_w = rect.width() / matrix.ncols() as f32;
        let cell_h = rect.height() / matrix.nrows() as f32;
        let cell_min = egui::pos2(
            rect.min.x + col as f32 * cell_w,
            rect.min.y + row as f32 * cell_h,
        );
        let cell_max = egui::pos2(cell_min.x + cell_w, cell_min.y + cell_h);
        let cell_rect = egui::Rect::from_min_max(cell_min, cell_max);

        let hover_stroke = match current_theme {
            ThemeContrastMode::HighContrast => egui::Stroke::new(2.0_f32, egui::Color32::YELLOW),
            ThemeContrastMode::Dark => egui::Stroke::new(1.5_f32, egui::Color32::WHITE),
            ThemeContrastMode::Light => egui::Stroke::new(1.5_f32, egui::Color32::BLACK),
        };
        ui.painter()
            .rect_stroke(cell_rect, 0.0_f32, hover_stroke, egui::StrokeKind::Middle);

        response.on_hover_ui(|ui| {
            ui.label(format!("Token: [Row {}, Col {}]", row, col));
            ui.label(format!("Weight: {}", val));
        });
    }
}

pub struct AttentionMapsTool {
    q_matrix: DMatrix<f64>,
    k_matrix: DMatrix<f64>,
    v_matrix: DMatrix<f64>,
    output_matrix: DMatrix<f64>,
    attention_weights: DMatrix<f64>,
    weights_texture: HeatmapTexture,
    output_texture: HeatmapTexture,
}

impl Default for AttentionMapsTool {
    fn default() -> Self {
        let seq_len = 4;
        let d_k = 4;

        let mut tool = Self {
            q_matrix: DMatrix::from_element(seq_len, d_k, 0.5),
            k_matrix: DMatrix::from_element(seq_len, d_k, 0.5),
            v_matrix: DMatrix::from_element(seq_len, d_k, 0.5),
            output_matrix: DMatrix::zeros(seq_len, d_k),
            attention_weights: DMatrix::zeros(seq_len, seq_len),
            weights_texture: HeatmapTexture::new(),
            output_texture: HeatmapTexture::new(),
        };
        tool.recalculate();
        tool
    }
}

impl AttentionMapsTool {
    #[cfg(test)]
    pub fn weights_texture(&self) -> &HeatmapTexture {
        &self.weights_texture
    }

    #[cfg(test)]
    pub fn output_texture(&self) -> &HeatmapTexture {
        &self.output_texture
    }

    #[cfg(test)]
    pub fn attention_weights(&self) -> &DMatrix<f64> {
        &self.attention_weights
    }

    #[cfg(test)]
    pub fn output_matrix(&self) -> &DMatrix<f64> {
        &self.output_matrix
    }

    pub fn recalculate(&mut self) {
        let (output, weights) =
            math_explorer::ai::transformer::attention::scaled_dot_product_attention(
                &self.q_matrix,
                &self.k_matrix,
                &self.v_matrix,
                None,
            );
        self.output_matrix = output;
        self.attention_weights = weights;
        self.weights_texture.mark_dirty();
        self.output_texture.mark_dirty();
    }

    fn draw_matrix_input(ui: &mut egui::Ui, name: &str, matrix: &mut DMatrix<f64>) -> bool {
        let mut changed = false;
        ui.label(name);
        egui::Grid::new(format!("{}_input_grid", name))
            .num_columns(matrix.ncols())
            .show(ui, |ui| {
                for r in 0..matrix.nrows() {
                    for c in 0..matrix.ncols() {
                        let mut val = matrix[(r, c)];
                        if ui
                            .add(
                                egui::DragValue::new(&mut val)
                                    .speed(0.1)
                                    .range(-10.0..=10.0),
                            )
                            .changed()
                        {
                            matrix[(r, c)] = val;
                            changed = true;
                        }
                    }
                    ui.end_row();
                }
            });
        changed
    }
}

impl InteractiveTool for AttentionMapsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Attention Maps"
    }

    fn show(&mut self, ctx: &egui::Context) {
        let input_mode = ctx.data(|d| {
            d.get_temp(egui::Id::new("INPUT_MODE"))
                .unwrap_or(InputMode::Mouse)
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Self-Attention Mechanism");

            ui.label("Explore how Queries (Q), Keys (K), and Values (V) interact.");

            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        let mut changed = false;
                        ui.group(|ui| {
                            changed |=
                                Self::draw_matrix_input(ui, "Queries (Q)", &mut self.q_matrix);
                        });
                        ui.group(|ui| {
                            changed |= Self::draw_matrix_input(ui, "Keys (K)", &mut self.k_matrix);
                        });
                        ui.group(|ui| {
                            changed |=
                                Self::draw_matrix_input(ui, "Values (V)", &mut self.v_matrix);
                        });

                        if changed {
                            self.recalculate();
                        }
                    });

                    ui.separator();

                    ui.vertical(|ui| {
                        ui.group(|ui| {
                            self.weights_texture.draw(
                                ui,
                                "Attention Weights (softmax(Q * K^T / sqrt(d_k)))",
                                &self.attention_weights,
                                input_mode,
                            );
                        });

                        ui.add_space(20.0);

                        ui.group(|ui| {
                            self.output_texture.draw(
                                ui,
                                "Output (Weights * V)",
                                &self.output_matrix,
                                input_mode,
                            );
                        });
                    });
                });
            });
        });
    }
}

impl TheoryDescribable for AttentionMapsTool {
    fn theory_description(&self) -> String {
        Transformer {
            encoder: Encoder::new(1, 4, 1, 4),
            decoder: Decoder::new(1, 4, 1, 4),
        }
        .theory_description()
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }
    fn theory_citation(&self) -> String {
        Transformer {
            encoder: Encoder::new(1, 4, 1, 4),
            decoder: Decoder::new(1, 4, 1, 4),
        }
        .theory_citation()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        Transformer {
            encoder: Encoder::new(1, 4, 1, 4),
            decoder: Decoder::new(1, 4, 1, 4),
        }
        .available_descriptions()
    }
}

// [cite:attention_is_all_you_need_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        name: "AttentionMapsTool",
        domain: "ai",
        tags: &[],
        build: || Box::new(AttentionMapsTool::default()),
    }
}

#[cfg(test)]
#[path = "attention_maps/tests.rs"]
mod tests;

