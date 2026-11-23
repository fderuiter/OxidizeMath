use crate::framework::{InputMode, InteractiveTool};
use eframe::egui;
use math_explorer::ai::transformer::{Decoder, Encoder, Transformer};
use nalgebra::DMatrix;
use scientific_metadata::theory::TheoryDescribable;

pub use crate::app::ThemeMode as ThemeContrastMode;
#[path = "attention_maps/heatmap.rs"]
mod heatmap;
#[cfg(test)]
pub use heatmap::cell_at_pos;
pub use heatmap::{compute_attention_cell_color, HeatmapTexture};

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
    fn domain(&self) -> &'static str {
        "ai"
    }
    fn save_state(&self) -> Option<String> {
        self.save_preset_state()
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.try_load_preset_state(state)
    }

    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::raster_only()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        preset::render_attention_raster(&self.attention_weights, &self.output_matrix)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let input_mode = parent_ui.ctx().data(|d| {
            d.get_temp(egui::Id::new("INPUT_MODE"))
                .unwrap_or(InputMode::Mouse)
        });

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
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
        id: "AttentionMapsTool",
        display_name: "Attention Maps",
        domain: "ai",
        tags: &["transformer", "self-attention", "softmax", "heatmap", "qkv"],
        build: || Box::new(AttentionMapsTool::default()),
    }
}

#[path = "attention_maps/preset.rs"]
mod preset;
#[cfg(test)]
#[path = "attention_maps/tests.rs"]
mod tests;
