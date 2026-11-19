use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::ai::transformer::tokenization::{
    CharTokenizer, PseudoEmbedding, Tokenizer, WordTokenizer,
};

/// A simple struct to represent a token and its pseudo-embedding
struct TokenData {
    text: String,
    embedding: Vec<f64>,
}

pub struct TokenizationTool {
    input_text: String,
    d_model: usize,
    tokens: Vec<TokenData>,
    use_subword: bool,
}

impl Default for TokenizationTool {
    fn default() -> Self {
        let initial_text = "Attention is all you need.".to_string();
        let mut tool = Self {
            input_text: initial_text,
            d_model: 8,
            tokens: Vec::new(),
            use_subword: false,
        };
        tool.recalculate_tokens();
        tool
    }
}

impl TokenizationTool {
    fn recalculate_tokens(&mut self) {
        let raw_tokens = if self.use_subword {
            CharTokenizer.tokenize(&self.input_text)
        } else {
            WordTokenizer.tokenize(&self.input_text)
        };

        let embedder = PseudoEmbedding {
            d_model: self.d_model,
        };

        self.tokens = raw_tokens
            .into_iter()
            .map(|text| {
                let embedding = embedder.embed_token(&text);
                TokenData { text, embedding }
            })
            .collect();
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TokenizationState {
    input_text: String,
    d_model: usize,
    use_subword: bool,
}

impl InteractiveTool for TokenizationTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Tokenization & Embeddings"
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::none()
    }
    fn domain(&self) -> &'static str {
        "ai"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&TokenizationState {
            input_text: self.input_text.clone(),
            d_model: self.d_model,
            use_subword: self.use_subword,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: TokenizationState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid TokenizationTool state: {e}"))?;
        if s.d_model == 0 {
            return Err("d_model must be positive in TokenizationTool".to_string());
        }
        self.input_text = s.input_text;
        self.d_model = s.d_model;
        self.use_subword = s.use_subword;
        self.recalculate_tokens();
        Ok(())
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Tokenization & Input Embeddings");
            ui.label(
                "Explore how text is broken down into discrete tokens and mapped \
                 to high-dimensional continuous vectors (embeddings) before entering a Transformer.",
            );
            ui.separator();

            let mut text_changed = false;
            let mut params_changed = false;

            // --- Controls Panel ---
            ui.horizontal(|ui| {
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.label(egui::RichText::new("Input Text").strong());
                        if ui
                            .add_sized(
                                [400.0, 60.0],
                                egui::TextEdit::multiline(&mut self.input_text)
                                    .hint_text("Type a sentence here..."),
                            )
                            .changed()
                        {
                            text_changed = true;
                        }
                    });
                });

                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.label(egui::RichText::new("Parameters").strong());

                        ui.horizontal(|ui| {
                            if ui.add(egui::Slider::new(&mut self.d_model, 4..=32).text("Embedding Dimension (d_model)"))
                                .changed()
                            {
                                params_changed = true;
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.label("Tokenizer Strategy:");
                            if ui.radio_value(&mut self.use_subword, false, "Word/Punctuation").changed() {
                                params_changed = true;
                            }
                            if ui.radio_value(&mut self.use_subword, true, "Character-level").changed() {
                                params_changed = true;
                            }
                        });
                    });
                });
            });

            if text_changed || params_changed {
                self.recalculate_tokens();
            }

            ui.add_space(10.0);
            ui.separator();

            // --- Tokens Display Panel ---
            ui.heading(format!("Tokens (Count: {})", self.tokens.len()));
            egui::ScrollArea::vertical().id_salt("tokens_scroll").max_height(100.0).show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
                    for token in &self.tokens {
                        // Render each token as a distinct visual block
                        let rect = egui::Frame::NONE
                            .fill(ui.visuals().widgets.inactive.bg_fill)
                            .stroke(ui.visuals().widgets.inactive.bg_stroke)
                            .corner_radius(4.0)
                            .inner_margin(6.0)
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new(&token.text).monospace().color(ui.visuals().strong_text_color()));
                            });

                        rect.response.accessible_hover_text(format!("Token ID: Hash of '{}'", token.text));
                    }
                });
            });

            ui.add_space(10.0);
            ui.separator();

            // --- Embeddings Matrix Panel ---
            ui.heading("Embedding Matrix");
            ui.label("Visual representation of the pseudo-embeddings mapped to each token. Red is negative, Green is positive.");

            egui::ScrollArea::both().id_salt("embeddings_scroll").show(ui, |ui| {
                egui::Grid::new("embeddings_grid")
                    .striped(true)
                    .num_columns(self.d_model + 1)
                    .spacing([8.0, 8.0])
                    .show(ui, |ui| {
                        // Header row
                        ui.label(egui::RichText::new("Token").strong());
                        for i in 0..self.d_model {
                            ui.label(egui::RichText::new(format!("Dim {}", i)).strong().small());
                        }
                        ui.end_row();

                        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());

                        // Data rows
                        for token in &self.tokens {
                            ui.label(egui::RichText::new(&token.text).monospace());

                            for &val in &token.embedding {
                                let color = embedding_cell_color(val, &palette);
                                let text_color = crate::app::contrasting_text_color(color);

                                egui::Frame::NONE
                                    .fill(color)
                                    .corner_radius(2.0)
                                    .inner_margin(4.0)
                                    .show(ui, |ui| {
                                        ui.label(
                                            egui::RichText::new(format!("{:.2}", val))
                                                .color(text_color)
                                                .small(),
                                        );
                                    });
                            }
                            ui.end_row();
                        }
                    });
            });
        });
    }
}

fn lerp_channel(a: u8, b: u8, t: f64) -> u8 {
    (f64::from(a) + (f64::from(b) - f64::from(a)) * t.clamp(0.0, 1.0)).round() as u8
}

pub(crate) fn embedding_cell_color(
    val: f64,
    palette: &crate::app::SemanticPalette,
) -> egui::Color32 {
    let base = palette.surface_fill;
    let target = if val < 0.0 {
        palette.error
    } else {
        palette.success
    };
    let t = val.abs().clamp(0.0, 1.0);
    egui::Color32::from_rgb(
        lerp_channel(base.r(), target.r(), t),
        lerp_channel(base.g(), target.g(), t),
        lerp_channel(base.b(), target.b(), t),
    )
}

// [cite:algorithmic_information_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "TokenizationTool",
        display_name: "Tokenization & Embeddings",
        domain: "ai",
        tags: &[],
        build: || Box::new(TokenizationTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for TokenizationTool {
    fn theory_description(&self) -> String {
        let mode = if self.use_subword {
            "Character/subword"
        } else {
            "Word"
        };
        format!(
            "{} tokenization mapping sequence tokens $t_i$ to discrete vocabulary indices and continuous vector representations $E(t_i) \\in \\mathbb{{R}}^{{d_{{\\text{{model}}}}}}$ with dimension $d_{{\\text{{model}}}} = {}$.",
            mode, self.d_model
        )
    }
    fn phonetic_description(&self) -> String {
        "Tokenization and embedding space theory".into()
    }
    fn theory_citation(&self) -> String {
        "Attention Is All You Need (Vaswani et al., 2017)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Byte-Pair Tokenizer",
    "domain": "ai",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "input_text": "Neural attention mechanisms",
    "d_model": 12,
    "use_subword": true
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "tokenization_starter",
        title: "Byte-Pair Tokenizer",
        description: "Subword lexical segmentation and token vocabulary mapping for natural language representations.",
        category: "NLP",
        icon: "🔤",
        domain: "ai",
        tool_name: Some("TokenizationTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
