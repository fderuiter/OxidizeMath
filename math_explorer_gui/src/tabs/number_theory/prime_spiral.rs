use crate::accessibility::AccessibleTheoryHover;
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui::{self, Color32, ColorImage, TextureHandle, TextureOptions};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::number_theory::primes::primes_up_to;

#[path = "prime_spiral/sonification.rs"]
mod sonification;

pub struct PrimeSpiralWidget {
    grid_size: usize,
    last_theme: Option<crate::app::ThemeMode>,
    texture: Option<TextureHandle>,
    pixels_buffer: Vec<Color32>,
    is_prime_lookup: Vec<bool>,
    seq_playing: bool,
    seq_cursor: usize,
    audio_settings: crate::audio::AcousticSettings,
}

impl Default for PrimeSpiralWidget {
    fn default() -> Self {
        Self {
            grid_size: 200,
            last_theme: None,
            texture: None,
            pixels_buffer: Vec::new(),
            is_prime_lookup: Vec::new(),
            seq_playing: false,
            seq_cursor: 2,
            audio_settings: crate::audio::AcousticSettings::default(),
        }
    }
}

use crate::framework::InteractiveTool;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PrimeSpiralState {
    grid_size: usize,
    #[serde(default, skip_serializing_if = "crate::audio::AcousticSettings::is_default")]
    audio: crate::audio::AcousticSettings,
}

impl InteractiveTool for PrimeSpiralWidget {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Prime Spiral (Ulam Spiral)"
    }
    fn domain(&self) -> &'static str {
        "number_theory"
    }

    fn state_revision(&self) -> u64 {
        self.compute_state_revision()
    }

    fn step_simulation(&mut self) {
        self.advance_spiral_cursor();
    }

    fn sonification_frame(&self) -> Option<crate::audio::SonificationFrame> {
        Some(self.compute_sonification_frame())
    }

    fn acoustic_settings(&self) -> crate::audio::AcousticSettings {
        self.audio_settings
    }

    fn set_acoustic_settings(&mut self, settings: crate::audio::AcousticSettings) {
        self.audio_settings = settings.sanitized();
    }

    fn sample_sonification_frames(&mut self, frame_count: usize) -> Vec<crate::audio::SonificationFrame> {
        self.render_wav_frames(frame_count)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&PrimeSpiralState {
            grid_size: self.grid_size,
            audio: self.audio_settings,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: PrimeSpiralState =
            serde_json::from_str(state).map_err(|e| format!("Invalid prime spiral state: {e}"))?;
        parsed.audio.validate()?;
        if !(10..=math_commons::registry::MAX_GRID_SIZE).contains(&parsed.grid_size) {
            return Err("grid_size out of allowed bounds".into());
        }
        self.grid_size = parsed.grid_size;
        self.audio_settings = parsed.audio;
        self.texture = None;
        Ok(())
    }

    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::raster_only().with_audio()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        let size = self.grid_size.max(2);
        if self.pixels_buffer.len() == size * size {
            return Some(ColorImage::new([size, size], self.pixels_buffer.clone()));
        }
        Some(Self::build_spiral_image(size))
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Prime Spiral (Ulam Spiral)");
            ui.label("A graphical depiction of the set of prime numbers...");
            ui.add_space(10.0);

            ui.vertical(|ui| {
                self.draw_transport_bar(ui);

                let changed = ui
                    .add(
                        egui::Slider::new(
                            &mut self.grid_size,
                            10..=math_commons::registry::MAX_GRID_SIZE,
                        )
                        .text("Grid Size (side length)"),
                    )
                    .changed();

                let theme = crate::app::ThemeMode::from_visuals(ui.visuals());
                if changed || self.texture.is_none() || self.last_theme != Some(theme) {
                    self.regenerate_texture(ui.ctx());
                }

                if let Some(texture) = &self.texture {
                    let alt_label = format!(
                        "Ulam Prime Spiral visualization ({0}x{0} grid)",
                        self.grid_size
                    );
                    let response = ui
                        .add(
                            egui::Image::new(texture)
                                .alt_text(alt_label)
                                .sense(egui::Sense::focusable_noninteractive()),
                        )
                        .accessible_theory_hover(self);
                    ui.painter().rect_stroke(
                        response.rect,
                        0.0,
                        ui.visuals().widgets.noninteractive.bg_stroke,
                        egui::StrokeKind::Inside,
                    );
                    crate::accessibility::paint_canvas_focus_ring(ui, &response);
                }
            });
        });
    }
}

impl PrimeSpiralWidget {
    fn draw_transport_bar(&mut self, ui: &mut egui::Ui) {
        if let Some(action) = PlaybackTransport::in_thread(
            "prime_spiral_transport",
            "Prime Spiral (Ulam Spiral)",
            self.seq_playing,
        )
        .with_sonification(&mut self.audio_settings)
        .show(ui)
        {
            match action {
                PlaybackAction::Play => self.seq_playing = true,
                PlaybackAction::Pause => self.seq_playing = false,
                PlaybackAction::Step => {
                    self.seq_playing = false;
                    self.advance_spiral_cursor();
                }
                PlaybackAction::Reset => {
                    self.seq_playing = false;
                    self.seq_cursor = 2;
                }
            }
        }
        if self.seq_playing {
            self.advance_spiral_cursor();
            ui.ctx().request_repaint();
        }
    }

    pub(crate) fn populate_ulam_pixels(
        size: usize,
        theme: crate::app::ThemeMode,
        lookup: &mut Vec<bool>,
        pixels: &mut Vec<Color32>,
    ) {
        let num_pixels = size * size;
        let bg = theme.visuals().extreme_bg_color;
        let palette = theme.palette();
        pixels.clear();
        pixels.resize(num_pixels, bg);

        let limit = num_pixels as u64;
        lookup.clear();
        lookup.resize((limit + 1) as usize, false);
        for p in primes_up_to(limit) {
            if (p as usize) < lookup.len() {
                lookup[p as usize] = true;
            }
        }

        let half = (size / 2) as i32;
        let (mut x, mut y, mut dx, mut dy) = (0i32, 0i32, 0i32, -1i32);
        for (i, &is_prime) in lookup.iter().enumerate().take(num_pixels + 1).skip(1) {
            if (-half <= x) && (x <= half) && (-half <= y) && (y <= half) {
                let (px, py) = ((half + x) as usize, (half + y) as usize);
                if px < size && py < size {
                    if is_prime {
                        pixels[flatten_2d_index(px, py, size)] = palette.node_stroke;
                    }
                    if i == 1 {
                        pixels[flatten_2d_index(px, py, size)] = palette.axis_x;
                    }
                }
            }
            if x == y || (x < 0 && x == -y) || (x > 0 && x == 1 - y) {
                let temp = dx;
                dx = -dy;
                dy = temp;
            }
            x += dx;
            y += dy;
        }
    }

    fn build_spiral_image(size: usize) -> ColorImage {
        let mut lookup = Vec::new();
        let mut pixels = Vec::new();
        Self::populate_ulam_pixels(size, crate::app::ThemeMode::Dark, &mut lookup, &mut pixels);
        ColorImage::new([size, size], pixels)
    }

    fn regenerate_texture(&mut self, ctx: &egui::Context) {
        let size = self.grid_size;
        let theme = crate::app::ThemeMode::from_visuals(&ctx.style().visuals);
        self.last_theme = Some(theme);
        Self::populate_ulam_pixels(
            size,
            theme,
            &mut self.is_prime_lookup,
            &mut self.pixels_buffer,
        );

        let image = ColorImage {
            size: [size, size],
            pixels: self.pixels_buffer.clone(),
            ..Default::default()
        };

        if let Some(texture) = &mut self.texture {
            texture.set(image, TextureOptions::NEAREST);
        } else {
            self.texture = Some(ctx.load_texture("ulam_spiral", image, TextureOptions::NEAREST));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prime_spiral_widget_buffer_reuse() {
        let ctx = egui::Context::default();
        let mut widget = PrimeSpiralWidget::default();

        assert_eq!(widget.grid_size, 200);
        assert!(widget.texture.is_none());
        assert!(widget.pixels_buffer.is_empty());
        assert!(widget.is_prime_lookup.is_empty());

        // Initial texture regeneration
        widget.regenerate_texture(&ctx);
        assert!(widget.texture.is_some());
        assert_eq!(widget.pixels_buffer.len(), 200 * 200);

        let initial_pixels_cap = widget.pixels_buffer.capacity();
        let initial_prime_cap = widget.is_prime_lookup.capacity();

        assert!(initial_pixels_cap >= 200 * 200);
        assert!(initial_prime_cap > 200 * 200);

        // Increase grid size to 300
        widget.grid_size = 300;
        widget.regenerate_texture(&ctx);
        assert_eq!(widget.pixels_buffer.len(), 300 * 300);

        let larger_pixels_cap = widget.pixels_buffer.capacity();
        let larger_prime_cap = widget.is_prime_lookup.capacity();

        assert!(larger_pixels_cap >= 300 * 300);

        // Decrease grid size back to 200; capacity should NOT decrease (buffer reused)
        widget.grid_size = 200;
        widget.regenerate_texture(&ctx);
        assert_eq!(widget.pixels_buffer.len(), 200 * 200);
        assert!(widget.pixels_buffer.capacity() >= larger_pixels_cap);
        assert!(widget.is_prime_lookup.capacity() >= larger_prime_cap);

        // Verify center pixel color logic (1 uses SemanticPalette axis_x)
        let center = 100;
        let center_idx = flatten_2d_index(center, center, 200);
        assert_eq!(
            widget.pixels_buffer[center_idx],
            crate::app::ThemeMode::Dark.palette().axis_x
        );
    }
}

// [cite:mmwave_radiotherapy_setup]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "PrimeSpiralWidget",
        display_name: "Prime Spiral (Ulam Spiral)",
        domain: "number_theory",
        tags: &[],
        build: || Box::new(PrimeSpiralWidget::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for PrimeSpiralWidget {
    fn theory_description(&self) -> String {
        format!(
            "Ulam prime spiral mapping natural numbers along a rectangular 2D spiral of size {}x{}, visualizing diagonal clustering of prime numbers along quadratic polynomials $f(n) = 4n^2 + bn + c$.",
            self.grid_size, self.grid_size
        )
    }
    fn phonetic_description(&self) -> String {
        "Ulam prime spiral and quadratic prime polynomial theory".into()
    }
    fn theory_citation(&self) -> String {
        "A Visual Display of Prime Numbers (Ulam, 1964); A visual display of prime numbers (Stein, Ulam, & Wells, 1964)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Ulam Prime Spiral",
    "domain": "number_theory",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "grid_size": 101
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "prime_spiral_starter",
        title: "Ulam Prime Spiral",
        description: "2D square spiral lattice revealing quadratic polynomial diagonal prime clustering alignments.",
        category: "Prime Distributions",
        icon: "🌀",
        domain: "number_theory",
        tool_name: Some("PrimeSpiralWidget"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
