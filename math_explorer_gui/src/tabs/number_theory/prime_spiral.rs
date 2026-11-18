use eframe::egui::{self, Color32, ColorImage, TextureHandle, TextureOptions};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::number_theory::primes::primes_up_to;

pub struct PrimeSpiralWidget {
    grid_size: usize,
    texture: Option<TextureHandle>,
    pixels_buffer: Vec<Color32>,
    is_prime_lookup: Vec<bool>,
}

impl Default for PrimeSpiralWidget {
    fn default() -> Self {
        Self {
            grid_size: 200,
            texture: None,
            pixels_buffer: Vec::new(),
            is_prime_lookup: Vec::new(),
        }
    }
}

use crate::framework::InteractiveTool;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PrimeSpiralState {
    grid_size: usize,
}

impl InteractiveTool for PrimeSpiralWidget {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Prime Spiral (Ulam Spiral)"
    }
    fn domain(&self) -> &'static str {
        "number_theory"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&PrimeSpiralState {
            grid_size: self.grid_size,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: PrimeSpiralState =
            serde_json::from_str(state).map_err(|e| format!("Invalid prime spiral state: {e}"))?;
        if !(10..=math_commons::registry::MAX_GRID_SIZE).contains(&parsed.grid_size) {
            return Err("grid_size out of allowed bounds".into());
        }
        self.grid_size = parsed.grid_size;
        self.texture = None;
        Ok(())
    }

    fn show(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Prime Spiral (Ulam Spiral)");
            ui.label("A graphical depiction of the set of prime numbers...");
            ui.add_space(10.0);

            ui.vertical(|ui| {
                let changed = ui
                    .add(egui::Slider::new(&mut self.grid_size, 10..=math_commons::registry::MAX_GRID_SIZE).text("Grid Size (side length)"))
                    .changed();

                if changed || self.texture.is_none() {
                    self.regenerate_texture(ui.ctx());
                }

                if let Some(texture) = &self.texture {
                    // Correctly use ui.image with 1 argument (impl Into<ImageSource>)
                    ui.image(texture);
                }
            });
        });
    }
}

impl PrimeSpiralWidget {
    fn regenerate_texture(&mut self, ctx: &egui::Context) {
        let size = self.grid_size;
        let num_pixels = size * size;

        self.pixels_buffer.clear();
        self.pixels_buffer.resize(num_pixels, Color32::BLACK);

        // Ulam Spiral Logic
        let center_x = (size / 2) as i32;
        let center_y = (size / 2) as i32;

        let mut x = 0;
        let mut y = 0;
        let mut dx = 0;
        let mut dy = -1;
        // removed unused 't' and 'max_i'

        let limit = (size * size) as u64;
        let primes = primes_up_to(limit);

        self.is_prime_lookup.clear();
        self.is_prime_lookup.resize((limit + 1) as usize, false);
        for &p in &primes {
            if (p as usize) < self.is_prime_lookup.len() {
                self.is_prime_lookup[p as usize] = true;
            }
        }

        let half_size = (size / 2) as i32;

        for (i, &is_prime) in self.is_prime_lookup
            .iter()
            .enumerate()
            .take(num_pixels + 1)
            .skip(1)
        {
            if (-half_size <= x) && (x <= half_size) && (-half_size <= y) && (y <= half_size) {
                let px = (center_x + x) as usize;
                let py = (center_y + y) as usize;

                if px < size && py < size {
                    if is_prime {
                        self.pixels_buffer[flatten_2d_index(px, py, size)] = Color32::WHITE;
                    }
                    if i == 1 {
                        self.pixels_buffer[flatten_2d_index(px, py, size)] = Color32::RED;
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

        // Use ..Default::default() to fill potentially new fields like source_size
        let image = ColorImage {
            size: [size, size],
            pixels: self.pixels_buffer.clone(),
            ..Default::default()
        };

        if let Some(texture) = &mut self.texture {
            texture.set(image, TextureOptions::NEAREST);
        } else {
            self.texture = Some(ctx.load_texture(
                "ulam_spiral",
                image,
                TextureOptions::NEAREST, // Nearest neighbor for crisp pixels
            ));
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

        // Verify center pixel color logic (1 is RED, primes are WHITE, non-primes are BLACK)
        let center = 100;
        let center_idx = flatten_2d_index(center, center, 200);
        assert_eq!(widget.pixels_buffer[center_idx], Color32::RED);
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
    fn phonetic_description(&self) -> String { "Ulam prime spiral and quadratic prime polynomial theory".into() }
    fn theory_citation(&self) -> String { "A Visual Display of Prime Numbers (Ulam, 1964); A visual display of prime numbers (Stein, Ulam, & Wells, 1964)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
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
