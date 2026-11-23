//! Interactive Ulam Prime Spiral widget with Euler quadratic ray ($4n^2 + bn + c$)
//! and spiral sweep polyrhythmic sonification playheads.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod playhead;

use std::hash::{Hash, Hasher};

use eframe::egui::{self, Color32, ColorImage, TextureHandle, TextureOptions};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::number_theory::primes::{is_prime, primes_up_to};

use crate::framework::{
    scoped_panel_id, ExportCapabilities, InteractiveTool, PaneSidePanel, StarterPresetMetadata,
    ToolMetadata,
};
use crate::web_io::PlatformInstant;
pub use playhead::{
    UlamPlayheadMode, DEFAULT_RAY_B, DEFAULT_RAY_C, DEFAULT_TEMPO_BPM, MAX_RAY_B, MAX_RAY_C,
    MAX_TEMPO_BPM, MAX_ULAM_CURSOR_STEP, MIN_RAY_B, MIN_RAY_C, MIN_TEMPO_BPM,
};

const fn default_ray_b() -> i64 {
    DEFAULT_RAY_B
}

const fn default_ray_c() -> i64 {
    DEFAULT_RAY_C
}

const fn default_tempo_bpm() -> f64 {
    DEFAULT_TEMPO_BPM
}

/// Interactive Ulam spiral explorer with quadratic ray and spiral sweep Web Audio playheads.
pub struct PrimeSpiralWidget {
    pub(crate) grid_size: usize,
    pub(crate) playhead_mode: UlamPlayheadMode,
    pub(crate) ray_b: i64,
    pub(crate) ray_c: i64,
    pub(crate) cursor_step: usize,
    pub(crate) tempo_bpm: f64,
    pub(crate) is_playing: bool,
    pub(crate) pulse_gate: bool,
    pub(super) beat_accum_sec: f64,
    pub(super) last_instant: Option<PlatformInstant>,
    pub(super) focused_canvas: Option<egui::Id>,
    last_theme: Option<crate::app::ThemeMode>,
    pub(crate) texture: Option<TextureHandle>,
    pub(crate) pixels_buffer: Vec<Color32>,
    pub(crate) is_prime_lookup: Vec<bool>,
    pub(crate) coord_lookup: Vec<(u16, u16)>,
    pub(crate) spiral_grid: Vec<u32>,
}

impl Default for PrimeSpiralWidget {
    fn default() -> Self {
        Self {
            grid_size: 200,
            playhead_mode: UlamPlayheadMode::QuadraticRay,
            ray_b: DEFAULT_RAY_B,
            ray_c: DEFAULT_RAY_C,
            cursor_step: 0,
            tempo_bpm: DEFAULT_TEMPO_BPM,
            is_playing: false,
            pulse_gate: true,
            beat_accum_sec: 0.0,
            last_instant: None,
            focused_canvas: None,
            last_theme: None,
            texture: None,
            pixels_buffer: Vec::new(),
            is_prime_lookup: Vec::new(),
            coord_lookup: Vec::new(),
            spiral_grid: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PrimeSpiralState {
    grid_size: usize,
    #[serde(default)]
    playhead_mode: UlamPlayheadMode,
    #[serde(default = "default_ray_b")]
    ray_b: i64,
    #[serde(default = "default_ray_c")]
    ray_c: i64,
    #[serde(default)]
    cursor_step: usize,
    #[serde(default = "default_tempo_bpm")]
    tempo_bpm: f64,
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

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (
            self.grid_size,
            self.playhead_mode,
            self.ray_b,
            self.ray_c,
            self.cursor_step,
            self.tempo_bpm.to_bits(),
        )
            .hash(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        (10..=math_commons::registry::MAX_GRID_SIZE).contains(&self.grid_size)
            && (MIN_RAY_B..=MAX_RAY_B).contains(&self.ray_b)
            && (MIN_RAY_C..=MAX_RAY_C).contains(&self.ray_c)
            && self.cursor_step <= MAX_ULAM_CURSOR_STEP
            && self.tempo_bpm.is_finite()
            && (MIN_TEMPO_BPM..=MAX_TEMPO_BPM).contains(&self.tempo_bpm)
    }

    fn clamp_parameters(&mut self) -> bool {
        self.grid_size = self
            .grid_size
            .clamp(10, math_commons::registry::MAX_GRID_SIZE);
        self.ray_b = self.ray_b.clamp(MIN_RAY_B, MAX_RAY_B);
        self.ray_c = self.ray_c.clamp(MIN_RAY_C, MAX_RAY_C);
        self.cursor_step = self.cursor_step.min(MAX_ULAM_CURSOR_STEP);
        self.tempo_bpm = if self.tempo_bpm.is_finite() {
            self.tempo_bpm.clamp(MIN_TEMPO_BPM, MAX_TEMPO_BPM)
        } else {
            DEFAULT_TEMPO_BPM
        };
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&PrimeSpiralState {
            grid_size: self.grid_size,
            playhead_mode: self.playhead_mode,
            ray_b: self.ray_b,
            ray_c: self.ray_c,
            cursor_step: self.cursor_step,
            tempo_bpm: self.tempo_bpm,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: PrimeSpiralState =
            serde_json::from_str(state).map_err(|e| format!("Invalid prime spiral state: {e}"))?;
        if !(10..=math_commons::registry::MAX_GRID_SIZE).contains(&parsed.grid_size) {
            return Err("grid_size out of allowed bounds".into());
        }
        if !(MIN_RAY_B..=MAX_RAY_B).contains(&parsed.ray_b)
            || !(MIN_RAY_C..=MAX_RAY_C).contains(&parsed.ray_c)
            || parsed.cursor_step > MAX_ULAM_CURSOR_STEP
            || !parsed.tempo_bpm.is_finite()
            || !(MIN_TEMPO_BPM..=MAX_TEMPO_BPM).contains(&parsed.tempo_bpm)
        {
            return Err("Ulam playhead parameters out of allowed bounds".into());
        }
        let grid_changed = self.grid_size != parsed.grid_size;
        self.grid_size = parsed.grid_size;
        self.playhead_mode = parsed.playhead_mode;
        self.ray_b = parsed.ray_b;
        self.ray_c = parsed.ray_c;
        self.cursor_step = parsed.cursor_step;
        self.tempo_bpm = parsed.tempo_bpm;
        self.is_playing = false;
        self.pulse_gate = true;
        if grid_changed {
            self.texture = None;
        }
        Ok(())
    }

    fn step_simulation(&mut self) {
        self.step_playhead_delta(1);
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::raster_only()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        let size = self.grid_size.max(2);
        if self.pixels_buffer.len() == size * size {
            return Some(ColorImage::new([size, size], self.pixels_buffer.clone()));
        }
        Some(Self::build_spiral_image(size))
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.advance_playhead_clock(ui.ctx());
        let theme = crate::app::ThemeMode::from_visuals(ui.visuals());
        if self.texture.is_none() || self.last_theme != Some(theme) {
            self.regenerate_texture(ui.ctx());
        }
        PaneSidePanel::left(scoped_panel_id(ui, "ulam_spiral_controls")).show_inside(ui, |ui| {
            self.render_playhead_controls(ui);
        });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.render_spiral_canvas(ui);
        });
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

impl PrimeSpiralWidget {
    #[inline]
    pub(crate) fn is_prime_fast(&self, n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if let Some(&cached) = self.is_prime_lookup.get(n as usize) {
            cached
        } else {
            is_prime(n)
        }
    }

    pub(crate) fn populate_ulam_pixels(
        size: usize,
        theme: crate::app::ThemeMode,
        lookup: &mut Vec<bool>,
        pixels: &mut Vec<Color32>,
    ) {
        let mut coords = Vec::new();
        let mut grid = Vec::new();
        Self::populate_ulam_buffers(size, theme, lookup, pixels, &mut coords, &mut grid);
    }

    pub(super) fn populate_ulam_buffers(
        size: usize,
        theme: crate::app::ThemeMode,
        lookup: &mut Vec<bool>,
        pixels: &mut Vec<Color32>,
        coords: &mut Vec<(u16, u16)>,
        grid: &mut Vec<u32>,
    ) {
        let num_pixels = size * size;
        let bg = theme.visuals().extreme_bg_color;
        let palette = theme.palette();
        pixels.clear();
        pixels.resize(num_pixels, bg);
        coords.clear();
        coords.resize(num_pixels + 1, (0, 0));
        grid.clear();
        grid.resize(num_pixels, 0);

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
        for (i, &prime_flag) in lookup.iter().enumerate().take(num_pixels + 1).skip(1) {
            if (-half <= x) && (x <= half) && (-half <= y) && (y <= half) {
                let (px, py) = ((half + x) as usize, (half + y) as usize);
                if px < size && py < size {
                    let flat = flatten_2d_index(px, py, size);
                    coords[i] = (px as u16, py as u16);
                    grid[flat] = i as u32;
                    if prime_flag {
                        pixels[flat] = palette.node_stroke;
                    }
                    if i == 1 {
                        pixels[flat] = palette.axis_x;
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

    pub(crate) fn regenerate_texture(&mut self, ctx: &egui::Context) {
        let size = self.grid_size;
        let theme = crate::app::ThemeMode::from_visuals(&ctx.style().visuals);
        self.last_theme = Some(theme);
        Self::populate_ulam_buffers(
            size,
            theme,
            &mut self.is_prime_lookup,
            &mut self.pixels_buffer,
            &mut self.coord_lookup,
            &mut self.spiral_grid,
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

inventory::submit! {
    ToolMetadata {
        id: "PrimeSpiralWidget",
        display_name: "Prime Spiral (Ulam Spiral)",
        domain: "number_theory",
        tags: &["ulam", "prime spiral", "euler quadratic", "polyrhythm", "sonification"],
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
    StarterPresetMetadata {
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

#[cfg(test)]
mod tests;

// [cite:mmwave_radiotherapy_setup]
