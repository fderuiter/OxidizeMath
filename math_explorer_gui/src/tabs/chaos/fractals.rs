use crate::accessibility::AccessibleTheoryHover;
use crate::framework::InteractiveTool;
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::chaos::fractals::{escape_time_julia, escape_time_mandelbrot};
use num_complex::Complex;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

#[derive(PartialEq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
enum FractalMode {
    Mandelbrot,
    Julia,
}

pub struct FractalViewer {
    center: Complex<f64>,
    zoom: f64,
    max_iter: u32,
    mode: FractalMode,
    julia_c: Complex<f64>,
    texture: Option<egui::TextureHandle>,
    dirty: bool,
}

impl Default for FractalViewer {
    fn default() -> Self {
        Self {
            center: Complex::new(-0.75, 0.0),
            zoom: 1.0,
            max_iter: 100,
            mode: FractalMode::Mandelbrot,
            julia_c: Complex::new(-0.4, 0.6),
            texture: None,
            dirty: true,
        }
    }
}

impl TheoryDescribable for FractalViewer {
    fn theory_description(&self) -> String {
        let name = match self.mode {
            FractalMode::Mandelbrot => "Mandelbrot set".to_string(),
            FractalMode::Julia => format!(
                "Julia set with c=({:.2}, {:.2})",
                self.julia_c.re, self.julia_c.im
            ),
        };
        format!(
            "Fractal viewer: {}, center: ({:.2}, {:.2}), zoom: {:.2}",
            name, self.center.re, self.center.im, self.zoom
        )
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        "Mandelbrot, B. B. (1982). The Fractal Geometry of Nature. W. H. Freeman; Julia, G. (1918). Mémoire sur l'itération des fonctions rationnelles. Journal de Mathématiques Pures et Appliquées, 8, 47-245.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("default".to_string(), "Fractal viewer".to_string());
        map
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct FractalState {
    center_re: f64,
    center_im: f64,
    zoom: f64,
    max_iter: u32,
    mode: FractalMode,
    julia_re: f64,
    julia_im: f64,
}

impl InteractiveTool for FractalViewer {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Fractal Viewer"
    }

    fn domain(&self) -> &'static str {
        "chaos"
    }

    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::raster_only()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        let (width, height) = (480, 320);
        let zoom = if self.zoom.is_finite() && self.zoom > 0.0 {
            self.zoom
        } else {
            1.0
        };
        let scale_y = 3.0 / zoom;
        let scale_x = scale_y * (width as f64 / height as f64);
        Some(self.generate_image(width, height, scale_x, scale_y))
    }

    fn step_simulation(&mut self) {
        self.zoom = (self.zoom * 1.06).min(10_000.0);
        self.dirty = true;
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&FractalState {
            center_re: self.center.re,
            center_im: self.center.im,
            zoom: self.zoom,
            max_iter: self.max_iter,
            mode: self.mode,
            julia_re: self.julia_c.re,
            julia_im: self.julia_c.im,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(s) = serde_json::from_str::<FractalState>(state) {
            self.center = Complex::new(s.center_re, s.center_im);
            self.zoom = s.zoom;
            self.max_iter = s.max_iter;
            self.mode = s.mode;
            self.julia_c = Complex::new(s.julia_re, s.julia_im);
            self.dirty = true;
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("fractal_settings").show(ctx, |ui| {
            ui.heading("Settings");
            ui.separator();

            ui.horizontal(|ui| {
                if ui
                    .radio_value(&mut self.mode, FractalMode::Mandelbrot, "Mandelbrot")
                    .clicked()
                {
                    self.dirty = true;
                    // Reset to standard Mandelbrot view
                    self.center = Complex::new(-0.75, 0.0);
                    self.zoom = 1.0;
                }
                if ui
                    .radio_value(&mut self.mode, FractalMode::Julia, "Julia")
                    .clicked()
                {
                    self.dirty = true;
                    // Reset to standard Julia view
                    self.center = Complex::new(0.0, 0.0);
                    self.zoom = 1.0;
                }
            });

            if self.mode == FractalMode::Julia {
                ui.separator();
                ui.label("Julia Constant (c):");
                let mut changed = false;
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.julia_c.re)
                            .speed(0.005)
                            .prefix("Re: "),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.julia_c.im)
                            .speed(0.005)
                            .prefix("Im: "),
                    )
                    .changed();
                if changed {
                    self.dirty = true;
                }
            }

            ui.separator();
            ui.label("View Controls:");

            // Zoom
            if ui
                .add(
                    egui::Slider::new(&mut self.zoom, 0.1..=10000.0)
                        .logarithmic(true)
                        .text("Zoom"),
                )
                .changed()
            {
                self.dirty = true;
            }

            // Center
            ui.label("Center:");
            let speed = 0.1 / self.zoom;
            let mut center_changed = false;
            center_changed |= ui
                .add(
                    egui::DragValue::new(&mut self.center.re)
                        .speed(speed)
                        .prefix("Re: "),
                )
                .changed();
            center_changed |= ui
                .add(
                    egui::DragValue::new(&mut self.center.im)
                        .speed(speed)
                        .prefix("Im: "),
                )
                .changed();

            if center_changed {
                self.dirty = true;
            }

            // Max Iterations
            if ui
                .add(
                    egui::Slider::new(
                        &mut self.max_iter,
                        10..=math_commons::registry::MAX_ITERATIONS as u32,
                    )
                    .logarithmic(true)
                    .text("Max Iter"),
                )
                .changed()
            {
                self.dirty = true;
            }

            ui.separator();
            if ui.button("↻ Reset View").clicked() {
                match self.mode {
                    FractalMode::Mandelbrot => {
                        self.center = Complex::new(-0.75, 0.0);
                    }
                    FractalMode::Julia => {
                        self.center = Complex::new(0.0, 0.0);
                    }
                }
                self.zoom = 1.0;
                self.dirty = true;
            }

            ui.label("Navigation:");
            ui.label("- Drag to Pan (Single Touch/Mouse)");
            ui.label("- Scroll/Pinch to Zoom");
            ui.label("- Two-Finger Pan (Touch)");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let available_size = ui.available_size();
            let (width, height) = (available_size.x as usize, available_size.y as usize);

            if width == 0 || height == 0 {
                return;
            }

            // Check if texture needs update due to resize or dirty state
            let texture_needs_update = if let Some(texture) = &self.texture {
                texture.size() != [width, height]
            } else {
                true
            };

            if self.dirty || texture_needs_update {
                // Determine aspect ratio to map pixels to complex plane correctly
                let aspect = width as f64 / height as f64;
                let scale_y = 3.0 / self.zoom; // Height in complex plane
                let scale_x = scale_y * aspect; // Width in complex plane

                let image = self.generate_image(width, height, scale_x, scale_y);

                self.texture =
                    Some(ctx.load_texture("fractal_texture", image, egui::TextureOptions::LINEAR));
                self.dirty = false;
            }

            if let Some(texture) = &self.texture {
                // Display the image
                let response = ui.image(texture).accessible_theory_hover(self);

                let multi_touch = ui.input(|i| i.multi_touch());

                // Handle multi-touch pinch-to-zoom and panning
                if let Some(touch) = multi_touch {
                    if touch.zoom_delta != 1.0 {
                        // Dampen zoom velocity to prevent erratic zooming on high-refresh screens
                        let dampened_zoom = 1.0 + (touch.zoom_delta - 1.0) * 0.5;
                        self.zoom *= dampened_zoom as f64;
                        self.dirty = true;
                    }

                    if touch.translation_delta != egui::Vec2::ZERO {
                        let aspect = width as f64 / height as f64;
                        let scale_y = 3.0 / self.zoom;
                        let scale_x = scale_y * aspect;

                        let dx = -touch.translation_delta.x as f64 / width as f64 * scale_x;
                        let dy = touch.translation_delta.y as f64 / height as f64 * scale_y;

                        self.center.re += dx;
                        self.center.im += dy;
                        self.dirty = true;
                    }
                } else if response.dragged() {
                    // Handle single-pointer drag to pan
                    let delta = response.drag_delta();
                    // Map pixel delta to complex delta
                    let aspect = width as f64 / height as f64;
                    let scale_y = 3.0 / self.zoom;
                    let scale_x = scale_y * aspect;

                    // Invert x because dragging right should move camera left (to see left content)
                    let dx = -delta.x as f64 / width as f64 * scale_x;
                    // Invert y because dragging down (+y screen) should move camera up (to see top content)
                    // Screen Y is down positive. Complex Im is up positive.
                    // Dragging down (+delta.y) means we want to shift the view so that what was above (higher Im) comes into view?
                    // No, dragging down usually means we want to see what is ABOVE.
                    // Wait, standard map drag: Drag down -> Map moves down. We see what was above.
                    // So center of view moves UP (higher Im).
                    let dy = delta.y as f64 / height as f64 * scale_y;

                    self.center.re += dx;
                    self.center.im += dy;

                    self.dirty = true;
                }

                // Handle scroll to zoom
                if response.hovered() {
                    let scroll_delta = ui.input(|i| i.raw_scroll_delta.y);
                    if scroll_delta != 0.0 {
                        // Zoom in (factor > 1) if scroll up (>0)
                        let zoom_factor = if scroll_delta > 0.0 { 1.1 } else { 0.9 };
                        self.zoom *= zoom_factor;
                        self.dirty = true;
                    }
                }
            }
        });
    }
}

impl FractalViewer {
    fn generate_image(
        &self,
        width: usize,
        height: usize,
        scale_x: f64,
        scale_y: f64,
    ) -> egui::ColorImage {
        let mut pixels = vec![0u8; width * height * 4];

        // Parallel iteration would be great here, but for now single threaded.
        // We iterate pixels and map to complex plane.

        let x_start = self.center.re - scale_x / 2.0;
        let y_start = self.center.im + scale_y / 2.0; // Top-left corner (max imaginary)

        for y in 0..height {
            // Map y to Im. y=0 -> Im=y_start. y=height -> Im=y_start - scale_y
            let im = y_start - (y as f64 / height as f64) * scale_y;

            for x in 0..width {
                // Map x to Re. x=0 -> Re=x_start. x=width -> Re=x_start + scale_x
                let re = x_start + (x as f64 / width as f64) * scale_x;

                let c = Complex::new(re, im);
                let iterations = match self.mode {
                    FractalMode::Mandelbrot => escape_time_mandelbrot(c, self.max_iter),
                    FractalMode::Julia => escape_time_julia(c, self.julia_c, self.max_iter),
                };

                let color = self.map_iter_to_color(iterations);

                let idx = flatten_2d_index(x, y, width) * 4;
                pixels[idx] = color[0];
                pixels[idx + 1] = color[1];
                pixels[idx + 2] = color[2];
                pixels[idx + 3] = color[3];
            }
        }

        egui::ColorImage::from_rgba_unmultiplied([width, height], &pixels)
    }

    fn map_iter_to_color(&self, iter: u32) -> [u8; 4] {
        if iter == self.max_iter {
            [0, 0, 0, 255] // Black inside
        } else {
            let rgb = math_commons::math_kernel::colormap::cyclic_cosine_palette(iter as f64);
            [rgb[0], rgb[1], rgb[2], 255]
        }
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "FractalViewer",
        display_name: "Fractal Viewer",
        domain: "chaos",
        tags: &[],
        build: || Box::new(FractalViewer::default()),
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Mandelbrot Explorer",
    "domain": "chaos",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "max_iterations": 150,
    "zoom": 1.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "fractal_viewer_starter",
        title: "Mandelbrot Explorer",
        description: "z_{n+1} = z_n^2 + c: Complex quadratic polynomial iteration and boundary fractal escape times.",
        category: "Fractals",
        icon: "❄️",
        domain: "chaos",
        tool_name: Some("FractalViewer"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
