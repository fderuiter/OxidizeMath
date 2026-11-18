use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui::{Color32, ColorImage, TextureOptions};
use egui_plot::{Plot, PlotImage, PlotPoint};
use math_commons::math_kernel::colormap::heatmap_color;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::medical::dose::kernel::{DoseKernel, ExponentialKernel};

pub struct DoseCalculationTool {
    amplitude: f64,
    beta: f64,
    width: usize,
    height: usize,
    texture: Option<egui::TextureHandle>,
}

impl Default for DoseCalculationTool {
    fn default() -> Self {
        Self {
            amplitude: 1.0,
            beta: 1.0,
            width: 200,
            height: 200,
            texture: None,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct DoseState {
    amplitude: f64,
    beta: f64,
    width: usize,
    height: usize,
}

impl InteractiveTool for DoseCalculationTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Dose Calculation"
    }

    fn domain(&self) -> &'static str {
        "medical"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::raster_only()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        Some(self.compute_dose_image())
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&DoseState {
            amplitude: self.amplitude,
            beta: self.beta,
            width: self.width,
            height: self.height,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(loaded) = serde_json::from_str::<DoseState>(state) {
            self.amplitude = loaded.amplitude;
            self.beta = loaded.beta;
            self.width = loaded.width;
            self.height = loaded.height;
            self.texture = None;
        }
    }

    fn show(&mut self, ctx: &egui::Context) {
        let mut changed = false;

        egui::SidePanel::left("dose_controls").show(ctx, |ui| {
            ui.heading("Dose Parameters");
            ui.separator();

            if ui
                .add(egui::Slider::new(&mut self.amplitude, 0.1..=10.0).text("Source Amplitude"))
                .changed()
            {
                changed = true;
            }

            if ui
                .add(egui::Slider::new(&mut self.beta, 0.1..=5.0).text("Attenuation (Beta)"))
                .changed()
            {
                changed = true;
            }

            ui.separator();
            ui.label("Kernel: Exponential");
            ui.label("K(r) = (A / r^2) * exp(-beta * r)");
            ui.small("Visualizing dose from a point source at (0,0)");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Recompute texture if needed or if params changed
            if self.texture.is_none() || changed {
                self.recompute_texture(ctx);
            }

            if let Some(texture) = &self.texture {
                Plot::new("dose_heatmap")
                    .data_aspect(1.0)
                    .view_aspect(1.0)
                    .show_grid(false)
                    .show_axes([true, true])
                    .show(ui, |plot_ui| {
                        plot_ui.image(PlotImage::new(
                            "dose_field",
                            texture.id(),
                            PlotPoint::new(0.0, 0.0),
                            [20.0, 20.0], // Physical dimensions (-10 to 10)
                        ));
                    });
            }
        });
    }
}

impl DoseCalculationTool {
    fn compute_dose_image(&self) -> ColorImage {
        let width = self.width.max(1);
        let height = self.height.max(1);
        let mut pixels = vec![Color32::BLACK; width * height];

        let kernel = ExponentialKernel::new(self.amplitude, self.beta);
        let range = 20.0;
        let dx = range / width as f64;
        let dy = range / height as f64;
        let max_dose = kernel.value_at(0.1).unwrap_or(10.0);

        for y in 0..height {
            let py = (y as f64 - height as f64 / 2.0) * dy;
            for x in 0..width {
                let px = (x as f64 - width as f64 / 2.0) * dx;
                let r = (px * px + py * py).sqrt();
                let dose = if r < 1e-3 {
                    max_dose
                } else {
                    kernel.value_at(r).unwrap_or(0.0)
                };
                let intensity = (dose / max_dose).clamp(0.0, 1.0);
                let rgb = heatmap_color(intensity);
                pixels[flatten_2d_index(x, y, width)] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
            }
        }

        ColorImage::new([width, height], pixels)
    }

    fn recompute_texture(&mut self, ctx: &egui::Context) {
        let image = self.compute_dose_image();
        self.texture = Some(ctx.load_texture("dose_field", image, TextureOptions::LINEAR));
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        name: "DoseCalculationTool",
        domain: "medical",
        tags: &[],
        build: || Box::new(DoseCalculationTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for DoseCalculationTool {
    fn theory_description(&self) -> String {
        format!(
            "Radiation therapy dose distribution computed via point spread function (PSF) kernel convolution: $D(\\vec{{r}}) = \\int \\Phi(\\vec{{r}}') K(\\vec{{r}} - \\vec{{r}}') d^3\\vec{{r}}'$, using an exponential kernel $K(r) = \\frac{{A}}{{r^2}} e^{{-\\beta r}}$ with amplitude $A = {:.2}$ and attenuation coefficient $\\beta = {:.2}$.",
            self.amplitude, self.beta
        )
    }
    fn phonetic_description(&self) -> String {
        "Radiation therapy dose calculation and convolution kernel theory".into()
    }
    fn theory_citation(&self) -> String {
        "The Physics of Radiation Therapy (Khan, 2014); Convolution/superposition dose calculation (Mackie et al., 1985)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "amplitude".into(),
            "Kernel primary energy fluence scaling factor A".into(),
        );
        map.insert(
            "beta".into(),
            "Radial dose attenuation and scattering decay coefficient beta".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Percentage Depth Dose (PDD)",
    "domain": "medical",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "depth_cm": 10.0,
    "ssd_cm": 100.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "dose_calculation_starter",
        title: "Percentage Depth Dose (PDD)",
        description: "Tissue-air ratio and depth attenuation profile for megavoltage photon radiation dosimetry.",
        category: "Dosimetry",
        icon: "🎯",
        domain: "medical",
        tool_name: Some("DoseCalculationTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
