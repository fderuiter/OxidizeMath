use crate::framework::InteractiveTool;
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
#[serde(deny_unknown_fields)]
struct DoseState {
    amplitude: f64,
    beta: f64,
    width: usize,
    height: usize,
}

impl InteractiveTool for DoseCalculationTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Dose Calculation"
    }

    fn domain(&self) -> &'static str {
        "medical"
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

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let loaded: DoseState =
            serde_json::from_str(state).map_err(|e| format!("Invalid DoseCalculation state: {e}"))?;
        if !loaded.amplitude.is_finite() || !(0.1..=10.0).contains(&loaded.amplitude) {
            return Err(format!("amplitude out of [0.1, 10.0]: {}", loaded.amplitude));
        }
        if !loaded.beta.is_finite() || !(0.1..=5.0).contains(&loaded.beta) {
            return Err(format!("beta out of [0.1, 5.0]: {}", loaded.beta));
        }
        if !(10..=1024).contains(&loaded.width) || !(10..=1024).contains(&loaded.height) {
            return Err("width and height must be in [10, 1024]".to_string());
        }
        self.amplitude = loaded.amplitude;
        self.beta = loaded.beta;
        self.width = loaded.width;
        self.height = loaded.height;
        self.texture = None;
        Ok(())
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
    fn recompute_texture(&mut self, ctx: &egui::Context) {
        let width = self.width;
        let height = self.height;
        let mut pixels = vec![Color32::BLACK; width * height];

        let kernel = ExponentialKernel::new(self.amplitude, self.beta);

        // Physical range: -10.0 to 10.0
        let range = 20.0;
        let dx = range / width as f64;
        let dy = range / height as f64;

        // Find max value for normalization (approx at r -> slightly > 0)
        // Since 1/r^2 diverges at 0, we clamp/saturate.
        // Let's sample at a small radius to get a "max" reference.
        let max_dose = kernel.value_at(0.1).unwrap_or(10.0);

        for y in 0..height {
            let py = (y as f64 - height as f64 / 2.0) * dy;
            for x in 0..width {
                let px = (x as f64 - width as f64 / 2.0) * dx;
                let r = (px * px + py * py).sqrt();

                let dose = if r < 1e-3 {
                    max_dose // Singularity handling
                } else {
                    kernel.value_at(r).unwrap_or(0.0)
                };

                // Map dose to color (Heatmap: Black -> Blue -> Red -> White)
                let intensity = (dose / max_dose).clamp(0.0, 1.0);
                let rgb = heatmap_color(intensity);
                pixels[flatten_2d_index(x, y, width)] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
            }
        }

        let image = ColorImage::new([width, height], pixels);

        self.texture = Some(ctx.load_texture("dose_field", image, TextureOptions::LINEAR));
    }
}

// [cite:graph_parameters_rust]


inventory::submit! {
    crate::framework::ToolMetadata {
        id: "DoseCalculationTool",
        display_name: "Dose Calculation",
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
    fn phonetic_description(&self) -> String { "Radiation therapy dose calculation and convolution kernel theory".into() }
    fn theory_citation(&self) -> String { "The Physics of Radiation Therapy (Khan, 2014); Convolution/superposition dose calculation (Mackie et al., 1985)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("amplitude".into(), "Kernel primary energy fluence scaling factor A".into());
        map.insert("beta".into(), "Radial dose attenuation and scattering decay coefficient beta".into());
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Point-Source Dose Kernel",
    "domain": "medical",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "amplitude": 2.5,
    "beta": 0.6,
    "width": 200,
    "height": 200
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "dose_calculation_starter",
        title: "Point-Source Dose Kernel",
        description: "K(r) = (A/r²)e^(-βr): 2D radial point-spread dose convolution heatmap.",
        category: "Dosimetry",
        icon: "🎯",
        domain: "medical",
        tool_name: Some("DoseCalculationTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
